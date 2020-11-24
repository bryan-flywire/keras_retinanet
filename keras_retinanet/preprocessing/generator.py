"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import numpy as np
import random
import threading
import time
import warnings
import math

import keras

from ..utils.image import preprocess_image, preprocess_mono_image, preprocess_gray_image, resize_image, random_transform, resize_and_fill
from ..utils.anchors import anchor_targets_bbox


class Generator(keras.utils.Sequence):
    def __init__(
        self,
        image_data_generator,
        batch_size=1,
        group_method='random',  # one of 'none', 'random', 'ratio'
        shuffle_groups=True,
        image_min_side=1080,
        image_max_side=1920,
        force_aspect_ratio=None,
        seed=None
    ):
        self.image_data_generator = image_data_generator
        self.batch_size           = int(batch_size)
        self.group_method         = group_method
        self.shuffle_groups       = shuffle_groups
        self.image_min_side       = image_min_side
        self.image_max_side       = image_max_side

        if force_aspect_ratio:
            self.force_aspect_ratio   = float(force_aspect_ratio)
        else:
            self.force_aspect_ratio = None

        if seed is None:
            seed = np.uint32((time.time() % 1)) * 1000
        np.random.seed(seed)

        self.group_index = 0
        self.lock        = threading.Lock()

        self.group_images()

    def on_epoch_end(self):
        if self.shuffle_groups:
            random.shuffle(self.groups)

    def size(self):
        raise NotImplementedError('size method not implemented')

    def num_classes(self):
        raise NotImplementedError('num_classes method not implemented')

    def name_to_label(self, name):
        raise NotImplementedError('name_to_label method not implemented')

    def label_to_name(self, label):
        raise NotImplementedError('label_to_name method not implemented')

    def image_aspect_ratio(self, image_index):
        raise NotImplementedError('image_aspect_ratio method not implemented')

    def load_image(self, image_index):
        raise NotImplementedError('load_image method not implemented')

    def load_annotations(self, image_index):
        raise NotImplementedError('load_annotations method not implemented')

    def load_annotations_group(self, group):
        return [self.load_annotations(image_index) for image_index in group]

    def filter_annotations(self, image_group, annotations_group, group):
        # test all annotations
        for index, (image, annotations) in enumerate(zip(image_group, annotations_group)):
            assert(isinstance(annotations, np.ndarray)), '\'load_annotations\' should return a list of numpy arrays, received: {}'.format(type(annotations))

            # test x2 < x1 | y2 < y1 | x1 < 0 | y1 < 0 | x2 <= 0 | y2 <= 0 | x2 >= image.shape[1] | y2 >= image.shape[0]
            invalid_indices = np.where(
                (annotations[:, 2] <= annotations[:, 0]) |
                (annotations[:, 3] <= annotations[:, 1]) |
                (annotations[:, 0] < 0) |
                (annotations[:, 1] < 0) |
                (annotations[:, 2] > image.shape[1]) |
                (annotations[:, 3] > image.shape[0])
            )[0]

            # delete invalid indices
            if len(invalid_indices):
                '''
                warnings.warn('Image with id {} (shape {}) contains the following invalid boxes: {}.'.format(
                    group[index],
                    image.shape,
                    [annotations[invalid_index, :] for invalid_index in invalid_indices]
                ))
                '''
                annotations_group[index] = np.delete(annotations, invalid_indices, axis=0)

        return image_group, annotations_group

    def load_image_group(self, group):
        return [self.load_image(image_index) for image_index in group]

    def force_aspect(self, image):
        """ Given an image; force it to be given aspect ratio prior to resizing """
        img_height = image.shape[0]
        img_width = image.shape[1]
        img_aspect = img_width / img_height
        if img_aspect < self.force_aspect_ratio:
            # This is when the image is boxier than the aspect ratio
            # so we add a black bar at the right side to compensate
            # this added bar does not effect annotation coordinates
            new_img_width = round(img_height * self.force_aspect_ratio)
            image,sf = resize_and_fill(image, (img_height, new_img_width))
        else:
            # This is when the image is narrower than the aspect ratio
            # so we add a black bar at the bottom to compensate
            # this added bar does not effect annotation coordinates
            new_img_height = round(img_width / self.force_aspect_ratio)
            image,sf = resize_and_fill(image, (new_img_height, img_width))
        assert math.isclose(sf[0],1.0) and math.isclose(sf[1],1.0)
        return image
    def resize_image(self, image):
        return resize_image(image, min_side=self.image_min_side, max_side=self.image_max_side)

    def preprocess_image(self, image):
        image_preprocessors = {
            'rgb' : preprocess_image,
            'mono': preprocess_gray_image
        }
        
        try:
            _ = self.image_type
        except AttributeError:
            self.image_type = 'rgb'

        return image_preprocessors[self.image_type](image, mean_image=self.mean_image)

    def preprocess_group_entry(self, image, annotations):
        """ Preprocess image and its annotations.
        """

        # preprocess the image
        image = self.preprocess_image(image)

        # force aspect ratio prior to resizing
        if self.force_aspect_ratio:
            aspect_ratio = self.image_max_side / self.image_min_side
            if not math.isclose(aspect_ratio, self.force_aspect_ratio):
                image = self.force_aspect(image)

        # resize image
        image, image_scale = self.resize_image(image)

        # apply resizing to annotations too
        annotations[:, :4] *= image_scale
        
        # randomly transform the image and annotations
        image, annotations = random_transform(image, annotations, self.image_data_generator)

        # convert to the wanted keras floatx
        image = keras.backend.cast_to_floatx(image)

        return image, annotations

    def preprocess_group(self, image_group, annotations_group):
        """ Preprocess each image and its annotations in its group.
        """
        assert(len(image_group) == len(annotations_group))

        for index in range(len(image_group)):
            # preprocess a single group entry
            image_group[index], annotations_group[index] = self.preprocess_group_entry(image_group[index], annotations_group[index])

        return image_group, annotations_group

    def group_images(self):
        # determine the order of the images
        order = list(range(self.size()))
        if self.group_method == 'random':
            random.shuffle(order)
        elif self.group_method == 'ratio':
            order.sort(key=lambda x: self.image_aspect_ratio(x))

        # divide into groups, one group = one batch
        self.groups = [[order[x % len(order)] for x in range(i, i + self.batch_size)] for i in range(0, len(order), self.batch_size)]

    def compute_inputs(self, image_group):
        # get the max image shape
        max_shape = tuple(max(image.shape[x] for image in image_group) for x in range(3))

        # construct an image batch object
        image_batch = np.zeros((self.batch_size,) + max_shape, dtype=keras.backend.floatx())

        # copy all images to the upper left part of the image batch object
        for image_index, image in enumerate(image_group):
            image_batch[image_index, :image.shape[0], :image.shape[1], :image.shape[2]] = image

        return image_batch

    def anchor_targets(
        self,
        image_shape,
        boxes,
        num_classes,
        mask_shape=None,
        negative_overlap=0.4,
        positive_overlap=0.5,
        **kwargs
    ):
        return anchor_targets_bbox(image_shape, boxes, num_classes, mask_shape, negative_overlap, positive_overlap, **kwargs)

    def compute_targets(self, image_group, annotations_group):
        # get the max image shape
        max_shape = tuple(max(image.shape[x] for image in image_group) for x in range(3))

        # compute labels and regression targets
        labels_group     = [None] * self.batch_size
        regression_group = [None] * self.batch_size
        for index, (image, annotations) in enumerate(zip(image_group, annotations_group)):
            labels_group[index], regression_group[index] = self.anchor_targets(max_shape, annotations, self.num_classes(), mask_shape=image.shape)

            # append anchor states to regression targets (necessary for filtering 'ignore', 'positive' and 'negative' anchors)
            anchor_states           = np.max(labels_group[index], axis=1, keepdims=True)
            regression_group[index] = np.append(regression_group[index], anchor_states, axis=1)

        labels_batch     = np.zeros((self.batch_size,) + labels_group[0].shape, dtype=keras.backend.floatx())
        regression_batch = np.zeros((self.batch_size,) + regression_group[0].shape, dtype=keras.backend.floatx())

        # copy all labels and regression values to the batch blob
        for index, (labels, regression) in enumerate(zip(labels_group, regression_group)):
            labels_batch[index, ...]     = labels
            regression_batch[index, ...] = regression

        return [regression_batch, labels_batch]

    def compute_input_output(self, group):
        # load images and annotations
        image_group       = self.load_image_group(group)
        annotations_group = self.load_annotations_group(group)

        # check validity of annotations
        image_group, annotations_group = self.filter_annotations(image_group, annotations_group, group)

        # perform preprocessing steps
        image_group, annotations_group = self.preprocess_group(image_group, annotations_group)

        # compute network inputs
        inputs = self.compute_inputs(image_group)

        # compute network targets
        targets = self.compute_targets(image_group, annotations_group)

        return inputs, targets

    def __len__(self):
        """
        Number of batches for generator.
        """

        return len(self.groups)

    def __getitem__(self, index):
        """
        Keras sequence method for generating batches.
        """
        group = self.groups[index]
        inputs, targets = self.compute_input_output(group)

        return inputs, targets

