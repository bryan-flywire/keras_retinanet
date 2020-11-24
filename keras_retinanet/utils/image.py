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

from __future__ import division
from PIL import Image
import time
import keras
import numpy as np
import cv2


def read_image_bgr(path):
    image = np.asarray(Image.open(path).convert('RGB'))
    return image[:, :, ::-1].copy()

def read_image_as_mono(path,channel=0):
    image = np.asarray(Image.open(path).convert('RGB'))
    return np.expand_dims(image[:,:,channel].copy(),axis=-1)

def read_image_as_grayscale(path):
    image = np.asarray(Image.open(path).convert('L'))
    return np.expand_dims(image.copy(), axis=-1)

def preprocess_gray_image(x, mean_image=None):
    x = x.astype(keras.backend.floatx())
    # image size check
    if keras.backend.image_data_format() == 'channels_first':
        if x.ndim == 3:
            if mean_image is not None:
                if not mean_image.shape == x.shape:
                    mean_image,_ = resize_image(mean_image, x.shape[1], x.shape[2])
                mean_image = mean_image[::-1]
                x[0, :, :] -= mean_image[0, :, :]
            else:
                x[0, :, :] -= 67.637 # MBARI specific number
        else:
            if mean_image is not None:
                if not mean_image.shape == x.shape:
                    mean_image,_ = resize_image(mean_image, x.shape[1], x.shape[2])
                x[:, 0, :, :] -= mean_image[0, :, :]
            else:
                x[:, 0, :, :] -= 67.637 # MBARI specific number
    else:
        if mean_image is not None:
            if not mean_image.shape == x.shape:
                mean_image,_ = resize_image(mean_image, x.shape[0], x.shape[1])
            x[..., 0] -= mean_image[:, :, 0]
        else:
            x[..., 0] -= 67.637 # MBARI specific number

    return x
    

def preprocess_mono_image(x, mean_image=None):
    x = x.astype(keras.backend.floatx())
    # image size check
    if keras.backend.image_data_format() == 'channels_first':
        if x.ndim == 3:
            if mean_image is not None:
                if not mean_image.shape == x.shape:
                    mean_image,_ = resize_image(mean_image, x.shape[1], x.shape[2])
                mean_image = mean_image[::-1]
                x[0, :, :] -= mean_image[0, :, :]
            else:
                x[0, :, :] -= 123.68 # Corresponds to red channel for ImageNet mean
        else:
            if mean_image is not None:
                if not mean_image.shape == x.shape:
                    mean_image,_ = resize_image(mean_image, x.shape[1], x.shape[2])
                x[:, 0, :, :] -= mean_image[0, :, :]
            else:
                x[:, 0, :, :] -= 123.68 # Corresponds to red channel for ImageNet mean
    else:
        if mean_image is not None:
            if not mean_image.shape == x.shape:
                mean_image,_ = resize_image(mean_image, x.shape[0], x.shape[1])
            x[..., 0] -= mean_image[:, :, 0]
        else:
            x[..., 0] -= 123.68 # Corresponds to red channel for ImageNet mean

    return x
    
def preprocess_image(x, mean_image=None):
    # mostly identical to "https://github.com/fchollet/keras/blob/master/keras/applications/imagenet_utils.py"
    # except for converting RGB -> BGR since we assume BGR already
    x = x.astype(keras.backend.floatx())
    # image size check
    if keras.backend.image_data_format() == 'channels_first':
        if x.ndim == 3:
            if mean_image is not None:
                if not mean_image.shape == x.shape:
                    mean_image,_ = resize_image(mean_image, x.shape[1], x.shape[2])
                mean_image = mean_image[::-1]
                x[0, :, :] -= mean_image[0, :, :]
                x[1, :, :] -= mean_image[1, :, :]
                x[2, :, :] -= mean_image[2, :, :]
            else:
                x[0, :, :] -= 103.939
                x[1, :, :] -= 116.779
                x[2, :, :] -= 123.68
        else:
            if mean_image is not None:
                if not mean_image.shape == x.shape:
                    mean_image,_ = resize_image(mean_image, x.shape[1], x.shape[2])
                mean_image = mean_image[::-1]
                x[:, 0, :, :] -= mean_image[0, :, :]
                x[:, 1, :, :] -= mean_image[1, :, :]
                x[:, 2, :, :] -= mean_image[2, :, :]
            else:
                x[:, 0, :, :] -= 103.939
                x[:, 1, :, :] -= 116.779
                x[:, 2, :, :] -= 123.68
    else:
        if mean_image is not None:
            if not mean_image.shape == x.shape:
                mean_image,_ = resize_image(mean_image, x.shape[0], x.shape[1])
            x[..., 0] -= mean_image[:, :, 0]
            x[..., 1] -= mean_image[:, :, 1]
            x[..., 2] -= mean_image[:, :, 2]
        else:
            x[..., 0] -= 103.939
            x[..., 1] -= 116.779
            x[..., 2] -= 123.68

    return x


def random_transform(
    image,
    boxes,
    image_data_generator,
    seed=None
):
    if seed is None:
        seed = np.random.randint(10000)

    image = image_data_generator.random_transform(image, seed=seed)

    # set fill mode so that masks are not enlarged
    fill_mode = image_data_generator.fill_mode
    image_data_generator.fill_mode = 'constant'
    invalid_boxes = []

    for index in range(boxes.shape[0]):
        # generate box mask and randomly transform it
        mask = np.zeros_like(image, dtype=np.uint8)
        b = boxes[index, :4].astype(int)

        if b[0] >= b[2] or b[1] >= b[3]:
            invalid_boxes.append(index)
            continue

        mask[b[1]:b[3], b[0]:b[2], :] = 255
        mask = image_data_generator.random_transform(mask, seed=seed)[..., 0]
        mask = mask.copy()  # to force contiguous arrays

        # find bounding box again in augmented image
        [i, j] = np.where(mask == 255)
        if (len(i)== 0) or (len(j) == 0):
           # print('Annotation transformed outside of new image: {}'.format(b))
           invalid_boxes.append(index)
           continue
        boxes[index, 0] = float(min(j))
        boxes[index, 1] = float(min(i))
        boxes[index, 2] = float(max(j)) + 1  # set box to an open interval [min, max)
        boxes[index, 3] = float(max(i)) + 1  # set box to an open interval [min, max)

    invalid_boxes = sorted(invalid_boxes,reverse=True)
    boxes = np.delete(boxes,invalid_boxes,axis=0)

    # restore fill_mode
    image_data_generator.fill_mode = fill_mode

    return image, boxes


def resize_image(img, min_side=600, max_side=1024):
    (rows, cols, _) = img.shape
    smallest_side = min(rows, cols)

    # rescale the image so the smallest side is min_side
    scale = min_side / smallest_side

    # check if the largest side is now greater than max_side, wich can happen
    # when images have a large aspect ratio
    largest_side = max(rows, cols)
    if largest_side * scale > max_side:
        scale = max_side / largest_side
    # resize the image with the computed scale
    img = cv2.resize(img, None, fx=scale, fy=scale)
    if len(img.shape) < 3:
        img = np.expand_dims(img,axis=-1)
    #img = cv2.resize(img, (max_side,min_side))
    return img, scale

def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])                     
    # compute the area of intersection rectangle
    interArea = (xB - xA + 1) * (yB - yA + 1)

    # compute the area of both the prediction and ground-truth rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection area and dividing it by the sum of prediction + ground-truth areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou

def resize_and_fill(image, desired_shape):
    """
    Resize an image to a desired shape (height,width) and maintaining
    the aspect ratio. Fill any extra areas with black.
    :param image: ndarray of the image
    :param desired_shape: tuple describing the output shape

    :returns image_resized,scaleFactor:
    image_resized is ndarray represented the scaled + padded image
    scaleFactor Given a pixel coordinate in the original this factor should
    be applied to land on the new image.
    """
    image_height=image.shape[0]
    image_width=image.shape[1]
    image_channels=image.shape[2]

    desired_width=desired_shape[1]
    desired_height=desired_shape[0]
    growth_factor=min(desired_height/image_height,
                           desired_width/image_width)
    new_height=int(image_height*growth_factor)
    new_width=int(image_width*growth_factor)
    image_resized,_=resize_image(image,new_height,new_width)
    added_rows=desired_height-new_height
    added_cols=desired_width-new_width

    if added_rows:
        black_bar=np.zeros((added_rows, new_width, image_channels))
        image_resized = np.append(image_resized,black_bar, axis=0)

    if added_cols:
        black_bar=np.zeros((new_height, added_cols, image_channels))
        image_resized = np.append(image_resized,black_bar, axis=1)

    scaleFactor=(float(new_height)/image_height,float(new_width)/image_width)
    return image_resized, scaleFactor
