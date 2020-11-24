#!/usr/bin/env python
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

import keras
import keras.preprocessing.image
from keras_retinanet.models.resnet import custom_objects
from keras_retinanet.utils.keras_version import check_keras_version
from keras_retinanet.utils.image import resize_image
from functools import partial
import tensorflow as tf
import numpy as np
import argparse
import time
import sys
import os
import re
import cv2
import json
import pickle
import multiprocessing as mp
import signal
import time

def signal_handler(stop_event, frame_stop_event,signal_received, frame):
    # Handle any cleanup here
    stop_event.set()
    frame_stop_event.set()
    print('SIGINT or CTRL-C detected. Exiting gracefully')
    for i in range(10):
        print(f'Shutting down in {10-i}')
        time.sleep(1)

def format_img(img,mean_image=None):
    #img = img[:, :, (2, 1, 0)]
    img = img.astype(np.float32)
    img, scale = resize_image(img, args.min_side, args.max_side)
    if mean_image is not None:
        mean_image,_ = resize_image(mean_image,args.min_side,args.max_side)
        img[:, :, 0] -= mean_image[:,:,0]
        img[:, :, 1] -= mean_image[:,:,1]
        img[:, :, 2] -= mean_image[:,:,2]
    else:
        img[..., 0] -= 103.939
        img[..., 1] -= 116.779
        img[..., 2] -= 123.68
    return img

def read_frames(img_path, raw_frame_queue, stop_event):
    ok = True
    vid = cv2.VideoCapture(img_path)
    if cv2.__version__ >= "3.2.0":
        vid_len = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = vid.get(cv2.CAP_PROP_FPS)
    else:
        vid_len = int(vid.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
        fps = vid.get(cv2.cv.CV_CAP_PROP_FPS)
    frame_num = 0
    while ok and not stop_event.is_set():
        if not raw_frame_queue.full():
            ok,img = vid.read()
            if ok:
                raw_frame_queue.put((img, frame_num))
                frame_num += 1
    stop_event.set()
    print("This thread should exit now read frames")
    return

def enqueue_frames(raw_frame_queue, processed_frame_queue, stop_event, mean_image):
    while not stop_event.is_set():
        try:
            img,frame_num = raw_frame_queue.get(timeout=5)
            X = format_img(img,mean_image)
            processed_frame_queue.put((X,frame_num))
        except:
            stop_event.set()
            print("This thread should exit now")
    return

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


def parse_args():
    parser = argparse.ArgumentParser(
            description='Testing script for testing video data.')
    parser.add_argument('model', help='Path to RetinaNet model.')
    parser.add_argument('video_path', 
            help='Path to video file')
    parser.add_argument('--gpu', 
            help='Id of the GPU to use (as reported by nvidia-smi).')
    parser.add_argument('--mean_image',
            help='Path to mean image to subtract',
            default=None)
    parser.add_argument('--score-threshold', 
            help='Threshold to filter detections', 
            default=0.7, 
            type=float)
    parser.add_argument('--min_side', 
            help='Image min side', 
            default=720, 
            type=int)
    parser.add_argument('--max_side', 
            help='Image max side', 
            default=1280, 
            type=int)
    parser.add_argument('--scale', 
            help='Image resized scale to original', 
            default = 0.666666, 
            type=float)

    return parser.parse_args()

if __name__ == '__main__':
    # parse arguments
    args = parse_args()
    if args.mean_image is not None:
        mean_image = np.load(args.mean_image)
    else:
        mean_image = None
    # make sure keras is the minimum required version
    check_keras_version()

    # optionally choose specific GPU
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    keras.backend.tensorflow_backend.set_session(get_session())

    # create the model
    print('Loading model, this may take a second...')
    model = keras.models.load_model(args.model, custom_objects=custom_objects)

    raw_queue = mp.Queue(50)
    frame_queue = mp.Queue(50)
    stop_event = mp.Event()
    frame_stop_event = mp.Event()
    signal.signal(
        signal.SIGINT, 
        partial(signal_handler, stop_event, frame_stop_event))
    p = mp.Process(target=read_frames, args=(args.video_path,raw_queue,stop_event))
    p.daemon = True
    p.start()
    for num in range(8):
        p = mp.Process(target=enqueue_frames, args=(raw_queue, frame_queue, frame_stop_event, mean_image))
        p.daemon = True
        p.start()
    id_ = 1
    results = []

    while True:
        st = time.time()
        try:
            image,frame = frame_queue.get(timeout=5)
        except:
            print("timed out")
            break
        print('Elapsed time pre-process = {}'.format(time.time() - st))

        _, _, detections = model.predict_on_batch(np.expand_dims(image, axis=0))

        print('Elapsed time model = {}'.format(time.time() - st))
        # clip to image shape
        detections[:, :, 0] = np.maximum(0, detections[:, :, 0])
        detections[:, :, 1] = np.maximum(0, detections[:, :, 1])
        detections[:, :, 2] = np.minimum(image.shape[1], detections[:, :, 2])
        detections[:, :, 3] = np.minimum(image.shape[0], detections[:, :, 3])

        # correct boxes for image scale
        scale = args.scale
        detections[0, :, :4] /= scale

        # change to (x, y, w, h) (MS COCO standard)
        detections[:, :, 2] -= detections[:, :, 0]
        detections[:, :, 3] -= detections[:, :, 1]

        # compute predicted labels and scores
        for detection in detections[0, ...]:
            label = int(detection[4])
            if float(detection[4 + label]) > args.score_threshold:
                image_result = {
                    'frame'       : frame,
                    'category_id' : label,
                    'scores'      : [float(det) for i,det in
                                      enumerate(detection) if i >=5],
                    'bbox'        : (detection[:4]).tolist(),
                }
                # append detection to results
                results.append(image_result)

    if len(results):
        # write output
        out_name = re.split(".mov",args.video_path.split('/')[-1],flags=re.IGNORECASE)[0]
        print(out_name)
        try:
            #json.dump(results, open('{}_bbox_results.json'.format(out_name), 'w'), indent=4)
            pickle.dump(results,open('{}_bbox_results_{}.pickle'.format(out_name, args.score_threshold),'wb'))
        except:
            pickle.dump(results,open('default_bbox_results.pickle','wb'))
            #json.dump(results, open('default_bbox_results.json', 'w'), indent=4)

    frame_stop_event.set()
    stop_event.set()
    print("Finished")
    sys.exit()
