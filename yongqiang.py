#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Yongqiang Cheng

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/..')
current_directory = os.path.dirname(os.path.abspath(__file__))

import numpy as np
# import tensorflow as tf
import cv2
import time


def inference(image_file, current_directory):
    img = cv2.imread(image_file, cv2.IMREAD_COLOR)

    # get dimensions of image
    dimensions = img.shape

    # height, width, number of channels in image
    height = img.shape[0]
    width = img.shape[1]
    channels = img.shape[2]

    print('Image Dimension    : ', dimensions)
    print('Image Height       : ', height)
    print('Image Width        : ', width)
    print('Number of Channels : ', channels)

    # (the center point (mass center) (x,y), (width, height), angle of rotation)
    center = (800, 400)
    size = (300, 600)
    angle = 45
    rect = (center, size, angle)
    print("(the center point (mass center) (x,y), (width, height), angle of rotation):", rect)
    box = cv2.boxPoints(rect)
    print("box = cv2.boxPoints(rect):", box)
    box = np.int0(box)
    print("box = np.int0(box):", box)
    img = cv2.drawContours(img, [box], 0, (255, 0, 0), 2)

    box0 = box[0]
    box1 = box[1]
    box2 = box[2]
    box3 = box[3]

    cv2.rectangle(img, pt1=(center[0] - 6, center[1] - 6), pt2=(center[0] + 6, center[1] + 6), color=(0, 0, 255),
                  thickness=-1)
    cv2.putText(img, text=" mass center: " + str(center), org=center, fontFace=0, fontScale=0.8, thickness=2,
                color=(0, 255, 0))

    cv2.putText(img, text="box0: " + str(box0), org=tuple(box0), fontFace=0, fontScale=0.8, thickness=2,
                color=(0, 255, 0))
    cv2.putText(img, text="box1: " + str(box1), org=tuple(box1), fontFace=0, fontScale=0.8, thickness=2,
                color=(0, 255, 0))
    cv2.putText(img, text="box2: " + str(box2), org=tuple(box2), fontFace=0, fontScale=0.8, thickness=2,
                color=(0, 255, 0))
    cv2.putText(img, text="box3: " + str(box3), org=tuple(box3), fontFace=0, fontScale=0.8, thickness=2,
                color=(0, 255, 0))

    tmp_directory = current_directory + "/tmp"
    if not os.path.exists(tmp_directory):
        os.makedirs(tmp_directory)

    cv2.namedWindow("Press ESC on keyboard to exit.", cv2.WINDOW_NORMAL)

    # Display the resulting frame
    cv2.imshow("Press ESC on keyboard to exit.", img)

    k = cv2.waitKey(0)
    if k == 27:  # wait for ESC key to exit
        pass
    elif k == ord('s'):  # wait for 's' key to save and exit
        image_name = "%s/%s.jpg" % (tmp_directory, "source_image")
        cv2.imwrite(image_name, img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    # When everything done, release the capture
    cv2.destroyAllWindows()


if __name__ == '__main__':
    image_file = "./tmp/000505.jpg"

    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    print("os.environ['CUDA_VISIBLE_DEVICES']:", os.environ['CUDA_VISIBLE_DEVICES'])

    inference(image_file, current_directory)
