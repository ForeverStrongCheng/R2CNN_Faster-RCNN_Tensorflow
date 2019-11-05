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

print(16 * "++--")
print("current_directory:", current_directory)

x = np.arange(9).reshape(1, 3, 3)

print('Array X:')
print(x)
print('\n')
y = np.squeeze(x)

print('Array Y:')
print(y)
print('\n')

print('The shapes of X and Y array:')
print(x.shape, y.shape)

'''
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

    print('Original Dimensions : ', img.shape)

    scale_percent = 150  # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dsize = (width, height)

    # resize image: dsize = (width, height)
    resized_img = cv2.resize(img, dsize, interpolation=cv2.INTER_AREA)

    print('Resized Dimensions : ', resized_img.shape)

    tmp_directory = current_directory + "/tmp"
    if not os.path.exists(tmp_directory):
        os.makedirs(tmp_directory)

    cv2.namedWindow("Press ESC on keyboard to exit.", cv2.WINDOW_NORMAL)

    # Display the resulting frame
    cv2.imshow("Press ESC on keyboard to exit.", resized_img)

    k = cv2.waitKey(0)
    if k == 27:  # wait for ESC key to exit
        pass
    elif k == ord('s'):  # wait for 's' key to save and exit
        image_name = "%s/%s.jpg" % (tmp_directory, "source_image")
        cv2.imwrite(image_name, resized_img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    # When everything done, release the capture
    cv2.destroyAllWindows()


if __name__ == '__main__':
    image_file = "./tmp/000505.jpg"

    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    print("os.environ['CUDA_VISIBLE_DEVICES']:", os.environ['CUDA_VISIBLE_DEVICES'])

    inference(image_file, current_directory)
'''
