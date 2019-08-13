#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/..')
current_directory = os.path.dirname(os.path.abspath(__file__))

import numpy as np
import tensorflow as tf
import cv2
import time


def short_side_resize_for_inference_data(img_tensor, target_shortside_len, is_resize=True):
    h, w, = tf.shape(img_tensor)[0], tf.shape(img_tensor)[1]

    img_tensor = tf.expand_dims(img_tensor, axis=0)

    if is_resize:
        new_h, new_w = tf.cond(tf.less(h, w),
                               true_fn=lambda: (target_shortside_len, target_shortside_len * w // h),
                               false_fn=lambda: (target_shortside_len * h // w, target_shortside_len))
        img_tensor = tf.image.resize_bilinear(img_tensor, [new_h, new_w], align_corners=True)

    return img_tensor  # [1, h, w, c]


def inference(image_file):
    # preprocess image
    img_placeholder = tf.placeholder(dtype=tf.uint8, shape=[None, None, 3])
    img_batch = tf.cast(img_placeholder, tf.float32)

    resize_image_batch = short_side_resize_for_inference_data(img_tensor=img_batch, target_shortside_len=512,
                                                              is_resize=True)

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        sess.run(init_op)

        img = cv2.imread(image_file)
        height, width, channels = img.shape

        size = (int(height), int(width))

        tmp_directory = current_directory + "/tmp"
        if not os.path.exists(tmp_directory):
            os.makedirs(tmp_directory)

        start = time.time()

        resized_img_batch = sess.run([resize_image_batch], feed_dict={img_placeholder: img})
        resized_img = np.squeeze(resized_img_batch, 0)

        print("tf.shape(img):", sess.run(tf.shape(img)))  # tf.shape(img): [1080 1920    3]
        print("tf.shape(resized_img):", sess.run(tf.shape(resized_img)))  # tf.shape(resized_img): [512 910   3]
        print("img type:", img.dtype)  # img type: uint8
        print("resized_img type:", resized_img.dtype)  # resized_img type: float32

        resized_img = np.asarray(resized_img, dtype='uint8')

        end = time.time()

        cv2.putText(img, text="source image", org=(10, 10), fontFace=1, fontScale=1, color=(0, 0, 255))
        cv2.putText(resized_img, text="resized image", org=(10, 10), fontFace=1, fontScale=1, color=(0, 0, 255))

        # Display the resulting frame
        cv2.imshow("Press ESC on keyboard to exit. 1", img)
        cv2.imshow("Press ESC on keyboard to exit. 2", resized_img)

        k = cv2.waitKey(0)
        if k == 27:  # wait for ESC key to exit
            cv2.destroyAllWindows()
        elif k == ord('s'):  # wait for 's' key to save and exit
            image_name = "%s/%s.jpg" % (tmp_directory, "resized_image")
            cv2.imwrite(image_name, resized_img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

            image_name = "%s/%s.jpg" % (tmp_directory, "source_image")
            cv2.imwrite(image_name, img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

        # When everything done, release the capture
        cv2.destroyAllWindows()


if __name__ == '__main__':
    image_file = "./tmp/000505.jpg"

    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    print("os.environ['CUDA_VISIBLE_DEVICES']:", os.environ['CUDA_VISIBLE_DEVICES'])

    inference(image_file)
