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
import argparse

from data.io.image_preprocess import short_side_resize_for_inference_data
from libs.configs import cfgs
from libs.networks import build_whole_network
from help_utils.tools import *
from libs.box_utils import draw_box_in_img
from help_utils import tools
from libs.box_utils import coordinate_convert


def capture_video_from_camera(device_index):
    cap = cv2.VideoCapture(device_index)

    while (cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()

        if True == ret:
            # Display the resulting frame
            cv2.imshow('Press q on keyboard to exit.', frame)

            # Press q on keyboard to exit.
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


def inference(det_net, device_index):
    # preprocess image
    img_placeholder = tf.placeholder(dtype=tf.uint8, shape=[None, None, 3])
    img_batch = tf.cast(img_placeholder, tf.float32)
    img_batch = img_batch - tf.constant(cfgs.PIXEL_MEAN)

    img_batch = short_side_resize_for_inference_data(img_tensor=img_batch, target_shortside_len=cfgs.IMG_SHORT_SIDE_LEN,
                                                     is_resize=False)

    det_boxes_h, det_scores_h, det_category_h, \
    det_boxes_r, det_scores_r, det_category_r = det_net.build_whole_detection_network(input_img_batch=img_batch,
                                                                                      gtboxes_h_batch=None,
                                                                                      gtboxes_r_batch=None)

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    restorer, restore_ckpt = det_net.get_restorer()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        sess.run(init_op)

        if not restorer is None:
            restorer.restore(sess, restore_ckpt)
            print('restore model')

        cap = cv2.VideoCapture(device_index)

        fps = cap.get(cv2.CAP_PROP_FPS)

        # size = (width, height)
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

        tmp_directory = current_directory + "/../tmp"
        if not os.path.exists(tmp_directory):
            os.makedirs(tmp_directory)

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_writer = cv2.VideoWriter('%s/OBB_camera_face.avi' % (tmp_directory), fourcc, fps, size)

        while (cap.isOpened()):
            # Capture frame-by-frame
            ret, frame = cap.read()

            if True != ret:
                break

            start = time.time()

            # print("tf.shape(frame):", sess.run(tf.shape(frame)))  # tf.shape(frame): [480 640   3]

            resized_img, det_boxes_h_, det_scores_h_, det_category_h_, det_boxes_r_, det_scores_r_, det_category_r_ = \
                sess.run(
                    [img_batch, det_boxes_h, det_scores_h, det_category_h, det_boxes_r, det_scores_r, det_category_r],
                    feed_dict={img_placeholder: frame})

            end = time.time()
            # det_detections_h = draw_box_in_img.draw_box_cv(np.squeeze(resized_img, 0),
            #                                                boxes=det_boxes_h_,
            #                                                labels=det_category_h_,
            #                                                scores=det_scores_h_)
            det_detections_r = draw_box_in_img.draw_rotate_box_cv(np.squeeze(resized_img, 0),
                                                                  boxes=det_boxes_r_,
                                                                  labels=det_category_r_,
                                                                  scores=det_scores_r_)

            # det_detections_h = cv2.resize(det_detections_h,
            #                               (det_detections_h.shape[0] // 2, det_detections_h.shape[1] // 2))
            # cv2.putText(det_detections_h, text="HBB - %3.2fps" % (1 / (end - start)), org=(0, 0), fontFace=3,
            #             fontScale=1, color=(255, 0, 0))
            # det_detections_r = cv2.resize(det_detections_r,
            #                               (det_detections_r.shape[0] // 2, det_detections_r.shape[1] // 2))

            cv2.putText(det_detections_r, text="OBB - %3.2fps" % (1 / (end - start)), org=(10, 10), fontFace=1,
                        fontScale=1, color=(0, 255, 0))

            video_writer.write(det_detections_r)

            # hmerge = np.hstack((det_detections_h, det_detections_r))  # 水平拼接

            # Display the resulting frame
            cv2.imshow("Press q on keyboard to exit.", det_detections_r)

            # Press q on keyboard to exit.
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # When everything done, release the capture
        cap.release()
        video_writer.release()
        cv2.destroyAllWindows()


def parsing_arguments():
    """
    Parsing arguments
    """

    # Creating a parser
    parser = argparse.ArgumentParser(description='Rotational Region CNN - R2CNN')

    # Adding arguments
    parser.add_argument('--gpu', dest='gpu', help='gpu_index', default='0', type=str)
    print("sys.argv:", sys.argv)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    # Parsing arguments
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    device_index = 0
    # capture_video_from_camera(device_index)

    print("sys.argv:", sys.argv)
    args = parsing_arguments()
    print('arguments:', args)
    print('args.gpu:', args.gpu)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    print("os.environ['CUDA_VISIBLE_DEVICES':", os.environ['CUDA_VISIBLE_DEVICES'])

    det_net = build_whole_network.DetectionNetwork(base_network_name=cfgs.NET_NAME, is_training=False)

    inference(det_net, device_index)
