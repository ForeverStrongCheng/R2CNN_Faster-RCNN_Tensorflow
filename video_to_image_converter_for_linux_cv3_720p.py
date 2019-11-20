# ============================================================================================
# Name        : video_to_image_converter_for_linux_cv3_720p_v1.py
# Author      : Yongqiang Cheng
# Version     : Feb 16, 2016
# Copyright   : Copyright 2016 ForeverStrong License
# Description : Video to Image Converter for Linux in Python, Ansi-style
# Description : Video to Image Converter converts all video frames to images from your video.
# ============================================================================================

import cv2
import os

video_set_dir = "/home/strong/dev_temp/videos/yongqiang_car"
frames_file = "/home/strong/dev_temp/videos/yongqiang_frames_2"

video_set_path = video_set_dir


def video_to_image(video_file, frames_file):
    video_name = video_file.strip()
    video_name_list = video_name.split('/')
    video_images_name = video_name_list[-1].split('.')[0]

    images_file = "%s/%s" % (frames_file, video_images_name)

    if not os.path.exists(images_file):
        os.makedirs(images_file)

    videocapture = cv2.VideoCapture(video_file)

    if not videocapture.isOpened():
        print('Cannot find video file:\n')

    source_fps = float(videocapture.get(cv2.CAP_PROP_FPS))
    interval = int(source_fps * 4)
    # interval = 1

    print(interval)

    frame_size = (
        int(videocapture.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(videocapture.get(cv2.CAP_PROP_FRAME_WIDTH)))

    min_frame_idx = 1
    fps = float(videocapture.get(cv2.CAP_PROP_FPS)) / interval
    max_frame_idx = int(videocapture.get(cv2.CAP_PROP_FRAME_COUNT) / interval)

    print('Frame Rate = ', fps, 'fps')

    for frame_idx in range(max_frame_idx):

        frame_err = 0
        for i in range(interval):
            vflag, frame_image = videocapture.read()
            if not vflag:
                frame_err += 1
        if vflag:
            # dsize = (width, height)
            dsize = (1280, 720)
            if frame_image.shape[1] > 1280:
                frame_image = cv2.resize(frame_image, dsize, interpolation=cv2.INTER_LINEAR)
            # cv2.imwrite(save_path + name, image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

            image_name = "%s/%s_%06d.jpg" % (images_file, video_images_name, int(frame_idx + 1))

            # cv2.imshow(video_images_name, frame_image)
            # cv2.imwrite(image_name, frame_image)
            cv2.imwrite(image_name, frame_image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

            keyboard = cv2.waitKey(5) & 0xFF

            # wait for ESC key to exit
            if keyboard == 27:
                break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    for folderName, subFolders, fileNames in os.walk(video_set_path):
        print("The current folder is " + folderName)

        for subfolder in subFolders:
            print("SUBFOLDER OF " + folderName + ': ' + subfolder)

        for filename in fileNames:
            print("FILE INSIDE " + folderName + ': ' + filename)

            target_filename = filename

            video_file = folderName + '/' + filename

            try:
                video_to_image(video_file, frames_file)
            except Exception as err:
                pass

            print('')
            print(video_file)
