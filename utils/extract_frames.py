# This file contains the functions and items necessary for extracting frames from videos
import cv2
import pandas as pd
import numpy as np
from utils import split_to_prompts


def extract_frames(video_path, frames_folder, frame_rate=0.5, threshold=60):
    vidcap = cv2.VideoCapture(video_path)
    sec = 0
    count = 0
    success = True
    frames = pd.DataFrame()
    videoId = video_path.split('/')[-1].replace('.mp4', '')
    while success:
        count = count + 1
        vidcap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
        success, image = vidcap.read()
        if success:
            frameId = videoId + '_frame_' + str(count)
            frame_path = frames_folder + '/' + video_path.replace('.mp4', '').split('/')[-1] + "_frame_" + \
                         str(count) + ".jpg"
            split_to_prompts.create_dir_if_not_exists(frames_folder)
            cv2.imwrite(frame_path, image)
            if image.shape == (1280, 720, 3):
                image = cv2.resize(image, (0,0), fx=0.5, fy=0.5)
            img = np.array(image)
            frameShape = img.shape[:2]
            frames = frames.append(pd.Series([frameId, videoId, count, str(frameShape), frame_path]), ignore_index=True)
        sec = sec + frame_rate
        if count > threshold:
            # Setting a threshold at 30 because the videos should be shorter or equal to baseline videos
            success = False
    frames.columns = ['frameId', 'videoId', 'frameNumber', 'frameShape', 's3path']

    return frames
