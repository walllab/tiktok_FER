import boto3
import pandas as pd
import os

from sqlalchemy import create_engine
from utils import extract_frames, split_to_prompts
from variables_and_constants import passwords, constants


def main_frames():
    # Get list of all TikTok videos fetched previously
    engine = create_engine("mysql+pymysql://{user}:{pw}@{host}/{db}"
                           .format(user=passwords.MY_SQL_TIKTOK_USERNAME,
                                   pw=passwords.MY_SQL_TIKTOK_PASSWORD,
                                   host=passwords.MY_SQL_TIKTOK_HOST,
                                   db="tiktok"))
    videos = pd.read_sql_table('videos', engine)
    list_of_video_paths = list(videos[videos.challenge == constants.CHALLENGE].s3path.unique())
    if constants.RUNTYPE == 'test':
        list_of_video_paths = list_of_video_paths[:4]
    engine.dispose()

    # For all the videos and for all prompts
    frames = pd.DataFrame()
    frames_folder = constants.CHALLENGE + '/frames'
    split_to_prompts.create_dir_if_not_exists(frames_folder)
    s3 = boto3.client('s3')

    for s3video_path in list_of_video_paths:
        video_path = s3video_path

        # Download video
        split_to_prompts.create_dir_if_not_exists('/'.join(video_path.split('/')[:-1]))
        s3.download_file('tiktok-fer-dataset', s3video_path, video_path)

        # Extract frames
        video_frames = extract_frames.extract_frames(video_path, frames_folder)
        video_frames['challenge'] = constants.CHALLENGE
        frames = frames.append(video_frames, ignore_index=True)

        # Upload frames
        for f in list(video_frames.s3path):
            if constants.RUNTYPE != 'test':
                s3.upload_file(
                    f, constants.BUCKET_NAME, f,
                    ExtraArgs={'Metadata': {'frameId': f.split('/')[-1].replace('.jpeg', ''),
                                            'videoId': video_path.split('/')[-1].replace('.mp4', ''),
                                            'challenge': constants.CHALLENGE}}
                )
            os.remove(f)
        os.remove(video_path)

    engine = create_engine("mysql+pymysql://{user}:{pw}@{host}/{db}"
                           .format(user=passwords.MY_SQL_TIKTOK_USERNAME,
                                   pw=passwords.MY_SQL_TIKTOK_PASSWORD,
                                   host=passwords.MY_SQL_TIKTOK_HOST,
                                   db="tiktok"))
    frames.columns = ['frameId', 'videoId', 'frameNumber', 'frameShape', 's3path', 'challenge']
    if constants.RUNTYPE != 'test':
        frames.to_sql("frames"
                      , con=engine
                      , if_exists='append'
                      , index=False)
    engine.dispose()

