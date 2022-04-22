import boto3
import pandas as pd
import os

from sqlalchemy import create_engine
from utils import split_to_prompts
from variables_and_constants import constants, passwords


def main_subvideos():
    # Download baseline video
    s3 = boto3.client('s3')
    split_to_prompts.create_dir_if_not_exists(constants.CHALLENGE + '/raw_videos/')
    baseline_video_path = constants.CHALLENGE + '/raw_videos/' + constants.BASELINE[constants.CHALLENGE] + '.mp4'
    s3.download_file('tiktok-fer-dataset', baseline_video_path, baseline_video_path)

    # Split baseline video based on prompts and extract subvideo baseline audios
    dict_of_prompt_timestamps = constants.BASELINE_PROMPT_TIMESTAMPS[constants.CHALLENGE]
    output_folder = constants.CHALLENGE + '/baseline_audios'
    baseline_audios = split_to_prompts.split_based_on_timestamp(baseline_video_path
                                                                , dict_of_prompt_timestamps
                                                                , output_folder
                                                                , element_type='audio')

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
    # output_folder = constants.CHALLENGE + '/split_videos'
    output_audio_folder = constants.CHALLENGE + '/audios'
    subvideos = pd.DataFrame()
    for s3video_path in list_of_video_paths:
        video_path = s3video_path

        # Download video
        s3.download_file('tiktok-fer-dataset', s3video_path, video_path)

        # Extract audio
        audio_file_path = split_to_prompts.extract_audio(video_path, output_audio_folder)

        # Detect audio prompt in audio of downloaded video
        dict_of_prompt_timestamps = {}
        for prompt in constants.BASELINE_PROMPT_TIMESTAMPS[constants.CHALLENGE]:
            baseline_audio_file_path = [i for i in baseline_audios if '/' + prompt + '/' in i][0]
            timeframe = split_to_prompts.check_subaudio(baseline_audio_file_path, audio_file_path, threshold=1)
            if ('face_challenge' == constants.CHALLENGE) and (
                    constants.BASELINE[constants.CHALLENGE] not in video_path):
                # Add reaction time for face challenge (not needed for baseline video)
                timeframe = [timeframe[0] + 1000, timeframe[0] + 2500]
            dict_of_prompt_timestamps[prompt] = timeframe

            subvideos = subvideos.append(pd.Series([video_path.split('/')[-1].replace('.mp4', '') + '_' + prompt,
                                                    video_path.split('/')[-1].replace('.mp4', ''),
                                                    constants.CHALLENGE,
                                                    prompt,
                                                    int(timeframe[0]),
                                                    int(timeframe[1])]), ignore_index=True)

        os.remove(video_path)
        os.remove(audio_file_path)

        # Commenting out the actual split of videos in order to only have the timestamps in a SQL table
        # split_videos = split_to_prompts.split_based_on_timestamp(video_path
        #                                                         , dict_of_prompt_timestamps
        #                                                         , output_folder
        #                                                         , element_type='video')

    for item in baseline_audios:
        os.remove(item)

    # Upload subvideos table
    engine = create_engine("mysql+pymysql://{user}:{pw}@{host}/{db}"
                           .format(user=passwords.MY_SQL_TIKTOK_USERNAME,
                                   pw=passwords.MY_SQL_TIKTOK_PASSWORD,
                                   host=passwords.MY_SQL_TIKTOK_HOST,
                                   db="tiktok"))
    subvideos.columns = ['subvideoId', 'videoId', 'challenge', 'prompt', 'startTime', 'endTime']
    if constants.RUNTYPE != 'test':
        subvideos.to_sql("subvideos"
                         , con=engine
                         , if_exists='append'
                         , index=False)
    engine.dispose()

