import boto3
import os
import pandas as pd

from utils import fetch_raw_data
from variables_and_constants import constants, passwords
from sqlalchemy import create_engine


def main_videos():
    # Make sure tiktok scrapper is set up
    fetch_raw_data.install_tiktok_scrapper()

    # Download videos from TikTok
    if constants.RUNTYPE == 'test':
        num_downloads = 4
    else:
        num_downloads = 5000
    list_of_paths = fetch_raw_data.download_tiktok_videos_from_music(constants.TIKTOK_MUSIC_IDS[constants.CHALLENGE]
                                                                     , constants.CHALLENGE + '/raw_videos'
                                                                     , number_download=num_downloads)

    # Export summary csv to SQL database
    csv_files = [x for x in os.listdir(constants.CHALLENGE + '/raw_videos') if x[-4:] == ".csv"]
    csv_files.sort()
    engine = create_engine("mysql+pymysql://{user}:{pw}@{host}/{db}"
                           .format(user=passwords.MY_SQL_TIKTOK_USERNAME,
                                   pw=passwords.MY_SQL_TIKTOK_PASSWORD,
                                   host=passwords.MY_SQL_TIKTOK_HOST,
                                   db="tiktok"))
    summary_file_path = constants.CHALLENGE + '/raw_videos/' + csv_files[-1]
    summary_file = pd.read_csv(summary_file_path)
    if constants.RUNTYPE != 'test':
        summary_file.to_sql("tiktok_scrapper_summary_file"
                            , con=engine
                            , if_exists='append'
                            , index=False)
    os.remove(summary_file_path)

    # Cover the trademarked TikTok logo and make sure TikTok handle is not recognizable
    # and upload videos to s3 file
    s3 = boto3.client('s3')
    list_of_paths_blur = []
    videos = pd.DataFrame()
    if constants.RUNTYPE == 'test':
        list_of_paths = list_of_paths[:4]
    for i in list_of_paths:
        if i.split('/')[-1].replace('.mp4', '') in constants.ADD_MUSICALLY[constants.CHALLENGE]:
            add_musically = True
        else:
            add_musically = False
        if i.split('/')[-1].replace('.mp4', '') in constants.ADDITIONAL_FOR_DUET[constants.CHALLENGE]:
            additional_needed_for_duet = True
        else:
            additional_needed_for_duet = False

        r_blur = 90
        i_b = i.replace('raw_videos', 'raw_videos_blur')
        list_of_paths_blur.append(i_b)
        duration = fetch_raw_data.cover_logo(i, i_b, r_blur, add_musically, additional_needed_for_duet)
        if constants.RUNTYPE != 'test':
            s3.upload_file(
                i_b, constants.BUCKET_NAME, i,
                ExtraArgs={'Metadata': {'videoId': i.split('/')[-1].replace('.mp4', ''),
                                        'challenge': i.split('/')[0]}}
            )
        videos = videos.append(pd.Series([i.split('/')[-1].replace('.mp4', ''),
                                          i.split('/')[0],
                                          i,
                                          duration]),
                               ignore_index=True)
        os.remove(i)
        os.remove(i_b)

    # Upload videos table
    videos.columns = ['videoId', 'challenge', 's3path', 'duration']
    if constants.RUNTYPE != 'test':
        videos.to_sql("videos"
                      , con=engine
                      , if_exists='append'
                      , index=False)
    engine.dispose()
