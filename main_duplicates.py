import pandas as pd
import boto3
import shutil

from imagededup.methods import PHash
from sqlalchemy import create_engine
from utils import split_to_prompts
from variables_and_constants import passwords, constants


def main_duplicates():
    # Get list of all TikTok faces fetched previously
    engine = create_engine("mysql+pymysql://{user}:{pw}@{host}/{db}"
                           .format(user=passwords.MY_SQL_TIKTOK_USERNAME,
                                   pw=passwords.MY_SQL_TIKTOK_PASSWORD,
                                   host=passwords.MY_SQL_TIKTOK_HOST,
                                   db="tiktok"))
    faces = pd.read_sql_table('faces', engine)
    faces = faces[faces.challenge == constants.CHALLENGE].copy()
    faces_with_final_ind = pd.read_sql_table('faces_with_final_ind', engine)
    faces_with_final_ind = faces_with_final_ind[faces_with_final_ind.challenge == constants.CHALLENGE].copy()

    s3 = boto3.client('s3')
    duplicates = {}
    for ind in faces_with_final_ind.finalIndividualId.unique():
        c = 0
        subset = faces_with_final_ind[faces_with_final_ind.finalIndividualId == ind].copy()
        phasher = PHash()
        for f in subset.faceId.unique():
            path = faces[faces.faceId == f].reset_index().s3path[0]
            if c == 0:
                folder = '/'.join(path.split('/')[:-1])
                split_to_prompts.create_dir_if_not_exists(folder)
                c += 1
            s3.download_file('tiktok-fer-dataset', path, path)
        encodings = phasher.encode_images(image_dir=folder)
        duplicates[ind] = phasher.find_duplicates_to_remove(encoding_map=encodings)
        shutil.rmtree(folder)

    to_remove = [i.replace('.jpg','') for i in sum(duplicates.values(), [])]
    faces['duplicate'] = faces.faceId.apply(lambda x: 1 if x in to_remove else 0)
    unique_faces = faces[faces.duplicate == 0][['faceId', 's3path', 'tensors3path', 'frameId', 'videoId',
                                                'coords', 'noseInside', 'leftEye', 'rightEye', 'nose', 'rightMouthEdge',
                                                'leftMouthEdge', 'challenge']]

    engine = create_engine("mysql+pymysql://{user}:{pw}@{host}/{db}"
                           .format(user=passwords.MY_SQL_TIKTOK_USERNAME,
                                   pw=passwords.MY_SQL_TIKTOK_PASSWORD,
                                   host=passwords.MY_SQL_TIKTOK_HOST,
                                   db="tiktok"))
    if constants.RUNTYPE != 'test':
        unique_faces.to_sql("unique_faces"
                           , con=engine
                           , if_exists='append'
                           , index=False)
    engine.dispose()