import boto3
import pandas as pd
import os

from sqlalchemy import create_engine
from utils import split_to_prompts, extract_faces
from variables_and_constants import passwords, constants


def main_faces():
    # Get list of all TikTok frames fetched previously
    engine = create_engine("mysql+pymysql://{user}:{pw}@{host}/{db}"
                           .format(user=passwords.MY_SQL_TIKTOK_USERNAME,
                                   pw=passwords.MY_SQL_TIKTOK_PASSWORD,
                                   host=passwords.MY_SQL_TIKTOK_HOST,
                                   db="tiktok"))
    frames = pd.read_sql_table('frames', engine)
    list_of_frames_paths = list(frames[frames.challenge == constants.CHALLENGE].s3path.unique())
    if constants.RUNTYPE == 'test':
        list_of_frames_paths = list_of_frames_paths[:4]
    frames_folder = constants.CHALLENGE + '/frames'
    split_to_prompts.create_dir_if_not_exists(frames_folder)

    # Creating faces folders
    faces = pd.DataFrame()
    nofaces = pd.DataFrame()
    faces_folder = constants.CHALLENGE + '/faces'
    split_to_prompts.create_dir_if_not_exists(faces_folder)
    alignedFaces_outputPath = constants.CHALLENGE + '/faces/alignedFaces'
    split_to_prompts.create_dir_if_not_exists(alignedFaces_outputPath)
    faceCoords_outputPath = constants.CHALLENGE + '/faces/faceTensors'
    split_to_prompts.create_dir_if_not_exists(faceCoords_outputPath)

    # Load face detectors
    faceEmbedder, faceDetector, cfg, device, colorMode = extract_faces.load_face_detectors()

    # For all the frames
    s3 = boto3.client('s3')
    for s3frame_path in list_of_frames_paths:
        frame_path = s3frame_path

        # Download frame
        s3.download_file('tiktok-fer-dataset', s3frame_path, frame_path)

        # Extract faces
        faces_extracted = extract_faces.extract_faces(frame_path, alignedFaces_outputPath, faceCoords_outputPath,
                                                      faceDetector, cfg, device, colorMode)
        if faces_extracted.shape[1] == 12:
            faces_extracted['challenge'] = constants.CHALLENGE

            # Upload faces
            for i, f in enumerate(list(faces_extracted.s3path)):
                t = list(faces_extracted.tensors3path)[i]
                if constants.RUNTYPE != 'test':
                    faceId = f.split('/')[-1].replace('.jpg', '')
                    faceNumber = faceId.split('_face_')[-1]
                    frameId = faceId.split('_face_')[0]
                    frameNumber = frameId.split('_frame_')[-1]
                    videoId = frameId.split('_frame_')[0]
                    s3.upload_file(
                        f, constants.BUCKET_NAME, f,
                        ExtraArgs={'Metadata': {'faceId': faceId,
                                                'faceNumber': faceNumber,
                                                'frameId': frameId,
                                                'frameNumber': frameNumber,
                                                'videoId': videoId,
                                                'challenge': constants.CHALLENGE}}
                    )
                    s3.upload_file(
                        t, constants.BUCKET_NAME, t,
                        ExtraArgs={'Metadata': {'faceId': faceId,
                                                'faceNumber': faceNumber,
                                                'frameId': frameId,
                                                'frameNumber': frameNumber,
                                                'videoId': videoId,
                                                'challenge': constants.CHALLENGE}}
                    )
                os.remove(f)
                os.remove(t)
            faces = faces.append(faces_extracted, ignore_index=True)
        else:
            frameId = frame_path.split('/')[-1].replace('.jpg', '')
            frameNumber = int(frameId.split('_frame_')[-1])
            videoId = frameId.split('_frame_')[0]
            nofaces = nofaces.append(pd.Series([frameId, frameNumber, videoId, constants.CHALLENGE])
                                     , ignore_index=True)

        os.remove(frame_path)

    engine = create_engine("mysql+pymysql://{user}:{pw}@{host}/{db}"
                           .format(user=passwords.MY_SQL_TIKTOK_USERNAME,
                                   pw=passwords.MY_SQL_TIKTOK_PASSWORD,
                                   host=passwords.MY_SQL_TIKTOK_HOST,
                                   db="tiktok"))
    if nofaces.shape[1] == 4:
        nofaces.columns = ['frameId', 'frameNumber', 'videoId', 'challenge']
    if constants.RUNTYPE != 'test':
        faces.to_sql("faces"
                     , con=engine
                     , if_exists='append'
                     , index=False)
        nofaces.to_sql("nofaces"
                     , con=engine
                     , if_exists='append'
                     , index=False)
    engine.dispose()
