import logging
import pandas as pd
import numpy as np
import boto3

#from clustering_algorithms.chinese_whispers import embedding2cosineSimilarityMatrix
from sqlalchemy import create_engine
#from utils import cluster_faces, extract_faces
from variables_and_constants import passwords, constants


def main_individuals():
    # Get list of all TikTok faces fetched previously
    engine = create_engine("mysql+pymysql://{user}:{pw}@{host}/{db}"
                           .format(user=passwords.MY_SQL_TIKTOK_USERNAME,
                                   pw=passwords.MY_SQL_TIKTOK_PASSWORD,
                                   host=passwords.MY_SQL_TIKTOK_HOST,
                                   db="tiktok"))
    faces = pd.read_sql_table('faces', engine)
    faces = faces[faces.challenge == constants.CHALLENGE].copy()


    #####################
    ## Intra-video clustering
    intra_ind_to_faces, intra_faces_to_ind = cluster_faces.intravideo_coordinate_clustering(faces)

    # Export to MySQL
    intra_faces_to_ind_df = pd.DataFrame()
    intra_faces_to_ind_df['faceId'] = pd.Series(list(intra_faces_to_ind.keys()))
    intra_faces_to_ind_df['intravideoIndividualId'] = pd.Series(list(intra_faces_to_ind.values()))
    faces_with_intra_ind = pd.merge(faces, intra_faces_to_ind_df, on='faceId', how='inner')
    faces_with_intra_ind['challenge'] = constants.CHALLENGE

    engine = create_engine("mysql+pymysql://{user}:{pw}@{host}/{db}"
                           .format(user=passwords.MY_SQL_TIKTOK_USERNAME,
                                   pw=passwords.MY_SQL_TIKTOK_PASSWORD,
                                   host=passwords.MY_SQL_TIKTOK_HOST,
                                   db="tiktok"))
    if constants.RUNTYPE != 'test':
        faces_with_intra_ind[['challenge', 'faceId', 'frameId'
            , 'videoId', 'intravideoIndividualId']].to_sql("faces_with_intra_ind"
                                                           , con=engine
                                                           , if_exists='append'
                                                           , index=False)
    #####################

    #####################
    ## Extra-video clustering
    # Fetch tensors
    s3 = boto3.client('s3')
    faces = cluster_faces.get_tensorFace(faces, s3)
    # Load face detectors
    faceEmbedder, faceDetector, cfg, device, colorMode = extract_faces.load_face_detectors()
    # Normalize features
    faces.reset_index(inplace=True)
    faces = cluster_faces.normalize_features(faces, faceEmbedder, device)
    faces_with_intra_ind = pd.merge(faces, faces_with_intra_ind, on='faceId', how='inner')

    # Get similarity matrix
    mean_ind_features = cluster_faces.get_meanFeaturesPerInd(faces_with_intra_ind)
    similarityMatrix = embedding2cosineSimilarityMatrix(mean_ind_features.modeFeat.tolist())
    xmax, xmin = similarityMatrix.max(), similarityMatrix.min()
    similarityMatrix = (similarityMatrix - xmin) / (xmax - xmin)

    # Create similarities table
    list_intra_inds = list(faces_with_intra_ind.intravideoIndividualId.unique())
    n = len(list_intra_inds)
    similarities = pd.DataFrame(0, index=np.arange(int(n * (n - 1) / 2)), columns=['intravideoIndividualId_x',
                                                                                   'intravideoIndividualId_y',
                                                                                   'similarity'])
    index_s = 0
    for c in range(n):
        similarities.iloc[index_s:index_s + (n - 1 - c), 2] = similarityMatrix[c + 1:, c]
        similarities.iloc[index_s:index_s + (n - 1 - c), 0] = list_intra_inds[c]
        similarities.iloc[index_s:index_s + (n - 1 - c), 1] = list_intra_inds[c + 1:]
        index_s += (n - 1 - c)
    similarities['challenge'] = constants.CHALLENGE

    # Export to MySQL
    engine = create_engine("mysql+pymysql://{user}:{pw}@{host}/{db}"
                           .format(user=passwords.MY_SQL_TIKTOK_USERNAME,
                                   pw=passwords.MY_SQL_TIKTOK_PASSWORD,
                                   host=passwords.MY_SQL_TIKTOK_HOST,
                                   db="tiktok"))
    if constants.RUNTYPE != 'test':
        similarities.to_sql("similarities"
                            , con=engine
                            , if_exists='append'
                            , index=False)
    #####################

    #####################
    ## Final step
    faces_with_intra_ind = pd.read_sql_table('faces_with_intra_ind', engine)
    faces_with_intra_ind = faces_with_intra_ind[faces_with_intra_ind.challenge == constants.CHALLENGE].copy()
    similarities = pd.read_sql_table('similarities', engine)
    similarities = similarities[similarities.challenge == constants.CHALLENGE].copy()

    # Match based on similarities
    same_extraind_sim = \
    similarities[similarities.similarity >= constants.EXTRAVIDEO_CLUSTERING[constants.CHALLENGE]['same']][
        ['intravideoIndividualId_x', 'intravideoIndividualId_y']].copy()
    # Adding manual matches from the "incertain" zone
    manuals_to_add = pd.DataFrame(constants.EXTRAVIDEO_CLUSTERING[constants.CHALLENGE]['mid_pairs'])
    manuals_to_add.columns = ['intravideoIndividualId_x', 'intravideoIndividualId_y']
    same_extraind_sim = same_extraind_sim.append(manuals_to_add)
    same_extraind_sim['intravideoIndividualId_x'] = same_extraind_sim.intravideoIndividualId_x.apply(lambda x: int(x))
    same_extraind_sim['intravideoIndividualId_y'] = same_extraind_sim.intravideoIndividualId_y.apply(lambda x: int(x))

    # Manuals to remove
    if constants.EXTRAVIDEO_CLUSTERING[constants.CHALLENGE]['breaks'] != []:
        manuals_to_break = pd.DataFrame(constants.EXTRAVIDEO_CLUSTERING[constants.CHALLENGE]['breaks'])
        manuals_to_break.columns = ['intravideoIndividualId_x', 'intravideoIndividualId_y']
        manuals_to_break['intravideoIndividualId_x'] = manuals_to_break.intravideoIndividualId_x.apply(lambda x: int(x))
        manuals_to_break['intravideoIndividualId_y'] = manuals_to_break.intravideoIndividualId_y.apply(lambda x: int(x))
        manuals_to_break['to_break'] = 1
        same_extraind_sim = pd.merge(same_extraind_sim, manuals_to_break,
                                 on=['intravideoIndividualId_x','intravideoIndividualId_y'],
                                 how='left')
        same_extraind_sim = same_extraind_sim[same_extraind_sim.to_break != 1][['intravideoIndividualId_x',
                                                                            'intravideoIndividualId_y']].copy()
    same_extraind_sim.reset_index(inplace=True)
    same_extraind_sim['challenge'] = constants.CHALLENGE

    engine = create_engine("mysql+pymysql://{user}:{pw}@{host}/{db}"
                           .format(user=passwords.MY_SQL_TIKTOK_USERNAME,
                                   pw=passwords.MY_SQL_TIKTOK_PASSWORD,
                                   host=passwords.MY_SQL_TIKTOK_HOST,
                                   db="tiktok"))
    if constants.RUNTYPE != 'test':
        same_extraind_sim[['intravideoIndividualId_x',
                           'intravideoIndividualId_y',
                           'challenge']].to_sql("intravideo_links"
                                                , con=engine
                                                , if_exists='append'
                                                , index=False)

    # Prep outputs
    extraind_to_intraind = {}
    intraind_to_extraind = {}
    t6 = same_extraind_sim.reset_index().copy()
    ind_ref = 0
    for i in range(t6.shape[0]):
        l1 = int(t6.intravideoIndividualId_x[i])
        l2 = int(t6.intravideoIndividualId_y[i])
        if (l1 not in intraind_to_extraind.keys()) and (l2 not in intraind_to_extraind.keys()):
            extraind_to_intraind[ind_ref] = [l1, l2]
            intraind_to_extraind[l1] = ind_ref
            intraind_to_extraind[l2] = ind_ref
            ind_ref += 1
        elif (l1 in intraind_to_extraind.keys()) and (l2 not in intraind_to_extraind.keys()):
            intraind_to_extraind[l2] = intraind_to_extraind[l1]
            extraind_to_intraind[intraind_to_extraind[l1]].append(l2)
        elif (l1 not in intraind_to_extraind.keys()) and (l2 in intraind_to_extraind.keys()):
            intraind_to_extraind[l1] = intraind_to_extraind[l2]
            extraind_to_intraind[intraind_to_extraind[l2]].append(l1)
        elif (l1 in intraind_to_extraind.keys()) and (l2 in intraind_to_extraind.keys()):
            old_ind_ref = max(intraind_to_extraind[l1], intraind_to_extraind[l2])
            new_ind_ref = min(intraind_to_extraind[l1], intraind_to_extraind[l2])
            if old_ind_ref != new_ind_ref:
                extraind_to_intraind[new_ind_ref] = extraind_to_intraind[new_ind_ref] + extraind_to_intraind[old_ind_ref]
                del (extraind_to_intraind[old_ind_ref])
                for h in extraind_to_intraind[new_ind_ref]:
                    intraind_to_extraind[h] = new_ind_ref

    #####################
    # Making final individual to face association
    finalind_to_face = {}
    face_to_finalind = {}
    faces.reset_index(inplace=True)
    for f in faces.faceId:
        intraind = faces_with_intra_ind[faces_with_intra_ind.faceId == f].reset_index().intravideoIndividualId[0]
        if (intraind in intraind_to_extraind.keys()):
            final_ind = 'extra_' + str(intraind_to_extraind[intraind])
        else:
            final_ind = 'intra_' + str(intraind)
        face_to_finalind[f] = final_ind
        if final_ind in finalind_to_face.keys():
            finalind_to_face[final_ind].append(f)
        else:
            finalind_to_face[final_ind] = [f]

    faces_with_final_ind = pd.DataFrame()
    faces_with_final_ind['faceId'] = pd.Series(list(face_to_finalind.keys()))
    faces_with_final_ind['finalIndividualId'] = pd.Series(list(face_to_finalind.values()))
    faces_with_final_ind = pd.merge(faces, faces_with_final_ind, on='faceId', how='inner')
    faces_with_final_ind['challenge'] = constants.CHALLENGE

    engine = create_engine("mysql+pymysql://{user}:{pw}@{host}/{db}"
                           .format(user=passwords.MY_SQL_TIKTOK_USERNAME,
                                   pw=passwords.MY_SQL_TIKTOK_PASSWORD,
                                   host=passwords.MY_SQL_TIKTOK_HOST,
                                   db="tiktok"))
    if constants.RUNTYPE != 'test':
        faces_with_final_ind[['challenge', 'faceId', 'frameId'
            , 'videoId', 'finalIndividualId']].to_sql("faces_with_final_ind"
                                                           , con=engine
                                                           , if_exists='append'
                                                           , index=False)
    engine.dispose()
    #####################
