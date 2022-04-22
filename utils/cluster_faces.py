#####
# This file contains all the functions to identify individuals inside and between videos
#####
import pandas as pd
import numpy as np
import torch
import os
from jupyterUtils import *
from sklearn.metrics.pairwise import euclidean_distances
from scipy.stats import mode
from utils import split_to_prompts
from variables_and_constants import constants


def Euclidean_Dist(x):
    if x is not np.nan:
        return max(euclidean_distances([x[0], x[1]])[0])
    else:
        return np.nan


def intravideo_coordinate_clustering(faces, threshold=20):
    faces['frameNumber'] = faces.frameId.apply(lambda x: int(x.split('_frame_')[-1]))
    faces['faceNumber'] = faces.faceId.apply(lambda x: int(x.split('_face_')[-1]))

    # Prep outputs
    ind_to_faces = {}
    faces_to_ind = {}
    ind_ref = 0

    # Select only records from videos with more than one individual
    multiple_people_vidids = faces[faces.faceNumber > 0].videoId.unique()
    multiple_people_data = faces[faces.videoId.isin(multiple_people_vidids)].copy()
    single_people_data = faces[~faces.videoId.isin(multiple_people_vidids)].copy()

    # Flagging frames with faces too close to eachother (and thus risk of mismatch)
    if multiple_people_data.shape[0] > 0:
        t0a = multiple_people_data[['faceId', 'frameId', 'videoId', 'coords']].copy()
        t0b = multiple_people_data[['faceId', 'frameId', 'videoId', 'coords']].copy()
        t0 = pd.merge(t0a, t0b, on=['videoId', 'frameId'])
        t0 = t0[(t0.coords_x != t0.coords_y)]
        t0['coords_x_full'] = t0.coords_x.apply(lambda x: eval(x))
        t0['coords_y_full'] = t0.coords_y.apply(lambda x: eval(x))
        t0['coords_x'] = t0.coords_x.apply(lambda x: eval(x)).apply(lambda x: (x[0], x[1]))
        t0['coords_y'] = t0.coords_y.apply(lambda x: eval(x)).apply(lambda x: (x[0], x[1]))
        t0['euclidean_dist'] = t0[['coords_x', 'coords_y']].apply(Euclidean_Dist, axis=1)
        t0 = t0[t0.euclidean_dist < threshold]
        list_of_frames_with_too_close_faces = list(t0.frameId.unique())
        multiple_people_data_filtered = multiple_people_data[
            ~multiple_people_data.frameId.isin(list_of_frames_with_too_close_faces)].copy()
        too_close_people_data = multiple_people_data[
            multiple_people_data.frameId.isin(list_of_frames_with_too_close_faces)].copy()

        # Matching based on distance of coords
        if multiple_people_data_filtered.shape[0] > 0:
            t1 = multiple_people_data_filtered[['faceId', 'frameId', 'videoId', 'frameNumber', 'coords']].copy()
            t1['join_frame'] = t1.frameNumber + 1
            max_faces_per_frame = pd.DataFrame(
                t1.groupby(['videoId', 'frameId', 'frameNumber'])['faceId'].nunique()).reset_index()
            max_faces_per_frame.columns = ['videoId', 'frameId', 'frameNumber', 'maxFaceNumberInFrame']
            t1 = pd.merge(t1, max_faces_per_frame, on=['videoId', 'frameId', 'frameNumber'])[
                ['faceId', 'frameId', 'videoId', 'frameNumber', 'coords', 'maxFaceNumberInFrame', 'join_frame']]
            t2 = multiple_people_data_filtered[['faceId', 'frameId', 'videoId', 'frameNumber', 'coords']].copy()
            t2['join_frame'] = t2.frameNumber
            max_faces_per_frame = pd.DataFrame(
                t2.groupby(['videoId', 'frameId', 'frameNumber'])['faceId'].nunique()).reset_index()
            max_faces_per_frame.columns = ['videoId', 'frameId', 'frameNumber', 'maxFaceNumberInFrame']
            t2 = pd.merge(t2, max_faces_per_frame, on=['videoId', 'frameId', 'frameNumber'])[
                ['faceId', 'frameId', 'videoId', 'frameNumber', 'coords', 'maxFaceNumberInFrame', 'join_frame']]
            t3 = pd.merge(t1, t2, on=['videoId', 'join_frame', 'maxFaceNumberInFrame'], how='inner')[
                ['videoId', 'faceId_x', 'frameId_x', 'coords_x',
                'faceId_y', 'frameId_y', 'coords_y']]
            t3['coords_x_full'] = t3.coords_x.apply(lambda x: eval(x) if x is not np.nan else np.nan)
            t3['coords_y_full'] = t3.coords_y.apply(lambda x: eval(x) if x is not np.nan else np.nan)
            t3['coords_x'] = t3.coords_x_full.apply(lambda x: (x[0], x[1]) if x is not np.nan else np.nan)
            t3['coords_y'] = t3.coords_y_full.apply(lambda x: (x[0], x[1]) if x is not np.nan else np.nan)
            t3['euclidean_dist'] = t3[['coords_x', 'coords_y']].apply(Euclidean_Dist, axis=1)
            t3['join_eucli'] = t3.euclidean_dist
            t4 = t3.groupby(['videoId', 'frameId_x', 'faceId_x', 'coords_x'])['euclidean_dist'].min().reset_index()
            t4.columns = ['videoId', 'frameId_x', 'faceId_x', 'coords_x', 'join_eucli']
            t5 = pd.merge(t3, t4, on=['videoId', 'frameId_x', 'faceId_x', 'coords_x', 'join_eucli']
                          , how='inner', suffixes=('', '_y'))[['videoId', 'frameId_x', 'faceId_x', 'coords_x',
                                                               'frameId_y', 'faceId_y', 'coords_y']]
            t5.drop_duplicates(inplace=True)

            # Creating individualIds based on chain associations
            t6 = t5.copy()
            for i in range(t6.shape[0]):
                l1 = t6.faceId_x[i]
                l2 = t6.faceId_y[i]
                if (l1 not in faces_to_ind.keys()) and (l2 not in faces_to_ind.keys()):
                    ind_to_faces[ind_ref] = [l1, l2]
                    faces_to_ind[l1] = ind_ref
                    faces_to_ind[l2] = ind_ref
                    ind_ref += 1
                elif (l1 in faces_to_ind.keys()) and (l2 not in faces_to_ind.keys()):
                    faces_to_ind[l2] = faces_to_ind[l1]
                    ind_to_faces[faces_to_ind[l1]].append(l2)
                elif (l1 not in faces_to_ind.keys()) and (l2 in faces_to_ind.keys()):
                    faces_to_ind[l1] = faces_to_ind[l2]
                    ind_to_faces[faces_to_ind[l2]].append(l1)
                elif (l1 in faces_to_ind.keys()) and (l2 in faces_to_ind.keys()):
                    old_ind_ref = max(faces_to_ind[l1], faces_to_ind[l2])
                    new_ind_ref = min(faces_to_ind[l1], faces_to_ind[l2])
                    ind_to_faces[new_ind_ref] = ind_to_faces[new_ind_ref] + ind_to_faces[old_ind_ref]
                    del (ind_to_faces[old_ind_ref])
                    for h in ind_to_faces[new_ind_ref]:
                        faces_to_ind[h] = new_ind_ref

        # Adding faces when face numbers change from one frame to another
        faceNumChanges = [f for f in multiple_people_data_filtered.faceId.unique() if f not in faces_to_ind.keys()]
        # Adding faces in frames with too close individuals
        for f in list(faceNumChanges) + list(too_close_people_data.faceId.unique()):
            ind_ref += 1
            faces_to_ind[f] = ind_ref
            ind_to_faces[ind_ref] = [f]

    # Adding faces from videos with only one individual
    for v in list(single_people_data.videoId.unique()):
        ind_ref += 1
        subdata = single_people_data[single_people_data.videoId == v].copy()
        ind_to_faces[ind_ref] = list(subdata.faceId.unique())
        for f in ind_to_faces[ind_ref]:
            faces_to_ind[f] = ind_ref

    return ind_to_faces, faces_to_ind


def get_tensorFace(faces, s3):
    split_to_prompts.create_dir_if_not_exists(constants.CHALLENGE + '/faces/faceTensors')

    tensors = {}
    prev_progress = 0
    progress = 0
    l = len(list(faces.tensors3path))
    print(progress)
    for t in list(faces.tensors3path):
        if int(100*progress/l) != prev_progress:
            print(int(100*progress/l))
            prev_progress = int(100*progress/l)
        s3.download_file('tiktok-fer-dataset', t, t)
        tensors[t] = torch.load(t)
        os.remove(t)
        progress += 1
    faces['tensorFace'] = pd.Series(list(tensors.values()))

    return faces


def normalize_features(faces, faceEmbedder, device):
    flipImage = True
    colorMode = 'gray'
    features = []
    for batch in chunks(faces['tensorFace'], 128):
        if colorMode == 'gray':
            inputs = np.concatenate(batch.tolist()).squeeze()
            inputs = torch.Tensor(inputs).unsqueeze(1).to(device)
        else:
            inputs = np.stack(batch.tolist()).squeeze()
            inputs = torch.Tensor(inputs).to(device)

        if flipImage:
            inputs = torch.cat([inputs, inputs.flip(dims=[3])])
        output = faceEmbedder(inputs)
        #inputs = inputs.detach().cpu().numpy()
        output = output.detach().cpu().numpy()
        if flipImage:
            fe_1 = output[:len(output)//2]
            fe_2 = output[len(output)//2:]
            output = np.hstack((fe_1, fe_2))
        features.append(output)

    features = np.concatenate(features)
    normalizedFeatures = features/np.linalg.norm(features,axis=1)[:,np.newaxis]
    faces['normalizedFeatures'] = normalizedFeatures.tolist()

    return faces


def get_meanFeaturesPerInd(faces_with_intra_ind):

    mode_ind_features = pd.DataFrame()
    for person in faces_with_intra_ind.intravideoIndividualId.unique():
        sub_data = faces_with_intra_ind[faces_with_intra_ind.intravideoIndividualId == person].reset_index()
        feats = sub_data.normalizedFeatures.tolist()
        #modeFeat = mode(np.array(feats)).mode[0]
        meanFeat = np.array(feats).mean(axis=0)
        meanFeat = meanFeat / np.linalg.norm(meanFeat)
        #maxFeat = np.array(feats).max(axis=0)
        #minFeat = np.array(feats).min(axis=0)
        #stdFeat = np.array(feats).std(axis=0)
        #allFeat = meanFeat+maxFeat+minFeat+modeFeat+stdFeat
        #representative_photo = sub_data.alignedFaces[0] #sub_data['faces'][0]
        mode_ind_features = mode_ind_features.append(pd.Series([person, meanFeat]), ignore_index = True)

    mode_ind_features.columns = ['intravideoIndividualId', 'modeFeat']

    return mode_ind_features

