import numpy as np
import pandas as pd
import torch
import time
import pickle
import os
import cv2
from layers.functions.prior_box import PriorBox
from utils.box_utils import decode, decode_landm
from utils.nms.py_cpu_nms import py_cpu_nms
from skimage import exposure
from collections import defaultdict

def getCASIAInfo(filename):
    ##### Load CASIA information for histogram transformation
    with open(filename,'rb') as file:
        casiaData = pickle.load(file)
        temp = []
        for image in casiaData['alignedFaces']:
            temp.append(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY))
        casiaData = np.array(temp)
        casiaHist, bin_edges = np.histogram(casiaData.flatten(),bins=256,density=True)
        casiaCDF = np.cumsum(casiaHist) # is a function from support [0,1] to range [0,1]
        x = np.linspace(0,1,256)
        xp = casiaCDF
        fp = np.linspace(0,1,256)
        finv = np.interp(x, xp, fp)
    return finv

def getLabels(labels_root):
    labelsDict = defaultdict(list)
    for family in os.listdir(labels_root):
        for video in os.listdir('/'.join([labels_root,family])):
            for identity in os.listdir('/'.join([labels_root,family,video])):
                if identity.endswith('.png'):
                    continue
                for imageName in os.listdir('/'.join([labels_root,family,video,identity])):
                    frameNum = imageName.split('_')[0][5:]
                    faceNum = imageName.split('_')[1].split('.')[0][4:]
                    labelsDict['family'].append(family)
                    labelsDict['video'].append(video)
                    labelsDict['frameNum'].append(frameNum)
                    labelsDict['faceNum'].append(faceNum)
                    labelsDict['identity'].append(identity)
                    labelsDict['uniqueID'].append(family+str(frameNum)+str(faceNum)+video)
    labelsDF = pd.DataFrame.from_dict(labelsDict)
    return labelsDF

def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    # print('Missing keys:{}'.format(len(missing_keys)))
    # print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    # print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    # print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    # print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)
    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    # return the list of (x, y)-coordinates
    return list(zip(*coords))

def align_face_new(face, landmarks, box, cfg, processed=False):
    """
    An implementation of the face alignment algorithm in "Face Search at Scale: 80 Million Gallery", 2015
    """
    margin = cfg.MARGIN

    # Extract box info
    startX, startY, endX, endY = box


    # Landmark locations
    if not processed:
        landmarks = landmarks - [[startX, startY]] * 5 + margin
    lEye, rEye, nose, mouthRight, mouthLeft = landmarks
        
    # Create the rotation matrix and rotate the face so that eyes are horizontally aligned
    eyes_center = (lEye + rEye) / 2.0
    dX, dY = lEye - rEye
    angle = np.degrees(np.arctan2(dY, dX)) - 180
    M = cv2.getRotationMatrix2D(tuple(eyes_center), angle, 1)
    face = cv2.warpAffine(face, M, face.shape[1::-1], flags=cv2.INTER_CUBIC)

    # Rotate the landmarks accordingly and extract new necessary ones
    landmarks = M.dot(np.concatenate([landmarks.T,np.ones((1,5))])).T
    lEye, rEye, nose, mouthRight, mouthLeft = landmarks
    center_point = np.mean(landmarks, axis=0)
    eyes_center = (lEye + rEye) / 2.0
    mouth_center = (mouthRight + mouthLeft) / 2.0

    # Perform the final cropping and resizing
    currentSpan = (mouth_center[1] - eyes_center[1]) / 0.4
    currentSpan *= 0.9
    xStart = int(eyes_center[1] - currentSpan * 0.3)
    xEnd = int(mouth_center[1] + currentSpan * 0.3)
    yStart = int(center_point[0] - currentSpan // 2)
    yEnd = int(center_point[0] + currentSpan // 2)
    if xStart < 0: xStart = 0
    if yStart < 0: yStart = 0
    if yEnd < yStart: yEnd = face.shape[0] 
    if xEnd < xStart: xEnd = face.shape[1] 

    face = face[xStart:xEnd, yStart:yEnd]

    try:
        face = cv2.resize(face, (cfg.desiredSize,cfg.desiredSize))
    except:
        print(face.shape, xStart, xEnd, yStart, yEnd)
        return None, None

    return face, landmarks



def align_face(face, landmarks, box, cfg):
    desiredSize, MARGIN, desiredLeftEye = cfg.desiredSize, cfg.MARGIN, cfg.desiredLeftEye

    # Extract box info
    startX, startY, endX, endY = box


    # Left/Right eye locations
    lEye, rEye = landmarks[0:2] - [[startX, startY], [startX, startY]] + MARGIN

    # Calculate angle, current distance between eyes and center
    dX, dY = lEye - rEye
    angle = np.degrees(np.arctan2(dY, dX)) - 180
    dist = np.sqrt((dX ** 2) + (dY ** 2))
    eyesCenter = tuple((lEye + rEye) // 2)

    # Calculate desired right eye location and necessary scaling
    desiredRightEyeX = desiredSize - desiredLeftEye[0]
    desiredDist = desiredRightEyeX - desiredLeftEye[0]
    scale = desiredDist / dist
        
    # Scale adjustment : Accounts for the artificial scaling due to faces seen from profile
    x1, y1 = eyesCenter
    x2, y2 = landmarks[2] - [startX, startY] + MARGIN
    xmax, ymax = face.shape[0:2]
    x1, x2, y1, y2 = ymax-y1, ymax-y2, xmax-x1, xmax-x2

    m = dX/(dY+0.000001)
    m = -1/m
    c = y1 - m*x1
    
    auxDist = abs(m*x2-y2+c)/np.sqrt(m**2+1)
    adjustment = np.exp(- auxDist / (4*dist)) 
    scale *= adjustment 

    # Rotation matrix for affine transform
    M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)
    tX = desiredSize * 0.5
    tY = desiredSize * desiredLeftEye[1] / float(desiredSize)
    M[0, 2] += (tX - eyesCenter[0])
    M[1, 2] += (tY - eyesCenter[1])

    # Apply the affine transformation
    face = cv2.warpAffine(face, M, (desiredSize, desiredSize), flags=cv2.INTER_CUBIC)
    return face


def transform_images(images, newTransform, finv=None):
    if newTransform:
        images = exposure.equalize_hist(images) * 255
        images = images.astype(int)
        images = finv[images]
        images = images * 2 - 1
    else:
        if images.dtype=='uint8':
            images = images.astype(np.float64)
        images -= 127.5
        images /= 127.5
    return images

def process_image(image, mode):
    if mode == 'gray':
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = image[np.newaxis, np.newaxis, :, :]
    else:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = image[np.newaxis, :, :, :]
        image = image.transpose(0,3,1,2)
    image = image.astype(np.float32, copy=False)
    return torch.Tensor(image)

def detect_faces(image, model, cfg, device, prior_data=None):
    # tic = time.time()
    img = np.float32(image)
    im_height, im_width, _ = img.shape
    scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0)
    img = img.to(device)
    scale = scale.to(device)

    
    loc, conf, landms = model(img)  # forward pass
    
    if prior_data is None:
        priorbox = PriorBox(cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(device)
        prior_data = priors.data
    
    boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
    boxes = boxes * scale / cfg['resize']
    boxes = boxes.cpu().numpy()
    scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
    landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
    scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                           img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                           img.shape[3], img.shape[2]])
    scale1 = scale1.to(device)
    landms = landms * scale1 / cfg['resize']
    landms = landms.cpu().numpy()

    # ignore low scores
    inds = np.where(scores > cfg['confidence_threshold'])[0]
    boxes = boxes[inds]
    landms = landms[inds]
    scores = scores[inds]

    # keep top-K before NMS
    order = scores.argsort()[::-1][:cfg['top_k']]
    boxes = boxes[order]
    landms = landms[order]
    scores = scores[order]

    # do NMS
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = py_cpu_nms(dets, cfg['nms_threshold'])
    dets = dets[keep, :]
    landms = landms[keep]

    # keep top-K faster NMS
    dets = dets[:cfg['keep_top_k'], :]
    landms = landms[:cfg['keep_top_k'], :]

    dets = np.concatenate((dets, landms), axis=1)
    # print('net forward time: {:.4f}'.format(time.time() - tic))
    return dets

def get_files(path):
    personDict = dict()
    for person in os.listdir(path):
        if not os.path.isdir('/'.join([path,person])):
            continue
        personDict[person] = ['/'.join([path,person,image]) for image in os.listdir('/'.join([path,person])) if image[0]!='.']
    return personDict


def load_image(img_path):
    image = cv2.imread(img_path, 0)
    if image is None:
        return None
    image = np.dstack((image, np.fliplr(image)))
    image = image.transpose((2, 0, 1))
    image = image[:, np.newaxis, :, :]
    image = image.astype(np.float32, copy=False)
    image -= 127.5
    image /= 127.5
    return image


def get_features(model, personDict, device, colorMode, cfg, batch_size=128):

    persons = personDict.keys()
    personCounts = dict(zip(persons,list(0 for _ in persons)))
    # personImages = dict(zip(persons,list([] for _ in persons)))
    images = []

    print('Loading the library ...',end='\r')
    for person in persons:
        for imagePath in personDict[person]:
            if imagePath[-4:] != '.jpg':
                continue
            image = cv2.imread(imagePath,1)
            if colorMode == 'gray':
                image = process_image(image, 'bgr2gray')
            else:
                image = cv2.resize(image,dsize=(cfg.desiredSize,cfg.desiredSize))
                image = process_image(image, 'color')

            images.append(image)
            personCounts[person] += 1
            # personImages[person].append(image)

    personIndices = dict(zip(np.cumsum(list(personCounts.values())),persons))

    inputs = torch.cat(images).to(device)
    outputs = []
    for batch in range(inputs.shape[0]//batch_size+1):
        output = model(inputs[(batch_size*batch):(batch_size*(batch+1))])
        output = output.detach().cpu().numpy()
        outputs.append(output)
    
    features = np.concatenate(outputs)
    print('Loading the library ...      Done !!')
    return features, personIndices

def pearsonCorrelation(x1, x2):
    return cosin_metric(x1-np.mean(x1), x2-np.mean(x2))

def cosin_metric(x1, x2):
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))

def cosin_metric_batch(x1, x2):
    # return np.squeeze(np.dot(x1, x2)) / (np.linalg.norm(x1,axis=1) * np.linalg.norm(x2))
    x1 = torch.Tensor(x1).to('cuda:0')
    x2 = torch.Tensor(x2).to('cuda:0')

    # return (torch.squeeze(torch.matmul(x1, x2)) / (torch.norm(x1,dim=1) * torch.norm(x2))).detach().cpu().numpy()
    return torch.matmul(x1, x2).cpu().numpy()

def areAllInside(landmarks, box):
    (startX, startY, endX, endY) = box
    return np.all(
        (landmarks[:,0] > startX) & 
        (landmarks[:,0] < endX) &
        (landmarks[:,1] > startY) &
        (landmarks[:,1] < endY)
        )

def isNoseInside(landmarks):
    from scipy.spatial import Delaunay
    nose = landmarks[2,:]
    corners = np.delete(landmarks, 2, axis=0)
    try:
        hull = Delaunay(corners)
        return hull.find_simplex(nose)>=0
    except:
        return False

def areLandmarksInFrame(landmarks, shape, margin):
    # Extract box info
    startX, startY, endX, endY = margin, margin, shape[1]-margin, shape[0]-margin
    # Landmark locations
    # lEye, rEye, nose, mouthRight, mouthLeft = landmarks

    return np.all((landmarks[:,0] > startX) & 
                  (landmarks[:,0] < endX) &
                  (landmarks[:,1] > startY) &
                  (landmarks[:,1] < endY))

def isBlurry(image, threshold=200):
    sharpness = measureSharpness(image)
    return sharpness < threshold

def measureSharpness(image):
    # compute the Laplacian of the image and then return the focus
    # measure, which is simply the variance of the Laplacian
    return cv2.Laplacian(image, cv2.CV_64F).var()

def measureSharpnessMax(image):
    # compute the Laplacian of the image and then return the focus
    # measure, which is simply the variance of the Laplacian
    return cv2.Laplacian(image, cv2.CV_64F).max()
