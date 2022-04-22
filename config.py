class Config(object):
    def __init__(self, mode='gray'):
        self.libraryPath = './data/faceLib'
        self.recognitionThreshold = 0.4
        self.detectionThreshold = 0.9
        self.frameSharpnessThreshold = 100
        self.MARGIN = 32
        # left eye, right eye, nose, right mouth edge, left mouth edge (all in BGR)
        self.LANDMARK_COLORS = [(255,0,0),(255,0,0),(0,255,0),(0,0,255),(0,0,255)] 

        if mode == 'gray':
            self.model_path = 'weights/resnet18_110.pth'
            self.use_se = False  
            self.desiredLeftEye = [41,37] # mean left eye location on LFW dataset
            self.desiredSize = 128     
        else:
            self.model_path = 'weights/model_ir_se50.pth'
            self.desiredSize = 112
            self.desiredLeftEye = [36,33]
            self.net_depth = 50
            self.drop_ratio = 0.6
            self.net_mode = 'ir_se' # or 'ir'

retinaConfig = {
    'trained_model': './weights/Resnet50_Final.pth',
    'nms_threshold': 0.4,
    'keep_top_k': 750,
    'top_k': 5000,
    'cpu': False,
    'confidence_threshold': 0.02,
    'name': 'Resnet50',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 24,
    'ngpu': 4,
    'epoch': 100,
    'decay1': 70,
    'decay2': 90,
    'image_size': 840,
    'pretrain': True,
    'return_layers': {'layer2': 1, 'layer3': 2, 'layer4': 3},
    'in_channel': 256,
    'out_channel': 256,
    'resize': 1
}

