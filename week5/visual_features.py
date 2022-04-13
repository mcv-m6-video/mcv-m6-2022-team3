import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch
import numpy as np
import math
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import scipy.io
import yaml
from tqdm import tqdm
from feat_extraction.reID.model import ft_net
import cv2

class ColorHistogramFeatures:
    # Color histogram as feature vector
    def __init__(self):
        pass

class DeepFeatures:
    # ResNet or whatever deep feature
    def __init__(self):
        pass
    
class ReIDNetwork:
    # Metric-learning based network: https://github.com/layumi/AICIty-reID-2020
    def __init__(self):
        model = ft_net(31605, 0, 1, None, 'avg')
        if os.path.exists("feat_extraction/reID/net_last.pth"):
            last_state_dict = torch.load("feat_extraction/reID/net_last.pth")
        else:
            last_state_dict = torch.load("week5/feat_extraction/reID/net_last.pth")
        last_state_dict = {".".join(k.split(".")[1:]):v for k,v in last_state_dict.items()}
        model.load_state_dict(last_state_dict)
        model.classifier.classifier = torch.nn.Sequential()
        self.model = model
        self.model.eval()
        
    def extract_features(self, car_patch):
        with torch.no_grad():
            to_tensor_t = torchvision.transforms.ToTensor()
            car_patch = cv2.resize(car_patch, (256, 256))
            car_patch_tensor = to_tensor_t(car_patch).unsqueeze(0)
            ff = self.model(car_patch_tensor)
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
            ff = ff.div(fnorm.expand_as(ff)) # L2 normalization
            return ff
    
    def return_similarity(self, feat1, feat2):
        return np.dot(feat1, feat2)