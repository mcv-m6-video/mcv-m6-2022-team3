import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import numpy as np
import math
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import scipy.io
import yaml
from tqdm import tqdm
from feature_extraction.reID.pretrained_model.model import PCB

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
        """ """
        model_structure = PCB(751)
        #self.model = load_network(model_structure)