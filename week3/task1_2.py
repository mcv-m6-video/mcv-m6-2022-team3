from genericpath import exists
import os
import time
from collections import defaultdict

import numpy as np
import cv2
import imageio

from tkinter import E
import cv2
from cv2 import cvtColor
import numpy as np
from argparse import ArgumentParser
from utils import read_annotations, image_to_tensor
from evaluation import show_annotations_and_predictions, voc_eval
import os
from tqdm import tqdm
from torchvision.ops import nms
from datasets import create_dataloaders
import torchvision

import torch
from models import load_model, train, evaluate

WAIT_TIME = 1
SAVE = False
CAR_LABEL_NUM = 3
FRAME_25PER = 510
EXPERIMENTS_FOLDER = "experiments"

def task1_2(architecture_name, video_path, annotations, run_name, finetune, train_model=False, use_gpu=True):
    """
    Finetune object detector
    """
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    transformations = torchvision.transforms.ToTensor()
    train_idxs, test_idxs = np.arange(0, FRAME_25PER), np.arange(FRAME_25PER, FRAME_25PER*4)
    train_loader, test_loader = create_dataloaders(annotations, video_path, train_idxs, test_idxs, transformations)
    model, device = load_model(architecture_name, use_gpu, finetune=finetune)

    model_folder_files = os.path.join(EXPERIMENTS_FOLDER, run_name)

    if not os.path.exists(EXPERIMENTS_FOLDER):
        os.mkdir(EXPERIMENTS_FOLDER)
    if not os.path.exists(model_folder_files):
        os.mkdir(model_folder_files)
    
    # if finetune:
    #     if train_model:
    #         train(model, train_loader, test_loader, device, num_epochs=10, save_path=model_folder_files)
    #         print("Training done")
    #     else:
    #         model.load_state_dict(torch.load(os.path.join(model_folder_files, "best.ckpt")))
    #         model.eval()
    #         mAP = evaluate(model, test_loader, device, run_name, save_path=model_folder_files)
    #         print("mAP for "+run_name+" is:", mAP)
    else:
        mAP = evaluate(model, test_loader, device, save_path=model_folder_files)
        print("mAP for "+run_name+" is:", mAP)
    
def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("-v",
                        dest="input_video",
                        required=True,
                        type=str,
                        help="Input video for analyzing mIou/mAP")
    parser.add_argument("-a",
                        dest="annotations",
                        required=True,
                        type=str,
                        help="XML Groundtruth annotations")
    parser.add_argument("-n",
                        dest="architecture_name",
                        required=True,
                        type=str,
                        help="Architecture name. Options: FasterRCNN / MaskRCNN / ... ")
    parser.add_argument("-r",
                        dest="run_name",
                        required=True,
                        type=str,
                        help="Name of the experiment")
    parser.add_argument("-t",
                        default=False, 
                        dest="train",
                        action="store_true",
                        help="Specify to train, otherwise will evaluate")
    parser.add_argument("-g", 
                        default=False, 
                        dest="use_gpu",
                        action="store_true",
                        help="Use GPU for model inference")
    args = parser.parse_args()

    return args.input_video, args.annotations, args.architecture_name, args.use_gpu, args.run_name, args.train
    
if __name__ == "__main__":
    input_video, annotations_path, architecture_name, use_gpu, run_name, train_model = parse_arguments()
    annotations = read_annotations(annotations_path)
    # (architecture_name, video_path, annotations, run_name, finetune, train_model=False, use_gpu=True)
    frame_ids, tot_boxes, confidences = task1_2(architecture_name, input_video, annotations, run_name, True, train_model=train_model, use_gpu=use_gpu)
    mAP = voc_eval([frame_ids, tot_boxes, confidences], annotations, ovthresh=0.5)
    print("Model ["+architecture_name+"] mAP:", mAP)
    

"""
python3 week3/task1_2.py -v /home/aszummer/Documents/MCV/M6/mcvm6team3/data/AICity_data/AICity_data/train/S03/c010/vdo.avi -a /home/aszummer/Documents/MCV/M6/mcvm6team3/data/ai_challenge_s03_c010-full_annotation.xml -n FasterRCNN -g -r "test"
"""