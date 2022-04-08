import cv2
import os
#import time
#from collections import defaultdict

import numpy as np
import cv2
#import imageio

from tkinter import E
import cv2
import numpy as np
from argparse import ArgumentParser

import scipy
from utils import read_annotations, image_to_tensor
#from evaluation import show_annotations_and_predictions, voc_eval
import os
from tqdm import tqdm
from torchvision.ops import nms
from datasets import AICityDataset
import utils
import sys

import torch
from models import load_model
from sort import Sort
from sort import convert_x_to_bbox
import motmetrics as mm
import random

WAIT_TIME = 1
SAVE = True
CAR_LABEL_NUM = 3
FRAME_25PER = 510
EXPERIMENTS_FOLDER = "experiments"
STORED_gt_boxesECTIONS_NAME = "gt_boxess.txt"
SHOW_THR = 0.5
RESULTS_FILENAME = "results"
COLORS = [(int(random.random() * 256), int(random.random() * 256), int(random.random() * 256)) for i in range(10000)]

def visualize_gt(video_path, gt_path = None):
    """
    Object tracking: tracking by Kalman
    3 parameters: gt_boxesection threshold, minimum iou to match track, and maximum frames to skip between tracked boxes.
    """

    # Prepare video capture
    cap = cv2.VideoCapture(video_path)
    last_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    length = last_frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    ret = True
    frame_number = 0
    
    ret, img = cap.read()

    # Create dataset
    dataset_path = os.path.dirname(os.path.dirname(os.path.dirname(video_path)))
    sequences = {os.path.basename(os.path.dirname(os.path.dirname(video_path))):
                 [os.path.basename(os.path.dirname(video_path))]}
    print("get_path before init = ",gt_path)
    dataset_det = AICityDataset(dataset_path, sequences, gt_path=gt_path)
    dataset = AICityDataset(dataset_path, sequences)
    #import pdb
    #pdb.set_trace()
    
    with torch.no_grad():
        with tqdm(total=length, file=sys.stdout) as pbar:
            while ret:
                img_draw = img.copy()
                if dataset.contains_gt_for_frame(frame_number):
                    gt = dataset.get_bboxes_of_frame_id(frame_number)
                    for i_gt_box in range(len(gt)):
                        gt_boxes = gt.iloc[i_gt_box, :]
                        track_id = int(gt_boxes["id"])
                        # 'left', 'top', 'width', 'height'
                        img_draw = cv2.rectangle(img_draw, (int(gt_boxes["left"]), int(gt_boxes["top"])), (int(gt_boxes["left"]+gt_boxes["width"]), int(gt_boxes["top"]+gt_boxes["height"])), COLORS[track_id], 2)
                        img_draw = cv2.rectangle(img_draw, (int(gt_boxes["left"]), int(gt_boxes["top"]-20)), (int(gt_boxes["left"]+gt_boxes["width"]), int(gt_boxes["top"])), COLORS[track_id], -2)
                        img_draw = cv2.putText(img_draw, str(track_id), (int(gt_boxes["left"]), int(gt_boxes["top"])), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)

                img_draw_det = img.copy()
                if dataset_det.contains_gt_for_frame(frame_number):
                    gt = dataset_det.get_bboxes_of_frame_id(frame_number)
                    for i_gt_box in range(len(gt)):
                        gt_boxes = gt.iloc[i_gt_box, :]
                        track_id = int(gt_boxes["id"])
                        # 'left', 'top', 'width', 'height'
                        img_draw_det = cv2.rectangle(img_draw_det, (int(gt_boxes["left"]), int(gt_boxes["top"])), (int(gt_boxes["left"]+gt_boxes["width"]), int(gt_boxes["top"]+gt_boxes["height"])), COLORS[track_id], 2)
                        img_draw_det = cv2.rectangle(img_draw_det, (int(gt_boxes["left"]), int(gt_boxes["top"]-20)), (int(gt_boxes["left"]+gt_boxes["width"]), int(gt_boxes["top"])), COLORS[track_id], -2)
                        img_draw_det = cv2.putText(img_draw_det, str(track_id), (int(gt_boxes["left"]), int(gt_boxes["top"])), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
                        
                
                cv2.imshow('Tracking results', cv2.resize(img_draw, (int(img_draw.shape[1]*0.5), int(img_draw.shape[0]*0.5))))
                cv2.imshow('Tracking results DET', cv2.resize(img_draw_det, (int(img_draw_det.shape[1]*0.5), int(img_draw_det.shape[0]*0.5))))


                k = cv2.waitKey(1)
                if k == ord('q'):
                    return
            
                frame_number += 1
                ret, img = cap.read()
                pbar.update(1)

    cv2.destroyAllWindows()

def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("-v",
                        dest="input_video",
                        required=True,
                        type=str,
                        help="Input video for analyzing mIou/mAP")
    parser.add_argument("-det_path",
                        dest="det_path",
                        required=False,
                        type=str,
                        help="Input video for analyzing mIou/mAP")
    args = parser.parse_args()
    return args.input_video, args.det_path
    
if __name__ == "__main__":
    input_video, det_path = parse_arguments()
    print(det_path)
    visualize_gt(input_video, det_path)
    