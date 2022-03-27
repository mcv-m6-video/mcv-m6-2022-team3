import cv2
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
import utils
import sys

import torch
from models import load_model
from tracks import TrackHandlerOverlap

WAIT_TIME = 1
SAVE = True
CAR_LABEL_NUM = 3
FRAME_25PER = 510
EXPERIMENTS_FOLDER = "experiments"
STORED_DETECTIONS_NAME = "dets.txt"
SHOW_THR = 0.5


def task2_1(architecture_name, video_path, annotations, run_name, first_frame=0, use_gpu=True, display=True):
    """
    Object tracking: tracking by overlap
    3 parameters: detection threshold, minimum iou to match track, and maximum frames to skip between tracked boxes.
    """
    detection_threshold = 0.4
    min_iou = 0.3
    max_frames_skip = 0
    track_handler = TrackHandlerOverlap(max_frame_skip=max_frames_skip, min_iou=min_iou)

    # Prepare model
    model, device = load_model(architecture_name, use_gpu)
    model_folder_files = os.path.join(EXPERIMENTS_FOLDER, run_name)
    ckpt_path = os.path.join(model_folder_files, run_name+"_best.ckpt")
    if not os.path.exists(ckpt_path):
        raise Exception("No pretrained weights for this experiment name.")
    model.load_state_dict(torch.load(ckpt_path))
    model.eval()
    
    # Prepare video capture
    cap = cv2.VideoCapture(video_path)
    last_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    length = last_frame-first_frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, first_frame)
    
    ret = True
    frame_number = first_frame
    
    ret, img = cap.read()
    
    # Check if detections have been saved previously
    det_path = os.path.join(model_folder_files, STORED_DETECTIONS_NAME)
    exists_det_file = os.path.exists(det_path)
    
    if exists_det_file:
        # Read detection files
        model_detections = utils.parse_predictions_rects(det_path)
    
    with torch.no_grad():
        with tqdm(total=length, file=sys.stdout) as pbar:
            while ret:
                # Model inference
                if not exists_det_file:
                    x = [image_to_tensor(img, device)]
                    output = model(x)
                    preds = output[0]

                    # Keep only car predictions
                    keep_cars_mask = preds['labels'] == CAR_LABEL_NUM
                    bboxes, scores = preds['boxes'][keep_cars_mask], preds['scores'][keep_cars_mask]
                    idxs = nms(bboxes, scores, 0.7)
                    final_dets, final_scores = bboxes[idxs].cpu().numpy(), scores[idxs].cpu().numpy()
                else:
                    frame_ids = model_detections[1][0]
                    final_dets, final_scores = model_detections[1][1][frame_ids == frame_number], model_detections[1][2][frame_ids == frame_number]
                    
                # Filter detections by score -> hyperparam
                dets_keep = final_dets[final_scores > detection_threshold]
                # Update tracker
                track_handler.update_tracks(dets_keep, frame_number)
                
                if display:
                    img_draw = img.copy()
                    for track in track_handler.live_tracks:
                        det, _ = track.last_detection()
                        img_draw = cv2.rectangle(img_draw, (int(det[0]), int(det[1])), (int(det[2]), int(det[3])), track.visualization_color, 2)
                        img_draw = cv2.rectangle(img_draw, (int(det[0]), int(det[1]-20)), (int(det[2]), int(det[1])), track.visualization_color, -2)
                        img_draw = cv2.putText(img_draw, str(track.id), (int(det[0]), int(det[1])), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
                        for detection in track.detections:
                            detection_center = ( int((detection[0]+detection[2])/2), int((detection[1]+detection[3])/2) )
                            img_draw = cv2.circle(img_draw, detection_center, 5, track.visualization_color, -1)
                            
                    cv2.imshow('Tracking results', cv2.resize(img_draw, (int(img_draw.shape[1]*0.5), int(img_draw.shape[0]*0.5))))
                    k = cv2.waitKey(1)
                    if k == ord('q'):
                        return
                        
                frame_number += 1        
                ret, img = cap.read()
                pbar.update(1)
                
                if SAVE and not exists_det_file:
                    with open(det_path, 'a') as f:
                        for idx in range(len(final_dets)):
                            detection = final_dets[idx]
                            f.write(f'{frame_number}, -1, {detection[0]}, {detection[1]}, {detection[2]-detection[0]}, {detection[3]-detection[1]}, {final_scores[idx]}, -1, -1, -1\n')

    # TODO: When IDF1 is implemented evaluate with different hyperparameters
    cv2.destroyAllWindows()

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
    parser.add_argument("-d", 
                        default=False, 
                        dest="display",
                        action="store_true",
                        help="Display predictions over the video")
    parser.add_argument("-g", 
                        default=False, 
                        dest="use_gpu",
                        action="store_true",
                        help="Use GPU for model inference")
    parser.add_argument("-r",
                        dest="run_name",
                        required=True,
                        type=str,
                        help="Run name of finetuned car detector experiment")
    args = parser.parse_args()

    return args.input_video, args.annotations, args.architecture_name, args.display, args.use_gpu, args.run_name
    
if __name__ == "__main__":
    input_video, annotations_path, architecture_name, display, use_gpu, run_name = parse_arguments()
    annotations = read_annotations(annotations_path)
    task2_1(architecture_name, input_video, annotations, run_name, first_frame=0, use_gpu=use_gpu, display=display)
    