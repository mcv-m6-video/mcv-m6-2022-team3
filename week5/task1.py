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
from datasets import AICityDatasetDetector
import utils
import sys

import torch
from models import load_model
from sort import Sort, DeepSORT
from sort import convert_x_to_bbox
import motmetrics as mm

WAIT_TIME = 1
SAVE = False
SAVE_DETS = True
CAR_LABEL_NUM = 3
FRAME_25PER = 510
EXPERIMENTS_FOLDER = "experiments"
STORED_DETECTIONS_NAME = "dets.txt"
SHOW_THR = 0.5
RESULTS_FILENAME = "results"


def task1(architecture_name, video_path, run_name, args, first_frame=0, use_gpu=True, display=True):
    """
    Object tracking: tracking by Kalman
    3 parameters: detection threshold, minimum iou to match track, and maximum frames to skip between tracked boxes.
    """
    detection_threshold = args.det_thr
    min_iou = args.min_iou
    max_frames_skip = args.frame_skip
    #track_handler = TrackHandlerOverlap(max_frame_skip=max_frames_skip, min_iou=min_iou)
    if False:
        track_handler = Sort(max_age=max_frames_skip, iou_threshold=min_iou)  # Sort max_age=1, here its 5
    else:
        track_handler = DeepSORT(max_age=max_frames_skip, iou_threshold=min_iou)

    # Check if detections have been saved previously
    model_folder_files = os.path.join(EXPERIMENTS_FOLDER, run_name)
    model_folder_files = os.path.join(model_folder_files, os.path.basename(os.path.dirname(os.path.dirname(video_path))), os.path.basename(os.path.dirname(video_path)))
    os.makedirs(model_folder_files, exist_ok=True)
    det_path = os.path.join(model_folder_files, STORED_DETECTIONS_NAME)
    exists_det_file = os.path.exists(det_path)

    # Create metrics accumulator
    acc = mm.MOTAccumulator(auto_id=True)

    # Create dataset
    dataset_path = os.path.dirname(os.path.dirname(os.path.dirname(video_path)))
    sequences = {os.path.basename(os.path.dirname(os.path.dirname(video_path))):
                 [os.path.basename(os.path.dirname(video_path))]}
    dataset = AICityDatasetDetector(dataset_path, sequences)

    if exists_det_file:
        # Read detection files
        print("Reading detections file")
        model_detections = utils.parse_predictions_rects(det_path)
        
    else:
        # Prepare model
        model, device = load_model(architecture_name, use_gpu)
    
        ckpt_path = os.path.join(model_folder_files, run_name+"_best.ckpt")
        if not os.path.exists(ckpt_path):
            print("No pretrained weights for this experiment name.")
        else:
            model.load_state_dict(torch.load(ckpt_path))
            print(f"Model {run_name} loaded")
        model.eval()
    
    # Prepare video capture
    cap = cv2.VideoCapture(video_path)
    last_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    length = last_frame-first_frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, first_frame)
    
    ret = True
    frame_number = first_frame
    
    ret, img = cap.read()
    
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
                dets_keep = np.hstack([dets_keep, final_scores[final_scores > detection_threshold][:,np.newaxis]])

                # Update tracker
                dets = track_handler.update(image=img, dets=dets_keep)

                _, gt = dataset[frame_number]
                if gt:
                    if "boxes" in list(gt.keys()):
                        #import pdb
                        #pdb.set_trace()
                        gt_boxes = gt['boxes']
                        gt_this_frame = [int(x) for x in gt['track_id']]
                        dets_this_frame = [int(det[4]) for det in dets]
                        dets_centers = np.vstack([(dets[:,0]+dets[:,2])/2, (dets[:,1]+dets[:,3])/2]).T
                        gt_centers = np.vstack([(gt_boxes[:,0]+gt_boxes[:,2])/2, (gt_boxes[:,1]+gt_boxes[:,3])/2]).T
                        dists = scipy.spatial.distance_matrix(dets_centers, gt_centers).T.tolist()
                        acc.update(
                            gt_this_frame,
                            dets_this_frame,
                            dists
                        )

                if display:
                    img_draw = img.copy()
                    for track in track_handler.trackers:
                        if not track.is_static():
                            det = convert_x_to_bbox(track.kf.x).squeeze()
                            #det, _ = track.last_detection()
                            img_draw = cv2.rectangle(img_draw, (int(det[0]), int(det[1])), (int(det[2]), int(det[3])), track.visualization_color, 2)
                            img_draw = cv2.rectangle(img_draw, (int(det[0]), int(det[1]-20)), (int(det[2]), int(det[1])), track.visualization_color, -2)
                            img_draw = cv2.putText(img_draw, str(track.id), (int(det[0]), int(det[1])), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
                            for detection in track.history:
                                detection_center = ( int((detection[0][0]+detection[0][2])/2), int((detection[0][1]+detection[0][3])/2) )
                                img_draw = cv2.circle(img_draw, detection_center, 5, track.visualization_color, -1)
                            
                    cv2.imshow('Tracking results', cv2.resize(img_draw, (int(img_draw.shape[1]*0.5), int(img_draw.shape[0]*0.5))))
                    k = cv2.waitKey(1)
                    if k == ord('q'):
                        return
                    
                    if SAVE:
                        track_exp_name = f"tracking_t{detection_threshold:03f}_i{min_iou}_fs{max_frames_skip}"
                        path_to_res_folder = os.path.join(model_folder_files, RESULTS_FILENAME, track_exp_name)
                        os.makedirs(path_to_res_folder,exist_ok=True)
                        cv2.imwrite(path_to_res_folder+'/image_'+str(frame_number-first_frame).zfill(4)+'.jpg', cv2.resize(img_draw, tuple(np.int0(0.5*np.array(img_draw.shape[:2][::-1])))))

                frame_number += 1
                ret, img = cap.read()
                pbar.update(1)
                
                if SAVE_DETS and not exists_det_file:
                    with open(det_path, 'a') as f:
                       for idx in range(len(final_dets)):
                            detection = final_dets[idx]
                            f.write(f'{frame_number}, -1, {detection[0]}, {detection[1]}, {detection[2]-detection[0]}, {detection[3]-detection[1]}, {final_scores[idx]}, -1, -1, -1\n')

    # TODO: When IDF1 is implemented evaluate with different hyperparameters
    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=mm.metrics.motchallenge_metrics, name='acc')
    print(summary)

    cv2.destroyAllWindows()

def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("-v",
                        dest="input_video",
                        required=True,
                        type=str,
                        help="Input video for analyzing mIou/mAP")
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
    
    # Tracking parameters
    parser.add_argument("-det_thr",
                        dest="det_thr",
                        type=float,
                        default=0.8,
                        help="Detection threshold for the car detector")
    parser.add_argument("-min_iou",
                        dest="min_iou",
                        default=0.4,
                        type=float,
                        help="Run name of finetuned car detector experiment")
    parser.add_argument("-fs",
                        dest="frame_skip",
                        default=5,
                        type=int,
                        help="Number of frames for which a track is still considerated lived")
    args = parser.parse_args()

    return args.input_video, args.architecture_name, args.display, args.use_gpu, args.run_name, args
    
if __name__ == "__main__":
    input_video, architecture_name, display, use_gpu, run_name, args = parse_arguments()
    task1(architecture_name, input_video, run_name, args, first_frame=0, use_gpu=use_gpu, display=display)
    
    
"""
python week5/task1.py -v /home/sergio/MCV/M6/data/aic19-track1/train/S03/c010/vdo.avi -n FasterRCNN -d -g -r FasterRCNN_finetune_S01_S04_e2

"""