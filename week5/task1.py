import string
from unittest import skip
import cv2
import os
#import time
#from collections import defaultdict
from itertools import product

import numpy as np
import cv2
import pandas as pd
#import imageio

from tkinter import E
import cv2
import numpy as np
from argparse import ArgumentParser
from pathlib import Path

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
STORED_DETECTIONS_NAME = "dets_all.txt"
STORED_FEATURES_NAME = "features_all.txt"
SHOW_THR = 0.5
RESULTS_FILENAME = "results"


def task1(architecture_name, video_path, run_name, args, first_frame=0, use_gpu=True, display=True, deep_sort = False, tracker = 'kalman', grid_search_dict = {}):
    """
    Object tracking: tracking by Kalman
    3 parameters: detection threshold, minimum iou to match track, and maximum frames to skip between tracked boxes.
    """
    detection_threshold = grid_search_dict['conf_thresh']
    min_iou = grid_search_dict['iou_threshold']
    max_frames_skip = grid_search_dict['skip_frames']

    #track_handler = TrackHandlerOverlap(max_frame_skip=max_frames_skip, min_iou=min_iou)
    if deep_sort == False:
        track_handler = Sort(online_filtering=True, max_age=max_frames_skip, iou_threshold=min_iou, tracker_type=tracker, beta = grid_search_dict['beta'])  # Sort max_age=1, here its 5
    else:
        track_handler = DeepSORT(max_age=max_frames_skip, iou_threshold=min_iou, tracker_type=tracker, alpha=grid_search_dict['alpha'], beta = grid_search_dict['beta'])

    # Check if detections have been saved previously
    model_folder_files = os.path.join(EXPERIMENTS_FOLDER, run_name)
    cam_det_path = os.path.join(model_folder_files, os.path.basename(os.path.dirname(os.path.dirname(video_path))), os.path.basename(os.path.dirname(video_path)))
    os.makedirs(cam_det_path, exist_ok=True)
    det_path = os.path.join(cam_det_path, STORED_DETECTIONS_NAME)
    exists_det_file = os.path.exists(det_path)

    features_path= os.path.join(cam_det_path, STORED_FEATURES_NAME)
    exists_features_file = os.path.exists(features_path)

    # Create metrics accumulator
    acc = mm.MOTAccumulator(auto_id=True)

    # Create dataset
    dataset_path = os.path.dirname(os.path.dirname(os.path.dirname(video_path)))
    sequences = {os.path.basename(os.path.dirname(os.path.dirname(video_path))):
                 [os.path.basename(os.path.dirname(video_path))]}
    dataset = AICityDatasetDetector(dataset_path, sequences)

    model_feature_vectors = []
    if exists_det_file:
        # Read detection files
        print("Reading detections file")
        model_detections = utils.parse_predictions_rects(det_path)

        if exists_features_file:
            print("Reading features file")
            model_feature_vectors = utils.parse_feature_vectors(features_path)
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

                # Update trackere
                if deep_sort == False:
                    dets = track_handler.update(image=img, dets=dets_keep, frame_number=frame_number)
                else:
                    if exists_features_file:
                        final_features = model_feature_vectors[1][frame_ids == frame_number]
                        frame_feature_vectors = final_features[final_scores > detection_threshold]
                    else:
                        frame_feature_vectors = np.empty((0,0))

                    dets, feature_vectors = track_handler.update(image=img, dets=dets_keep, frame_feature_vectors = frame_feature_vectors, frame_number=frame_number)

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
                        if track.time_since_update < 2:
                            if not track.is_static():
                                det = track.get_state()[0]
                                #det, _ = track.last_detection()
                                img_draw = cv2.rectangle(img_draw, (int(det[0]), int(det[1])), (int(det[2]), int(det[3])), track.visualization_color, 2)
                                img_draw = cv2.rectangle(img_draw, (int(det[0]), int(det[1]-20)), (int(det[2]), int(det[1])), track.visualization_color, -2)
                                img_draw = cv2.putText(img_draw, str(track.id), (int(det[0]), int(det[1])), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
                                for detection in track.history:
                                    if detection.ndim == 2:
                                        detection = detection[0]
                                    detection_center = ( int((detection[0]+detection[2])/2), int((detection[1]+detection[3])/2) )
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
                       for idx in range(len(dets_keep)):
                            detection = dets_keep[idx]
                            f.write(f'{frame_number}, -1, {detection[0]}, {detection[1]}, {detection[2]-detection[0]}, {detection[3]-detection[1]}, {final_scores[idx]}, -1, -1, -1')
                            # f.write(str(feature_vectors[idx]).replace('\n',''))
                            f.write('\n')

                    if deep_sort:
                        feature_vectors = [feat.numpy()[0].round(4) for feat in feature_vectors]
                        with open(features_path, 'a') as f:
                            for idx in range(len(feature_vectors)):
                                # f.write(str(feature_vectors[idx]).replace('\n',''))
                                f.write(f'{frame_number},')
                                f.write(','.join([str(round(x,4)) for x in feature_vectors[idx].tolist()]))
                                f.write('\n')

    # TODO: When IDF1 is implemented evaluate with different hyperparameters
    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=mm.metrics.motchallenge_metrics, name='acc')
    print(summary)

    cv2.destroyAllWindows()

    return summary

def generate_all_features(architecture_name, input_video, run_name, args, first_frame, use_gpu, display, deep_sort):
    path_to_seqs = Path("/home/aszummer/Documents/MCV/M6/mcvm6team3/data/aic19-track1-mtmc-train/train")
    paths_to_seqs = list(path_to_seqs.iterdir())

    deep_sort = True
    display = False

    for sequences in paths_to_seqs:
        paths_to_cams = list(sequences.iterdir())
        print(sequences)

        if sequences.name == 'S01':
            run_name = 'FasterRCNN_finetune_S03_S04_e2'
            for cams in paths_to_cams:
                input_video = str(cams/'vdo.avi')
                print(input_video)
                display = False
                task1(architecture_name, input_video, run_name, args, first_frame=0, use_gpu=use_gpu, display=display, deep_sort=deep_sort)

        if sequences.name == 'S03':
            run_name = 'FasterRCNN_finetune_S01_S04_e2'
            for cams in paths_to_cams:
                input_video = str(cams/'vdo.avi')
                print(input_video)
                display = False
                task1(architecture_name, input_video, run_name, args, first_frame=0, use_gpu=use_gpu, display=display, deep_sort=deep_sort)

        if sequences.name == 'S04':
            run_name = 'FasterRCNN_finetune_S01_S03_e2'
            for cams in paths_to_cams:
                input_video = str(cams/'vdo.avi')
                print(input_video)
                display = False
                task1(architecture_name, input_video, run_name, args, first_frame=0, use_gpu=use_gpu, display=display, deep_sort=deep_sort)

    print('done')

def grid_search(architecture_name, input_video, run_name, args, first_frame, use_gpu, display, deep_sort):

    # alpha_values = list(np.arange(0.3,0.9,0.1))
    # beta_values = list(np.arange(0.3,0.9,0.1))
    # skip_frames_values = list(np.arange(5,50,5))
    # conf_thresh_values = list(np.arange(0.3,0.9,0.1))
    # iou_threshold_values = list(np.arange(0.3,0.9,0.1))

    # alpha_values = list(np.arange(0.5,0.7,0.1))
    # beta_values = list(np.arange(0.5,0.7,0.1))
    # skip_frames_values = list(np.arange(5,10,5))
    # conf_thresh_values = list(np.arange(0.5,0.7,0.1))
    # iou_threshold_values = list(np.arange(0.3,0.4,0.1))

    alpha_values = [0.3,0.4,0.7]
    beta_values = [0.3,0.4,0.7]
    skip_frames_values = [15,20]
    conf_thresh_values = [0.4, 0.7, 0.8]
    iou_threshold_values = [0.2, 0.5]

    deep_sort_values = [True]
    tracker_values = ['kalman']

    columns = ['sequence', 'camera','alpha', 'beta', 'skip_frames', 'conf_thresh', 'iou_threshold', 'deep_sort', 'tracker']

    df = pd.DataFrame(columns=columns)
    # df.to_csv('grid_search_data.csv', index = False)


    for alpha, beta, skip_frames, conf_thresh, iou_threshold, deep_sort, tracker in product(alpha_values, beta_values, skip_frames_values, conf_thresh_values, iou_threshold_values, deep_sort_values, tracker_values):
        if os.path.exists('grid_search_data_3.csv'):
            df = pd.read_csv('grid_search_data_3.csv')

        grid_search_dict = {
            'alpha' : alpha,
            'beta' : beta,
            'skip_frames' : skip_frames,
            'conf_thresh' : conf_thresh,
            'iou_threshold' : iou_threshold,
            'deep_sort' : deep_sort,
            'tracker' : tracker
        }

        path_to_seqs = Path("/home/aszummer/Documents/MCV/M6/mcvm6team3/data/aic19-track1-mtmc-train/train")
        paths_to_seqs = list(path_to_seqs.iterdir())
        paths_to_seqs.sort(reverse=True)

        display = False

        for sequences in paths_to_seqs:
            paths_to_cams = list(sequences.iterdir())
            print(sequences)

            if sequences.name == 'S03':
                run_name = 'FasterRCNN_finetune_S01_S04_e2'
                for cams in tqdm(paths_to_cams):
                    input_video = str(cams/'vdo.avi')
                    print(input_video)
                    summary = task1(architecture_name, input_video, run_name, args, first_frame=0, 
                    use_gpu=use_gpu, display=display, deep_sort=deep_sort, tracker=tracker, grid_search_dict = grid_search_dict)
                    
                    data_dict = grid_search_dict
                    data_dict['sequence'] = sequences.name
                    data_dict['camera'] = cams.name
                    data_dict['run_name'] = run_name
                    data_dict_summary = dict(zip(summary.columns,summary.values[0]))
                    data_dict = {**data_dict,**data_dict_summary}
                    df = df.append(data_dict, ignore_index= True)
                    df.to_csv('grid_search_data_3.csv', index = False)



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
    parser.add_argument("-deep",
                    dest="deep_sort",
                    action="store_true",
                    default=False,
                    help="Enable deep sort")
    parser.add_argument("-t",
                        dest="tracker",
                        type=str,
                        default='kalman',
                        help="Use kalman, IoU or kcf")
    args = parser.parse_args()

    return args.input_video, args.architecture_name, args.display, args.use_gpu, args.run_name, args.deep_sort, args.tracker, args
    
if __name__ == "__main__":
    input_video, architecture_name, display, use_gpu, run_name, deep_sort, tracker, args= parse_arguments()
    task1(architecture_name, input_video, run_name, args, first_frame=0, use_gpu=use_gpu, display=display, deep_sort= deep_sort, tracker= tracker)
    # generate_all_features(architecture_name, input_video, run_name, args, first_frame=0, use_gpu=use_gpu, display=display, deep_sort=deep_sort)
    # grid_search(architecture_name, input_video, run_name, args, first_frame=0, use_gpu=use_gpu, display=display, deep_sort=deep_sort)
    
    
"""
python week5/task1.py -v /home/sergio/MCV/M6/data/aic19-track1/train/S03/c010/vdo.avi -n FasterRCNN -d -g -r FasterRCNN_finetune_S01_S04_e2

"""