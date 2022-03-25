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

import torch
from models import load_model

WAIT_TIME = 1
SAVE = False
CAR_LABEL_NUM = 3

def task1_1(architecture_name, video_path, annotations, first_frame=0, use_gpu=True, display=True):
    """
    Off-the-shelf object detector 
    """

    # Prepare model
    model, device = load_model(architecture_name, use_gpu)
    
    # Prepare video capture
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, first_frame)
    ret = True
    frame_number = first_frame
    
    predictions, frame_numbers, total_scores = [], [], []
    ret, img = cap.read()

    with torch.no_grad():
        while ret:
            # Model inference
            x = [image_to_tensor(img, device)]
            output = model(x)
            preds = output[0]

            # Keep only car predictions
            keep_cars_mask = preds['labels'] == CAR_LABEL_NUM
            bboxes, scores = preds['boxes'][keep_cars_mask], preds['scores'][keep_cars_mask]
            idxs = nms(bboxes, scores, 0.7)
            final_dets, final_scores = bboxes[idxs], scores[idxs]

            for i in range(len(final_dets)):
                predictions.append(final_dets[i])
                total_scores.append(final_scores[i])
            
            frame_numbers.append([frame_number for _ in range(len(final_dets))])
            
            if display:
                if frame_number in list(annotations.keys()):
                    show_annotations_and_predictions(img, annotations[frame_number], final_dets[final_scores > 0.3])
                else:
                    show_annotations_and_predictions(img, [], final_dets[final_scores > 0.3])

            """ if SAVE:
                if not os.path.exists(EXPERIMENTS_FOLDER):
                    os.mkdir(EXPERIMENTS_FOLDER)
                experiment_run_folder = os.path.join(EXPERIMENTS_FOLDER, run_name)
                if not os.path.exists(experiment_run_folder):
                    os.mkdir(experiment_run_folder)
                
                cv2.imwrite(experiment_run_folder+'/image_'+str(frame_number-frame_25per).zfill(4)+'.jpg', cv2.resize(img, tuple(np.int0(0.5*np.array(img.shape[:2][::-1])))))
                cv2.imwrite(experiment_run_folder+'/classification_'+str(frame_number-frame_25per).zfill(4)+'.jpg', cv2.resize(frame, tuple(np.int0(0.5*np.array(img.shape[:2][::-1]))))) """
            
            frame_number += 1
            ret, img = cap.read()
    
    cap.release()
    cv2.destroyAllWindows()
    frame_numbers, predictions = np.concatenate(frame_numbers), np.concatenate(predictions)
    tot_predictions = [frame_numbers.astype(np.int64), predictions.reshape((len(frame_numbers), 4)), total_scores]
    return tot_predictions

def task1_2(finetune=True, architecture='maskrcnn', save_path=None):
    """
    Finetune object detector
    """
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    train_loader, test_loader = get_data_loaders(root='data')
    model = get_model(architecture=architecture, finetune=finetune, num_classes=len(train_loader.dataset.classes))
    model.to(device)

    if finetune:
        train(model, train_loader, test_loader, device, save_path=save_path)
    else:
        evaluate(model, test_loader, device, save_path=save_path)
    
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
    args = parser.parse_args()

    return args.input_video, args.annotations, args.architecture_name, args.display, args.use_gpu
    
if __name__ == "__main__":
    input_video, annotations_path, architecture_name, display, use_gpu = parse_arguments()
    annotations = read_annotations(annotations_path)
    frame_ids, tot_boxes, confidences = task1_1(architecture_name, input_video, annotations, first_frame=0, use_gpu=use_gpu, display=display)
    mAP = voc_eval([frame_ids, tot_boxes, confidences], annotations, ovthresh=0.5)
    print("Model ["+architecture_name+"] mAP:", mAP)
    
    