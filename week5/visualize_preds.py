import cv2

import numpy as np
import cv2

from tkinter import E
import cv2
import numpy as np
from argparse import ArgumentParser

from tqdm import tqdm
import sys

import torch
import random


random.seed(a=2022)
COLORS = [(int(random.random() * 256), int(random.random() * 256), int(random.random() * 256)) for i in range(10000)]


def visualize_preds(video_path, cam_info_file, first_frame=0):
    """
    Visualize predictions.
        - video_path
        - cam_info_file: a .npz file containing rows of [frame_id, track_id, bbox (x4), feature_vector (x256)]
    """

    cam_info = np.load(cam_info_file)
    cap = cv2.VideoCapture(video_path)
    last_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    length = last_frame-first_frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, first_frame)

    frame_id = first_frame
    ret, img = cap.read()

    with torch.no_grad():
        with tqdm(total=length, file=sys.stdout) as pbar:
            while ret:
                img_draw = img.copy()
                frame_mask = cam_info[:,0] == frame_id
                ids = cam_info[frame_mask, 1].astype(np.int)
                dets = cam_info[frame_mask, 2:6].astype(np.int)  # left, top, width, height
                for track_id, det in zip(ids, dets):
                    img_draw = cv2.rectangle(img_draw, (det[0], det[1]), (det[2], det[3]), COLORS[track_id], 2)
                    img_draw = cv2.rectangle(img_draw, (det[0], det[1]-20), (det[2], det[1]), COLORS[track_id], -2)
                    img_draw = cv2.putText(img_draw, str(track_id), (det[0], det[1]), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
                
                cv2.imshow('Tracking results', cv2.resize(img_draw, (int(img_draw.shape[1]*0.5), int(img_draw.shape[0]*0.5))))

                k = cv2.waitKey(1)
                if k == ord('q'):
                    return
            
                frame_id += 1
                ret, img = cap.read()
                pbar.update(1)

    cv2.destroyAllWindows()


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("-v",
                        dest="input_video",
                        required=True,
                        type=str,
                        help="Input video")
    parser.add_argument("-c",
                        dest="cam_info_file",
                        required=False,
                        type=str,
                        help="Input detections file (.npy)")
    parser.add_argument("-f",
                        dest="first_frame",
                        required=False,
                        default=0,
                        type=int,
                        help="Frame at which to start the video")
    args = parser.parse_args()
    return args.input_video, args.cam_info_file, args.first_frame
    
if __name__ == "__main__":
    input_video, cam_info_file, first_frame = parse_arguments()
    print(cam_info_file)
    visualize_preds(input_video, cam_info_file, first_frame)
    
