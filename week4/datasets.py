from random import shuffle
import torch
import cv2
import numpy as np
import pandas as pd
import os


class AICityDataset(torch.utils.data.Dataset):
    def __init__(self, path, sequences, transformations=None):
        """
        sequences can be:
            - list of strings: ['S01', 'S03']. This will load all videos from all cameras
            - dict: {'S01':['c001', 'c002']}. So we can choose the cameras
        """
        self.transformations = transformations
        self.path = path
        self.videos = []
        self.gts = []
        lengths = []
        if isinstance(sequences, list):
            dict_seqs = {}
            for sequence in sequences:
                dict_seqs[sequence] = sorted(os.listdir(os.path.join(path, sequence)))
            sequences = dict_seqs

        for sequence, camera_ids in sequences.items():
            for camera_id in camera_ids:
                if os.path.isdir(cam_folder):
                    cam_folder = os.path.join(path, sequence, camera_id)
                    gt = pd.read_csv(os.path.join(cam_folder, 'gt','gt.txt'),
                                    names=['frame', 'id', 'left', 'top', 'width', 'height',
                                            '1','2','3','4'])  # extra useless cols
                    cap = cv2.VideoCapture(os.path.join(cam_folder, 'vdo.avi'))
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    self.gts.append(gt)
                    self.videos.append(cap)
                    lengths.append(frame_count)
        self.video_starts = np.array(lengths).cumsum()
        self.length = sum(lengths)

    def __getitem__(self, idx):
        vid_idx = (self.video_starts > idx).argmax()

        ret, img = self.videos[vid_idx].read()
        if self.transformations:
            img = self.transformations(img)

        # Check if video is labelled
        frame_id = idx - self.video_starts[:vid_idx].sum()
        if frame_id not in self.gts[vid_idx]['frame'].unique():
            return img, None
        else:
            bboxes = self.gts[vid_idx][self.gts[vid_idx]['frame'] == frame_id][['left','top','width','height','id']].to_numpy()
            target = {'boxes': bboxes[:,:4], 'labels': bboxes[:,4], 'image_id': torch.tensor([idx])}
            return img[:,:,::-1], bboxes
        
    def __len__(self):
        return self.length
    
def collate_dicts_fn(batch):
    return tuple(zip(*batch))
