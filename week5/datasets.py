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
        self.class_car_number = 3
        lengths = [0]
        if isinstance(sequences, list):
            dict_seqs = {}
            for sequence in sequences:
                dict_seqs[sequence] = sorted(os.listdir(os.path.join(path, sequence)))
            sequences = dict_seqs

        for sequence, camera_ids in sequences.items():
            for camera_id in camera_ids:
                cam_folder = os.path.join(path, sequence, camera_id)
                if os.path.isdir(cam_folder):
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
        self.lengths = lengths

    def __getitem__(self, idx):
        vid_idx = (self.video_starts[1:] >= idx).argmax()

        self.videos[vid_idx].set(cv2.CAP_PROP_POS_FRAMES, idx - self.video_starts[vid_idx])
        ret, img = self.videos[vid_idx].read()
        if ret:
            if self.transformations:
                img = self.transformations(img[:,:,::-1].copy())

            # Check if video is labelled
            frame_id = idx - self.video_starts[:vid_idx].sum()
            if frame_id not in self.gts[vid_idx]['frame'].unique():
                return img, None
            else:
                bboxes = self.gts[vid_idx][self.gts[vid_idx]['frame'] == frame_id][['left','top','width','height','id']].to_numpy()
                bboxes[:,2] = bboxes[:,0] + bboxes[:,2]
                bboxes[:,3] = bboxes[:,1] + bboxes[:,3]
                labels = torch.ones((len(bboxes)), dtype=torch.int64)*self.class_car_number
                target = {'boxes': torch.tensor(bboxes[:,:4], dtype=torch.float32), 'track_id':torch.tensor(bboxes[:,4], dtype=torch.float32), 'labels': labels, 'image_id': torch.tensor([idx])}
                return img, target
        else:
            return torch.ones((24,24,3)), None
        
    def __len__(self):
        return self.length
    
def collate_dicts_fn(batch):
    return tuple(zip(*batch))
