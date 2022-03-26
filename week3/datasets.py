from random import shuffle
import torch
import cv2
import torchvision
import os

class AICityDataset(torch.utils.data.Dataset):
    def __init__(self, annotations, video_path, frame_idxs, transformations=None):
        self.video_path = video_path
        self.transformations = transformations
        self.frame_idxs = frame_idxs
        self.annotations = annotations
        self.video_cap = cv2.VideoCapture(self.video_path)
        self.class_car_number = 3
        
    def __getitem__(self, idx):
        idx = self.frame_idxs[idx]
        self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        _, img = self.video_cap.read()

        if idx in list(self.annotations.keys()):
            boxes = torch.tensor(self.annotations[idx], dtype=torch.float32)    
        else:
            boxes = torch.tensor([])
        
        labels = torch.ones((len(boxes)), dtype=torch.int64)*self.class_car_number
        
        if self.transformations is not None:
            img = self.transformations(img)

        target = {'boxes': boxes, 'labels': labels, 'image_id': torch.tensor([idx])}

        return img, target

    def __len__(self):
        return len(self.frame_idxs)
    
def collate_dicts_fn(batch):
    return tuple(zip(*batch))
    
def create_dataloaders(annotations, video_path, train_idxs, test_idxs, transformations, batch_size):
    # use our dataset and defined transformations
    train_dataset = AICityDataset(annotations, video_path, train_idxs, transformations=transformations)
    test_dataset = AICityDataset(annotations, video_path, test_idxs, transformations=transformations)

    # define training and validation data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=1, shuffle=True, collate_fn=collate_dicts_fn)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=1, shuffle=True, collate_fn=collate_dicts_fn)

    return train_loader, test_loader
