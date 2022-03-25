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
        
        labels = torch.ones((len(boxes)))*self.class_car_number
        
        if self.transformations is not None:
            img = self.transformations(img)

        return img, boxes, labels

    def __len__(self):
        return len(self.frame_idxs)
    
def get_data_loaders(annotations, video_path, train_idxs, test_idxs, transformations):
    # use our dataset and defined transformations
    train_dataset = AICityDataset(annotations, video_path, train_idxs, transformations=transformations)
    test_dataset = AICityDataset(annotations, video_path, test_idxs, transformations=transformations)

    # define training and validation data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, num_workers=1)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8, num_workers=1)

    return train_loader, test_loader
