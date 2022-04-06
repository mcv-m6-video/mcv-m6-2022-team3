import os

#import numpy as np

#from tkinter import E
import numpy as np
from argparse import ArgumentParser
import os
import torchvision

import torch
from models import load_model, train#, evaluate
from datasets import AICityDataset, collate_dicts_fn


WAIT_TIME = 1
SAVE = False
CAR_LABEL_NUM = 3
FRAME_25PER = 510
EXPERIMENTS_FOLDER = "experiments"


def finetune(architecture_name, dataset_path, sequences, run_name, use_gpu=True):
    """
    Finetune object detector
    """
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('device:', device)

    transformations = torchvision.transforms.Compose([
                        torchvision.transforms.ToTensor(),
                        torchvision.transforms.RandomAutocontrast(p=0.5),
                        torchvision.transforms.RandomHorizontalFlip(p=0.5)]
                        )
    # transformations = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    train_dataset = AICityDataset(dataset_path, sequences, transformations=transformations)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, num_workers=1, shuffle=True, collate_fn=collate_dicts_fn)
    print('loader created')

    model, device = load_model(architecture_name, use_gpu, finetune=True)
    print('model loaded')

    model_folder_files = os.path.join(EXPERIMENTS_FOLDER, run_name)

    if not os.path.exists(EXPERIMENTS_FOLDER):
        # os.mkdir(EXPERIMENTS_FOLDER)
        os.makedirs(EXPERIMENTS_FOLDER,exist_ok=True)
    if not os.path.exists(model_folder_files):
        # os.mkdir(model_folder_files)
        os.makedirs(model_folder_files,exist_ok=True)

    log_bool=True
    num_epochs = 3
    batch_size = 1

    print('strating training')
    train(model, train_loader, device, architecture_name,
                    num_epochs=num_epochs, 
                    batch_size=batch_size,
                    save_path=model_folder_files, log_bool=log_bool, run_name=run_name)
    print("Training done")


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("-d",
                        dest="dataset_path",
                        required=True,
                        type=str,
                        help="Path of the dataset.")
    parser.add_argument("-s",
                        dest="sequencies",
                        required=True,
                        type=str,
                        nargs='+')
    parser.add_argument("-n",
                        dest="architecture_name",
                        required=True,
                        type=str,
                        help="Architecture name. Options: FasterRCNN / MaskRCNN / ... ")
    parser.add_argument("-r",
                        dest="run_name",
                        required=True,
                        type=str,
                        help="Name of the experiment")
    parser.add_argument("-g", 
                        default=False, 
                        dest="use_gpu",
                        action="store_true",
                        help="Use GPU for model inference")
    
    ## Model training parameters
    parser.add_argument("-b", 
                        default=8, 
                        dest="batch_size",
                        type=int,
                        help="Batch size")
    
    args = parser.parse_args()

    return args.dataset_path, args.sequencies, args.architecture_name, args.use_gpu, args.run_name
    
if __name__ == "__main__":
    path_dataset, sequencies, architecture_name, use_gpu, run_name= parse_arguments()
    # (architecture_name, video_path, annotations, run_name, finetune, train_model=False, use_gpu=True)
    finetune(architecture_name, path_dataset, sequencies, run_name, use_gpu=use_gpu)
    

"""
python3 week3/task1_2.py -v /home/aszummer/Documents/MCV/M6/mcvm6team3/data/AICity_data/AICity_data/train/S03/c010/vdo.avi -a /home/aszummer/Documents/MCV/M6/mcvm6team3/data/ai_challenge_s03_c010-full_annotation.xml -n FasterRCNN -g -r "test"
python3 task1_2.py -v /home/aszummer/Documents/MCV/M6/mcvm6team3/data/AICity_data/AICity_data/train/S03/c010/vdo.avi -a /home/aszummer/Documents/MCV/M6/mcvm6team3/data/ai_challenge_s03_c010-full_annotation.xml -n MaskRCNN -g -r "test_MASKRCNN_b2_lr0.002_10_epoch -t
"""
