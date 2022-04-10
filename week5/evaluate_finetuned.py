import os

#import numpy as np

#from tkinter import E
import numpy as np
from argparse import ArgumentParser
import os
import torchvision

import torch
from models import load_model, train, evaluate #, evaluate
from datasets import AICityDatasetValidation, collate_dicts_fn


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

    transformations = torchvision.transforms.ToTensor()
    # transformations = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    test_dataset = AICityDatasetDetector(dataset_path, sequences, transformations=transformations)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, num_workers=1, shuffle=False, collate_fn=collate_dicts_fn)
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
    
    ckpt_path = os.path.join(model_folder_files, run_name+"_best.ckpt")
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path))
    else:
        print("No finetuned weights for this experiment name, using pretrained model...")
    model.eval()

    log_bool = True
    num_epochs = 3
    batch_size = 1

    print('Starting mAP evaluation')
    evaluate(model, test_loader, device)
    print("Evaluation done")


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
    args = parser.parse_args()

    return args.dataset_path, args.sequencies, args.architecture_name, args.use_gpu, args.run_name
    
if __name__ == "__main__":
    path_dataset, sequencies, architecture_name, use_gpu, run_name= parse_arguments()
    # (architecture_name, video_path, annotations, run_name, finetune, train_model=False, use_gpu=True)
    finetune(architecture_name, path_dataset, sequencies, run_name, use_gpu=use_gpu)
    