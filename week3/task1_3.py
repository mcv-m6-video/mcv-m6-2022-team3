import os
import numpy as np
from argparse import ArgumentParser

import torch
import torchvision

from utils import read_annotations
from datasets import create_dataloaders
from models import load_model, train, evaluate


WAIT_TIME = 1
SAVE = False
CAR_LABEL_NUM = 3
FRAME_25PER = 510
EXPERIMENTS_FOLDER = "experiments"


def run_tine_tuning(train_idxs, test_idxs, architecture_name, video_path, annotations, run_name, finetune, train_model, use_gpu, transformations, fold=1):
    train_loader, test_loader = create_dataloaders(annotations, video_path, train_idxs, test_idxs, transformations, 
                                                    batch_size=2)
    model, device = load_model(architecture_name, use_gpu, finetune=finetune)

    model_folder_files = os.path.join(EXPERIMENTS_FOLDER, run_name)

    if not os.path.exists(EXPERIMENTS_FOLDER):
        os.makedirs(EXPERIMENTS_FOLDER,exist_ok=True)
    if not os.path.exists(model_folder_files):
        os.makedirs(model_folder_files,exist_ok=True)
    
    mAP = evaluate(model, test_loader, device, annotations)
    print("Initial mAP:", mAP)

    log_bool=True
    num_epochs = 3
    batch_size = 2

    if finetune:
        if train_model:
            train(model, train_loader, test_loader, device, annotations, architecture_name,
                         num_epochs=num_epochs, 
                         batch_size=batch_size,
                         save_path=model_folder_files, log_bool=log_bool, run_name=run_name)
            print(f"[fold {fold}] Training done")
            mAP = evaluate(model, test_loader, device, annotations)
            print(f"[fold {fold}] Final mAP:", mAP)
        else:
            model.load_state_dict(torch.load(os.path.join(model_folder_files, "best.ckpt")))
            model.eval()
            mAP = evaluate(model, test_loader, device, annotations)
            print(f"[fold {fold}] mAP for "+run_name+" is:", mAP)
    else:
        mAP = evaluate(model, test_loader, device, annotations)
        print(f"[fold {fold}] mAP for "+run_name+" is:", mAP)



def task1_3(architecture_name, video_path, annotations, run_name, finetune, train_model=False, use_gpu=True, cv_strategy='A'):
    """
    Finetune object detector
    """
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    transformations = torchvision.transforms.Compose([
                        torchvision.transforms.ToTensor(),
                        torchvision.transforms.RandomAutocontrast(p=0.5),
                        torchvision.transforms.RandomHorizontalFlip(p=0.5)]
                        )
    # transformations = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    if cv_strategy == 'A':
        train_idxs, test_idxs = np.arange(0, FRAME_25PER), np.arange(FRAME_25PER, FRAME_25PER*4)
        run_tine_tuning(train_idxs, test_idxs, architecture_name, video_path, annotations, run_name, finetune, train_model, use_gpu, transformations, 1)
    
    elif cv_strategy == 'B':
        # FOLD 1
        train_idxs, test_idxs = np.arange(0, FRAME_25PER), np.arange(FRAME_25PER, FRAME_25PER*4)
        run_tine_tuning(train_idxs, test_idxs, architecture_name, video_path, annotations, run_name, finetune, train_model, use_gpu, transformations, 1)
        # FOLD 2
        train_idxs, test_idxs = np.arange(FRAME_25PER, FRAME_25PER*2), np.concatenate([np.arange(0, FRAME_25PER), np.arange(FRAME_25PER*2, FRAME_25PER*4)])
        run_tine_tuning(train_idxs, test_idxs, architecture_name, video_path, annotations, run_name, finetune, train_model, use_gpu, transformations, 2)
        # FOLD 3
        train_idxs, test_idxs = np.arange(FRAME_25PER*2, FRAME_25PER*3), np.concatenate([np.arange(0, FRAME_25PER*2), np.arange(FRAME_25PER*3, FRAME_25PER*4)])
        run_tine_tuning(train_idxs, test_idxs, architecture_name, video_path, annotations, run_name, finetune, train_model, use_gpu, transformations, 3)
        # FOLD 4
        train_idxs, test_idxs = np.arange(FRAME_25PER*3, FRAME_25PER*4), np.arange(0, FRAME_25PER*3)
        run_tine_tuning(train_idxs, test_idxs, architecture_name, video_path, annotations, run_name, finetune, train_model, use_gpu, transformations, 4)
    
    elif cv_strategy == 'C':
        all_idxs = np.arange(0, FRAME_25PER*4)
        for fold in range(1,5):
            train_idxs = np.random.choice(all_idxs, FRAME_25PER, replace=False)
            test_idxs = np.array(list(set(all_idxs).difference(set(train_idxs))))
            run_tine_tuning(train_idxs, test_idxs, architecture_name, video_path, annotations, run_name, finetune, train_model, use_gpu, transformations, fold)
    
    else:
        print("Wrong strategy. Options: A, B, C.")

    
    
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
    parser.add_argument("-r",
                        dest="run_name",
                        required=True,
                        type=str,
                        help="Name of the experiment")
    parser.add_argument("-t",
                        default=False, 
                        dest="train",
                        action="store_true",
                        help="Specify to train, otherwise will evaluate")
    parser.add_argument("-g", 
                        default=False, 
                        dest="use_gpu",
                        action="store_true",
                        help="Use GPU for model inference")
    parser.add_argument("-s",
                        dest="strategy",
                        default="A",
                        type=str,
                        help="Strategy used for cross-validation")
    
    ## Model training parameters
    parser.add_argument("-b", 
                        default=8, 
                        dest="batch_size",
                        type=int,
                        help="Batch size")
    
    args = parser.parse_args()

    return args.input_video, args.annotations, args.architecture_name, args.use_gpu, args.run_name, args.train, args.strategy #, args
    
if __name__ == "__main__":
    input_video, annotations_path, architecture_name, use_gpu, run_name, train_model, strategy = parse_arguments()
    annotations = read_annotations(annotations_path)
    task1_3(architecture_name, input_video, annotations, run_name, True, train_model=train_model, use_gpu=use_gpu, cv_strategy=strategy)
