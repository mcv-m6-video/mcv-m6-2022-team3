import os
from time import time
from itertools import product
import imageio
from tqdm import tqdm

import numpy as np
import cv2
import pandas as pd
from argparse import ArgumentParser

from optical_flow import block_matching_flow, get_opticalflow_model
from utils import load_optical_flow, display_OpticalFlow
from evaluation import eval_opticalflow

def task1_2(random_params, run_times):
    # Optical Flow with Block Matching
    img_prev = cv2.imread('imgs/000045_10.png', cv2.IMREAD_GRAYSCALE)
    img_next = cv2.imread('imgs/000045_11.png', cv2.IMREAD_GRAYSCALE)

    # Image 157
    mask, u_gt, v_gt = load_optical_flow('imgs/000045_10_gt.png')
    _, u, v = load_optical_flow('imgs/LKflow_000045_10.png')

    # Parameter combinations
    os.makedirs("results", exist_ok=True)
    if not random_params:
        run_times = 1
    
    optical_flow_methods = ["fb", "hs"]
    data = []
    for optical_flow_method_name in optical_flow_methods:
        for _ in range(run_times):
            optical_flow_method = get_opticalflow_model(optical_flow_method_name, random_params=random_params)
            t0 = time()
            # Estimate block matching with: motion_type, search_area, block_size
            flow = optical_flow_method.estimate_optical_flow(img_prev, img_next)
            u, v = flow[:,:,0], flow[:,:,1]
            t1 = time()
            msen, pepn = eval_opticalflow(u, v, u_gt, v_gt, mask)
            print(f"Method '{optical_flow_method_name}' | MSEN:", msen, "PEPN:", pepn)
            display_OpticalFlow(img_prev, u, v, f"results/{optical_flow_method_name}_{optical_flow_method.return_params_string()}.png", "arrows", divisor=16, plot=False)
            data.append([optical_flow_method_name, optical_flow_method.return_params_string(), t1 - t0, msen, pepn])

    df = pd.DataFrame(data, columns=["method_name", "params", "exe_time", "msen", "pepn"])
    df.to_csv("results.csv", index=None, sep=',')
    
def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("-n",
                        dest="num_runs",
                        default=1,
                        type=int,
                        help="Number of runs if using random parameters")
    parser.add_argument("-r", 
                        default=False, 
                        dest="random_params",
                        action="store_true",
                        help="Use random parameters")
    args = parser.parse_args()

    return args.num_runs, args.random_params
        
if __name__ == "__main__":
    num_runs, random_params = parse_arguments()
    task1_2(random_params, num_runs)
    