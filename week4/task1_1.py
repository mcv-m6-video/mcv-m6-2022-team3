import os
from time import time
from itertools import product
import imageio
from tqdm import tqdm

import numpy as np
import cv2
import pandas as pd

from optical_flow import block_matching_flow
from utils import load_optical_flow, display_OpticalFlow
from evaluation import eval_opticalflow

def task1_1():
    # Optical Flow with Block Matching
    img_prev = cv2.imread('imgs/000045_10.png', cv2.IMREAD_GRAYSCALE)
    img_next = cv2.imread('imgs/000045_11.png', cv2.IMREAD_GRAYSCALE)

    # Image 157
    mask, u_gt, v_gt = load_optical_flow('imgs/000045_10_gt.png')
    _, u, v = load_optical_flow('imgs/LKflow_000045_10.png')

    # Parameter combinations
    motion_types = ["forward"]
    search_areas = [64]
    block_sizes = [32]
    step_sizes= [1]
    kinds = ['ncc']
    os.makedirs("results", exist_ok=True)

    data = []
    for motion_type, search_area, block_size, step_size, kind in product(motion_types, search_areas, block_sizes, step_sizes, kinds):
        t0 = time()
        # Estimate block matching with: motion_type, search_area, block_size
        flow = block_matching_flow(img_prev, img_next, block_size, search_area, step_size, motion_type, kind)
        u, v = flow[:,:,0], flow[:,:,1]
        t1 = time()
        msen, pepn = eval_opticalflow(u, v, u_gt, v_gt, mask)
        display_OpticalFlow(img_prev, u, v, f"results/arrows_col_seq045_m{motion_type}_s{search_area}_b{block_size}_st{step_size}_k{kind}.png", "color_wheel", divisor=16, plot=False)
        print(f"motion type {motion_type}, search area {search_area}, block size {block_size}, step size {step_size}, kind {kind} | MSEN:", msen, "PEPN:", pepn)
        data.append([motion_type, search_area, block_size, step_size, kind, t1 - t0, msen, pepn])

    df = pd.DataFrame(data, columns=["motion_type", "search_area", "block_size", "step_size", "kind", "exe_time", "msen", "pepn"])
    df.to_csv("results.csv", index=None, sep=',')
    
def task1_2():
    # Evaluate Optical Flow algorithms
    a = 4
    
if __name__ == "__main__":
    task1_1()
    