import os
from time import time
from itertools import product
import imageio
from tqdm import tqdm

import numpy as np
import cv2
import pandas as pd

from optical_flow import block_matching_flow
from utils import OpticalFlow, eval_opticalflow

def task1_1():
    # Optical Flow with Block Matching
    img_prev = cv2.imread('imgs/000045_10.png', cv2.IMREAD_GRAYSCALE)
    img_next = cv2.imread('imgs/000045_11.png', cv2.IMREAD_GRAYSCALE)
    optical_flow = OpticalFlow(".")

    # Image 157
    mask, u_gt, v_gt = optical_flow[45]
    _, u, v = optical_flow.load_optical_flow('imgs/LKflow_000045_10.png')

    motion_types = ['forward', 'backward']
    search_areas = [2**i for i in range(4, 8)]
    block_sizes = [2**i for i in range(2, 6)]

    data = []
    for motion_type, search_area, block_size in product(motion_types, search_areas, block_sizes):
        t0 = time()
        # Estimate block matching with: motion_type, search_area, block_size
        #flow = block_matching_flow(img_prev, img_next, motion_type=m, search_area=p, block_size=n, algorithm='corr')
        flow = 4
        u, v = flow
        t1 = time()
        msen, pepn = eval_opticalflow(u, v, u_gt, v_gt, mask)
        data.append([motion_type, search_area, block_size, t1 - t0, msen, pepn])
    
    # Write pandas dataframe with result
    # motion_type = forward / backward, search_area = int, block_size = int, msen, pepn, runtime
    df = pd.DataFrame(data, columns=["motion_type", "search_area", "block_size", "exe_time", "msen", "pepn"])
    df.to_csv("results.csv", index=None, sep=' ')
    
if __name__ == "__main__":
    task1_1()
    