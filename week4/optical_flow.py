from random import random
import numpy as np
import math
import cv2
import numpy as np
from scipy.signal import convolve2d

class OpticalFlowEstimator:
    def estimate_optical_flow(self, image_prev, image_next):
        pass

""" class LucasKanade(OpticalFlowEstimator):
    def __init__(self):
        # params for ShiTomasi corner detection
        self.feature_params = dict( maxCorners = 100,
                            qualityLevel = 0.3,
                            minDistance = 7,
                            blockSize = 7 )
        # Parameters for lucas kanade optical flow
        self.lk_params = dict( winSize  = (15, 15),
                        maxLevel = 2,
                        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        
    def estimate_optical_flow(self, image_prev, image_next):
        image_prev_grayscale = cv2.cvtColor(image_prev, cv2.COLOR_BGR2GRAY)
        p0 = cv2.goodFeaturesToTrack(image_prev_grayscale, mask = None, **self.feature_params)
        
        # calculate optical flow
        image_next_grayscale = cv2.cvtColor(image_next, cv2.COLOR_BGR2GRAY)
        p1, st, err = cv2.calcOpticalFlowPyrLK(image_prev_grayscale, image_next_grayscale, p0, None, **self.lk_params)
        pass """

class HornSchunk(OpticalFlowEstimator):
    # Source: https://github.com/scivision/pyoptflow
    def __init__(self, random_params=False):
        self.HSKERN = np.array(
            [[1 / 12, 1 / 6, 1 / 12], [1 / 6, 0, 1 / 6], [1 / 12, 1 / 6, 1 / 12]], float
        )
        self.kernelX = np.array([[-1, 1], [-1, 1]]) * 0.25  # kernel for computing d/dx
        self.kernelY = np.array([[-1, -1], [1, 1]]) * 0.25  # kernel for computing d/dy
        self.kernelT = np.ones((2, 2)) * 0.25
    
        # alpha: float    regularization constant
        # Niter: int      number of iteration
        if random_params:
            self.alpha = np.random.uniform(0.01, 100, 1)[0]
            self.Niter = 100
        else:
            self.alpha = 10
            self.Niter = 100
    
    def estimate_optical_flow(self, image_prev, image_next):
        im1 = image_prev
        im2 = image_next
            
        im1 = im1.astype(np.float32)
        im2 = im2.astype(np.float32)

        # set up initial velocities
        uInitial = np.zeros([im1.shape[0], im1.shape[1]], dtype=np.float32)
        vInitial = np.zeros([im1.shape[0], im1.shape[1]], dtype=np.float32)

        # Set initial value for the flow vectors
        U = uInitial
        V = vInitial

        # Estimate derivatives
        [fx, fy, ft] = self.computeDerivatives(im1, im2)

        # Iteration to reduce error
        for _ in range(self.Niter):
            # Compute local averages of the flow vectors
            uAvg = convolve2d(U, self.HSKERN, "same")
            vAvg = convolve2d(V, self.HSKERN, "same")
            # common part of update step
            der = (fx * uAvg + fy * vAvg + ft) / (self.alpha ** 2 + fx ** 2 + fy ** 2)
            # iterative step
            U = uAvg - fx * der
            V = vAvg - fy * der

        return np.concatenate([U[:,:,np.newaxis], V[:,:,np.newaxis]], -1)
    
    def computeDerivatives(self, im1, im2):

        fx = convolve2d(im1, self.kernelX, "same") + convolve2d(im2, self.kernelX, "same")
        fy = convolve2d(im1, self.kernelY, "same") + convolve2d(im2, self.kernelY, "same")

        # ft = im2 - im1
        ft = convolve2d(im1, self.kernelT, "same") + convolve2d(im2, -self.kernelT, "same")

        return fx, fy, ft
        
    def return_params_string(self):
        return f"{self.alpha}_{self.Niter}"
        
class FarneBack(OpticalFlowEstimator):
    def __init__(self, random_params=False):
        if random_params:
            self.pyr_scale = np.random.uniform(0.25, 0.75, 1)[0]
            self.levels = np.random.randint(2, 5)
            self.win_size = np.random.randint(5, 17)
            self.iterations = np.random.randint(1, 5)
            self.poly_n = np.random.randint(2, 7)
            self.poly_sigma = np.random.uniform(1.1, 1.5, 1)[0]
            self.use_initial_flow = 0
        else:
            self.pyr_scale = 0.5
            self.levels = 3
            self.win_size = 15
            self.iterations = 3
            self.poly_n = 5
            self.poly_sigma = 1.2
            self.use_initial_flow = 0
    
    def estimate_optical_flow(self, image_prev, image_next):
        image_prev_grayscale = image_prev
        image_next_grayscale = image_next
        flow = cv2.calcOpticalFlowFarneback(image_prev_grayscale, image_next_grayscale, None, self.pyr_scale, self.levels, self.win_size, self.iterations, self.poly_n, self.poly_sigma, self.use_initial_flow)
        return flow
    
    def return_params_string(self):
        return f"{self.pyr_scale}_{self.levels}_{self.win_size}_{self.iterations}_{self.poly_n}_{self.poly_sigma}_{self.use_initial_flow}"

def get_opticalflow_model(method_name, random_params=False):
    if method_name == "lk": # Lucas-Kanade
        pass
    elif method_name == "fb": # FarneBack
        return FarneBack(random_params=random_params)
    elif method_name == "hs": # Horn-Schunk
        return HornSchunk(random_params=random_params)
    else:
        raise ValueError(f"Method '{method_name}' not found.")

def matching_cost(patch1, patch2, kind="ssd"):
    if kind == "ssd":
        return np.sum( (patch1.astype(np.float32)-patch2.astype(np.float32))**2 )
    elif kind == "sad":
        return np.sum( np.abs(patch1.astype(np.float32)-patch2.astype(np.float32)) )
    elif kind == "ncc":
        patch1 = patch1.astype(np.float32)
        patch2 = patch2.astype(np.float32)
        epsilon = 1e-6
        mean_I1 = np.mean(patch1)
        mean_I2 = np.mean(patch2)
        std_I1 = max(np.sqrt(np.sum((patch1 - mean_I1)**2)), epsilon)
        std_I2 = max(np.sqrt(np.sum((patch2 - mean_I2)**2)), epsilon)
        cost = np.sum((patch1 - mean_I1)*(patch2 - mean_I2)) / (std_I1*std_I2)
        return -cost
    else:
        raise Exception("Not found")

def block_matching_flow(img_prev, img_next, block_size, search_area, step_size, motion_type, kind):
    """
    Compute block-matching based motion estimation
    """

    if motion_type == 'forward':
        reference = img_prev
        target = img_next

        height, width = reference.shape[:2]
        flow_field = np.zeros((height, width, 2), dtype=float)

        blocks_list = []
        width = img_prev.shape[1]
        height = img_prev.shape[0]

        # For each block in the reference image
        for i in range(0, width, block_size):
            for j in range(0, height, block_size):
                if reference.shape[0] > block_size+j and reference.shape[1] > block_size+i: 
                    #print(j,block_size+j, i,block_size+i)
                    block = reference[j:block_size+j, i:block_size+i]
                    
                    # Search in the target image
                    us = [u for u in np.arange(-search_area//2, search_area//2 + 1, step_size) if 0 <= u+i < target.shape[1] and 0 <= u+i+block_size < target.shape[1]]
                    vs = [v for v in np.arange(-search_area//2, search_area//2 + 1, step_size) if 0 <= v+j < target.shape[0] and 0 <= v+j+block_size < target.shape[0]]
                    
                    min_cost = math.inf
                    opt_u, opt_v = None, None
                    for u in us:
                        for v in vs:
                            block_to_compare = target[j+v:block_size+j+v, i+u:block_size+i+u]
                            #print(block.shape, block_to_compare.shape)
                            cost = matching_cost(block, block_to_compare, kind)
                            if cost < min_cost:
                                min_cost = cost
                                opt_u, opt_v = u, v
                    
                    # Assign optical flow to block u / v order?
                    flow_field[j:block_size+j, i:block_size+i, 0] = opt_u
                    flow_field[j:block_size+j, i:block_size+i, 1] = opt_v

    elif motion_type == 'backward':
        reference = img_next
        target = img_prev

        height, width = reference.shape[:2]
        flow_field = np.zeros((height, width, 2), dtype=float)

        blocks_list = []
        width = img_prev.shape[1]
        height = img_prev.shape[0]

        # For each block in the reference image
        for i in range(0, width, block_size):
            for j in range(0, height, block_size):
                if reference.shape[0] > block_size+j and reference.shape[1] > block_size+i: 
                    #print(j,block_size+j, i,block_size+i)
                    block = reference[j:block_size+j, i:block_size+i]
                    
                    # Search in the target image
                    us = [u for u in np.arange(-search_area//2, search_area//2 + 1, step_size) if 0 <= u+i < target.shape[1] and 0 <= u+i+block_size < target.shape[1]]
                    vs = [v for v in np.arange(-search_area//2, search_area//2 + 1, step_size) if 0 <= v+j < target.shape[0] and 0 <= v+j+block_size < target.shape[0]]
                    
                    min_cost = math.inf
                    opt_u, opt_v = None, None
                    for u in us:
                        for v in vs:
                            block_to_compare = target[j+v:block_size+j+v, i+u:block_size+i+u]
                            #print(block.shape, block_to_compare.shape)
                            cost = matching_cost(block, block_to_compare)
                            if cost < min_cost:
                                min_cost = cost
                                opt_u, opt_v = u, v
                    
                    # Assign optical flow to block u / v order?
                    flow_field[j-opt_u:block_size+j-opt_u, i-opt_v:block_size+i-opt_v, 0] = -opt_u
                    flow_field[j-opt_u:block_size+j-opt_u, i-opt_v:block_size+i-opt_v, 1] = -opt_v

    return flow_field