import numpy as np
import math

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