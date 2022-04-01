import numpy as np

def block_matching_flow(img_prev, img_next, block_size, search_area, motion_type):
    """
    Compute block-matching based motion estimation
    """
    if motion_type == 'forward':
        reference = img_prev
        target = img_next
    elif motion_type == 'backward':
        reference = img_next
        target = img_prev
    
    height, width = reference.shape[:2]
    flow_field = np.zeros((height, width, 2), dtype=float)

    return flow_field