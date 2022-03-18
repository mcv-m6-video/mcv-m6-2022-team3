from distutils.log import debug
import numpy as np
import cv2
from matplotlib import pyplot as plt
import copy


def fill_holes(mask):
    im_floodfill = mask.astype(np.uint8).copy()
    h, w = im_floodfill.shape[:2]
    filling_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(im_floodfill, filling_mask, (0, 0), 1)
    return mask.astype(np.uint8) | (1 - im_floodfill)


def filter_noise(mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50, 50))
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    mask = cv2.medianBlur(mask, 7)
    return mask

def filter_noise1(mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    mask = cv2.medianBlur(mask, 5)
    #kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    #mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    return mask

def closing2(mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 50))
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    return mask

def morphological_filtering(mask, debug_ops=True):
    """
    Apply morphological operations to prepare pixel candidates to be selected as
    a traffic sign or not.
    """
    mask_filtered = filter_noise1(mask)
    if debug_ops:
        cv2.imshow("filtered_fg1", mask_filtered.astype(np.uint8)*255); cv2.waitKey(0)

    #mask_filled = fill_holes(mask_filtered)
    if debug_ops:
        cv2.imshow("filled_fg", mask_filtered.astype(np.uint8)*255); cv2.waitKey(0)
    mask_finetuned = closing2(mask_filtered)
    
    #mask_filled = fill_holes(mask_finetuned)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 10))
    mask_final = cv2.morphologyEx(mask_finetuned.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    
    """mask_filtered = filter_noise(mask_filled)
    if debug_ops:
        cv2.imshow("filtered_fg", mask_filtered.astype(np.uint8)*255); cv2.waitKey(0)"""
    return mask_final


def granulometry(mask, steps, dict_kernels):
    """
        Granulometry study used to choose the size of kernels
    """
    new_mask = copy.deepcopy(mask.astype(np.uint8))
    g_curve = np.zeros(steps)
    for i in range(steps - 1):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (i + 1, i + 1))
        remain = cv2.morphologyEx(new_mask, cv2.MORPH_OPEN, kernel)
        g_curve[i + 1] = np.sum(np.abs(np.count_nonzero(remain)))
        new_mask = remain

    g_curve[0] = g_curve[1]
    pecstrum = -np.gradient(g_curve)
    pecstrum = np.array(pecstrum)
    plt.plot(pecstrum)

    for i in range(2):
        peak = np.where(pecstrum == max(pecstrum))[0][0]
        if peak in dict_kernels.keys():
            dict_kernels[peak] += 1
        else:
            dict_kernels[peak] = 1
        pecstrum[peak] = min(pecstrum)
    return dict_kernels