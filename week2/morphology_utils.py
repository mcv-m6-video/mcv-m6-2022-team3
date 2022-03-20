from distutils.log import debug
import numpy as np
import cv2
from matplotlib import pyplot as plt
import copy
from scipy import ndimage


"""def morphological_filtering(mask, debug_ops=True):
    debug_ops = False
    mask = mask.astype(np.uint8)
    # median blur
    mask = cv2.medianBlur(mask, 5)

    if debug_ops:
        cv2.imshow("filtered_fg3", mask.astype(np.uint8)*255); cv2.waitKey(0)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 30))
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    if debug_ops:
        cv2.imshow("filtered_fg4", mask.astype(np.uint8)*255); cv2.waitKey(0)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    hull_list = []
    for i in range(len(contours)):
        hull = cv2.convexHull(contours[i])
        hull_list.append(hull)
    
    mask2 = np.zeros_like(mask)
    for i in range(len(hull_list)):
        mask2 = cv2.drawContours(mask2, hull_list, i, color=(1), thickness=cv2.FILLED)
    #mask = np.max((mask, mask2), axis=0)
    mask = mask2
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1))
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    if debug_ops:
        cv2.imshow("filtered_fg4", mask.astype(np.uint8)*255); cv2.waitKey(0)
    # Then label
    
    #mask = fill_holes(mask)
    #kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 20))
    #filled = cv2.erode(mask, kernel1, iterations=1)
    ## Dilate
    #filled = cv2.dilate(filled, kernel, iterations=1)
    return mask"""

""" def morphological_filtering(mask, debug_ops=True):
    debug_ops = False
    mask = mask.astype(np.uint8)
    # median blur
    mask = cv2.medianBlur(mask, 5)
    # opening
    if debug_ops:
        cv2.imshow("filtered_fg1", mask.astype(np.uint8)*255); cv2.waitKey(0)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    if debug_ops:
        cv2.imshow("filtered_fg2", mask.astype(np.uint8)*255); cv2.waitKey(0)
    # closing
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    if debug_ops:
        cv2.imshow("filtered_fg3", mask.astype(np.uint8)*255); cv2.waitKey(0)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 10))
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    if debug_ops:
        cv2.imshow("filtered_fg4", mask.astype(np.uint8)*255); cv2.waitKey(0)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    hull_list = []
    for i in range(len(contours)):
        hull = cv2.convexHull(contours[i])
        hull_list.append(hull)
    
    mask2 = np.zeros_like(mask)
    for i in range(len(hull_list)):
        mask2 = cv2.drawContours(mask2, hull_list, i, color=(1), thickness=cv2.FILLED)
    #mask = np.max((mask, mask2), axis=0)
    mask = mask2
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1))
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    if debug_ops:
        cv2.imshow("filtered_fg4", mask.astype(np.uint8)*255); cv2.waitKey(0)
    # Then label
    
    #mask = fill_holes(mask)
    #kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 20))
    #filled = cv2.erode(mask, kernel1, iterations=1)
    ## Dilate
    #filled = cv2.dilate(filled, kernel, iterations=1)
    return mask """

def morphological_filtering(mask, debug_ops=True):
    debug_ops = False
    mask = mask.astype(np.uint8)
    # median blur
    mask = cv2.medianBlur(mask, 5)
    # opening
    if debug_ops:
        cv2.imshow("filtered_fg1", mask.astype(np.uint8)*255); cv2.waitKey(0)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    if debug_ops:
        cv2.imshow("filtered_fg2", mask.astype(np.uint8)*255); cv2.waitKey(0)
    # closing
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    if debug_ops:
        cv2.imshow("filtered_fg3", mask.astype(np.uint8)*255); cv2.waitKey(0)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 10))
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    if debug_ops:
        cv2.imshow("filtered_fg4", mask.astype(np.uint8)*255); cv2.waitKey(0)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    hull_list = []
    for i in range(len(contours)):
        hull = cv2.convexHull(contours[i])
        hull_list.append(hull)
    
    mask2 = np.zeros_like(mask)
    for i in range(len(hull_list)):
        mask2 = cv2.drawContours(mask2, hull_list, i, color=(1), thickness=cv2.FILLED)
    #mask = np.max((mask, mask2), axis=0)
    mask = mask2
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 50))
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    if debug_ops:
        cv2.imshow("filtered_fg4", mask.astype(np.uint8)*255); cv2.waitKey(0)
    # Then label
    
    #mask = fill_holes(mask)
    #kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 20))
    #filled = cv2.erode(mask, kernel1, iterations=1)
    ## Dilate
    #filled = cv2.dilate(filled, kernel, iterations=1)
    return mask

