import numpy as np
import cv2
from matplotlib import pyplot as plt

def morphological_filtering(mask):
    # 1. Remove noise
    mask = mask.astype(np.uint8)
    mask = cv2.medianBlur(mask, 5)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)

    # 2. Connect regions and remove shadows
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 10))
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    
    # 3. Fill convex hull of connected components
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    hull_list = []
    for i in range(len(contours)):
        hull = cv2.convexHull(contours[i])
        hull_list.append(hull)
    
    mask2 = np.zeros_like(mask)
    for i in range(len(hull_list)):
        mask2 = cv2.drawContours(mask2, hull_list, i, color=(1), thickness=cv2.FILLED)
    mask = mask2
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 50))
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    return mask

