import cv2
from skimage.measure import label, regionprops
from matplotlib import pyplot as plt

def obtain_bboxes(mask, min_h=20, max_h=500, min_w=30, max_w=600, min_area=0.4, min_aspect_ratio=0.25, max_aspect_ratio=1.25):
    regions = regionprops(label(mask))
    
    detection = []
    for region in regions:
        bbox = region.bbox
        y1, x1, y2, x2 = bbox
        w,h = x2-x1,y2-y1
        if valid_bbox(mask, bbox, min_h, max_h, min_w, max_w, min_aspect_ratio, max_aspect_ratio, min_area):
            detection.append([x1, y1, x2, y2])
    
    if len(detection) > 4:
        detection = detection[:4]

    return detection

def valid_bbox(mask, bbox, min_h, max_h, min_w, max_w, min_ratio, max_ratio, min_area):
    box_h, box_w = bbox[2] - bbox[0], bbox[3] - bbox[1]
    if box_h > max_h or box_w > max_w:
        return False
    if box_h < min_h or box_w < min_w:
        return False
    if box_h/box_w > max_ratio or box_h/box_w < min_ratio:
        return False
    if mask[bbox[0]:bbox[2], bbox[1]:bbox[3]].astype(int).sum()/(box_h*box_w) < min_area:
        return False
    return True
