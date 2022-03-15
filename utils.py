from bs4 import BeautifulSoup
import numpy as np
from tqdm import tqdm
import pandas as pd
import random
import copy

WIDTH = 1920
HEIGHT = 1080


def generate_noisy_bboxes(gt_bboxes, rank=10, std_coords=0, resizing_factor=1, prob_drop_detections=0, prob_new_detections=0 , dropout = 0):
    """Generate noisy bounding boxes from groundtruth annotations.

    Args:
        gt_bboxes (Dict): GT annotations in format: 
            {
                'imagename': {
                    'bbox': [['x1', 'y1', 'x2', 'y2'], ['x1', 'y1', 'x2', 'y2']],
                    'difficult':[True, False],
                    'det': [False, False]
                }, ...
            }
        rank (int > 0): How many random rankings are generated 
        std_coords (float): Standard deviation for Gaussian noise
        resizing_factor (1 > float > 0): Resizing factor of the bounding box
        prob_drop_detections (1 > float > 0): Probability of dropping bounding box in the frame
        prob_new_detections (1 > float > 0): Probability of adding new bounding box in the frame (how to determine width/height)
    
    Return:
        [np.ndarray of frame ids (number of frames,), np.ndarray of bounding box detections (number of frames, 4), list of confidences (rank, number of frames)]
    """
    gt_bboxes = copy.deepcopy(gt_bboxes)
    tot_boxes = []
    confidences = np.array([])
    frame_ids = []
    for frame_id in sorted(list(gt_bboxes.keys())):
        frame_bboxes = gt_bboxes[frame_id]
        # Modify frame_bboxes / add new ones / remove

        if std_coords != 0:
            noise = np.random.normal(0, std_coords, size=(len(frame_bboxes), 4))
            for i in range(0, len(frame_bboxes)):
                x0, y0, x1, y1 = frame_bboxes[i]
                frame_bboxes[i] = [x0-noise[i,0], y0-noise[i,1], x1+noise[i,2], y1+noise[i,3]] 

        #resize the bounding box by factor 0-1
        if resizing_factor != 1:
            for i in range(0,len(frame_bboxes)):
                x0, y0, x1, y1 = frame_bboxes[i]
                w, h = x1-x0, y1-y0
                resized_item = np.asarray([x0-(w*(resizing_factor - 1)/2), y0-(h*(resizing_factor - 1)/2), x1+(w*(resizing_factor - 1)/2), y1+(h*(resizing_factor - 1)/2)])
                resized_item = np.round(resized_item, decimals = 2)
                frame_bboxes[i] = list(resized_item)

        #adding new bounding box with prob_new_detection probability
        if prob_new_detections != 0:
            if random.random() < prob_new_detections:
                x1 = random.randint(0,WIDTH)    
                x2 = random.randint(x1,WIDTH)
                y1 = random.randint(0,HEIGHT)
                y2 = random.randint(y1,HEIGHT)
                frame_bboxes.append([x1,y1,x2,y2])
        
        #deleting bounding box with prob_drop_detection probability
        # if prob_drop_detections != 0:
        #     frame_bboxes = np.array(frame_bboxes)
        #     dets_to_keep = np.random.random(len(frame_bboxes)) > prob_drop_detections
        #     frame_bboxes = frame_bboxes[dets_to_keep]
        #     frame_bboxes = frame_bboxes.tolist()
                #deleting bounding box with prob_drop_detection probability
        if prob_drop_detections!=0:
            if random.random() < prob_drop_detections:
                frame_bboxes.pop(random.randint(0,len(frame_bboxes)-1))

        #deletes random index X times as defined by dropout amount, 0.5 deletes half of indexes
        if dropout != 0:
            boxes_to_delete = int(len(frame_bboxes)*dropout)
            for i in range(boxes_to_delete):
                frame_bboxes.pop(random.randint(0,len(frame_bboxes)-1))
        
        # Add bboxes, ids and confidences to total bboxes, ids and confidences
        if len(frame_bboxes) > 0:
            tot_boxes.append(np.array(frame_bboxes))
            frame_ids = frame_ids + [frame_id]*len(frame_bboxes)
    
    frame_ids = np.array(frame_ids)
    confidences = np.random.random((rank, len(frame_ids)))
    tot_boxes = np.concatenate(tot_boxes, 0)
    
    # Return in required format for evaluation
    return frame_ids, tot_boxes, confidences
    
    

def parse_xml_reacts(path_to_anno):
    
    # Reading the data inside the xml
    # file to a variable under the name
    # data
    with open(path_to_anno, 'r') as f:
        data = f.read()
    
    # Passing the stored data inside
    # the beautifulsoup parser, storing
    # the returned object
    Bs_data = BeautifulSoup(data, "xml")
    
    # Finding all instances of tag
    # `unique`
    tracks_all = Bs_data.find_all('track')
    frame_dict = {}
    for item in tqdm(tracks_all):
        if item.get('label') == 'car':
            boxs_in_track = item.find_all('box')
            for box in boxs_in_track:
                # print(box.get('frame'))
                frame_n = int(box.get('frame')) 
                if frame_n not in frame_dict:
                    frame_dict[frame_n] = []
                frame_dict[frame_n].append([float(box.get('xtl')), float(box.get('ytl')), 
                    float(box.get('xbr')), float(box.get('ybr'))])
    return frame_dict

def parse_predictions_rects(path):
    """
    Input:
        - Path to annotation txt where each row contains information about:
            frame number, bounding_box_coords, confidence
    Output format
        dict[frame_num] = [[x1, y1, x2, y2]]
    """
    # COL_NAMES = ['frame', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf', 'x', 'y', 'z']
    COL_NAMES = ['frame', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf']
    
    predictions_dict = {}
    df = pd.read_csv(path, delimiter=',', usecols=[0, 2, 3, 4, 5, 6], names=COL_NAMES)
    df.iloc[:,0] = df.iloc[:,0] - 1
    df.iloc[:,3] = df.iloc[:,3] + df.iloc[:,1]
    df.iloc[:,4] = df.iloc[:,4] + df.iloc[:,2]
    
    last_frame = 0
    bboxes = []
    for i in range(len(df)):
        frame = df['frame'][i]
        if last_frame != frame:
            predictions_dict[last_frame] = bboxes
            last_frame = frame
            bboxes = []
        
        bbox_p_conf = df.iloc[i, 1:-1]
        bboxes.append(list(bbox_p_conf.values))
    
    predictions_dict[frame] = bboxes
    
    bboxes = df[['bb_left', 'bb_top', 'bb_width', 'bb_height']].values
    
    return predictions_dict, [df['frame'].values, bboxes, df['conf'].values]
    
def predictions_to_gt_format(frame_ids, tot_boxes):
    preds_formatted = {}
    for i in np.unique(frame_ids):
        preds_formatted[i] = tot_boxes[frame_ids == i]
    return preds_formatted

def get_rect_iou(a, b):
    """Return iou for a single a pair of boxes"""
    x11, y11, x12, y12 = a
    x21, y21, x22, y22 = b

    xA = max(x11,x21)
    yA = max(y11,y21)
    xB = min(x12,x22)
    yB = min(y12,y22)
     
    # respective area of ​​the two boxes
    boxAArea=(x12-x11)*(y12-y11)
    boxBArea=(x22-x21)*(y22-y21)
     
     # overlap area
    interArea=max(xB-xA,0)*max(yB-yA,0)
     
     # IOU
    return interArea/(boxAArea+boxBArea-interArea)

def get_frame_iou(gt_rects, det_rects):
    """Return iou for a frame"""
    list_iou = []

    for gt in gt_rects:
        max_iou = 0
        for det in det_rects:
            iou = get_rect_iou(det, gt)
            if iou > max_iou:
                max_iou = iou
        
        if max_iou != 0:
            list_iou.append(max_iou)

    return np.mean(list_iou)