import os
from bs4 import BeautifulSoup
import numpy as np
from tqdm import tqdm
import pandas as pd
import cv2
import random
import copy
import flow_vis
from PIL import Image
import matplotlib.pyplot as plt
import torch

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

def parse_xml_reacts(path_to_anno, discard_parked=True, return_ids=False):

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
        track_id = int(item.get('id'))
        if item.get('label') == 'car':
            boxs_in_track = item.find_all('box')
            for box in boxs_in_track:
                # print(box.get('frame'))
                if discard_parked:
                    if box.attribute.text == "false":
                        frame_n = int(box.get('frame'))
                        if frame_n not in frame_dict:
                            frame_dict[frame_n] = []
                        if return_ids:
                            frame_dict[frame_n].append([float(box.get('xtl')), float(box.get('ytl')),
                                float(box.get('xbr')), float(box.get('ybr')), track_id])
                        else:
                            frame_dict[frame_n].append([float(box.get('xtl')), float(box.get('ytl')),
                                float(box.get('xbr')), float(box.get('ybr'))])
                else:
                    frame_n = int(box.get('frame'))
                    if frame_n not in frame_dict:
                        frame_dict[frame_n] = []
                    if return_ids:
                        frame_dict[frame_n].append([float(box.get('xtl')), float(box.get('ytl')),
                            float(box.get('xbr')), float(box.get('ybr')), track_id])
                    else:
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

def parse_feature_vectors(path):
    """
    Input:
        - path to annotation txt where 512 values per row of feature vectors are stored
    Output:
        list of torch.tensors of shape [1,512]
    """

    COL_NAMES = [str(x) for x in range(0,512)]
    COL_NAMES.insert(0,'frame')
    df = pd.read_csv(path, delimiter=',', names=COL_NAMES)

    list_of_feature_vectors = []
    for row in range(df.shape[0]):
        tab = np.array(df.iloc[row,1:])
        tab = tab.reshape((1,512))
        tab_torch = torch.from_numpy(tab).type(torch.float32)
        list_of_feature_vectors.append(tab_torch)
    
    return [df['frame'].values, np.array(list_of_feature_vectors)]


def predictions_to_gt_format(frame_ids, tot_boxes):
    preds_formatted = {}
    for i in np.unique(frame_ids):
        preds_formatted[i] = tot_boxes[frame_ids == i]
    return preds_formatted

def read_annotations(annotations, return_ids=False):
    # gt_rect format: 
    # {frame_number: [[x_min1, y_min1, x_max1, y_max1], [x_min2, y_min2, x_max2, y_max2], [], ...], ...}
    print("Reading annotations from:", annotations)
    gt_rect = parse_xml_reacts(annotations, discard_parked=False, return_ids=return_ids)
    return gt_rect

def image_to_tensor(img, device):
    img = torch.tensor(img.astype(np.float32) / 255).to(device)
    img = img.permute(2, 0, 1)
    return img

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


## Optical flow utils

class OpticalFlow:
    """Helper class to load optical flow files and also work with KITTI"""
    def __init__(self, dataset_path=None):
        self.dataset_path = dataset_path
        self.image_dir = os.path.join(dataset_path or "", "training", "image_0")
        self.flow_dir = os.path.join(dataset_path or "", "training", "flow_noc")

    @staticmethod
    def load_optical_flow(img_path):
        flow_data = cv2.imread(img_path, cv2.IMREAD_UNCHANGED).astype('float')
        mask = (flow_data[:,:,0]).astype('bool')
        v = (flow_data[:,:,1] - 2**15)/64
        u = (flow_data[:,:,2] - 2**15)/64

        return mask, u, v

    def load_kitti_image(self, idx):
        img_path = os.path.join(self.image_dir, f"{idx:06}_10.png")
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        return img

    def __getitem__(self, idx):
        if self.dataset_path is None:
            raise ValueError("Please initialize with the dataset: KITTI(dataset_path)")

        # Dataset paths
        frame1 = f"{idx:06}_10.png"
        frame2 = f"{idx:06}_11.png"
        img1_path = os.path.join(self.image_dir, frame1)
        img2_path = os.path.join(self.image_dir, frame2)
        flow_path = os.path.join(self.flow_dir, frame1)

        # Load images and optical flow
        img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
        mask, u, v = self.load_optical_flow(flow_path)

        return mask, u, v


def load_optical_flow(img_path):
        flow_data = cv2.imread(img_path, cv2.IMREAD_UNCHANGED).astype('float')
        mask = (flow_data[:,:,0]).astype('bool')
        v = (flow_data[:,:,1] - 2**15)/64
        u = (flow_data[:,:,2] - 2**15)/64

        return mask, u, v

from PIL import Image
import matplotlib.pyplot as plt

def display_OpticalFlow(image, u, v, name, type_plot, divisor=3, h_parts=14, v_parts=4, plot=False):
    """
    Save image with a visualisation of a flow over the top.
    A divisor controls the density of the quiver plot.

    Args:
        image (cv2): imatge over which we want to plot the Optical Flow.
        u,v (HxW arrays): Optical Flow that we want to display over the image.
        name (str): path where to save the image with the Optical Flow over it.
        type_plot (str): "arrows", "color_wheel" or "simplification".
        divisor (int): every how many pixels we want to display the Optical Flow vector.
                        Used in type_plot="arrows".
        h_parts (int): number of horizontal cells to divide the image. Used in type_plot="simplification".
        v_parts (int): number of vertical cells to divide the image. Used in type_plot="simplification".
        plot (bool): if we want to display the plots or not.
    """
    flow = np.dstack((u,v))
    if type_plot == "arrows":
        OpticalFlow_arrows(image, flow, divisor, name)
    elif type_plot == "color_wheel":
        flow_color = flow_vis.flow_uv_to_colors(u, v, convert_to_bgr=False)
        blend_imgs(image, flow_color, name)
    elif type_plot == "simplification":
        OpticalFlow_simplification(image, u, v, h_parts, v_parts, name)
    else:
        print('ERROR: Incorrect plot type. Options: arrows or color_wheel.')

    if plot:
        flow_image = Image.open(name)
        plt.figure(figsize=(12,4))
        plt.imshow(flow_image)


def OpticalFlow_arrows(image, flow, divisor, name):
    """
    Save image with a visualisation of a flow over the top.
    A divisor controls the density of the quiver plot.

    Args:
        image (cv2): imatge over which we want to plot the Optical Flow.
        flow (HxWx2 array): Optical Flow that we want to display over the image. Format: np.dstack((u,v))
        divisor (int): every how many pixels we want to display the Optical Flow vector.
        name (str): path where to save the image with the Optical Flow over it.
    """
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    picture_shape = np.shape(image)
    # determine number of quiver points there will be
    Imax = int(picture_shape[0]/divisor)
    Jmax = int(picture_shape[1]/divisor)

    max_val = np.sqrt(np.max(flow[:,:,0].reshape(-1)**2 + flow[:,:,1].reshape(-1)**2))

    # draw the arrows into the image
    for i in range(1, Imax):
        for j in range(1, Jmax):
            X1 = (i)*divisor
            Y1 = (j)*divisor
            X2 = int(X1 + flow[X1,Y1,1])
            Y2 = int(Y1 + flow[X1,Y1,0])
            X2 = np.clip(X2, 0, picture_shape[0])
            Y2 = np.clip(Y2, 0, picture_shape[1])

            magnitude = np.sqrt(flow[X1,Y1,1]**2 + flow[X1,Y1,0]**2)
            col = int(255 * (magnitude / max_val))
            
            #add all the lines to the image
            image = cv2.arrowedLine(image, (Y1,X1),(Y2,X2), [255, col, 0], int(1 + 6*(col/255)))

    cv2.imwrite(name, image)


def blend_imgs(img1, img2, name):
    """Blend two images and saves the resulting one."""
    background = Image.fromarray(img1)
    overlay = Image.fromarray(img2)

    background = background.convert("RGBA")
    overlay = overlay.convert("RGBA")

    new_img = Image.blend(background, overlay, 0.8)
    new_img.save(name,"PNG")


def OpticalFlow_simplification(image, u, v, h_parts, v_parts, name):
    """
    Save image with a simplified visualisation of a flow over the top.
    h_parts and v_parts controls in how many cells we want to devide the image.

    Args:
        image (cv2): imatge over which we want to plot the Optical Flow.
        u,v (HxW arrays): Optical Flow that we want to display over the image.
        h_parts (int): number of horizontal cells to divide the image.
        v_parts (int): number of vertical cells to divide the image.
        name (str): path where to save the image with the Optical Flow over it.
    """
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # size of the cells
    f_box = round(u.shape[0] / v_parts)
    c_box = round(u.shape[1] / h_parts)

    # draw the grid
    for i in range(1,v_parts):
        image = cv2.line(image, (0, i*f_box), (u.shape[1], i*f_box), (255, 255, 255), 1, 1)

    for i in range(1,h_parts):
        image = cv2.line(image, (i*c_box, 0), (i*c_box, u.shape[0]), (255, 255, 255), 1, 1)

    # compute the centers of the cells
    centers = [(x,y) for y in range(round(f_box/2),u.shape[0],f_box) for x in range(round(c_box/2),u.shape[1],c_box)]

    # compute the mean direction of each cell
    directions = []
    u_max = -1; v_max = -1
    for i in range(0,u.shape[0],f_box):
        for j in range(0,u.shape[1],c_box):
            u_mean = np.mean(u[i:i+f_box, j:j+c_box])
            v_mean = np.mean(v[i:i+f_box, j:j+c_box])
            directions.append((u_mean,v_mean))
            if abs(u_mean) > u_max:
                u_max = abs(u_mean)
            if abs(v_mean) > v_max:
                v_max = abs(v_mean)

    # normalize the directions
    directions = [(uu/u_max*(c_box/2), vv/v_max*(f_box/2)) for uu,vv in directions]

    # draw the arrows
    for c,d in zip(centers, directions):
        image = cv2.arrowedLine(image, c, (int(round(c[0]+d[0])), int(round(c[1]+d[1]))), [0, 0, 255], 2, 2, 0, 0.2)

    # save
    cv2.imwrite(name, image)
