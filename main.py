from cProfile import run
from multiprocessing.connection import wait
import utils,os
import cv2
from argparse import ArgumentParser
import numpy as np
from evaluation import voc_eval
import matplotlib.pyplot as plt
from utils import generate_noisy_bboxes

ANNOTATIONS = 'Lab1/ai_challenge_s03_c010-full_annotation.xml'
DATA_FRAMES = 'data'
VIDEO_PATH = '/home/aszummer/Documents/MCV/M6/Team3-Project/Lab1/AICity_data/AICity_data/train/S03/c010/vdo.avi'
GT_COLOR = (0, 255, 0)
PREDICTED_COLOR = (255, 0, 0)
PREDICTIONS_MODELS = {
    'mask_rcnn': 'det_mask_rcnn.txt',
    'ssd512': 'det_ssd512.txt',
    'yolov3': 'det_yolo3.txt'
}
EXPERIMENTS_FOLDER = "experiments"
# python main.py -v /home/sergio/MCV/M6/data/AICity_data/train/S03/c010/vdo.avi -a /home/sergio/MCV/M6/data/ai_challenge_s03_c010-full_annotation.xml

display = True

def show_boxes(input_video, gt_rect, predictions):

    cap = cv2.VideoCapture(input_video)
    frame_n = 0
    wait_time = 1
    gt_frame_keys, pred_frame_keys = list(gt_rect.keys()), list(predictions.keys())
    mIoU = []

    while(True):
        ret, img = cap.read()
        if frame_n in gt_frame_keys:
            for r in gt_rect[frame_n]:
                img = cv2.rectangle(img, (int(r[0]), int(r[1])),  (int(r[2]), int(r[3])), GT_COLOR, 2)
        
        if frame_n in pred_frame_keys:
            for r in predictions[frame_n]:
                img = cv2.rectangle(img, (int(r[0]), int(r[1])),  (int(r[2]), int(r[3])), PREDICTED_COLOR, 2)
        
        if not ret: # End of video sequence
            break
        
        if display:
            display_frame = cv2.resize(img, tuple(np.int0(0.5*np.array(img.shape[:2][::-1]))))
            cv2.imshow('frame',display_frame)
            k = cv2.waitKey(wait_time)
            if k == ord('q'):
                break
            elif k == ord('p'):
                wait_time = int(not(bool(wait_time)))
        
        gt_bboxes = gt[frame_n]
        det_bboxes = predictions[frame_n]
        mIoU.append(utils.get_frame_iou(gt_bboxes, det_bboxes))

        frame_n+=1
    print('Mean IoU over time for',run_name,':',np.mean(mIoU))
    
    plt.figure()
    plt.plot(mIoU, c="blue")
    plt.xlim([0, len(gt_frame_keys)])
    plt.ylim([0, 1])
    plt.title("Mean IoU over time")
    plt.xlabel('Frame number')
    plt.ylabel('Mean IoU')
    plt.title('Mean IoU over time')
    plt.tight_layout()
    plt.savefig(EXPERIMENTS_FOLDER+'/'+str(run_name)+'/'+str(run_name)+'.png')

    cap.release()
    cv2.destroyAllWindows()

    return np.mean(mIoU)

def read_annotations(annotations, predictions_folder):
    # gt_rect format: 
    # {frame_number: [[x_min1, y_min1, x_max1, y_max1], [x_min2, y_min2, x_max2, y_max2], [], ...], ...}
    print("Reading annotations from:", annotations)
    gt_rect = utils.parse_xml_reacts(annotations)
    
    # pred_rect format: 
    # { "rcnn": {frame_number: [[x_min1, y_min1, x_max1, y_max1], [x_min2, y_min2, x_max2, y_max2], [], ...], ...},
    #   "yolo": {frame_number: [[x_min1, y_min1, x_max1, y_max1], [x_min2, y_min2, x_max2, y_max2], [], ...], ...},
    # }
    print("Reading predictions from:", predictions_folder)
    predictions = {model_name: utils.parse_predictions_rects(os.path.join(predictions_folder, PREDICTIONS_MODELS[model_name])) for model_name in list(PREDICTIONS_MODELS.keys())}
    
    return gt_rect, predictions

def save_reranking_mAP(reranking_mAP, mIoU, run_name):
    if not os.path.exists(EXPERIMENTS_FOLDER):
        os.mkdir(EXPERIMENTS_FOLDER)
    if not os.path.exists(EXPERIMENTS_FOLDER+"/"+str(run_name)):
        os.mkdir(EXPERIMENTS_FOLDER+"/"+str(run_name))
    name = f"{EXPERIMENTS_FOLDER}/{str(run_name)}/{run_name}.txt"
    
    with open(name,'w') as f:
        f.writelines("mAP = "+str(reranking_mAP))
        f.writelines("\nmIoU = "+str(mIoU))


def show_iou_over_time_and_compute(video_path, gt, pred, run_name, save=True):
    print("Computing mIoU")
    
    if not os.path.exists(EXPERIMENTS_FOLDER):
        os.mkdir(EXPERIMENTS_FOLDER)
    if not os.path.exists(EXPERIMENTS_FOLDER+"/"+str(run_name)):
        os.mkdir(EXPERIMENTS_FOLDER+"/"+str(run_name))
    
    mIoU = []
    save = False

    cap = cv2.VideoCapture(video_path)
    frame_num = 0
    ret, img = cap.read()
    total_frames = len(list(gt.keys()))

    while(True):
        gt_bboxes = gt[frame_num]
        det_bboxes = pred[frame_num]
        mIoU.append(utils.get_frame_iou(gt_bboxes, det_bboxes))

        #display_frame = cv2.resize(img, tuple(np.int0(0.5*np.array(img.shape[:2][::-1]))))
        
        for r in gt_bboxes:
            img = cv2.rectangle(img, (int(r[0]), int(r[1])),  (int(r[2]), int(r[3])), GT_COLOR, 2)
    
        for r in det_bboxes:
            img = cv2.rectangle(img, (int(r[0]), int(r[1])),  (int(r[2]), int(r[3])), PREDICTED_COLOR, 2)
        
        if save:
            plt.plot(mIoU, c="black")
            plt.xlabel('Frame number')
            plt.ylabel('Mean IoU')
            plt.xlim([0, total_frames])
            plt.ylim([0, 1])
            plt.title("Mean IoU over time")
            plt.tight_layout()
            plt.savefig(EXPERIMENTS_FOLDER+'/'+str(run_name)+'/iou_plt_'+str(frame_num)+'.png')
        
        ret, img = cap.read()
        frame_num += 1
        
        if not ret:
            break

    cap.release()
    cv2.destroyAllWindows()

    # Compute mean IoU over time
    print('Mean IoU over time for',run_name,':',np.mean(mIoU))

    plt.figure()
    plt.plot(mIoU, c="black")
    plt.xlim([0, total_frames])
    plt.ylim([0, 1])
    plt.title("Mean IoU over time")
    plt.xlabel('Frame number')
    plt.ylabel('Mean IoU')
    plt.title('Mean IoU over time')
    plt.tight_layout()
    plt.savefig(EXPERIMENTS_FOLDER+'/'+str(run_name)+'/iou_plt_final.png')

def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("-v",
                        dest="input_video",
                        required=True,
                        type=str,
                        help="input_video")
    parser.add_argument("-a",
                        dest="annotations",
                        required=True,
                        type=str,
                        help="annotations")
    parser.add_argument("-s",
                        dest="save_plot",
                        required=False,
                        default=False,
                        type=bool,
                        help="Save mIoU plot")
    parser.add_argument("-r",
                        dest="run_name",
                        required=True,
                        default="default",
                        type=str,
                        help="Name of experiment")
    args = parser.parse_args()

    return args.input_video, args.annotations, args.run_name, args.save_plot

if __name__ == "__main__":
    input_video, annotations, run_name, save_plot = parse_arguments()
    gt, predictions = read_annotations(annotations, os.path.join("/".join(input_video.split("/")[:-1]), "det"))

    # Show bboxes
    # show_boxes(input_video, gt, predictions["yolov3"][0])
    
    # Compute mAP
    print("MaskRCNN mAP 0.5:", voc_eval(predictions["mask_rcnn"][1], gt, ovthresh=0.5))
    print("YoloV3 mAP 0.5:", voc_eval(predictions["yolov3"][1], gt, ovthresh=0.5))
    print("SSD512 mAP 0.5:", voc_eval(predictions["ssd512"][1], gt, ovthresh=0.5))

    if not os.path.exists(EXPERIMENTS_FOLDER):
        os.mkdir(EXPERIMENTS_FOLDER)
    if not os.path.exists(EXPERIMENTS_FOLDER+"/"+str(run_name)):
        os.mkdir(EXPERIMENTS_FOLDER+"/"+str(run_name))
    
    # Evaluating noisy AP
    rank = 10
    frame_ids, tot_boxes, confidences = generate_noisy_bboxes(gt, rank=rank, std_coords=0, resizing_factor=1, prob_drop_detections=0, prob_new_detections=0, dropout = 0)
    preds_formatted = utils.predictions_to_gt_format(frame_ids, tot_boxes)

    reranking_mAP = np.mean(np.array([voc_eval([frame_ids, tot_boxes, confidences[i]], gt, ovthresh=0.5) for i in range(rank)]))
    mIoU = show_boxes(input_video, gt, preds_formatted)
    save_reranking_mAP(reranking_mAP,mIoU, run_name)
    
    print("Default reranking mAP:", reranking_mAP)
    
    
    # Show AP over time ?
    if save_plot:
        # Show IoU over time + Compute mean IoU
        show_iou_over_time_and_compute(input_video, gt, predictions["yolov3"][0], run_name+"_yolov3", save=save_plot)
        show_iou_over_time_and_compute(input_video, gt, predictions["mask_rcnn"][0], run_name+"_maskrcnn", save=save_plot)
        show_iou_over_time_and_compute(input_video, gt, predictions["ssd512"][0], run_name+"_ssd512", save=save_plot)
        