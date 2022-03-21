from tkinter import E
import cv2
from cv2 import cvtColor
import numpy as np
from argparse import ArgumentParser
from background_models import GaussianDynamicModel, GaussianStaticModel
from io_utils import read_annotations
from evaluation import show_annotations_and_predictions, voc_eval
import os
from tqdm import tqdm

# video_path = "/home/aszummer/Documents/MCV/M6/mcv-m6-2022-team3/lab1-data/AICity_data/AICity_data/train/S03/c010/vdo.avi"
EXPERIMENTS_FOLDER = "experiments"

def obtain_predictions_from_model(model, run_name, video_path, annotations, display=True):
    frame_25per = 510
    model.fit()

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_25per)
    ret = True
    wait_time = 1
    save = False
    frame_number = frame_25per
    
    predictions = []
    frame_numbers = []
    ret, img = cap.read()
    
    while(ret):
        
        dets, frame = model.infer(img)
        for det in dets:
            predictions.append(det)
        frame_numbers.append([frame_number for _ in range(len(dets))])
        
        if display:
            display_frame = cv2.resize(frame, tuple(np.int0(0.5*np.array(img.shape[:2][::-1]))))
            cv2.imshow('frame',display_frame)
            cv2.imshow('frame_color',cv2.resize(img, tuple(np.int0(0.5*np.array(img.shape[:2][::-1])))))
            k = cv2.waitKey(1)
            if k == ord('q'):
                break
            elif k == ord('p'):
                wait_time = int(not(bool(wait_time)))
                
            if frame_number in list(annotations.keys()):
                show_annotations_and_predictions(img, annotations[frame_number], dets)
            else:
                show_annotations_and_predictions(img, [], dets)

        if save:
            if not os.path.exists(EXPERIMENTS_FOLDER):
                os.mkdir(EXPERIMENTS_FOLDER)
            experiment_run_folder = os.path.join(EXPERIMENTS_FOLDER, run_name)
            if not os.path.exists(experiment_run_folder):
                os.mkdir(experiment_run_folder)
            
            cv2.imwrite(experiment_run_folder+'/image_'+str(frame_number-frame_25per).zfill(4)+'.jpg', cv2.resize(img, tuple(np.int0(0.5*np.array(img.shape[:2][::-1])))))
            cv2.imwrite(experiment_run_folder+'/classification_'+str(frame_number-frame_25per).zfill(4)+'.jpg', cv2.resize(frame, tuple(np.int0(0.5*np.array(img.shape[:2][::-1])))))
    
        frame_number += 1
        ret, img = cap.read()
    
    cap.release()
    cv2.destroyAllWindows()
    frame_numbers, predictions = np.concatenate(frame_numbers), np.concatenate(predictions)
    tot_predictions = [frame_numbers.astype(np.int64), predictions.reshape((len(frame_numbers), 4)), np.random.random((10, len(frame_numbers)))]
    return tot_predictions
    
def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("-v",
                        dest="input_video",
                        required=True,
                        type=str,
                        help="Input video for analyzing mIou/mAP")
    parser.add_argument("-a",
                        dest="annotations",
                        required=True,
                        type=str,
                        help="XML Groundtruth annotations")
    parser.add_argument("-r",
                        dest="run_name",
                        required=True,
                        default="default",
                        type=str,
                        help="Name of experiment")
    parser.add_argument("-d", 
                        default=False, 
                        dest="display",
                        action="store_true",
                        help="Display predictions over the video")
    args = parser.parse_args()

    return args.input_video, args.annotations, args.run_name, args.display
    
if __name__ == "__main__":
    input_video, annotations_path, run_name, display = parse_arguments()
    annotations = read_annotations(annotations_path)
    roi_path = os.path.join("/".join(input_video.split("/")[:-1]), "roi.jpg")
    
    model = GaussianStaticModel(input_video, roi_path, alpha=6.5, color_format="grayscale", num_frames_training=510)
    frame_ids, tot_boxes, confidences = obtain_predictions_from_model(model, run_name, input_video, annotations, display=display)
    reranking_mAP = np.mean(np.array([voc_eval([frame_ids, tot_boxes, confidences[i]], annotations, ovthresh=0.5) for i in range(len(confidences))]))
    print("Static gaussian mAP:", reranking_mAP)
   
    mAPs = []
    for alpha in tqdm(np.arange(3, 8, 0.25)):
        model = GaussianStaticModel(input_video, roi_path, alpha=alpha, color_format="grayscale", num_frames_training=510)
        frame_ids, tot_boxes, confidences = obtain_predictions_from_model(model, run_name, input_video, annotations, display=display)
        reranking_mAP = np.mean(np.array([voc_eval([frame_ids, tot_boxes, confidences[i]], annotations, ovthresh=0.5) for i in range(len(confidences))]))
        mAPs.append(reranking_mAP)
        print("map @ alpha=", alpha, ":", reranking_mAP)
    
    print("STATIC mAPs:")
    print(mAPs)
    
    