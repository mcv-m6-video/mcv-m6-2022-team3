import cv2
import numpy as np
from argparse import ArgumentParser
from background_models import BackgroundModelCV2, GaussianDynamicModel, GaussianStaticModel
from io_utils import read_annotations
from evaluation import voc_eval
import os
from task_1 import obtain_predictions_from_model


EXPERIMENTS_FOLDER = "experiments"

    
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
    parser.add_argument("-m",
                        dest="method",
                        required=True,
                        type=str,
                        help="Method to segment the background")
    parser.add_argument("-d", 
                        default=False, 
                        dest="display",
                        action="store_true",
                        help="Display predictions over the video")
    args = parser.parse_args()

    return args.input_video, args.annotations, args.run_name, args.display, args.method


if __name__ == "__main__":
    input_video, annotations_path, run_name, display, method = parse_arguments()
    annotations = read_annotations(annotations_path)
    roi_path = os.path.join("/".join(input_video.split("/")[:-1]), "roi.jpg")

    if method == "MOG":
        background_substractor = cv2.bgsegm.createBackgroundSubtractorMOG()
    if method == "MOG2":
        background_substractor = cv2.createBackgroundSubtractorMOG2()
    if method == "LSBP":
        background_substractor = cv2.bgsegm.createBackgroundSubtractorLSBP()

    if method == "BSUV":
        model = GaussianDynamicModel(input_video, roi_path, alpha=6.5, color_format="grayscale", num_frames_training=510, apply_morphology=True)
        model.bg_model_name = "bg_gaussian_bsuv"
    else:
        model = BackgroundModelCV2(input_video, roi_path, background_substractor=background_substractor, num_frames_training=510)
    frame_ids, tot_boxes, confidences = obtain_predictions_from_model(model, run_name, input_video, annotations, display=display)
    reranking_mAP = np.mean(np.array([voc_eval([frame_ids, tot_boxes, confidences[i]], annotations, ovthresh=0.5) for i in range(len(confidences))]))
    print(f"Method {method} mAP: {reranking_mAP}")

    