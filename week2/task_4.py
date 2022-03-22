import os
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser
from background_models import GaussianDynamicModel, GaussianStaticModel
from io_utils import read_annotations
from evaluation import voc_eval
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
    parser.add_argument("-c",
                        dest="color_space",
                        required=True,
                        type=str,
                        help="Color space used to model the background")
    parser.add_argument("-y", 
                        default=False,
                        dest="dynamic",
                        action="store_true",
                        help="Use dynamic model for the background (static otherwise)")
    parser.add_argument("-d", 
                        default=False, 
                        dest="display",
                        action="store_true",
                        help="Display predictions over the video")
    args = parser.parse_args()

    return args.input_video, args.annotations, args.run_name, args.display, args.color_space, args.dynamic


if __name__ == "__main__":
    input_video, annotations_path, run_name, display, color_space, dynamic = parse_arguments()
    annotations = read_annotations(annotations_path)
    roi_path = os.path.join("/".join(input_video.split("/")[:-1]), "roi.jpg")
   
    if dynamic:
        print(f'DYNAMIC MODEL {color_space}')
        mAPs = []
        rho = 0.04
        for alpha in tqdm([0.5, 1, 1.5, 2, 3, 4, 5, 6, 6.75, 7, 8]):
            model = GaussianDynamicModel(input_video, roi_path, rho=rho, alpha=alpha, color_format=color_space, num_frames_training=510)
            frame_ids, tot_boxes, confidences = obtain_predictions_from_model(model, run_name, input_video, annotations, display=display)
            reranking_mAP = np.mean(np.array([voc_eval([frame_ids, tot_boxes, confidences[i]], annotations, ovthresh=0.5) for i in range(len(confidences))]))
            mAPs.append(reranking_mAP)
            print("map @ alpha=", alpha, ":", reranking_mAP)
        
        print(f"DYNAMIC {color_space} mAPs:")
        print(mAPs)
    
    else:
        print(f'STATIC MODEL {color_space}')
        mAPs = []
        for alpha in tqdm(np.arange(0.5, 8, 0.5)):
            model = GaussianStaticModel(input_video, roi_path, alpha=alpha, color_format=color_space, num_frames_training=510)
            frame_ids, tot_boxes, confidences = obtain_predictions_from_model(model, run_name, input_video, annotations, display=display)
            reranking_mAP = np.mean(np.array([voc_eval([frame_ids, tot_boxes, confidences[i]], annotations, ovthresh=0.5) for i in range(len(confidences))]))
            mAPs.append(reranking_mAP)
            print("map @ alpha=", alpha, ":", reranking_mAP)
        
        print(f"STATIC {color_space} mAPs:")
        print(mAPs)
    
    