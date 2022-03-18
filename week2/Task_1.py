import cv2
from cv2 import cvtColor
import numpy as np
from argparse import ArgumentParser
from background_models import GaussianStaticModel

# video_path = "/home/aszummer/Documents/MCV/M6/mcv-m6-2022-team3/lab1-data/AICity_data/AICity_data/train/S03/c010/vdo.avi"

def main(video_path, annotations):
    frame_25per = 510
    model = GaussianStaticModel(video_path, alpha=4, color_format="grayscale", num_frames_training=frame_25per)
    model.fit()

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_25per)
    ret = True
    wait_time = 1
    display = True

    while(ret):
        ret, img = cap.read()
        frame = model.infer(img)
        
        if display:
            display_frame = cv2.resize(frame, tuple(np.int0(0.5*np.array(img.shape[:2][::-1]))))
            # display_frame = frame
            cv2.imshow('frame',display_frame)
            k = cv2.waitKey(wait_time)
            if k == ord('q'):
                break
            elif k == ord('p'):
                wait_time = int(not(bool(wait_time)))

        
        
        if not ret:
            break

    cap.release()
    cv2.destroyAllWindows()
    
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
    args = parser.parse_args()

    return args.input_video, args.annotations
    
if __name__ == "__main__":
    input_video, annotations = parse_arguments()
    main(input_video, annotations)