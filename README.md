# Video Surveillance for Road Traffic Monitoring

The goal of this project is to learn the basic concepts and techniques related to video sequences processing, mainly for surveillance applications. Road traffic monitoring and Advanced Driver Assistance Systems (ADAS) are aimed to improve safety, efficiency and comfort at road transportation by means of information technologies.

## Week 1

The contents of the first week are in the folder `week1`:

Instructions for task 1 and 2:
```
usage: python eval_iou_and_map.py -v INPUT_VIDEO -a ANNOTATIONS [-s SAVE_PLOT] -r RUN_NAME

arguments:
  -v INPUT_VIDEO  Input video for analyzing mIou/mAP
  -a ANNOTATIONS  XML Groundtruth annotations
  -s SAVE_PLOT    Whether to save the mIoU plot over time
  -r RUN_NAME     Name of experiment
```

See the notebook `optical_flow_evaluation_and_plot.ipynb` for tasks 3 and 4.

## Week 2
The contents of the second week are in the folder `week2`:

The first task consists of modeling the background with a per-pixel Gaussian distribution. For running the code that evaluates the mAP using this background model and evaluates different values of alpha, run:
```
usage: task_1.py [-h] -v INPUT_VIDEO -a ANNOTATIONS -r RUN_NAME [-d]

optional arguments:
  -h, --help      show this help message and exit
  -v INPUT_VIDEO  Input video for analyzing mIou/mAP
  -a ANNOTATIONS  XML Groundtruth annotations
  -r RUN_NAME     Name of experiment
  -d              Display predictions over the video
```

To try the different methods mentioned for task 3, we can use the `task_3.py` script the same way as the `task_1.py` above, but with an additional `-m --method` argument which can be one of `MOG`, `MOG2` or `LSBP`.
To get the results for the BSUV-net method, we have to first perform inference with their [released weights](https://github.com/ozantezcan/BSUV-Net-inference), and then feed the masked video to the `task_3.py` script, using `BSUV` as the chosen method. The already processed video sequence is available on [this link](https://drive.google.com/file/d/1xGEcGX39hYitts1rpiXmCX9tV132zeJv/view?usp=sharing).

The last task consists of generalizing both the adaptive and the non-adaptive modelings for the background from the firsts tasks to be used in color sequences. For running the code that evaluates the mAP using this background models (static or dynamic) with different color spaces and evaluates different values of alpha, run:
```
usage: task_4.py [-h] -v INPUT_VIDEO -a ANNOTATIONS -r RUN_NAME -c COLOR_SPACE [-y] [-d]

optional arguments:
  -h, --help      show this help message and exit
  -v INPUT_VIDEO  Input video for analyzing mIou/mAP
  -a ANNOTATIONS  XML Groundtruth annotations
  -r RUN_NAME     Name of experiment
  -c COLOR_SPACE  Color space used to model the background
  -y              Use dynamic model for the background (static otherwise)
  -d              Display predictions over the video
```

## Week 3
The contents of the second week are in the folder `week3`:

Task 1.1 evaluates mAP on off-the-shelf detectors:
```
usage: task1_1.py -v INPUT_VIDEO -a ANNOTATIONS -n ARCHITECTURE_NAME [-d] [-g]

optional arguments:
  -v INPUT_VIDEO        Input video for analyzing mIou/mAP
  -a ANNOTATIONS        XML Groundtruth annotations
  -n ARCHITECTURE_NAME  Architecture name. Options: FasterRCNN / MaskRCNN / ...
  -d                    Display predictions over the video
  -g                    Use GPU for model inference
```
Example: python week3/task1_1.py -v /home/sergio/MCV/M6/data/AICity_data/train/S03/c010/vdo.avi -a /home/sergio/MCV/M6/data/ai_challenge_s03_c010-full_annotation.xml -n FasterRCNN -g

Task 1.2 finetunes and evaluates finetuned detectors on part of the sequence:
```
...
```
Example: python week3/task1_2.py -v /home/sergio/MCV/M6/data/AICity_data/train/S03/c010/vdo.avi -a /home/sergio/MCV/M6/data/ai_challenge_s03_c010-full_annotation.xml -n FasterRCNN -g -r fasterRCNN_finetune -t