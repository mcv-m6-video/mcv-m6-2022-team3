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

