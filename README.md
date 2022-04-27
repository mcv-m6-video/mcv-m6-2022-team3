# Video Surveillance for Road Traffic Monitoring

The goal of this project is to learn the basic concepts and techniques related to video sequences processing, mainly for surveillance applications. Road traffic monitoring and Advanced Driver Assistance Systems (ADAS) are aimed to improve safety, efficiency and comfort at road transportation by means of information technologies.

# Team 3

| Members | Contact |
| :---         |   :---    | 
| Sergio Montoya   | sergio.montoyadepaco@e-campus.uab.cat | 
| Laia Albors    | laia.albors@e-campus.uab.cat  |
| Ibrar Malik    | ibrar.malik@e-campus.uab.cat  |
| Adam Szummer | adam.szummer@e-campus.uab.cat |

Project presentation: [Google Slides](https://docs.google.com/presentation/d/1Kaw4ZFY4qFNT922YqpbG1HjQ76cZtwgx9F3WxoYMo3g/edit?usp=sharing)

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
The contents of the third week are in the folder `week3`:

### Task 1.1 evaluates mAP on off-the-shelf detectors:
```
usage: python task1_1.py -v INPUT_VIDEO -a ANNOTATIONS -n ARCHITECTURE_NAME [-d] [-g]

optional arguments:
  -v INPUT_VIDEO        Input video for analyzing mIou/mAP
  -a ANNOTATIONS        XML Groundtruth annotations
  -n ARCHITECTURE_NAME  Architecture name. Options: FasterRCNN / MaskRCNN / ...
  -d                    Display predictions over the video
  -g                    Use GPU for model inference
```
Example: python week3/task1_1.py -v /home/sergio/MCV/M6/data/AICity_data/train/S03/c010/vdo.avi -a /home/sergio/MCV/M6/data/ai_challenge_s03_c010-full_annotation.xml -n FasterRCNN -g

### Task 1.2 finetunes and evaluates finetuned detectors on part of the sequence:
```
optional arguments:
  -v INPUT_VIDEO        Input video for analyzing mIou/mAP
  -a ANNOTATIONS        XML Groundtruth annotations
  -n ARCHITECTURE_NAME  Architecture name. Options: FasterRCNN / MaskRCNN / ...
  -d                    Display predictions over the video
  -g                    Use GPU for model inference
  -t                    train
  -r                    name of the run
```
...
```
Example: python week3/task1_2.py -v /home/sergio/MCV/M6/data/AICity_data/train/S03/c010/vdo.avi -a /home/sergio/MCV/M6/data/ai_challenge_s03_c010-full_annotation.xml -n FasterRCNN -g -r fasterRCNN_finetune -t
```

List of architectures available to run:
- MaskRCNN
- FasterRCNN
- FasterRCNN_mobile
- RetinaNet
- SSD
- SSDlite
- FCOS

Train parameters are to be set in the code in the model.py file in train function.
If you wish to see the interminent log in a dashboard you need to create account in wandb.ai and provide your username in WANDB_ENTITY parameter meanwhile all results are printed in the console as well.

### Task 1.3 can be used to apply different strategies of cross-validation when fine-tuning a given model:

usage:
```
python task1_3.py -v INPUT_VIDEO -a ANNOTATIONS -n ARCHITECTURE_NAME -r RUN_NAME [-s STRATEGY] [-g] [-t]

optional arguments:
  -v INPUT_VIDEO        Input video for analyzing mIou/mAP
  -a ANNOTATIONS        XML Groundtruth annotations
  -n ARCHITECTURE_NAME  Architecture name. Options: FasterRCNN / MaskRCNN / ...
  -r RUN_NAME           Name of the experiment
  -s STRATEGY           Strategy used for cross-validation. Options: A, B, C. A by default
  -g                    Use GPU for model inference
  -t                    Specify to train, otherwise will evaluate
```

### Task 2.1 and 2.2 can be used to perform tracking over the video sequence, using either Maximum Overlap (2.1) or a Kalman filter (2.2):

usage:
```
python [task2_1.py|task2_2.py] -v INPUT_VIDEO -a ANNOTATIONS -n ARCHITECTURE_NAME -r RUN_NAME [-g] [-d]

optional arguments:
  -v INPUT_VIDEO        Input video for analyzing mIou/mAP
  -a ANNOTATIONS        XML Groundtruth annotations
  -n ARCHITECTURE_NAME  Architecture name. Options: FasterRCNN / MaskRCNN / ...
  -r RUN_NAME           Name of the experiment
  -g                    Use GPU for model inference
  -d                    Display the tracked boxes and history
```

## Week 4
The contents of the fourth week are in the folder `week4`:


### Task 1.1 optical flow with block matching:
```
usage: python task1_1.py
```
Runs with optimal parameters that have been found using notebook.ipynb grid search that took 1200 min

### Task 1.2 off-the-shelf optical flow:
For Farne Back and Horn Schunk run:

```
usage: python task1_2.py -n -r

optional arguments:
  - n                   number of runs
  - r                   random parameters search
```

For PyFlow run as per ReadMe in Pyflow folder

For MaskFlowNet run as per ReadMe in MaskFlowNet


### Task 2 Multi-target single-camera (MTSC) tracking:
First we can fine-tune the architecture that we want with the specified sequences:
```
usage: python finetune.py -d DATASET_PATH -s LIST_OF_SEQUENCES -n ARCHITECTURE_NAME -r RUN_NAME [-g]

optional arguments:
  -d DATASET_PATH       Path to the dataset, train folder
  -s LIST_OF_SEQUENCES  The sequences we want to use for training. e.g. S01 S03 S04
  -n ARCHITECTURE_NAME  Architecture name. Options: FasterRCNN / MaskRCNN / RetinaNet ...
  -r RUN_NAME           Name of the experiment
  -g                    Use GPU for model inference
```

Then we can evaluate the object tracking task using the SORT implementation like this:
```
usage: python task2.py -v INPUT_VIDEO -n ARCHITECTURE_NAME -r RUN_NAME [-g]

optional arguments:
  -v INPUT_VIDEO        Input video for analyzing mIou/mAP
  -n ARCHITECTURE_NAME  Architecture name. Options: FasterRCNN / MaskRCNN / RetinaNet ...
  -r RUN_NAME           Name of the experiment
  -g                    Use GPU for model inference
```
If we want to use a fine-tuned model, we have to use the same RUN_NAME as in `finetune.py`. 

## Week 5
The contents are in the folder `week5`:

### Task 1: Multi-target single-camera tracking (MTSC)

- Finetune car detection network on sequences S01 and S04:
```
usage: python finetune.py -d DATASET_PATH -s S01 S04 -n ARCHITECTURE_NAME -r RUN_NAME [-g]

optional arguments:
  -d DATASET_PATH       Path to the dataset, train folder
  -s LIST_OF_SEQUENCES  The sequences we want to use for training. e.g. S01 S03 S04
  -n ARCHITECTURE_NAME  Architecture name. Options: FasterRCNN / MaskRCNN / RetinaNet ...
  -r RUN_NAME           Name of the experiment
  -g                    Use GPU for model inference
```

- Evaluate car finetune car detector on sequence S03:
```
usage: python finetune.py -d DATASET_PATH -s S03 -n ARCHITECTURE_NAME -r RUN_NAME [-g]

optional arguments:
  -d DATASET_PATH       Path to the dataset, train folder
  -s LIST_OF_SEQUENCES  The sequences we want to use for training. e.g. S01 S03 S04
  -n ARCHITECTURE_NAME  Architecture name. Options: FasterRCNN / MaskRCNN / RetinaNet ...
  -r RUN_NAME           Name of the experiment
  -g                    Use GPU for model inference
```

- Run tracker on sequence S03: ...

- For using ReIdentification network for visual features download the model from: https://drive.google.com/file/d/1wUbYm5-EJs0W-LAGS69yvb33D6NkFWpH/view
Then put the network weights "net_last.pth" in `week5/feat_extraction/reID`.

### Task 1.2


### Task 2: Multi-target multi-camera tracking (MTMC)

We have three algorithms to do multi-target multi-camera (MTMC) tracking. To run them, first we should compute the tracklets files with the `task1.py -s PATH_TRACKLETS` script, to ease up computing.

- Use the **online** algorithm and evaluate it.
```
python mtmc_online.py -s SEQUENCE_PATH -t PATH_TRACKLETS

optional arguments:
  -i SIM_THR         Similarity threshold
  -u MIN_IOU         IOU threshold for tracking
```

- Use the **offline** algorithm that finds the connected components and evaluate it:
```
usage: python mtmc_v1.py -d DATASET_PATH -s S01 S04 -n ARCHITECTURE_NAME -r RUN_NAME [-g]

optional arguments:
  -s SEQUENCE           Sequence of videos that we want to use. Options: S01, S03, S04
  -t PATH_TRACKLETS     Path to the folder where the thacklets are
  -g PATH_GT            Path to the ground truth files
  -i SIM_THR            Similarity threshold
  -a PATH_SAVE          Path to the folder where to save the detections with the new IDs
```

- Use the **offline** algorithm that uses our version of the connected components algorithm and evaluate it:
```
usage: python mtmc_v2.py -d DATASET_PATH -s S01 S04 -n ARCHITECTURE_NAME -r RUN_NAME [-g]

optional arguments:
  -s SEQUENCE           Sequence of videos that we want to use. Options: S01, S03, S04
  -t PATH_TRACKLETS     Path to the folder where the thacklets are
  -g PATH_GT            Path to the ground truth files
  -i SIM_THR            Similarity threshold
  -a PATH_SAVE          Path to the folder where to save the detections with the new IDs
```
