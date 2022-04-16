#from argon2 import Parameters
import torchvision
import torch
import logging_utils
import tv_utils
import numpy as np
import os
from torchvision.ops import nms
from evaluation import show_annotations_and_predictions, voc_eval
from tqdm import tqdm
from copy import deepcopy
import wandb

CAR_LABEL_NUM = 3
WANDB_ENTITY = "aszummer"

def load_model(architecture_name, use_gpu, finetune=False, trainable_backbone_layers=3):
    if architecture_name == 'FasterRCNN':
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        if finetune:
            in_features = model.roi_heads.box_predictor.cls_score.in_features
    #         # Keep first 4 classes so that car is still classified as class 3
    #         model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, 4)
    elif architecture_name == 'FasterRCNN_mobile':
        model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=True,pretrained_backbone=True,trainable_backbone_layers=trainable_backbone_layers)
    elif architecture_name == 'MaskRCNN':
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True,pretrained_backbone=True,trainable_backbone_layers=trainable_backbone_layers)
        if finetune == False:
            model.roi_heads.mask_roi_pool = None
            model.roi_heads.mask_head = None
            model.roi_heads.mask_predictor = None
        if finetune:
            model.roi_heads.mask_roi_pool = None
            model.roi_heads.mask_head = None
            model.roi_heads.mask_predictor = None
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            # # Keep first 4 classes so that car is still classified as class 3
            # model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, 4)
    elif architecture_name == 'RetinaNet':
        model = torchvision.models.detection.retinanet_resnet50_fpn(pretrained=True, trainable_backbone_layers=trainable_backbone_layers)
        if finetune:
            model = torchvision.models.detection.retinanet_resnet50_fpn(pretrained=True, trainable_backbone_layers=trainable_backbone_layers)
    
    elif architecture_name == 'SSD':
        model = torchvision.models.detection.ssd300_vgg16(pretrained=True,pretrained_backbone=True,trainable_backbone_layers=trainable_backbone_layers)
        if finetune:
            model = torchvision.models.detection.ssd300_vgg16(pretrained=True,pretrained_backbone=True,trainable_backbone_layers=trainable_backbone_layers)

    elif architecture_name == 'SSDlite':
        model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(pretrained=True,pretrained_backbone=True,trainable_backbone_layers=trainable_backbone_layers)
        if finetune:
            model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(pretrained=True,pretrained_backbone=True,trainable_backbone_layers=trainable_backbone_layers)

    elif architecture_name == 'FCOS':
        model = torchvision.models.detection.fcos_resnet50_fpn(pretrained=True,pretrained_backbone=True,trainable_backbone_layers=trainable_backbone_layers)
        if finetune:
            model = torchvision.models.detection.fcos_resnet50_fpn(pretrained=True,pretrained_backbone=True,trainable_backbone_layers=trainable_backbone_layers)


    elif architecture_name == 'more':
        # ...
        pass
    else:
        raise ValueError(architecture_name+ " not found.")
    
    device = torch.device('cuda') if torch.cuda.is_available() and use_gpu else torch.device('cpu')
    model.to(device)
    model.eval()
    return model, device

def evaluate(model, data_loader, device):

    model.eval()

    predictions, frame_numbers, total_scores = [], [], []
    annotations = {}
    with torch.no_grad():
        for images, targets in tqdm(data_loader):
            images = [image.to(device) for image in images]
            outputs = model(images)
            
            for el in range(len(outputs)):
                preds = outputs[el]
                keep_cars_mask = preds['labels'] == CAR_LABEL_NUM
                bboxes, scores = preds['boxes'][keep_cars_mask], preds['scores'][keep_cars_mask]
                idxs = nms(bboxes, scores, 0.7)
                final_dets, final_scores = bboxes[idxs].cpu().numpy(), scores[idxs].cpu().numpy()

                for i in range(len(final_dets)):
                    predictions.append(final_dets[i])
                    total_scores.append(final_scores[i])
                
                image_id = targets[el]["image_id"].cpu().numpy()[0]
                # Add annotation only if there are boxes
                if "boxes" in list(targets[el].keys()):
                    annotations[image_id] = targets[el]["boxes"].cpu().numpy()
                if len(final_dets) > 0:
                    frame_numbers.append([image_id for _ in range(len(final_dets))])
    
    # eval mAP
    annotations = deepcopy(annotations)
    frame_numbers, predictions = np.concatenate(frame_numbers), np.concatenate(predictions)
    filtered_annotations = {frame_n:annotations[frame_n] for frame_n in list(annotations.keys()) if frame_n in np.unique(frame_numbers)}
    
    tot_predictions = [frame_numbers.astype(np.int64), predictions.reshape((len(frame_numbers), 4)), np.array(total_scores)]
    frame_ids, tot_boxes, confidences = tot_predictions
    mAP = voc_eval([frame_ids, tot_boxes, confidences], filtered_annotations, ovthresh=0.5)
    return mAP
    
def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):

    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)

def train(model, train_loader, test_loader, device, architecture_name,
                 num_epochs=1,
                 batch_size=1,
                 save_path=None, log_bool=False, run_name='test'):
        # MODEL TRAINING
    if log_bool:
        # pass
        #   # 1. Start a new run
        wandb.init(project='MCVM6', entity=WANDB_ENTITY)
        wandb.run.name = run_name
        wandb.run.save()

    lr = 0.005
    step_size = 5
    print_freq = 10
    params = [p for p in model.parameters() if p.requires_grad]
    print("Params to train:", len(params))
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)

    if (architecture_name == 'SSD') or (architecture_name == 'SSDlite'):
        lr = 0.00004
        optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=0.0005)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    mAPs = [0]
    # or compute initial mAP before finetuning
    mAPs = [evaluate(model, test_loader, device)]

    config_dict = {
            "learning_rate": lr,
            "epochs": num_epochs,
            "batch_size": batch_size,
            "step_size": step_size
        }

    if log_bool:
        # pass
        wandb.config = config_dict

    # TODO: Add tensorboard logging or something
    for epoch in range(num_epochs):
        model.train()
        #metric_logger = logging_utils.MetricLogger(delimiter="  ")
        #metric_logger.add_meter('lr', logging_utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        header = 'Epoch: [{}]'.format(epoch)

        """ lr_scheduler = None
        if epoch == 0:
            warmup_iterations = min(500, len(train_loader) - 1)
            warmup_factor = 1. / warmup_iterations
            lr_scheduler = warmup_lr_scheduler(optimizer, warmup_iterations, warmup_factor) """
        loss_reduced_all = []

        for images, targets in tqdm(train_loader):#metric_logger.log_every(train_loader, print_freq, header):
            idxs_keep = [i for i in range(len(images)) if not targets[i] is None]
            
            images = images[idxs_keep]
            targets = targets[idxs_keep]
            if len(targets) == 0:
                continue
            
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)

            losses = sum(loss for loss in loss_dict.values())

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = tv_utils.reduce_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            loss_reduced_all.append = losses_reduced

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            #metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
            #metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        else:
            model.eval()
            mAP = evaluate(model, test_loader, device)
            mAPs.append(mAP)

            losses_reduced_mean = np.array(loss_reduced_all).mean()

            log_dict = {
            'epoch': epoch + 1,
            'train_loss': losses_reduced_mean,
            'lr': optimizer.param_groups[0]["lr"],
            'mAP': mAP
            }
            for k, v in loss_dict_reduced.items():
                if isinstance(v, torch.Tensor):
                    v = v.item()
                    log_dict[k]=v

            print("log_dict = ", log_dict)
            print("config_dict = ", config_dict)
            if log_bool:
                wandb.log(log_dict)
    

        # model.train()
    
        if save_path is not None and mAPs[-1] > np.max(np.array(mAPs[:-1])):
            print("Saved Model to ", save_path)
            torch.save(model.state_dict(), os.path.join(save_path, f"{run_name}_best.ckpt"))
