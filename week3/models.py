import torchvision
import torch
import logging_utils
import tv_utils
import numpy as np
import os

def load_model(architecture_name, use_gpu, finetune=False, trainable_backbone_layers=None):
    if architecture_name == 'FasterRCNN':
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        if finetune:
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            # Keep first 4 classes so that car is still classified as class 3
            model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, 4)
    elif architecture_name == 'FasterRCNN_mobilenet':
        model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=True, pretrained_backbone=True)

    elif architecture_name == 'MaskRCNN':
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
        if finetune:
            model.roi_heads.mask_roi_pool = None
            model.roi_heads.mask_head = None
            model.roi_heads.mask_predictor = None
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            # Keep first 4 classes so that car is still classified as class 3
            model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, 4)
    elif architecture_name == 'RetinaNet':
        model = torchvision.models.detection.retinanet_resnet50_fpn(pretrained=True, trainable_backbone_layers=trainable_backbone_layers)
        if finetune:
            model = torchvision.models.detection.retinanet_resnet50_fpn(pretrained=True, trainable_backbone_layers=trainable_backbone_layers,num_classes=4)
    
    elif architecture_name == 'SSD':
        model = torchvision.models.detection.ssd300_vgg16(pretrained=True,pretrained_backbone=True,trainable_backbone_layers=trainable_backbone_layers)
        if finetune:
            model = torchvision.models.detection.ssd300_vgg16(pretrained=True,pretrained_backbone=True,trainable_backbone_layers=trainable_backbone_layers, num_classes=4)

    elif architecture_name == 'SSDlite':
        model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(pretrained=True,pretrained_backbone=True,trainable_backbone_layers=trainable_backbone_layers)
        if finetune:
            model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(pretrained=True,pretrained_backbone=True,trainable_backbone_layers=trainable_backbone_layers, num_classes=4)

    elif architecture_name == 'FCOS':
        model = torchvision.models.detection.fcos_resnet50_fpn(pretrained=True,pretrained_backbone=True,trainable_backbone_layers=trainable_backbone_layers)
        if finetune:
            model = torchvision.models.detection.fcos_resnet50_fpn(pretrained=True,pretrained_backbone=True,trainable_backbone_layers=trainable_backbone_layers, num_classes=4)


    elif architecture_name == 'more':
        # ...
        pass
    else:
        raise ValueError(architecture_name+ " not found.")
    
    device = torch.device('cuda') if torch.cuda.is_available() and use_gpu else torch.device('cpu')
    model.to(device)
    model.eval()
    return model, device

@torch.no_grad()
def evaluate(model, data_loader, device, save_path=None):
    print("eval")
    return 1
    """ n_threads = torch.get_num_threads()

    y_true = []
    y_pred = []
    for image, targets in metric_logger.log_every(data_loader, 100, header):
        image = list(img.to(device) for img in image)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        torch.cuda.synchronize()

        model_time = time.time()
        outputs = model(image)
        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time
        metric_logger.update(model_time=model_time)

        for target, output in zip(targets, outputs):
            frame = target['image_id'].item()

    # gather the stats from all processes
    print("Averaged stats:", metric_logger)

    torch.set_num_threads(n_threads) """
    
def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):

    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)

def train(model, train_loader, test_loader, device, num_epochs=1, save_path=None):
    print_freq = 10
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    mAPs = [0]

    # TODO: Add tensorboard logging or something
    for epoch in range(num_epochs):
        model.train()
        metric_logger = logging_utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter('lr', logging_utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        header = 'Epoch: [{}]'.format(epoch)

        """ lr_scheduler = None
        if epoch == 0:
            warmup_iterations = min(500, len(train_loader) - 1)
            warmup_factor = 1. / warmup_iterations
            lr_scheduler = warmup_lr_scheduler(optimizer, warmup_iterations, warmup_factor) """

        for images, targets in metric_logger.log_every(train_loader, print_freq, header):
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)

            losses = sum(loss for loss in loss_dict.values())

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = tv_utils.reduce_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            if lr_scheduler is not None:
                lr_scheduler.step()

            metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        lr_scheduler.step()
        mAP = evaluate(model, test_loader, device)
        if mAP < np.min(np.array(mAPs)):
            if save_path is not None:
                torch.save(model.state_dict(), os.path.join(save_path, "best.ckpt"))
            
        mAPs.append(mAP)
        
        print("Current mAPs by epoch:", mAPs)
        
    return mAP