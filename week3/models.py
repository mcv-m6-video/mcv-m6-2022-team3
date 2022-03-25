import torchvision
import torch

def load_model(architecture_name, use_gpu):
    if architecture_name == 'FasterRCNN':
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    elif architecture_name == 'MaskRCNN':
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    elif architecture_name == 'more':
        # ...
        pass
    else:
        raise ValueError(architecture_name+ " not found.")
    
    device = torch.device('cuda') if torch.cuda.is_available() and use_gpu else torch.device('cpu')
    model.to(device)
    model.eval()
    return model, device