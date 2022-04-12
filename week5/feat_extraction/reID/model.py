import argparse
import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
from torch.autograd import Variable
from feature_extraction.reID.pretrained_model.losses import AngleLinear, ArcLinear
from torch.nn import functional as F

######################################################################
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in') # For old pytorch, you may use kaiming_normal.
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)

def fix_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True
# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|
class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, droprate, relu=False, extra_bn=False, bnorm=True, num_bottleneck=512, linear=True, return_f = False):
        super(ClassBlock, self).__init__()
        self.return_f = return_f
        add_block = []
        if extra_bn:
            add_block += [nn.BatchNorm1d(input_dim)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if linear:
            add_block += [nn.Linear(input_dim, num_bottleneck)]
        else:
            num_bottleneck = input_dim
        if bnorm:
            add_block += [nn.BatchNorm1d(num_bottleneck)]
        if droprate>0:
            add_block += [nn.Dropout(p=droprate)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier
    def forward(self, x):
        x = self.add_block(x)
        if self.return_f:
            f = x
            x = self.classifier(x)
            return x,f
        else:
            x = self.classifier(x)
            return x


# Define the ResNet50-based Model
class ft_net(nn.Module):

    def __init__(self, class_num, droprate=0.5, stride=2, init_model=None, pool='avg'):
        super(ft_net, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        # avg pooling to global pooling
        if stride == 1:
            model_ft.layer4[0].downsample[0].stride = (1,1)
            model_ft.layer4[0].conv2.stride = (1,1)

        self.pool = pool
        if pool =='avg+max':
            model_ft.avgpool2 = nn.AdaptiveAvgPool2d((1,1))
            model_ft.maxpool2 = nn.AdaptiveMaxPool2d((1,1))
            self.model = model_ft
            self.classifier = ClassBlock(4096, class_num, droprate)
        elif pool=='avg':
            model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
            self.model = model_ft
            self.classifier = ClassBlock(2048, class_num, droprate)

        self.flag = False
        if init_model!=None:
            self.flag = True
            self.model = init_model.model
            self.pool = init_model.pool
            self.classifier.add_block = init_model.classifier.add_block
            self.new_dropout = nn.Sequential(nn.Dropout(p = droprate))
        # avg pooling to global pooling

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        if self.pool == 'avg+max':
            x1 = self.model.avgpool2(x)
            x2 = self.model.maxpool2(x)
            x = torch.cat((x1,x2), dim = 1)
            x = x.view(x.size(0), x.size(1))
        elif self.pool == 'avg':
            x = self.model.avgpool(x)
            x = x.view(x.size(0), x.size(1))
        if self.flag:
            x = self.classifier.add_block(x)
            x = self.new_dropout(x)
            x = self.classifier.classifier(x)
        else:
            x = self.classifier(x)
        return x

# Define the ResNet50  Model with angle loss
# The code is borrowed from https://github.com/clcarwin/sphereface_pytorch
class ft_net_angle(nn.Module):

    def __init__(self, class_num, droprate=0.5, stride=2):
        super(ft_net_angle, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        # avg pooling to global pooling
        if stride == 1:
            model_ft.layer4[0].downsample[0].stride = (1,1)
            model_ft.layer4[0].conv2.stride = (1,1)
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.model = model_ft
        self.classifier = ClassBlock(2048, class_num, droprate)
        #self.classifier.classifier=nn.Sequential()
        self.classifier.classifier = AngleLinear(512, class_num)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = x.view(x.size(0), x.size(1))
        x = self.classifier(x)
        #x = self.fc(x)
        return x

class ft_net_arc(nn.Module):

    def __init__(self, class_num, droprate=0.5, stride=2):
        super(ft_net_arc, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        # avg pooling to global pooling
        if stride == 1:
            model_ft.layer4[0].downsample[0].stride = (1,1)
            model_ft.layer4[0].conv2.stride = (1,1)
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.model = model_ft
        self.classifier = ClassBlock(2048, class_num, droprate)
        #self.classifier.classifier=nn.Sequential()
        self.classifier.classifier = ArcLinear(512, class_num)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = x.view(x.size(0), x.size(1))
        x = self.classifier(x)
        #x = self.fc(x)
        return x

# Define the DenseNet121-based Model
class ft_net_dense(nn.Module):

    def __init__(self, class_num, droprate=0.5, stride=2, init_model=None, pool='avg'):
        super().__init__()
        model_ft = models.densenet121(pretrained=True)
        if stride == 1:
            model_ft.features.transition3.pool.stride = 1
        model_ft.fc = nn.Sequential()

        self.pool = pool
        if pool =='avg+max':
            model_ft.features.avgpool = nn.Sequential()
            model_ft.avgpool2 = nn.AdaptiveAvgPool2d((1,1))
            model_ft.maxpool2 = nn.AdaptiveMaxPool2d((1,1))
            self.model = model_ft
            self.classifier = ClassBlock(2048, class_num, droprate)
        elif pool=='avg':
            model_ft.features.avgpool = nn.AdaptiveAvgPool2d((1,1))
            self.model = model_ft
            self.classifier = ClassBlock(1024, class_num, droprate)

        self.flag = False
        if init_model!=None:
            self.flag = True
            self.model = init_model.model
            self.pool = init_model.pool
            self.classifier.add_block = init_model.classifier.add_block
            self.new_dropout = nn.Sequential(nn.Dropout(p = droprate))

    def forward(self, x):
        if self.pool == 'avg':
            x = self.model.features(x)
        elif self.pool == 'avg+max':
            x = self.model.features(x)
            x1 = self.model.avgpool2(x)
            x2 = self.model.maxpool2(x)
            x = torch.cat((x1,x2), dim = 1)
        x = x.view(x.size(0), x.size(1))
        if self.flag:
            x = self.classifier.add_block(x)
            x = self.new_dropout(x)
            x = self.classifier.classifier(x)
        else:
            x = self.classifier(x)
        return x


# Define the ResNet50-based Model (Middle-Concat)
# In the spirit of "The Devil is in the Middle: Exploiting Mid-level Representations for Cross-Domain Instance Matching." Yu, Qian, et al. arXiv:1711.08106 (2017).
class ft_net_middle(nn.Module):

    def __init__(self, class_num, droprate=0.5):
        super(ft_net_middle, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        # avg pooling to global pooling
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.model = model_ft
        self.classifier = ClassBlock(2048+1024, class_num, droprate)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        # x0  n*1024*1*1
        x0 = self.model.avgpool(x)
        x = self.model.layer4(x)
        # x1  n*2048*1*1
        x1 = self.model.avgpool(x)
        x = torch.cat((x0,x1),1)
        x = x.view(x.size(0), x.size(1))
        x = self.classifier(x)
        return x

# Part Model proposed in Yifan Sun etal. (2018)
class PCB(nn.Module):
    def __init__(self, class_num, init_model=None ):
        super(PCB, self).__init__()

        self.part = 2  # We cut the pool5 to 2 parts
        model_ft = models.resnet50(pretrained=True)
        self.model = model_ft
        self.part_avgpool1 = nn.AdaptiveAvgPool2d((3, 1))
        self.part_maxpool1 = nn.AdaptiveMaxPool2d((3, 1))
        self.part_avgpool2 = nn.AdaptiveAvgPool2d((1, 3))
        self.part_maxpool2 = nn.AdaptiveMaxPool2d((1, 3))
        self.avgpool2 = nn.AdaptiveAvgPool2d((1,1))
        self.maxpool2 = nn.AdaptiveMaxPool2d((1,1))
        self.dropout = nn.Dropout(p=0.5)
        # remove the final downsample
        self.model.layer4[0].downsample[0].stride = (1,1)
        self.model.layer4[0].conv2.stride = (1,1)
        # define 6 classifiers
        self.classifier0 = ClassBlock(1024, class_num, droprate=0.2, relu=False, bnorm=True, extra_bn=True, num_bottleneck=512)
        self.classifier1 = ClassBlock(2048, class_num, droprate=0.2, relu=False, bnorm=True, extra_bn=True, num_bottleneck=512)
        for i in range(2, self.part + 2): # add one original
            name = 'classifier'+str(i)
            setattr(self, name, ClassBlock(2048*3, class_num, droprate=0.2, relu=True, bnorm=True, extra_bn=True, num_bottleneck=512))

        if init_model!=None:
            self.flag = True
            self.model = init_model.model
            self.pool = init_model.pool

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x0 = self.avgpool2(x) + self.maxpool2(x)
        x = self.model.layer4(x)
        x1 = self.avgpool2(x) + self.maxpool2(x)
        x2 = self.part_avgpool1(x) + self.part_maxpool1(x)
        x3 = self.part_avgpool2(x) + self.part_maxpool2(x)
        part = {}
        predict = {}
        # get six part feature batchsize*2048*6
        predict[0] = self.classifier0(x0.view(x0.size(0), x0.size(1)))
        predict[1] = self.classifier1(x1.view(x1.size(0), x1.size(1)))
        predict[2] = self.classifier2(x2.view(x2.size(0), 2048*3))
        predict[3] = self.classifier3(x3.view(x3.size(0), 2048*3))

        y = []
        for i in range(self.part+2):
            y.append(predict[i])
        return y

class PCB_test(nn.Module):
    def __init__(self,model):
        super(PCB_test,self).__init__()
        self.part = 2
        self.model = model.model
        self.avgpool = nn.AdaptiveAvgPool2d((self.part,1))
        # remove the final downsample
        self.model.layer4[0].downsample[0].stride = (1,1)
        self.model.layer4[0].conv2.stride = (1,1)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.avgpool(x)
        y = x.view(x.size(0),x.size(1),x.size(2))
        return y

'''
# debug model structure
# Run this code with:
python model.py
'''
if __name__ == '__main__':
# Here I left a simple forward function.
# Test the model, before you train it. 
    net = PCB(751)
    #net = ft_net_SE(751)
    print(net)
    input = Variable(torch.FloatTensor(4, 3, 256, 256))
    output = net(input)
    print('net output size:')
    #print(output.shape)
