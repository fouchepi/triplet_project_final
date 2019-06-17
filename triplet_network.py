from __future__ import print_function
import torch.nn.functional as F
import torch.nn as nn
from torchvision import models
from densenet import DenseNet
import torch

class MnistTripletNet(nn.Module):
    """
    First triplet network by radhakrishna.achanta@epfl.ch

    """
    def __init__(self):
        super(MnistTripletNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv1_drop = nn.Dropout2d(p=0)
        self.conv2_drop = nn.Dropout2d(p=0)
        #self.conv3_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(128*3*3, 50)
        self.fc2 = nn.Linear(50, 10)
        self.fc3 = nn.Linear(10, 2)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1_drop(self.conv1(x)), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        # x = x.view(-1, 128*3*3) # Returns a new tensor with the same data but different size.
        x = x.view(-1, self.num_flat_features(x))# Returns a new tensor with the same data but different size.
        x = F.relu(self.fc1(x))
        # x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

def densenetcustom(out_features, pretrained=True):
    """
    generate pytorch densenet121 model with custom classifier layer

    :param out_features: the size of the descriptors
    :param pretrained: if we use a pretrained model on imagenet
    :return: the model
    """
    model_ft = models.densenet121(pretrained=pretrained)
    num_ftrs = model_ft.classifier.in_features
    print("num features : ", num_ftrs)
    model_ft.classifier = nn.Linear(num_ftrs, out_features)
    return model_ft

def resnetcustom(out_features, pretrained=True):
    """
    generate pytorch resnet18 model with custom classifier layer

    :param out_features: the size of the descriptors
    :param pretrained: if we use a pretrained model on imagenet
    :return: the model
    """
    model_ft = models.resnet18(pretrained=pretrained)
    num_ftrs = model_ft.fc.in_features
    print("num features : ", num_ftrs)
    model_ft.fc = nn.Linear(num_ftrs, out_features)
    return model_ft

def densenet_mem_eff(out_features, growth_rate, depth, small_inputs=True, efficient=False):
    """
    Generate memory efficient densenet model from https://github.com/gpleiss/efficient_densenet_pytorch
    :param out_features: size of the descriptors
    :param growth_rate:
    :param depth:
    :param small_inputs: image size to use 32 or 224
    :param efficient: memory efficient or not
    :return: the model
    """
    # Get densenet configuration
    if (depth - 4) % 3:
        raise Exception('Invalid depth')
    block_config = [(depth - 4) // 6 for _ in range(3)]

    model = DenseNet(
        growth_rate=growth_rate,
        block_config=block_config,
        num_classes=out_features,
        small_inputs=small_inputs,
        efficient=efficient,
    )

    return model

class CustomDensenet121(nn.Module):
    def __init__(self, out_features):
        super(CustomDensenet121, self).__init__()
        self.densenet121 = models.densenet121(pretrained=True)
        self.in_features = self.densenet121.classifier.in_features

        self.features = nn.Sequential(*list(self.densenet121.children())[:-1])
        self.fc1 = nn.Linear(self.in_features, out_features)

    def forward(self, x):
        x = self.features(x)
        print(x.shape)
        x = x.view(x.size(0), -1)
        print(x.shape)
        x1 = self.fc1(x)
        return x1

class CustomResnet(nn.Module):
    def __init__(self, out_features):
        super(CustomResnet, self).__init__()
        self.resnet50 = models.resnet50(pretrained=True)
        self.in_features = self.resnet50.fc.in_features

        self.features = nn.Sequential(*list(self.resnet50.children())[:-2])

        self.maxpool = nn.MaxPool2d(kernel_size=7, stride=1, padding=0)
        self.fc1 = nn.Linear(self.in_features, out_features)

    def forward(self, x):
        x = self.features(x)
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x1 = self.fc1(x)
        return x1

class CustomResnet2(nn.Module):
    """
    Mostly used resnet18 model.
    Keep all layers of the resnet model but the last two
    and replace them in order to change the size of the descriptor
    and the last activation function.
    """
    def __init__(self, out_features):
        super(CustomResnet2, self).__init__()
        self.resnet18 = models.resnet18(pretrained=True)
        self.in_features = self.resnet18.fc.in_features

        self.features = nn.Sequential(*list(self.resnet18.children())[:-2])

        self.maxpool = nn.MaxPool2d(kernel_size=7, stride=1, padding=0)
        self.fc1 = nn.Linear(self.in_features, out_features)
        self.fc2 = nn.Linear(out_features, 2)

    def forward(self, x):
        x = self.features(x)
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x1 = self.fc1(x)
        x2 = self.fc2(x1)
        return x1, x2

class CustomResnet4(nn.Module):
    def __init__(self, out_features):
        super(CustomResnet4, self).__init__()
        self.resnet18 = models.resnet18(pretrained=True)
        self.in_features = self.resnet18.fc.in_features

        self.features = nn.Sequential(*list(self.resnet18.children())[:-1])

        self.conv1 = nn.Sequential(
            nn.Conv2d(self.in_features, self.in_features*2, kernel_size=3),
            nn.Dropout2d(p=0),
            nn.BatchNorm2d(self.in_features*2),
            nn.MaxPool2d(2),
            nn.Tanh())

        self.conv2 = nn.Sequential(
            nn.Conv2d(self.in_features*2, self.in_features*4, kernel_size=3, padding=1),
            nn.Dropout2d(p=0),
            nn.BatchNorm2d(self.in_features*4),
            nn.MaxPool2d(2),
            nn.Tanh())

        self.fc1 = nn.Sequential(
            nn.Linear(self.in_features*4, 1024),
            nn.ReLU())

        self.fc2 = nn.Sequential(
            nn.Linear(self.in_features, 256),
            nn.Tanh())

        self.fc3 = nn.Linear(256, out_features)

    def forward(self, x):
        x = self.features(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x1 = self.fc2(x)
        x2 = self.fc3(x1)
        return x1, x2


