import torch
from torch import nn
import torchvision
from torch.nn import Conv2d,MaxPool2d,Flatten,Linear,Sequential
from torch.utils.data import DataLoader
from torchvision.models import vgg16, Weights

#dataset = torchvision.datasets.ImageNet(root="imageset",split="train",download=True,transform=torchvision.transforms.ToTensor())
#vgg16_false = torchvision.models.vgg16()

vgg16_true = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights)
print(vgg16_true)#查看VGG的网络架构
vgg16_true.classifier.add_module("add_linear",Linear(1000,10))
print(vgg16_true)
#vgg16_true.classifier[6] = Linear(4096,10)