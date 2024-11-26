import torch
import torchvision
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear

vgg16 = torchvision.models.vgg16()
#保存方式1 模型结构+模型参数
torch.save(vgg16,"vgg16_method1.pth")
#保存方式2 模型参数(占用空间小，便于保存大模型)
torch.save(vgg16.state_dict(),"vgg16_method2.pth")

#陷阱
class model(nn.Module):

    def __init__(self):
        super(model,self).__init__()
        self.mode = Sequential(
            Conv2d(3, 32, 5, 1, 2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, 1, 2),
            MaxPool2d(2), Conv2d(32, 64, 5, 1, 2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )
    def forward(self,x):
        x = self.mode(x)
        return x
model3 =model()
torch.save(model3,"111.pth")#这样保存之后，在另一个文件中加载时需要将上面class后面的再定义一遍，直接加载会报错。不过也可以通过在另一个文件的头部直接from这个py文件import*
