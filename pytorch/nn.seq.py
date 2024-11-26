import torch
import torchvision
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.nn import Conv2d,MaxPool2d,Flatten,Linear
from torch.utils.tensorboard import SummaryWriter

#dataset = torchvision.datasets.CIFAR10(root="dataset",train=False,download=True,
#                                       transform=torchvision.transforms.ToTensor())
#dataloader = DataLoader(dataset,batch_size=64,shuffle=True,drop_last=True)
class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()
        self.conv1 = Conv2d(3,32,5,1,2)#这里需要根据官方文档中的公式去计算stride和padding
        self.maxpool1 = MaxPool2d(2)
        self.conv2 = Conv2d(32,32,5,1,2)
        self.maxpool2 = MaxPool2d(2)
        self.conv3 = Conv2d(32,64,5,1,2)
        self.maxpool3 = MaxPool2d(2)
        self.flatten = Flatten()
        self.linear1 = Linear(1024,64)#1024=64*4*4
        self.linear2 = Linear(64,10)

        self.seq = nn.Sequential(
            Conv2d(3,32,5,1,2),MaxPool2d(2),Conv2d(32,32,5,1,2),MaxPool2d(2),Conv2d(32,64,5,1,2),MaxPool2d(2),
            Flatten(),Linear(1024,64),Linear(64,10)
        )

    def forward(self,input):
        input = self.conv1(input)
        input = self.maxpool1(input)
        input = self.conv2(input)
        input = self.maxpool2(input)
        input = self.conv3(input)
        input = self.maxpool3(input)
        input = self.flatten(input)
        input = self.linear1(input)
        output = self.linear2(input)
#       output = self.seq(input)
        return output
writer =SummaryWriter("logs")
model1 = model()
print(model1)
input = torch.ones((64,3,32,32))
output = model1(input)
print(output.shape)
writer.add_graph(model1,input)
writer.close()
#for data in dataloader:
  #  input,target = data
  #  output = model1(output)

