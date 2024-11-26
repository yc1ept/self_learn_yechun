import torch
from torch import nn
import torchvision
from torch.nn import Conv2d,MaxPool2d,Flatten,Linear,Sequential
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10(root="dataset",train=False,download=True,
                                       transform=torchvision.transforms.ToTensor())
dataloader =DataLoader(dataset,batch_size=8
                       ,shuffle=True,drop_last=True)

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
loss = nn.CrossEntropyLoss()#交叉熵
model1 = model()
optim = torch.optim.SGD(model1.parameters(),lr=0.01)#随机梯度下降,lr是学习速率
for epoch in range(20):
    running_loss = 0.0
    for data in dataloader:
        inputs,target = data
        outputs = model1(inputs)
        result_loss = loss(outputs,target)
        optim.zero_grad()#把初始梯度设置为0
        result_loss.backward()#反向传播
        optim.step()
        running_loss = running_loss + result_loss
    print(running_loss)
