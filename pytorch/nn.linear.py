import torch
import torchvision
from torch.utils.data import DataLoader
import torch.nn as nn

dataset = torchvision.datasets.CIFAR10(root="dataset",train=False,download=True,
                                       transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset,batch_size=64,shuffle=True,drop_last=True)
class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()
        self.linear = nn.Linear(196608,10)#这里的196608是通过下面output那里reshape拉伸之后，那个-1自动算出来的。
    def forward(self,input):
        output = self.linear(input)
        return output

model1 = model()
for data in dataloader:
    input,target = data
    print(input.shape)
    #output =torch.reshape(input,(1,1,1,-1))
    output = torch.flatten(input)
    print(output.shape)
    output = model1(output)
    print(output.shape)