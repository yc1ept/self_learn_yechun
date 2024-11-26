import torch
import torchvision
import torch.nn as nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10(root="dataset",train=False,transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset,batch_size=64,shuffle=True,drop_last=True)
class model(nn.Module):

    def __init__(self):
        super(model,self).__init__()
        self.conv1 =Conv2d(in_channels=3,out_channels=6,kernel_size=3,stride=1,padding=1)

    def forward(self,x):
        x = self.conv1(x)
        return x
writer = SummaryWriter("dataloader")
model1 = model()
print(model1)
i = 0
for data in dataloader:
    imgs,target = data
    output = model1(imgs)
    print(imgs.shape)
    print(output.shape)
    writer.add_images("origin", imgs,global_step=i)
    #torch.Size([64, 6, 32, 32])->([???,3,32,32])
    output = torch.reshape(output,(-1,3,32,32))#第一个值不清楚填多少的时候可以填-1，会自动调整为合适值
    writer.add_images("conv2d", output, global_step=i)
    i = i + 1
writer.close()
