import torch
import torch.nn as nn
from torch.nn import MaxPool2d
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

#input =torch.tensor([[1,2,0,3,1],
#                     [0,1,2,3,1],
#                     [1,2,1,0,0],
#                     [5,2,3,1,1],
#                     [2,1,0,1,1]])

input = torch.reshape(input,(-1,1,5,5))

dataset = torchvision.datasets.CIFAR10(root="dataset",train=False,transform=torchvision.transforms.ToTensor()
                                       ,download=True)
dataloader = DataLoader(dataset=dataset,batch_size=64,shuffle=True,num_workers=0,drop_last=False)
writer = SummaryWriter("dataloader")
class model(nn.Module):
    def __init__(self):
        super(model,self).__init__()
        self.maxpool1 = MaxPool2d(kernel_size=3,ceil_mode=True)
    def forward(self,x):
        x=self.maxpool1(x)
        return x

model1 = model()
step = 0
for data in dataloader:
    input,target=data
    output =model1(input)
    print(input.shape)
    print(output.shape)
    writer.add_images("origin",input,step)
    writer.add_images("maxpool2d",output,step)
    step = step + 1
#print(output)
writer.close()