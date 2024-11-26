import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
#input = torch.tensor([[1,-0.5],
#                      [-1,3]])
#input = torch.reshape(input,(-1,1,2,2))
dataset = torchvision.datasets.CIFAR10(root="dataset",train=False,transform=torchvision.transforms.ToTensor()
                                       ,download=True)
dataloader = DataLoader(dataset,batch_size=64,shuffle=True)
writer = SummaryWriter("dataloader")

class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        #x = self.relu(x)
        x = self.sigmoid(x)
        return x
model1 =model()
step = 0
for data in dataloader:
    input,target = data
    output = model1(input)
    writer.add_images("origin",input,step)
    writer.add_images("sigmoid",output,step)
    step = step + 1
#print(output)
writer.close()