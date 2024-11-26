import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

test_set = torchvision.datasets.CIFAR10(root="dataset",train=False,transform=torchvision.transforms.ToTensor(),download=True)

test_loader = DataLoader(dataset=test_set,batch_size=64,shuffle=True,num_workers=0,drop_last=False)#num_workers取0是只有一个主进程，>0可能会报错，drop_last则是如果有余数是否取净
img,target = test_set[0]
print(img.shape)
writer = SummaryWriter("dataloader")
step=0
for data in test_loader:
    imgs,targets =data
    writer.add_images("test_data",imgs,step)
    step=step+1

writer.close()
