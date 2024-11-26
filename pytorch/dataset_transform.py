import torchvision
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
writer = SummaryWriter("logs")
dataset_trans = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Resize(512)
])

train_set = torchvision.datasets.CIFAR10(root="dataset",train=True,transform=dataset_trans,download=True)
test_set = torchvision.datasets.CIFAR10(root="dataset",train=False,transform=dataset_trans,download=True)
#print(type(test_set[0]))通过这个可以看出返回的是tuple元组类型的数据
img,target=test_set[0]#所以不能直接用img=test_set[0]
print(test_set.classes[target])
#img.show()转换成tensor格式后就不能show了
for i in range(10):
    img, target = test_set[i]
    writer.add_image("testset",img,i)

writer.close()