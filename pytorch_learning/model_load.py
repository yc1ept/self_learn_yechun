import torch
import torchvision

#加载方式1
model1 = torch.load("vgg16_method1.pth")
print(model1)
#加载方式2
vgg16 = torchvision.models.vgg16()
#model2 = torch.load("vgg16_method2.pth")#这样print输出的为字典型，并非模型
vgg16.load_state_dict(torch.load("vgg16_method2.pth"))
print(vgg16)