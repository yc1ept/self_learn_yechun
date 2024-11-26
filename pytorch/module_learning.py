import torch
import torch.nn as nn


class model(nn.Module):
    def __init__(self) :
        super().__init__()#super() 是一个内置函数，用于调用父类（超类）的一个方法。
    def forward(self,input):
        output = input + 1
        return output

model1 = model()#类的实例化
x = torch.tensor(1.0)
output = model1(x)
print(output)

