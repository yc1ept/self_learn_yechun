import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn import Sequential,Conv2d,MaxPool2d,Flatten,Linear
from torch.utils.tensorboard import SummaryWriter

#1.准备数据集（训练+测试）

device = torch.device("cuda")#定义训练设备

train_data = torchvision.datasets.CIFAR10(root="dataset",train=True,download=True,
                                          transform=torchvision.transforms.ToTensor())
test_data = torchvision.datasets.CIFAR10(root="dataset",train=False,download=True,
                                          transform=torchvision.transforms.ToTensor())
train_data_size = len(train_data)
test_data_size = len(test_data)
#print(train_data_size)
#print(test_data_size)
print("训练集的长度为：{}".format(train_data_size))
print("测试集的长度为：{}".format(test_data_size))

#2.利用dataloader加载数据集
train_dataloader = DataLoader(train_data,64)
test_dataloader = DataLoader(test_data,64)

#3.搭建神经网络（因为cifar10有十个类别，所以最后需要的输出有10个通道数）
class model(nn.Module):
    def __init__(self):
        super(model,self).__init__()
        self.seq = Sequential(
            Conv2d(3, 32, 5, 1, 2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, 1, 2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, 1, 2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )
    def forward(self,input):
        output = self.seq(input)
        return output
#4.创建网络模型
model1 = model()
model1 = model1.to(device)
#5.损失函数（这里采用的依然是交叉熵）
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)
#6.优化器(随机梯度下降）
learning_rate = 0.001
optimizer =torch.optim.SGD(model1.parameters(),lr=learning_rate)

#7.设置训练网络的参数
total_train_step = 0 #记录训练次数
total_test_step = 0 #记录测试次数
epoch = 20 #训练轮数
writer = SummaryWriter("../traindata")

for i in range(epoch):
    print("第{}轮训练开始".format(i+1))
    #训练步骤开始
    #model1.train()将神经网络设置成训练模式，它和后面的eval()一样，都只对部分层失效，像是dropout、batchnorm
    for data in train_dataloader:
        inputs,targets = data
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = model1(inputs)
        loss = loss_fn(outputs,targets)
        #优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
            print("训练次数：{},loss:{}".format(total_train_step,loss))
            writer.add_scalar("train_loss",loss.item(),total_train_step)
    #测试步骤开始
    #model1.eval()
    total_test_loss = 0
    total_accuracy = 0 #正确率
    with torch.no_grad():
        for data in test_dataloader:
            inputs,targets = data
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model1(inputs)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss
            accuracy = (outputs.argmax(1) == targets).sum()#argmax参数为0则为纵向
            total_accuracy = total_accuracy + accuracy
    print("整体测试集上的loss：{}".format(total_test_loss))
    print("整体测试集上的正确率：{}".format(total_accuracy/test_data_size))
    writer.add_scalar("total_test_loss",total_test_loss,total_test_step)
    writer.add_scalar("total_accuracy", total_accuracy/test_data_size, total_test_step)
    total_test_step = total_test_step + 1

    #保存模型
    #torch.save(model,"model_{}.pth".format(i))
    #print("模型已保存")

writer.close()