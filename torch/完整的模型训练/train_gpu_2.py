import torch.optim
import torchvision
from torch import nn
from torch.nn import Sequential
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import *

#准备数据集
train_data = torchvision.datasets.CIFAR10("../dataset_offical",transform=torchvision.transforms.ToTensor(),train=True,download=True)


# 准备测试数据集

test_data = torchvision.datasets.CIFAR10("../dataset_offical",transform=torchvision.transforms.ToTensor(),train=False,download=True)



#查看数据集

train_data_size = len(train_data)
test_data_size = len(test_data)

print("训练数据集的长度为{}".format(train_data_size))
print("测试数据集的长度为{}".format(test_data_size))


#用dataloader 加载数据集

train_dataloader = DataLoader(train_data,batch_size=64)
test_dataloader = DataLoader(test_data,batch_size=64)


#定义训练设备
# device = torch.device("cpu")
# device = torch.device("cuda")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device('mps' if torch.backends.mps.is_available() else "cpu")
# print(device)

class Image2ten(nn.Module):
    def __init__(self):
        super().__init__()
        self.module = Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),  # padding
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.Linear(64, 10)

        )

    def forward(self, input):
        ouput = self.module(input)
        return ouput
#搭建神经网络  10分类的网络

my_module = Image2ten()
#cuda
# my_module = my_module.cuda()
my_module = my_module.to(device)

#损失函数
loss_func = nn.CrossEntropyLoss()

# loss_func = loss_func.cuda()
loss_func = loss_func.to(device)

#优化器
# learing_rate = 0.01
learing_rate = 1e-2
optimizer = torch.optim.SGD(my_module.parameters(),lr=learing_rate)


#设置训练网路的一些参数

#训练次数
total_train_step = 0

#测试次数
total_test_step = 0

#训练的轮数
epoch = 200

writer = SummaryWriter("logs")

for i in range(epoch):
    print("第{}轮训练开始".format(i+1))
    epoch_loss = 0
    #训练步骤开始
    for data in train_dataloader:
        imgs, targets = data

        # imgs = imgs.cuda()
        imgs = imgs.to(device)
        # targets = targets.cuda()
        targets = targets.to(device)
        outputs = my_module(imgs)
        loss_output = loss_func(outputs,targets)

        #优化器优化模型
        # 梯度清零
        optimizer.zero_grad()

        #反向传播
        loss_output.backward()

        #优化
        optimizer.step()

        total_train_step += 1

        epoch_loss+=loss_output.item()#loss_output是tensor类型，.item()转数字

        if total_train_step % 100 ==0:
            writer.add_scalar("train_loss",epoch_loss,total_train_step)


    #测试步骤开始
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        #测试每个epoch的loss及准确率
        for data in test_dataloader:
            imgs,targets = data
            # imgs = imgs.cuda()
            imgs = imgs.to(device)
            targets = targets.to(device)
            # targets = targets.cuda()
            outputs = my_module(imgs)
            loss =loss_func(outputs,targets)
            total_test_loss = total_test_loss + loss
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy+=accuracy

    print("正确率：{}".format(total_accuracy/train_data_size))
    writer.add_scalar("test_loss",total_test_loss,i)
    print("第{}轮loss累积为{}".format(i,epoch_loss))

    torch.save(my_module,"my_model{}".format(epoch))
    print("模型已保存")
writer.close()


