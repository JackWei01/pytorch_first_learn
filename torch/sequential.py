import torch
import torchvision
from torch import nn
from torch.nn import Linear, Flatten, Sequential
from torch.utils.data import DataLoader

# A sequential container.

# dataset = torchvision.datasets.CIFAR10("data-offical",train=True,transform=torchvision.transforms.ToTensor(),download=True)
#
#
# dataloader = DataLoader(dataset,batch_size=64,shuffle=True,drop_last=False)
#
# for data in dataloader:
#     imgs, targets = data
#
# print(imgs.dim())

## 方法1：
# class My_module(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Conv2d(in_channels= 3, out_channels= 32, kernel_size=5,stride=1,padding=2)# stride=1,padding=2 是根据你要的卷积后的层的channel h,w 计算出来的
#         self.maxpool1 = nn.MaxPool2d(kernel_size=2)
#         self.conv2 = nn.Conv2d(in_channels= 32, out_channels= 32, kernel_size=5,stride=1,padding=2)# stride=1,padding=2 是根据你要的卷积后的层的channel h,w 计算出来的
#         self.maxpool2 = nn.MaxPool2d(kernel_size=2)
#         self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
#         self.maxpool3 = nn.MaxPool2d(kernel_size=2)
#         # self.flatten1 = Flatten(start_dim=4,end_dim=1)
#         self.flatten1 = Flatten()
#         self.linear1 = Linear(in_features=1024,out_features=64)
#         self.linear2 = Linear(in_features=64,out_features=10)
#
#     def forward(self,input):
#         input = self.conv1(input)
#         input = self.maxpool1(input)
#         input = self.conv2(input)
#         input = self.maxpool2(input)
#         input = self.conv3(input)
#         input = self.maxpool3(input)
#         input = self.flatten1(input)
#         input = self.linear1(input)
#         output = self.linear2(input)
#
#         return  output

## 方法2：sequential
class My_module(nn.Module):
    def __init__(self):
        super().__init__()

        self.model1 = Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),
            # stride=1,padding=2 是根据你要的卷积后的层的channel h,w 计算出来的
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2),
            # stride=1,padding=2 是根据你要的卷积后的层的channel h,w 计算出来的
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2),

            Flatten(),
            Linear(in_features=1024, out_features=64),
            Linear(in_features=64, out_features=10)
        )


    def forward(self,input):

        output = self.model1(input)

        return  output






myfirst_module = My_module()

input = torch.ones((64,3,32,32))


output = myfirst_module(input)

print(output)