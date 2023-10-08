import torch
import torchvision
from torch import nn
from torch.nn import Linear
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10("dataset_offical",train=False,transform=torchvision.transforms.ToTensor(),download=True)


dataloader = DataLoader(dataset,batch_size=64,drop_last=True)

class MyLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = Linear(196608,10)

    def forward(self,input):

        return  self.linear1(input)



mylinear = MyLinear()


step = 0
for data in dataloader:
    imgs, targets = data

    output = torch.reshape(imgs,(1,1,1,-1)) #上下两个的区别 ，此处是为了将数据展成一个通道一行
    # torch.reshape(input, (-1, 1, 5, 5))  # 四维 #torch.Size([1, 1, 5, 5])

    print(imgs.shape)
    print(output.shape)

    output = mylinear(output)  #flatten 展平成一行 与reshape相比较
    print(output.shape)
    print(output)

    if step == 0:
        break

