import torch
import torchvision
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter\


maxpool = MaxPool2d(kernel_size=3,stride=3,padding=0,ceil_mode=True)
class MyModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,input):
        return maxpool.forward(input)


dataset = torchvision.datasets.CIFAR10("dataset_offical",train=False,transform=torchvision.transforms.ToTensor(),download=True)


dataloader = DataLoader(dataset=dataset,batch_size=64,shuffle=True,drop_last=True)

maxpool_obj = MyModule()

setup = 1
writer = SummaryWriter("logs")
for data in dataloader:
    imgs,targets = data
    writer.add_images("pre_imgs",imgs,setup)
    print(imgs.shape)

    #池化操作
    output = maxpool_obj(imgs)
    print(output.shape)
    writer.add_images("maxpool_imgs", output,setup)

    setup+= 1


writer.close()