import torch
import torchvision
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter




dataset = torchvision.datasets.CIFAR10("dataset_offical",train=False,transform=torchvision.transforms.ToTensor(),download=True)


dataloader = DataLoader(dataset=dataset,batch_size=64,shuffle=True)


#
class MyMoudle(nn.Module):
    def __init__(self):
        super().__init__()
        #定义卷积层
        self.conv1 = Conv2d(in_channels=3,out_channels= 6,kernel_size= 3,stride= 1,padding= 0)


    def forward(self,x):
        x = self.conv1(x)
        return  x

writer = SummaryWriter("logs")

my_module = MyMoudle()

print(my_module)

step = 0
for data in dataloader:
    imgs,targets = data
    output = my_module(imgs)
    # print(output.shape)
    # print(imgs.shape)
    writer.add_images("conv_input",imgs,step) #add_images 不是 add_image 否则AssertionError: size of input tensor and input format are different.         tensor shape: (64, 3, 32, 32), input_format: CHW


    #方法1：
    #writer.add_image("conv_output",output,step) #次数的output torch.Size([64, 6, 32, 32]) 不是常规彩色图像的3层，而是6层，不能直接在tensorboard显示出来

    #reshape
    output = torch.reshape(output,(-1,3,30,30)) #此处意思是将 torch.Size([64, 3, 32, 32])  原来6合1个channel，分成6个 1个Chanel的图像放到batch里


    #方法2
    # 重新排列输出通道以便正确显示
    # output = output.permute(0, 2, 3, 1)  # 将通道维度移到最后
    # output = output[:, :, :, :3]  # 只保留前3个通道，以适应彩色图像显示

    writer.add_images("conv_output",output,step) #次数的output torch.Size([64, 6, 32, 32]) 不是常规彩色图像的3层，而是6层，不能直接在tensorboard显示出来


    step+=1

writer.close()