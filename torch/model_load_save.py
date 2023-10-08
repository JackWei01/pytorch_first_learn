import torch
import torchvision
from torch import nn
from torch.nn import Sequential, Flatten, Linear

vgg16 = torchvision.models.vgg16(pretrained=False)


#保存方式1
torch.save(vgg16,"vgg16_method1.pth")#保存vgg16的网络结构+参数

#保存方式2 官方推荐 大小较小
torch.save(vgg16.state_dict(),"vgg16_method2.pth")#只保存vgg16网络的参数到字典


#模型加载->保存方式1

model1 =torch.load("vgg16_method1.pth")
print(model1)

#模型加载->保存方式2
vgg16_model = torchvision.models.vgg16(pretrained=False) #加载vgg16的网络结构
vgg16_model.load_state_dict(torch.load("vgg16_method2.pth"))
# model2 = torch.load("vgg16_method2.pth")
print(vgg16_model)


### 陷阱

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

mymodule = My_module()

torch.save(mymodule,"mymoudle.pth")

#此时想要加载自己定一个模型，要在torch.load之前重新定义一下 class Mymodule(),但不需要再实力化对象mymodule = My_module()

