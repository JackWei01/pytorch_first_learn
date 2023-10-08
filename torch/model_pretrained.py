import torchvision
from torch import nn

# train_dataset = torchvision.datasets.ImageNet("./data_image_net",split="train",transform=torchvision.transforms.ToTensor()
#                                               ,download=True)

vgg16 = torchvision.models.vgg16(pretrained=False) #模型的参数是我们自己代码中预设的？
vgg16_true = torchvision.models.vgg16(pretrained=True) #模型中的参数是训练好的
print("ok")

print(vgg16_true)


#利用现有的vgg16模型（可以分类1000个），改动结构，变成10分类


#增加层数
vgg16.add_module("linear_1000_10",nn.Linear(1000,10))

vgg16.classifier.add_module("linear_1000_10",nn.Linear(1000,10)) #添加的层次 与上面的不同
print(vgg16)

#修改现有的层

vgg16.classifier[6] = nn.Linear(4096,10)

print(vgg16)