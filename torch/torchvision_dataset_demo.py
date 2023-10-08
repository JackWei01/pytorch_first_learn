# 数据集+ transform 处理
import torchvision
from torch.utils.tensorboard import SummaryWriter

'''
train_set = torchvision.datasets.CIFAR10(root="./dataset_offical",train=True,download=True)

test_set = torchvision.datasets.CIFAR10(root="./dataset_offical",train=False,download=True)



print(test_set[0])
print(test_set.classes) # (<PIL.Image.Image image mode=RGB size=32x32 at 0x282848DF850>, 3) 可知test_set[0] 对应两个数据 第一个是PIL.Image.Image，第二个是数字



img, target = test_set[0]

img.show() #图片太小，结合之前的 transform resize 放大！
print(img)

print(target)
print("test_set[0]的标签是:{}".format(test_set.classes[target]))

'''



#读取数据集，并转为tensor格式

#写法1：在compose中直接定义类
# dataset_transform =  torchvision.transforms.Compose([torchvision.transforms.ToTensor()])


#写法2：先定义类，在填入参数中去
trans_ToTensor = torchvision.transforms.ToTensor()
dataset_transform =  torchvision.transforms.Compose([trans_ToTensor])

train_set = torchvision.datasets.CIFAR10(root="./dataset_offical",train=True, transform=dataset_transform, download=True)

test_set = torchvision.datasets.CIFAR10(root="./dataset_offical",train=False, transform=dataset_transform, download=True)


# print(test_set[0])

writer = SummaryWriter("logs")

for i in range(50,100):

    writer.add_image("train_set001",train_set[i][0],i) #train_set[i][0] 是 tensor类型 train_set数据集 在下载的同时，也完成了transform=dataset_transform TOttensor的转换

writer.close()