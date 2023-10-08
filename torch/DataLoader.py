# dataset 数据集
# dataloader 数据加载器，把数据加载到神经网络中
import torchvision
from torch.utils.data import  DataLoader
from torch.utils.tensorboard import SummaryWriter

#准备测试数据集
test_dataset = torchvision.datasets.CIFAR10("dataset_offical",train=False,download=True,transform=torchvision.transforms.ToTensor())


# test_loader = DataLoader(dataset=test_dataset,batch_size=4,shuffle=True,sampler=None,num_workers=0,drop_last=False) #sampler=None 默认 采用随即抓取，batch_size=4,四个图片一组
# test_loader = DataLoader(dataset=test_dataset,batch_size=4,shuffle=True,sampler=None,num_workers=0,drop_last=False) #shuffle=True ,多次的epoch，数据的抽取顺序顺序是否一致 shuffle=True 为不一致
# test_loader = DataLoader(dataset=test_dataset,batch_size=64,shuffle=True,sampler=None,num_workers=0,drop_last=False) #sampler=None 默认 采用随即抓取 # drop_last 是否舍去最后不能凑成64的那个批次的数据
test_loader = DataLoader(dataset=test_dataset,batch_size=64,shuffle=True,sampler=None,num_workers=0,drop_last=True) #s 舍去最后不能凑成64的那个批次的数据


#测试数据集中第一张图片及其target
img, target = test_dataset[0]  ## 原理是魔法方法  __getitem__(self, index: int) -> Tuple[Any, Any]:
print(img.shape)
print(target)
print(test_dataset.classes)


#test_loader ，


writer = SummaryWriter("logs")
step = 0


for batch_data in test_loader:
    imgs, targets = batch_data

    # print(imgs.shape) #torch.Size([4, 3, 32, 32])  4张图片 3个通道  32X32像素
    # print(targets)   #tensor([1, 1, 4, 0]) # 4张图片的 target
    # print(batch_data)
    #单张图片添加
    # for temp_data in batch_data[0]:
    #     writer.add_image("test_dataloader", temp_data ,step)

    #一堆图片，一次添加
    writer.add_images("test_dataloader005",imgs,step)


    step += 1

writer.close()
