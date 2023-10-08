# transforms的结构及用法
import cv2
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

#tensor 数据类型

#通过transform.ToTensor
#1.transform如何使用
# 2.为什么需要tensor数据类型
from PIL import  Image
# img_path = r'hymenoptera_data/train/ants/0013035.jpg'
img_path = r'C:\Users\Gaming\PycharmProjects\Deeplearn\torch\hymenoptera_data\train\ants\175998972.jpg'
img_PIL = Image.open(img_path)
img_ndarray = cv2.imread(img_path)
# print(img)
#创建class
tensor_trans = transforms.ToTensor()

tensor_img1 = tensor_trans(img_PIL) #此处调用了 __call__ 方法
tensor_img2 = tensor_trans(img_ndarray) #此处调用了 __call__ 方法

print(tensor_img1)
print(tensor_img2)


writer = SummaryWriter("logs")

# writer.add_scalar()
writer.add_image("tesor_img",tensor_img1,1)
writer.add_image("tesor_img",tensor_img2,1)# 假设这里的tensor_img2 是一张不同于tensor_img1 的图片 再次运行脚本 也不会替换掉tesor_img 第一步的图片~！~
writer.close() #一定记得close 不然不写入