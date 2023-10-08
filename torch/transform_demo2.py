import cv2
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms


#


img_path = r'C:\Users\Gaming\PycharmProjects\Deeplearn\torch\hymenoptera_data\train\ants\175998972.jpg'
img_PIL = Image.open(img_path)


writer = SummaryWriter("logs")



def To_tensor(img):
    tensor_trans = transforms.ToTensor()
    img_tensor = tensor_trans(img)
    return img_tensor


def To_PILimage():
    pass

#标准化  正太分布
def  Normalize(img_tensor):
    '''


      Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation in-place.
    :return:
     output[channel] = (input[channel] - mean[channel]) / std[channel]
    '''

    print(To_tensor(img_tensor)[0][0][0])
    trans_norm = transforms.Normalize([1,0.5,3],[0.5,2,3])
    img_norm_tensor = trans_norm.forward(To_tensor(img_tensor))
    print(img_norm_tensor)
    writer.add_image("IMG_Normalize",img_norm_tensor,1)
    writer.close()
    return img_norm_tensor


def Resize(img_tensor):
    #放缩至固定的长宽
    # trans_resize = transforms.Resize(640)
    print("原尺寸 {}".format(img_PIL.size))
    trans_resize = transforms.Resize((80,80))
    img_resize = trans_resize.forward(img_tensor)
    print(img_resize)
    # img_resize.show()
    img_resize_tensor = To_tensor(img_resize)
    print(img_resize_tensor.size())

    writer.add_image("IMG_Resize", img_resize_tensor, 2)
    writer.close()
    return img_resize_tensor

def Resize_proportional(img_tensor):
    tensor_trans = transforms.ToTensor()
    trans_resize_proportional = transforms.Resize(80)
    trans_compose = transforms.Compose([trans_resize_proportional,tensor_trans]) #transforms.Compose 可以理解为流水线方法
    img_resize_proportional = trans_compose(img_tensor)
    writer.add_image("IMG_Resize_proportional", img_resize_proportional, 0)
    writer.close()


def RandomCrop(img_tensor):
    tensor_trans = transforms.ToTensor()
    # trans_randomcrop = transforms.RandomCrop(110)#指要小于原图像的最小边（宽高）
    trans_randomcrop = transforms.RandomCrop(120,100)#指要小于原图像的最小边（宽高）
    trans_compose = transforms.Compose([trans_randomcrop,tensor_trans])
    for i in range(10):
        img_crop = trans_compose(img_tensor)
        writer.add_image("RandomCrop",img_crop,i)

if __name__ == '__main__':
    Normalize(img_PIL)
    Resize(img_PIL)
    Resize_proportional(img_PIL)
    RandomCrop(img_PIL)