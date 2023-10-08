from torch.utils.tensorboard import SummaryWriter





def add_scalar():
    writer = SummaryWriter("logs")
    for i in range(100):
        #第一个参数是title(名字唯一会被持续记录，想清除，就删掉对应的logs里的文件
        writer.add_scalar("y=2x",2*i,i)#定义y=2x
    writer.close()


def add_image():
    '''
    tag (str): Data identifier
    img_tensor (torch.Tensor, numpy.ndarray, or string/blobname): Image data
    global_step (int): Global step value to record
    '''
    writer = SummaryWriter("logs")
    img_path = r'hymenoptera_data/train/ants/0013035.jpg'

    #方法1使用Image和numpy
    from PIL import   Image
    import numpy as np
    img_PIL = Image.open(img_path)
    img_arry = np.array(img_PIL)
    img_arry = np.array(img_PIL)
    #print(img_arry.shape) #此处的shape信息为(512, 768, 3) 必须设置dataformats 为 对应的 CHW,HWC,HW中的一个

    #方法2 使用opencv
    # import cv2
    # img = cv2.imread(img_path)
    # writer.add_image("test",img,1)

    # 第一个参数是title(名字唯一会被持续记录，想清除，就删掉对应的logs里的文件) 第三个参数 表示step
    writer.add_image("test", img_arry, 1,dataformats='HWC')
    writer.add_image("test", img_arry, 2,dataformats='HWC') #当然实际中是不同的图片，显示训练的效果
    writer.add_image("test", img_arry, 3,dataformats='HWC')
    writer.add_image("test", img_arry, 4,dataformats='HWC')

    writer.add_image("test002", img_arry, 1, dataformats='HWC')


    # for i in range(100):
    #     writer.add_scalar("y=2x",2*i,i)#定义y=2x

    writer.close()




if __name__ == '__main__':
    add_scalar()
    add_image()