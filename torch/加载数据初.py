# Dataset  提供一种方式去获取数据及其label

# Dataloader  为后面的网络提供不同的数据形式

import os
from PIL import Image
import cv2
from torch.utils.data import Dataset



class MyData(Dataset):
    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.imge_path_list = os.listdir(self.path)
        # self.imge_path_list =os.listdir(r'C:\Users\Gaming\PycharmProjects\Deeplearn\torch\hymenoptera_data\train\ants')

    def __getitem__(self, index):
        img_name = self.imge_path_list[index]
        img_item_path = os.path.join(self.path, img_name)
        img = Image.open(img_item_path)
        label = self.label_dir
        # print(img.size)
        return img, label

    def __len__(self):
        return (len(self.imge_path_list))




# dir_path = './hymenoptera_data/train/ants'
dir_absou_path = r'C:\Users\Gaming\PycharmProjects\Deeplearn\torch\hymenoptera_data\train\ants'

# img_path_list = os.listdir(dir_path)
#这里的root_dir 要看项目实际运行时os的当前路径等情况而定
root_dir = 'hymenoptera_data/train'
label_ants_dir = 'ants'
label_bees_dir = 'bees'
# path = os.path.join(root_dir, label_dir)


ants_dataset = MyData(root_dir, label_ants_dir)
bees_dataset = MyData(root_dir, label_bees_dir)

img,label = ants_dataset[0]
img.show()
img,label = ants_dataset[1]
img.show()
### 数据集合并
train_data = ants_dataset + bees_dataset
