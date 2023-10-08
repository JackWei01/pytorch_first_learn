import torch
import torchvision
from torch import nn
from torch.nn import MaxPool2d

#
# input = torch.tensor([[],
#                       [],
#                       [],
#
#                       ])

input = torch.rand((5,5)) #二维 torch.Size([5, 5])
print(input.shape)
#形状变化：原始二维张量是形状为(5, 5)，而重塑后的四维张量形状为(N, 1, 5, 5)，其中N是由原始张量的元素数量和新形状的其他维度推断得出的。

# 通道维度的添加：重塑操作在原始二维张量的基础上添加了一个额外的通道维度，这是一个大小为1的维度，用于表示通道数。这在深度学习中经常用于处理图像数据，其中通道维度通常表示图像的颜色通道（例如，RGB图像有3个颜色通道）。

# 数据保持不变：重塑操作没有改变原始数据的内容，只是将数据重新组织为四维张量的形式。原始二维张量的元素在新的四维张量中保持不变。

# 这个操作通常用于将数据准备为输入卷积神经网络（CNN）的格式，因为CNN通常要求输入是四维张量，其中通道维度表示不同的特征图（特征通道）。
input = torch.reshape(input,(-1,1,5,5)) #四维 #torch.Size([1, 1, 5, 5])


print(input.shape)

class Tudui(nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool1 = MaxPool2d(kernel_size=3,ceil_mode=True)

    def forward(self,input):
        if __name__ == '__main__':
            output = self.maxpool1(input)
            return output

my_model = Tudui()

# output = my_model.forward(input) 区别在于class中有没有定义 __call__
output = my_model(input)
print(output)




