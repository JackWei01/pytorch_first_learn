import torch
import torch.nn as nn
import torch.nn.functional as F

# 创建输入张量，5x5的矩阵
input_tensor = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0],
                             [6.0, 7.0, 8.0, 9.0, 10.0],
                             [11.0, 12.0, 13.0, 14.0, 15.0],
                             [16.0, 17.0, 18.0, 19.0, 20.0],
                             [21.0, 22.0, 23.0, 24.0, 25.0]], requires_grad=True)

# 创建3x3的高斯卷积核
gaussian_kernel = torch.tensor([[1, 2, 1],
                                [2, 4, 2],
                                [1, 2, 1]], dtype=torch.float32)


print(input_tensor.shape) #torch.Size([5, 5])  只有两个数字，不满足 nn.Conv2d 的shape 4个数字的要求  #iput size (N,C,H,W)
print(gaussian_kernel.shape) #torch.Size([5, 5])

# 将输入张量和卷积核转换为合适的形状
# input_tensor = input_tensor.view(1, 1, 5, 5)  # (batch_size, channels, height, width)
# gaussian_kernel = gaussian_kernel.view(1, 1, 3, 3)  # (out_channels, in_channels, kernel_height, kernel_width)
input_tensor = torch.reshape(input_tensor,(1,1,5,5))
gaussian_kernel = torch.reshape(gaussian_kernel,(1,1,3,3))

print(input_tensor.shape) #torch.Size([1, 1, 5, 5])  #iput size (N,C,H,W)
print(gaussian_kernel.shape) #torch.Size([1, 1, 5, 5])



# torch.nn.functional.conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1)
'''


'''


output = F.conv2d(input_tensor,gaussian_kernel,stride=1)
output = F.conv2d(input_tensor,gaussian_kernel,stride=2)
print(output)

'''
# 创建卷积层
conv_layer = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0, bias=False)

# 将卷积核的权重设置为高斯卷积核
with torch.no_grad():  #todo:高级语法
    conv_layer.weight[0] = nn.Parameter(gaussian_kernel)

# 执行卷积操作
output = conv_layer(input_tensor)

# 输出结果
print("原始输入:")
print(input_tensor)
print("\n卷积核:")
print(gaussian_kernel)
print("\n卷积结果:")
print(output)
'''

