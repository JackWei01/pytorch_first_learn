# Non-linear Activations
import torch
from torch import nn

# RELU

# y=x  while x>0
# y = 0 while x<=0

m = nn.ReLU(inplace=True)
m = nn.ReLU(inplace=False) #是否用激活函数的输出，替换输入的x



input = torch.randn(2)

output = m.forward(input) #任意维度的输入，输出与输入同一纬度

print(input)
print(output)


# SIGMOID
m = nn.Sigmoid()
input = torch.randn(2)
output = m.forward(input)

print(input)
print(output)



