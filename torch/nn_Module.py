import torch
import torch.nn as nn
import torch.nn.functional as F


## 定义一个简单的模型？还是叫简单的神经网络
class Tudui(nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, input):
        output = input + 1
        return output


x = torch.tensor(1.0)
print(x)
tudui = Tudui()
# output  = tudui.forward(x) #或者定义————__cal__魔法方法
output = tudui(x)


print(output)







class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5) #卷积
        self.conv2 = nn.Conv2d(20, 20, 5)

    def forward(self, x):
        x = F.relu(self.conv1(x))  #F.relu 冲激函数
        return F.relu(self.conv2(x))