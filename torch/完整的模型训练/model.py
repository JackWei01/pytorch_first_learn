import torch
from torch import nn
from torch.nn import Sequential


class Image2ten(nn.Module):
    def __init__(self):
        super().__init__()
        self.module = Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),  # padding
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.Linear(64, 10)

        )

    def forward(self, input):
        ouput = self.module(input)
        return ouput

if __name__ == '__main__':

    my_module = Image2ten()

    input = torch.ones(64,3,32,32)
    output = my_module(input)

    print(output)
    print(output.shape)