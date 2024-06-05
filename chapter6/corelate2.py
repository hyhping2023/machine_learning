import torch
from torch import nn

# 为了方便起见，我们定义了一个计算卷积层的函数。
# 此函数初始化卷积层权重，并对输入和输出提高和缩减相应的维数
def comp_conv2d(conv2d, x):
    # 这里的（1，1）表示批量大小和通道数都是1
    x = x.reshape((1,1)+x.shape)
    y = conv2d(x)
    return y.reshape(y.shape[2:])

conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1)
x = torch.rand(size=(8, 8))
y = comp_conv2d(conv2d, x)
print(y)

conv2d = nn.Conv2d(1, 1, kernel_size=(5, 3), padding=(2, 1))
y = comp_conv2d(conv2d, x)
print(y)

conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=2)
y = comp_conv2d(conv2d, x)
print(y)
