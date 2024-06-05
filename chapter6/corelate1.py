import torch
from torch import nn
from d2l import torch as d2l


def corr2d(x, k):
    h, w = k.shape
    y = torch.zeros(x.shape[0] - h + 1, x.shape[1] - w + 1)
    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            y[i, j] = (x[i:i+h, j:j+w]*k).sum()
    return y

X = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
K = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
print(corr2d(X, K))
print(X[0:2][0:1])
print(X[0:2, 0:2])

class Conv2D(nn.Module):
    def __init__(self, kernel_size) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias
    
x = torch.ones((6,8))
x[:, 2:6] = 0
print(x)

k = torch.tensor([[1.0, -1.0]])
y = corr2d(x, k)
print(y)
print(corr2d(x.t(), k))

conv2d = nn.Conv2d(1,1, kernel_size=(1,2), bias=False)

x = x.reshape((1,1,6,8))
y = y.reshape((1,1,6,7))
lr = 3e-2

for i in range(20):
    y_hat = conv2d(x)
    l = (y_hat - y)**2
    conv2d.zero_grad()
    l.sum().backward()
    conv2d.weight.data[:] -= lr * conv2d.weight.grad
    if (i + 1) % 2 == 0:
        print(f'epoch{i+1}: loss = {l.sum():.10f}')

print(conv2d.weight.data.reshape(1,2))