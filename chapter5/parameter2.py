import torch
from torch import nn

net = nn.Sequential(nn.Linear(4,8), 
                    nn.ReLU(), nn.Linear(8,1))

def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.normal(m.weight, mean=0, std =0.01)
        nn.init.zeros_(m.bias)

def init_constant(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 1)
        nn.init.zeros_(m.bias)

net.apply(init_normal)
print(net[0].weight.data, net[0].bias.data)

net.apply(init_constant)
print(net[0].weight.data, net[0].bias.data)

def init_xavier(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
def init_42(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 42)

net[0].apply(init_xavier)
net[2].apply(init_42)
print(net[0].weight.data[0])
print(net[2].weight.data)

#绑定参数
shared = nn.Linear(8,8)

net = nn.Sequential(nn.Linear(4,8),nn.ReLU(), shared, nn.ReLU(), shared, 
                    nn.ReLU(), nn.Linear(8,1))

# 检查参数是否相同
print(net[2].weight.data[0] == net[4].weight.data[0])
net[2].weight.data[0, 0] = 100
# 确保它们实际上是同一个对象，而不只是有相同的值
print(net[2].weight.data[0] == net[4].weight.data[0])