import torch
from torch import nn

net = nn.Sequential(nn.Linear(4, 8),nn.ReLU(), nn.Linear(8, 1))
x = torch.rand(size=(2,4))
print(net(x))
print(net[2].state_dict())
print(net[2].bias,net[2].bias.data)
print(net[2].bias.grad == None)
print(*[(name, param.shape) for name, param in net[0].named_parameters()])
print(*[(name, param.shape) for name, param in net.named_parameters()])
print(net.state_dict()['2.bias'].data)

def block1():
    return nn.Sequential(nn.Linear(4,8),nn.ReLU(),
                         nn.Linear(8,4),nn.ReLU())

def block2():
    net = nn.Sequential()
    for i in range(4):
        net.add_module(f"block{i}",block1())
    return net

rgnet = nn.Sequential(block2(), nn.Linear(4,1))
print(rgnet(x))
print(rgnet)
print(rgnet[0][1][0].bias.data)


