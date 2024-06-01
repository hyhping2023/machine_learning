import torch 
from torch import nn
from torch.nn import functional as F

class MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.hidden = nn.Linear(20, 256)# 隐藏层
        self.out = nn.Linear(256,10)

    def forward(self, x):
        return self.out(F.relu(self.hidden(x)))

x = torch.rand(2,20)

print("Origin:",x)
net = MLP()
print("MLP: ",net(x))

class MySequential(nn.Module):
    def __init__(self,*args) -> None:
        super().__init__()
        for idx, module in enumerate(args):
            # 这里，module是Module子类的一个实例。我们把它保存在'Module'类的成员
            # 变量_modules中。_module的类型是OrderedDict
            self._modules[str(idx)] = module

    def forward(self, x):
        for block in self._modules.values():
            x = block(x)
        return x

net = MySequential(nn.Linear(20,256), nn.ReLU(), nn.Linear(256,10))
print("MySequential: ",net(x))


class FixedHiddenMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.rand_weight = torch.rand((20,20),requires_grad=False)
        self.linear = nn.Linear(20,20)

    def forward(self,x):
        x = self.linear(x)
        x = F.relu(torch.mm(x,self.rand_weight)+1)
        x = self.linear(x)
        while x.abs().sum()>1:
            x/=2
        return x.sum()
net = FixedHiddenMLP()
print("FixedHidden: ",net(x))

class NestMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(20,64),nn.ReLU(),
                                 nn.Linear(64,32),nn.ReLU())
        self.linear = nn.Linear(32,16)
    
    def forward(self, x):
        return self.linear(self.net(x))
    
chimera = nn.Sequential(NestMLP() ,nn.Linear(16,20) ,FixedHiddenMLP())
print("NestMLP",chimera(x))