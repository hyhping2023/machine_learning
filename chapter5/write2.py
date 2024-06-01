import torch 
from torch import nn
from torch.nn import functional as F

class MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.output = nn.Linear(256, 10)
    
    def forward(self, x):
        return self.output(F.relu(self.hidden(x)))
    
net = MLP()
x = torch.randn(size=(2, 20))
y = net(x)

print(y.data)

torch.save(net.state_dict(), 'mip.params')

clone = MLP()
clone.load_state_dict(torch.load('mip.params'))
clone.eval()
print(clone)

y_clone = clone(x)
print(y_clone == y)