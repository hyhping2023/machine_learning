import torch 
from torch import nn
from torch.nn import functional as f

x = torch.arange(4)
torch.save(x, 'x-file')

x2 = torch.load('x-file')
print(x2)

y = torch.zeros(5)
torch.save([x, y], 'x-file')
x2, y2 = torch.load('x-file')
print(x2, y2)