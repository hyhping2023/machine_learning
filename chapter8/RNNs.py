import torch
from d2l import torch as d2l

x, W_xh = torch.normal(0, 1, (3, 1)), torch.normal(0, 1, (1, 4))
h, W_hh = torch.zeros(3, 4), torch.normal(0, 1, (4, 4))
print(torch.matmul(x, W_xh) + torch.matmul(h, W_hh))