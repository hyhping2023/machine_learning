import torch
from torch import nn
import d2l

print(torch.device('cpu'),torch.cuda.device_count()
      , torch.device('cuda'))

x = torch.ones(2, 3, device='cuda:0')
print(x.device)
# print(torch.__version__)
# print(torch.cuda.is_available())

cuda = 'cuda:0'

net = nn.Sequential(nn.Linear(3, 1))
net = net.to(device=cuda)

print(net(x))


