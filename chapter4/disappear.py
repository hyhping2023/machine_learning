import torch
from d2l import torch as d2l

x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
y =torch.sigmoid(x)
#y.backward(torch.ones_like(x))
#x.grad.data.zero_()
y.sum().backward()

d2l.plot(x.detach().numpy(), [y.detach().numpy(), x.grad.numpy()],
         legend=['sigmoid', 'gradient'], figsize=(15,12) )
#d2l.plt.show()

m = torch.normal(0, 1 , size=(4,4))
print('一个矩阵：\n',m)
for i in range(100):
    m = torch.mm(m,torch.normal(0,1,size=(4,4)))
print('第100次矩阵：\n',m)