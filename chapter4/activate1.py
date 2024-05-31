import torch
from d2l import torch as d2l

x = torch.arange(-8.0,8.0,0.1,requires_grad=True)
y = torch.relu(x)
d2l.plot(x.detach(),y.detach(),'x','relu(x)',figsize=(5,2.5))

y.backward(torch.ones_like(x),retain_graph=True)
d2l.plot(x.detach(),x.grad,'x','grad of relu',figsize=(5,2.5))

#挤压函数
y = torch.sigmoid(x)
d2l.plot(x.detach(),y.detach(),'x','sigmoid(x)',figsize=(5,2.5))

y = torch.tanh(x)
d2l.plot(x.detach(),y.detach(),'x','tanh(x)',figsize=(5,2.5))
x.grad.data.zero_()
#y.backward(torch.ones_like(x),retain_graph=True)
y.sum().backward()
d2l.plot(x.detach(),x.grad,'x','grad of tanh',figsize=(5,2.5))

#pRelu
alpha = torch.tensor(-0.2)
y = torch.prelu(x,alpha)
d2l.plot(x.detach(),y.detach(),'x','prelu(alpha=0.2)(x)',figsize=(5,2.5))
d2l.plt.show()
x.grad.data.zero_()
y.backward(torch.ones_like(x))
#表示创建了一个与loss同样大小的全1张量，另loss与torch.ones_like(loss)点乘后再进行反向传播，
#也相当于对loss求和后再进行反向传播。
d2l.plot(x.detach(),x.grad,'x','grad of prelu',figsize=(5,2.5))

d2l.plt.show()