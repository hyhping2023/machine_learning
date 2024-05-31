from normal import normal
from timer import Timer
import numpy as np
import torch
from d2l import torch as d2l

n = 10000
a = torch.ones([n])
b = torch.ones([n])

c = torch.zeros([n])
timer = Timer()
for i in range(n):
    c[i]=a[i]+b[i]

print(f'for循环:{timer.stop():.5f} sec')

timer = Timer()
d = a + b
print(f'重载的+运算符来计算按元素的和:{timer.stop():.5f} sec')
print("------------------------------------------------------")

'''
----------------------------------------------------------------------
'''

# 生成数据
x = np.arange(-7,7,0.01)

# 生成模型
params = [(0,1),(0,2),(3,1)]
d2l.plot(x, [normal(x, mu, sigma) for mu, sigma in params], xlabel='x', ylabel='p(x)', legend=[f'mean {mu}, std {sigma}' for mu, sigma in params], figsize=(4.5, 2.5))
d2l.plt.show()

