import torch

x = torch.arange(12)
print(x)

x = x.reshape(3,4)
print(x)

x = torch.zeros((2,3,4)) 
print(x)

x = torch.ones(3,4)
print(x)

x = torch.randn(4,5)
print(x)

x = torch.tensor([1,2,4,8])
y = torch.tensor([2,2,2,2])

print(x+y, x-y, x*y, x/y, x**y, torch.exp(x))

print(x == y)

x = torch.tensor([[1,2,3,4],
                 [2,3,4,5],
                 [3,4,5,6]])
y = torch.tensor([[2,2,2,2],
                 [2,2,2,2],
                 [2,2,2,2]])

print(torch.cat((x,y),dim = 0), torch.cat((x,y),dim = 1))

a = torch.arange(6).reshape(3,2)
b = torch.arange(2).reshape(1,2)
print(a+b) # broadcasting

a[0:1,:] = 114514
print(a)
print(list(a))