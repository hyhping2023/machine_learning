import torch

x= torch.tensor([1.0,2.0])
y = torch.tensor([2.0,3.0])

print(x+y, x-y, x*y, x/y, x**y)

x = torch.arange(4)
print(x)

x= torch.randn(4)
print(x)

print(x[3])

x = torch.arange(12).reshape(3,4)

print(x,len(x),len(x[1]),x.shape)

A = torch.arange(20).reshape(5,4)

print(A,A.T)

B = torch.randn(5,4)

print(torch.arange(5).sum())
print(A)
print(A.sum(axis=1),A.sum(axis=0),A.sum())

print(B,B.mean())

print(A*B)
C = A*B
print(C.sum(axis=0))

A = torch.arange(5)
B = torch.arange(5)
print(A,B,(A*B).sum(),torch.dot(A,B),)

A = torch.arange(20).reshape(4,5)
B = torch.arange(5)
print(A,B,torch.mv(A,B))

A = torch.randn((5,4))
B = torch.randn((4,5))
print(A,B,torch.mm(A,B))
print(torch.norm(A)) # Frobenius norm 或者范数