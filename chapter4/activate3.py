import torch
from d2l import torch as d2l
from torch import nn

net = nn.Sequential(nn.Flatten(),nn.Linear(784, 256),nn.Tanh(),nn.Linear(256, 10))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights)

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

loss = nn.CrossEntropyLoss(reduction='none')
trainer = torch.optim.SGD(net.parameters(), lr=0.1)

num_epochs = 10
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
d2l.predict_ch3(net, test_iter)
d2l.plt.show()