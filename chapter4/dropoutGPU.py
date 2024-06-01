import torch
from torch import nn
from d2l import torch as d2l
from tqdm import tqdm
cuda = 'cuda:0'
dropout1, dropout2, lr= 0.5, 0.2, 0.5
num_epochs, lr, batch_size, wd= 50, 0.5, 256, 0

def toCuda(items):
    for item in items:
        item.to(device=cuda)

net = nn.Sequential(nn.Flatten(),
                    nn.Linear(784,256),
                    nn.ReLU(),
                    nn.Dropout(dropout1),
                    nn.Linear(256,256),
                    nn.ReLU(),
                    nn.Dropout(dropout2),
                    nn.Linear(256,10)
                    )
#net = nn.Sequential(nn.Flatten(), nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights)

loss = nn.CrossEntropyLoss(reduction='none')
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
#trainer = torch.optim.SGD(net.parameters(),lr=lr)
trainer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=wd)


d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer, cuda)
d2l.predict_ch3(net, test_iter,15)
d2l.plt.show()