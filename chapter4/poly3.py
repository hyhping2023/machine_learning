import torch
from torch import nn
from d2l import torch as d2l
from tqdm import tqdm

n_train, n_test, num_imputs, batch_size = 20, 100, 200 ,5
true_w, true_b = torch.ones((num_imputs, 1))*0.01, 0.05
train_data = d2l.synthetic_data(true_w, true_b, n_train)
train_iter = d2l.load_array(train_data, batch_size)
test_data = d2l.synthetic_data(true_w, true_b, n_test)
test_iter = d2l.load_array(test_data, batch_size, is_train=False)

def init_params():
    w = torch.normal(0,1,size=(num_imputs,1),requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    return [w,b]

def train_concise(wd):
    net = nn.Sequential(nn.Linear(num_imputs,1))
    for param in net.parameters():
        param.data.normal_()
    loss = nn.MSELoss(reduction='mean')
    num_epochs, lr = 1000, 0.003
    trainer = torch.optim.SGD([{"params":net[0].weight,'weight_decay':wd}
                              ,{"params":net[0].bias}],lr=lr)
    # trainer_l1 = torch.optim.SGD([{"params":net[0].weight,'l1_weight_decay':wd}
    #                           ,{"params":net[0].bias}],lr=lr)
    animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log',
                            xlim=[5, num_epochs], legend=['train', 'test'])
    for epoch in tqdm(range(num_epochs)):
        for x,y in train_iter:
            trainer.zero_grad()
            l = loss(net(x),y)
            l.backward()
            trainer.step()
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1,
                         (d2l.evaluate_loss(net, train_iter, loss),
                          d2l.evaluate_loss(net, test_iter, loss)))
    print('w的L2范数：', net[0].weight.norm().item())
# train_concise(0)
# d2l.plt.show()

train_concise(1)
d2l.plt.show()

train_concise(3)
d2l.plt.show()

train_concise(10)
d2l.plt.show()