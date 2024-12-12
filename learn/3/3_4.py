import torch
from torch import nn
from d2l import torch as d2l

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

def train(num_epochs, net, train_iter, loss, trainer):
    loss_sum = 0
    for i in range(num_epochs):
        for x, y in train_iter:
            l = loss(net(x), y)
            trainer.zero_grad()
            l.backward()
            trainer.step()

def test():


if __name__ == '__main__':
    batch_size = 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

    # PyTorch不会隐式地调整输入的形状。因此，
    # 我们在线性层前定义了展平层（flatten），来调整网络输入的形状
    net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))

    net.apply(init_weights);
    # apply 方法：通过 net.apply(init_weights)，每一层的权重和偏置都会被 init_weights 函数初始化。
    # apply 会遍历网络的所有层，并对每一层的参数应用这个初始化方法。

    loss = nn.CrossEntropyLoss(reduction='mean')  # 计算每个样本的交叉熵损失，而不是求整个批次的平均损失。
    trainer = torch.optim.SGD(net.parameters(), lr=0.1)
    num_epochs = 10

    train(num_epochs, net, train_iter, loss, trainer)