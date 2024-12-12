import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l
# nn是神经网络的缩写
from torch import nn

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)

def load_array(data_arrays, batch_size, is_train=True):  #@save
    """构造一个PyTorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

batch_size = 10
data_iter = load_array((features, labels), batch_size)

# print(next(iter(data_iter)))

net = nn.Sequential(nn.Linear(2, 1))

loss = nn.MSELoss()

trainer = torch.optim.SGD(net.parameters(), lr=0.03)

num_epochs = 4
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X) ,y)
        trainer.zero_grad() #每次计算梯度之前，需要将上一轮的梯度清零（因为 PyTorch 会累加梯度）。
        l.backward() #计算损失函数的偏导数
        trainer.step() #更新模型的参数
        #trainer（优化器）主要有两个作用:
        # 管理模型的参数和梯度：它知道哪些参数需要更新，并存储对应的梯度。
        # 更新模型的参数.
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')