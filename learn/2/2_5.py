import torch

x = torch.arange(4.0)
print(x)
x.requires_grad_(True)  # 等价于x=torch.arange(4.0,requires_grad=True)
print(x.grad)  # 默认值是None
y = 2 * torch.dot(x, x)
#y = 2x^2
y.backward() #开始计算梯度
print(x.grad)

# 在默认情况下，PyTorch会累积梯度，我们需要清除之前的值
x.grad.zero_() #梯度清零
y = x.sum()
y.backward()
print(x.grad)

y = x * x