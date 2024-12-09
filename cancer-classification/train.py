import os
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from dataset import CustomDataset
import pandas as pd
from net import CustomNet

# 训练循环
def train(epochs, net_name, loss_csv):
    net = CustomNet(net_name)
    net = net.__net__()

    # 检查是否有GPU并加载模型到设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)

    # 3. 数据预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # ResNet输入固定224x224
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 4. 加载数据集（示例使用ImageFolder）
    train_dataset = CustomDataset('train', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # 5. 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # 前向传播
            outputs = net(images)
            loss = criterion(outputs, labels)
            # 反向传播与优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}')
        dataframe = pd.DataFrame({'epoch': [epoch+1], 'loss': [running_loss/len(train_loader)]})
        if os.path.exists(loss_csv):
            min_loss = running_loss/len(train_loader)
            dataframe.to_csv(loss_csv, mode='a', header=False, index=False)
        else:
            dataframe.to_csv(loss_csv, index=False)
        if min_loss > running_loss/len(train_loader):
            min_loss = running_loss/len(train_loader)
            model_name = net_name+'model_weights_'+str(min_loss)+'.pth'
            torch.save(net.state_dict(), model_name)
    print(net_name+"训练完成！")

if __name__ == '__main__':
    epochs = 100
    # 'resnet18' 'resnet34' 'vgg16' 'efficientnet_v2_m' 'inception3'
    net_list = ['resnet18', 'resnet34', 'vgg16', 'efficientnet_v2_m', 'inception3']
    for net_name in net_list:
        loss_csv = net_name+'loss.csv'
        train(epochs, net_name, loss_csv)