import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from dataset import CustomDataset

# 1. 加载预训练的ResNet18模型
resnet = models.resnet18(pretrained=False)

# 2. 修改最后的全连接层为二分类
num_classes = 2  # 二分类
resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)

# 检查是否有GPU并加载模型到设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet.to(device)

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
optimizer = torch.optim.Adam(resnet.parameters(), lr=0.001)

# 6. 训练循环
epochs = 10
for epoch in range(epochs):
    resnet.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # 前向传播
        outputs = resnet(images)
        loss = criterion(outputs, labels)
        # 反向传播与优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    print(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}')
    torch.save(resnet.state_dict(), 'model_weights.pth')
print("训练完成！")