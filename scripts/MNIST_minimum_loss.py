import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import MultiStepLR
import pandas as pd
import os
import torch.nn.functional as F

# 设置随机种子以保证可重复性
torch.manual_seed(42)

# 检查是否有GPU可用，并设置device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 超参数设置
input_size = 784       # MNIST图像大小 (28x28)
hidden_size = 2048     # 隐藏层大小
num_classes = 10       # 类别数 (0-9)
num_epochs = 1000       # 训练轮数
batch_size = 128      # 批量大小
initial_lr = 1e-1       # 初始学习率
milestones = [50, 100, 150]  # 学习率衰减的epoch
gamma = 0.1            # 学习率衰减因子

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),                     # 转换为Tensor
    transforms.Normalize((0.1307,), (0.3081,)) # MNIST数据集的均值和标准差
])

# 加载MNIST数据集
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=12, pin_memory=True)

# 定义全连接网络
class SimpleFCN(nn.Module):
    def __init__(self):
        super(SimpleFCN, self).__init__()
        # 扩大隐藏层宽度（显著增加模型容量）
        self.fc1 = nn.Linear(784, 1024)          # 第一层：784 → 1024
        self.fc2 = nn.Linear(1024, 1024)         # 第二层：1024 → 1024
        self.fc3 = nn.Linear(1024, 10)           # 输出层：1024 → 10
        
        # He初始化（适配ReLU激活函数）
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc3.weight, nonlinearity='linear')  # 输出层用线性初始化

    def forward(self, x):
        x = x.view(x.size(0), -1)                # 展平为一维向量
        x = F.relu(self.fc1(x))                  # 第一层 + ReLU
        x = F.relu(self.fc2(x))                  # 第二层 + ReLU
        x = self.fc3(x)                          # 输出层（无激活函数）
        return x

# 初始化模型、损失函数和优化器
model = SimpleFCN().to(device)  # 将模型移动到GPU
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=initial_lr)
scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

# 创建保存loss的目录（如果不存在）
output_dir = '/root/GanLuo/ICML2025_project/outputs/linear_speedup'
os.makedirs(output_dir, exist_ok=True)

def compute_normalized_global_gradient_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.detach().norm(2).item()  # 获取梯度的L2范数
            total_norm += param_norm ** 2
    total_norm = total_norm ** 0.5  # 计算全局梯度范数
    return total_norm  # 返回未归一化的总梯度范数

# 初始化loss列表
loss_list = []
grad_list = []

# 训练循环
# 训练循环
for epoch in range(num_epochs):
    model.train()  # 设置模型为训练模式
    running_loss = 0.0
    
    for i, (images, labels) in enumerate(train_loader):
        # 将图像展平为向量 (batch_size, 784)，并移动到GPU
        images = images.reshape(-1, input_size).to(device)
        labels = labels.to(device)
        
        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # 反向传播
        optimizer.zero_grad()  # 清除历史梯度
        loss.backward()        # 计算梯度
        
        # 计算当前batch的梯度范数
        grad_norm = compute_normalized_global_gradient_norm(model)
        
        # 优化器更新参数
        optimizer.step()
        
        # 记录损失和梯度范数
        running_loss += loss.item()
    
    # 计算当前epoch的平均损失和梯度范数
    epoch_loss = running_loss / len(train_loader)
    epoch_grad_norm = grad_norm  # 使用最后一个batch的梯度范数（或可以改为累积平均）
    
    loss_list.append(epoch_loss)
    grad_list.append(epoch_grad_norm)
    
    # 打印当前epoch的损失、梯度范数和学习率
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.6f}, Grad_norm: {epoch_grad_norm}")

    # 将loss_list和grad_list合并存储为DataFrame
    combined_df = pd.DataFrame({
        "Epoch": list(range(1, epoch + 2)),
        "Loss": loss_list,
        "Grad_norm": grad_list
    })
    combined_df.to_csv(os.path.join(output_dir, 'lr=1e-1,batch_size = 128,单机训练的最小loss和grad_norm.csv'), index=False)