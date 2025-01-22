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
initial_lr = 5e-1       # 初始学习率
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

import copy
import torch
from torch.utils.data import DataLoader

def compute_full_gradient_norm(model, train_dataset, criterion, batch_size=None, device="cuda"):
    """
    计算模型在整个训练集上的梯度F范数（所有参数梯度元素的平方和开根号）
    
    参数:
        model: 原始模型（不会被修改）
        train_dataset: 训练集数据集
        criterion: 损失函数
        batch_size: 如果为None则使用全量数据，否则使用指定batch_size（内存不足时使用）
        device: 计算设备
    
    返回:
        float: 全局梯度范数（Frobenius范数）
    """
    # 深拷贝模型以避免影响原始模型
    model_copy = copy.deepcopy(model).to(device)
    model_copy.train()
    
    # 自动确定batch_size（全量数据或用户指定）
    use_full_batch = batch_size is None
    effective_bs = len(train_dataset) if use_full_batch else batch_size
    
    # 创建数据加载器（全量数据时关闭shuffle和drop_last）
    loader = DataLoader(train_dataset,
                        batch_size=effective_bs,
                        shuffle=False,
                        drop_last=False,
                        num_workers=12,
                        pin_memory=True)
    
    # 初始化梯度缓冲区
    for param in model_copy.parameters():
        param.grad = torch.zeros_like(param.data)
    
    total_samples = 0  # 已处理样本计数
    for inputs, labels in loader:
        # 将数据转移到设备
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        # 展平图像（适配全连接网络）
        batch_size = inputs.shape[0]
        inputs = inputs.view(batch_size, -1)
        
        # 前向传播
        outputs = model_copy(inputs)
        loss = criterion(outputs, labels)
        
        # 反向传播（计算梯度并累加）
        loss.backward()
        
        # 记录已处理样本数
        total_samples += batch_size
    
    # 验证数据完整性
    assert total_samples == len(train_dataset), "数据存在缺失"
    
    # 计算全局梯度范数（所有参数梯度的平方和开根号）
    total_norm_sq = 0.0
    for param in model_copy.parameters():
        if param.grad is not None:
            grad = param.grad.data
            total_norm_sq += torch.sum(grad ** 2).item()
    
    return total_norm_sq ** 0.5

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
        #grad_norm = compute_normalized_global_gradient_norm(model)
        
        # 优化器更新参数
        optimizer.step()
        
        # 记录损失和梯度范数
        running_loss += loss.item()
    
    # 计算当前epoch的平均损失和梯度范数
    epoch_loss = running_loss / len(train_loader)
    epoch_grad_norm = compute_full_gradient_norm(model, train_dataset, criterion, batch_size=None, device=device)
    
    loss_list.append(epoch_loss)
    grad_list.append(epoch_grad_norm)
    
    # 打印当前epoch的损失、梯度范数和学习率
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.6f}, Grad_norm: {epoch_grad_norm:6f}")

    # 将loss_list和grad_list合并存储为DataFrame
    combined_df = pd.DataFrame({
        "Epoch": list(range(1, epoch + 2)),
        "Loss": loss_list,
        "Grad_norm": grad_list
    })
    combined_df.to_csv(os.path.join(output_dir, f'/root/GanLuo/ICML2025_project/outputs/linear_speedup/单一节点训练_计算grad需要在整个trainset上采样/lr={initial_lr}, bs={batch_size}.csv'), index=False)