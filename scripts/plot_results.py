""" import pandas as pd
import matplotlib.pyplot as plt
import os

# 定义文件路径
file_paths = [
    "/root/GanLuo/ICML2025_project/outputs/linear_speedup/test_for_best_lr/ring, PullDiag_GT, lr=0.003, n_nodes=40, batch_size=500, 2025-01-21.csv",
    "/root/GanLuo/ICML2025_project/outputs/linear_speedup/test_for_best_lr/ring, PullDiag_GT, lr=0.01, n_nodes=40, batch_size=500, 2025-01-21.csv"
]

# 定义要比较的列
col = 'train_accuracy(average)'
#可选：'train_loss(average)', 'train_accuracy(average)', 'test_loss(average)', 'test_accuracy(average)'

# 创建一个空的DataFrame列表，用于存储每个文件的数据
dataframes = []

# 读取每个CSV文件并存储到DataFrame列表中
for file_path in file_paths:
    df = pd.read_csv(file_path)
    dataframes.append(df)

# 创建一个新的图形
plt.figure(figsize=(10, 6))

# 绘制每个文件的train_loss(average)曲线
for i, df in enumerate(dataframes):
    lr = file_paths[i].split('lr=')[1].split(',')[0]  # 从文件名中提取学习率
    plt.plot(df['epoch'], df[col], label=f'lr={lr}')

# 添加图例、标题和标签
plt.title('Comparison')
plt.xlabel('Epoch')
plt.ylabel(f'{col}')
plt.legend()

# 保存图像到指定路径
output_image_path = f"/root/GanLuo/ICML2025_project/outputs/linear_speedup/image/当n=40时,最优的lr是什么?比较{col}.png"
os.makedirs(os.path.dirname(output_image_path), exist_ok=True)  # 确保目录存在
plt.savefig(output_image_path)

# 显示图像
plt.show() """

import pandas as pd
import matplotlib.pyplot as plt

# 读取CSV文件
csv_path = '/root/GanLuo/ICML2025_project/outputs/linear_speedup/单一节点训练_计算grad需要在整个trainset上采样/lr=0.1, bs=128.csv'
data = pd.read_csv(csv_path)

# 创建画布和子图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))  # 1行2列的子图布局

# 左边的图：用log scale画Loss
ax1.plot(data['Epoch'], data['Loss'], label='Loss', color='blue')
ax1.set_yscale('log')  # 设置y轴为对数刻度
ax1.set_title('Loss, lr=0.1, bs=128, n=1')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss (log scale)')
ax1.legend()

# 右边的图：用log scale画Grad_norm
ax2.plot(data['Epoch'], data['Grad_norm'], label='Grad_norm', color='red')
ax2.set_yscale('log')  # 设置y轴为对数刻度
ax2.set_title('Grad_norm, lr=0.1, bs=128, n=1')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Grad_norm (log scale)')
ax2.legend()

# 调整布局
plt.tight_layout()

# 保存图表
output_path = '/root/GanLuo/ICML2025_project/outputs/linear_speedup/image/在整体的训练集上采样计算梯度, lr=0.1,bs=128.png'
plt.savefig(output_path)

# 显示图表
plt.show()












# 读取CSV文件
csv_path = '/root/GanLuo/ICML2025_project/outputs/linear_speedup/单一节点训练_计算grad需要在整个trainset上采样/lr=0.1, bs=512.csv'
data = pd.read_csv(csv_path)

# 创建画布和子图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))  # 1行2列的子图布局

# 左边的图：用log scale画Loss
ax1.plot(data['Epoch'], data['Loss'], label='Loss', color='blue')
ax1.set_yscale('log')  # 设置y轴为对数刻度
ax1.set_title('Loss, lr=0.1, bs=512, n=1')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss (log scale)')
ax1.legend()

# 右边的图：用log scale画Grad_norm
ax2.plot(data['Epoch'], data['Grad_norm'], label='Grad_norm', color='red')
ax2.set_yscale('log')  # 设置y轴为对数刻度
ax2.set_title('Grad_norm,lr=0.1, bs=512, n=1')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Grad_norm (log scale)')
ax2.legend()

# 调整布局
plt.tight_layout()

# 保存图表
output_path = '/root/GanLuo/ICML2025_project/outputs/linear_speedup/image/在整体的训练集上采样计算梯度, lr=0.1,bs=512.png'
plt.savefig(output_path)

# 显示图表
plt.show()



















# 读取CSV文件
csv_path = '/root/GanLuo/ICML2025_project/outputs/linear_speedup/单一节点训练_计算grad需要在整个trainset上采样/lr=0.1, bs=512.csv'
data = pd.read_csv(csv_path)

# 创建画布和子图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))  # 1行2列的子图布局

# 左边的图：用log scale画Loss
ax1.plot(data['Epoch'], data['Loss'], label='Loss', color='blue')
ax1.set_yscale('log')  # 设置y轴为对数刻度
ax1.set_title('Loss, lr=0.1, bs=512, n=1')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss (log scale)')
ax1.legend()

# 右边的图：用log scale画Grad_norm
ax2.plot(data['Epoch'], data['Grad_norm'], label='Grad_norm', color='red')
ax2.set_yscale('log')  # 设置y轴为对数刻度
ax2.set_title('Grad_norm,lr=0.1, bs=512, n=1')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Grad_norm (log scale)')
ax2.legend()

# 调整布局
plt.tight_layout()

# 保存图表
output_path = '/root/GanLuo/ICML2025_project/outputs/linear_speedup/image/在整体的训练集上采样计算梯度, lr=0.1,bs=512.png'
plt.savefig(output_path)

# 显示图表
plt.show()