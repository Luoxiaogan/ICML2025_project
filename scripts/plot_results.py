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
import re

# 读取CSV文件路径
csv_path = '/root/GanLuo/ICML2025_project/outputs/linear_speedup/csv/special_train,_在整个训练集上计算grad_norm_PullDiag_GT_lr=0.5_n=5_bs=128_2025-01-23.csv'

# 从文件名中提取 lr, n, bs 的值
filename = csv_path.split('/')[-1]
lr = re.search(r'lr=([\d.]+)', filename).group(1)
n = re.search(r'n=(\d+)', filename).group(1)
bs = re.search(r'bs=(\d+)', filename).group(1)

# 读取CSV文件
data = pd.read_csv(csv_path)

# 创建画布和子图
fig, ax = plt.subplots(figsize=(8, 6))  # 单个图

# 画图：用log scale画global_gradient_norm(average)
ax.plot(data['iteration'], data['global_gradient_norm(average)'], label='Gradient Norm', color='blue')
ax.set_yscale('log')  # 设置y轴为对数刻度
ax.set_title(f'lr={lr}, n={n}, bs={bs}')  # 设置标题为 lr, n, bs 的值
ax.set_xlabel('Iteration')
ax.set_ylabel('Global Gradient Norm (log scale)')
ax.legend()

# 调整布局
plt.tight_layout()

# 保存图表
output_path = f'/root/GanLuo/ICML2025_project/outputs/linear_speedup/image/从整体训练集_分布式, lr={lr},n={n},bs={bs}.png'
plt.savefig(output_path)

# 显示图表
plt.show()