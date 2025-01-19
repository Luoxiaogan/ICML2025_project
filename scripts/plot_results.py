import pandas as pd
import matplotlib.pyplot as plt
import os

# 定义文件路径
file_paths = [
    "/root/GanLuo/ICML2025_project/outputs/Multi_Gossip_test/csv/MG=1, ring1, PullDiag_GT, lr=0.008, n_nodes=15, batch_size=128, 2025-01-19.csv",
    "/root/GanLuo/ICML2025_project/outputs/Multi_Gossip_test/csv/MG=2, ring1, PullDiag_GT, lr=0.016, n_nodes=15, batch_size=256, 2025-01-19.csv",
    "/root/GanLuo/ICML2025_project/outputs/Multi_Gossip_test/csv/MG=5, ring1, PullDiag_GT, lr=0.04, n_nodes=15, batch_size=640, 2025-01-19.csv",
    "/root/GanLuo/ICML2025_project/outputs/Multi_Gossip_test/csv/MG=5, ring1, PullDiag_GT, lr=0.05, n_nodes=15, batch_size=640, 2025-01-19.csv"
]

# 定义要比较的列
col = 'test_accuracy(average)'

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
plt.ylabel('test_accuracy(average)')
plt.legend()

# 保存图像到指定路径
output_image_path = f"/root/GanLuo/ICML2025_project/outputs/Multi_Gossip_test/image/MG_com,{col}_comparison.png"
os.makedirs(os.path.dirname(output_image_path), exist_ok=True)  # 确保目录存在
plt.savefig(output_image_path)

# 显示图像
plt.show()