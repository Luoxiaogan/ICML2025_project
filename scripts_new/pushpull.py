import sys
import os
import subprocess

# 将项目根目录添加到 sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 从原始脚本导入必要的函数和变量
from training import train
from utils import ring1, show_row
from network_utils import get_matrixs_from_exp_graph
import numpy as np
import torch

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # 设置学习率列表
# lr_list = [3e-2, 2e-2, 9e-3, 8e-3]

# # 准备矩阵数据，和原脚本保持一致
# n = 2
# A, B = get_matrixs_from_exp_graph(n=n, seed=48)
# show_row(A)
# print(A.shape)

# # 方法1: 直接在当前进程中循环调用train函数
# print("开始使用不同学习率训练模型...")
# for lr in lr_list:
#     print(f"使用学习率: {lr}")
#     train(
#         algorithm="PushPull",
#         lr=lr,
#         A=A,
#         B=B,
#         dataset_name="MNIST",
#         batch_size=128,
#         num_epochs=200,
#         remark=f"Exp_test",  # 自动标记不同学习率的实验
#     )
#     print(f"学习率 {lr} 的训练已完成")

# print("所有学习率的训练均已完成!")






# 设置学习率列表
lr_list = [3e-2, 2e-2, 9e-3, 8e-3, 7e-3, 6e-3, 5e-3, 4e-3]

# 准备矩阵数据，和原脚本保持一致
n = 4
A, B = get_matrixs_from_exp_graph(n=n, seed=48)
show_row(A)
print(A.shape)

# 方法1: 直接在当前进程中循环调用train函数
print("开始使用不同学习率训练模型...")
for lr in lr_list:
    print(f"使用学习率: {lr}")
    train(
        algorithm="PushPull",
        lr=lr,
        A=A,
        B=B,
        dataset_name="MNIST",
        batch_size=128,
        num_epochs=200,
        remark=f"Exp_test",  # 自动标记不同学习率的实验
    )
    print(f"学习率 {lr} 的训练已完成")

print("所有学习率的训练均已完成!")













# 设置学习率列表
lr_list = [3e-2, 2e-2, 9e-3, 8e-3, 7e-3, 6e-3, 5e-3, 4e-3]

# 准备矩阵数据，和原脚本保持一致
n = 8
A, B = get_matrixs_from_exp_graph(n=n, seed=48)
show_row(A)
print(A.shape)

# 方法1: 直接在当前进程中循环调用train函数
print("开始使用不同学习率训练模型...")
for lr in lr_list:
    print(f"使用学习率: {lr}")
    train(
        algorithm="PushPull",
        lr=lr,
        A=A,
        B=B,
        dataset_name="MNIST",
        batch_size=128,
        num_epochs=200,
        remark=f"Exp_test",  # 自动标记不同学习率的实验
    )
    print(f"学习率 {lr} 的训练已完成")

print("所有学习率的训练均已完成!")










# 方法2: 子进程方式（可选，通过注释/取消注释使用）
'''
print("开始使用子进程方式训练模型...")
script_path = os.path.join(current_dir, 'run_training_pushpull.py')

for lr in lr_list:
    print(f"使用学习率: {lr}")
    # 准备修改的脚本内容
    with open(script_path, 'r') as f:
        script_content = f.read()
    
    # 修改学习率并保存到临时脚本
    temp_script = script_content.replace('lr=1e-2', f'lr={lr}')
    temp_script = temp_script.replace('remark=f"Exp_test"', f'remark=f"Exp_lr_{lr}"')
    
    temp_script_path = os.path.join(current_dir, f'temp_training_lr_{lr}.py')
    with open(temp_script_path, 'w') as f:
        f.write(temp_script)
    
    # 执行修改后的脚本
    subprocess.run([sys.executable, temp_script_path])
    
    # 清理临时脚本
    os.remove(temp_script_path)
    print(f"学习率 {lr} 的训练已完成")

print("所有学习率的训练均已完成!")
'''