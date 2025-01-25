# scripts/run_training.py

import sys
import os

# 将项目根目录添加到 sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import sys
import os
import numpy as np
import pandas as pd

from training import train, train_per_iteration
import torch
from utils import ring1, show_row 

import numpy as np

def Row(matrix):
    """行归一化函数，确保每行和为1"""
    M = matrix.copy()
    row_sums = np.sum(M, axis=1)
    for i in range(M.shape[0]):
        M[i, :] /= row_sums[i]
    return M

def grid_16(seed=None):
    """
    生成4x4网格图的行随机矩阵，包含自环和随机权重。
    参数:
        seed (int): 控制随机权重生成的种子
    返回:
        A (np.ndarray): 行随机矩阵，shape=(16,16)
    """
    np.random.seed(seed)
    n = 16
    A = np.zeros((n, n))
    
    # 生成4x4网格的拓扑结构（含自环）
    for row in range(4):
        for col in range(4):
            node = row * 4 + col  # 当前节点编号（0~15）
            
            # 添加自环权重（1~10）
            A[node, node] = np.random.randint(1, 11)
            
            # 横向连接（右邻居）
            if col < 3:
                right_neighbor = node + 1
                A[node, right_neighbor] = np.random.randint(1, 11)
                A[right_neighbor, node] = np.random.randint(1, 11)  # 反向连接
            
            # 纵向连接（下邻居）
            if row < 3:
                down_neighbor = node + 4
                A[node, down_neighbor] = np.random.randint(1, 11)
                A[down_neighbor, node] = np.random.randint(1, 11)  # 反向连接
    
    # 行归一化
    A = Row(A)
    return A

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n=16
A= grid_16(seed=1)
k = 10
A = np.linalg.matrix_power(A, k)
show_row(A)
print(A.shape)

train_per_iteration(
    algorithm="PullDiag_GT",
    lr=3e-2,
    A=A,
    dataset_name="MNIST",
    batch_size=128,
    num_epochs=10,
    remark=f"MG={k}, GRID16",
)