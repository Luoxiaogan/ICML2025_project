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
from utils import ring1, show_row , ring3, ring2, ring4, Row

def get_xinmeng_matrix(n=5):
    M = np.zeros((n, n))

    # 主对角线上的元素
    M[np.diag_indices(n)] = 1 / 3 * np.ones(n)
    M[n - 1, n - 1] = M[n - 1, n - 1] + 1 / 3

    # 次对角线上的元素
    for i in range(n - 1):
        M[i + 1, i] = M[i + 1, i] + 1 / 3

    # 第一行上的元素
    M[0, :] = M[0, :] + 1 / 3

    return M


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n=35
#A, B = ring1(n)
#A = generate_row_stochastic_matrix_with_self_loops(seed=48)
A = get_xinmeng_matrix(n=n).T
B = np.eye(n)
k = 5
A = np.linalg.matrix_power(A, k)
show_row(A)
print(A.shape)

train(
    algorithm="PullDiag_GT",
    lr=1e-2,
    A=A,
    B=B,# 实际没用用到
    dataset_name="MNIST",
    batch_size=128,
    num_epochs=250,
    remark=f"MG={k},Xinmeng",
)