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
from utils import ring1, show_row , ring3, ring2, ring4


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n=16
#A, B = ring1(n)
#A = generate_row_stochastic_matrix_with_self_loops(seed=48)
A, B = ring1(n=n)
k = 1 # 初始就设置为2哈哈
A = np.linalg.matrix_power(A, k)
show_row(A)
print(A.shape)

train(
    algorithm="PullDiag_GT",
    lr=5e-3,
    A=A,
    B=B,# 实际没用用到
    dataset_name="CIFAR10",
    batch_size=128,
    num_epochs=250,
    remark=f"MG={k}, Ring16",
)