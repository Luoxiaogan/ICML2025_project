# scripts/run_training.py

import sys
import os

# 将项目根目录添加到 sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from training import train, train_per_iteration
import torch
from utils import ring1, show_row 
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

n=16
A, B = ring1(n=n)
k = 20
A = np.linalg.matrix_power(A, k)
show_row(A)
print(A.shape)

train_per_iteration(
    algorithm="PullDiag_GT",
    lr=5e-3,
    A=A,
    dataset_name="MNIST",
    batch_size=128,
    num_epochs=10,
    remark=f"MG={k}, ring1",
)