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

from training import train, train_per_iteration, train_high_hetero
import torch
from utils import ring1, show_row , ring3, ring2, ring4
from network_utils import get_matrixs_from_exp_graph


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n=16
A,B = ring1(n=n)
k = 1
A = np.linalg.matrix_power(A, k)
show_row(A)
print(A.shape)

lr_list = [6e-3, 7e-3, 5e-3, 4e-3, 3e-3, 1e-2]

for lr in lr_list:
    train_high_hetero(
        algorithm="PullDiag_GT",
        lr=lr,
        A=A,
        B=B,# 实际没用用到
        dataset_name="MNIST",
        batch_size=128,
        num_epochs=300,
        remark=f"MG={k}, RING 16, HIGH HETERO",
    )