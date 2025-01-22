# scripts/run_training.py

import sys
import os

# 将项目根目录添加到 sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from training import train, train_per_iteration, special_train
import torch
from utils import ring1, ring2, ring3, ring4, show_row

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

n=35

import numpy as np
A = np.full((n, n), 1/n)

train_per_iteration(
    algorithm="PullDiag_GT",
    lr=1e-1,
    A=A,
    dataset_name="MNIST",
    batch_size=128,
    num_epochs=600,
    remark="special_train, 在整个训练集上计算grad_norm",
)