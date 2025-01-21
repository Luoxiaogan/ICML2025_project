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
from utils import ring1, ring2, ring3, ring4, show_row

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

n=5

import numpy as np
A = np.full((n, n), 1/n)

train(
    algorithm="PullDiag_GT",
    lr=5e-2,
    A=A,
    dataset_name="MNIST",
    batch_size=500,
    num_epochs=600,
    remark="全连接A, 取小学习率, 看看收敛到的noise是多少",
)