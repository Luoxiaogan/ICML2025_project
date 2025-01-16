# scripts/run_training.py

import sys
import os

# 将项目根目录添加到 sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from training import train
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n=5
A = torch.full((n, n), 1.0 / n, device=device)

train(
    algorithm="PullDiag_GD",
    lr=3e-2,
    A=A,
    dataset_name="MNIST",
    batch_size=256,
    num_epochs=100,
)