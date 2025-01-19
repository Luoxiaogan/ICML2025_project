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
from utils import ring1, ring2, ring3, ring4, show_row

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 第一组 MG=1, ring1, PullDiag_GT, lr=8e-3, n_nodes=15, batch_size=128

n=15
A, B = ring1(n=n)
show_row(A)
A = torch.from_numpy(A).float()
print(A.shape)

train(
    algorithm="PullDiag_GT",
    lr=8e-3,
    A=A,
    dataset_name="MNIST",
    batch_size=128,
    num_epochs=100,
    remark="MG=1, ring1",
)

# 第二组 MG=2, ring1, PullDiag_GT, lr=1.6e-2, n_nodes=15, batch_size=256

n=15
A, B = ring1(n=n)
show_row(A)
A = torch.from_numpy(A).float()
A = torch.matmul(A, A)
print(A.shape)

train(
    algorithm="PullDiag_GT",
    lr=1.6e-2,
    A=A,
    dataset_name="MNIST",
    batch_size=256,
    num_epochs=100,
    remark="MG=2, ring1",
)

# 第三组 MG=5, ring1, PullDiag_GT, lr=4e-2, n_nodes=15, batch_size=640

n=15
A, B = ring1(n=n)
show_row(A)
A = torch.from_numpy(A).float()
A = torch.matrix_power(A, 5)
print(A.shape)

train(
    algorithm="PullDiag_GT",
    lr=4e-2,
    A=A,
    dataset_name="MNIST",
    batch_size=640,
    num_epochs=100,
    remark="MG=5, ring1",
)