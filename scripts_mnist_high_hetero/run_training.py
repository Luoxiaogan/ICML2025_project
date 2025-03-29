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
k = 10
A = np.linalg.matrix_power(A, k)
show_row(A)
print(A.shape)


# 在 Python 程序的最后执行 shell 命令
os.chdir("/root/GanLuo/ICML2025_project")
os.system("source /etc/network_turbo")
os.system("git add .")
os.system('git commit -m "add all this"')
os.system("git push")

lr_list = [1.5e-2, 2e-2,3e-2, 4e-2, 5e-2, 6e-2, 7e-2, 8e-2, 9e-2]
for lr in lr_list:
    print(f"lr = {lr}")
    train(
        algorithm="PullDiag_GT",
        lr=lr,
        A=A,
        B=B,# 实际没用用到
        dataset_name="MNIST",
        batch_size=128,
        num_epochs=400,
        remark=f"MG={k}, RING 16, HIGH HETERO",
    )

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n=16
A,B = ring1(n=n)
k = 5
A = np.linalg.matrix_power(A, k)
show_row(A)
print(A.shape)

lr_list = [6e-3, 7e-3,8e-3,9e-3,1e-2]
for lr in lr_list:
    print(f"lr = {lr}")
    train(
        algorithm="PullDiag_GT",
        lr=lr,
        A=A,
        B=B,# 实际没用用到
        dataset_name="MNIST",
        batch_size=128,
        num_epochs=400,
        remark=f"MG={k}, RING 16, HIGH HETERO",
    )


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n=16
A,B = get_matrixs_from_exp_graph(n=n,seed=42)
k = 10
A = np.linalg.matrix_power(A, k)
show_row(A)
print(A.shape)
lr_list = [1.5e-2, 2e-2,3e-2, 4e-2, 5e-2, 6e-2, 7e-2, 8e-2, 9e-2, 1e-1,2e-1]
for lr in lr_list:
    print(f"lr = {lr}")
    train(
        algorithm="PullDiag_GT",
        lr=lr,
        A=A,
        B=B,# 实际没用用到
        dataset_name="MNIST",
        batch_size=128,
        num_epochs=400,
        remark=f"MG={k}, RING 16, HIGH HETERO",
    )

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n=16
A,B = get_matrixs_from_exp_graph(n=n,seed=42)
k = 5
A = np.linalg.matrix_power(A, k)
show_row(A)
print(A.shape)
lr_list = [1e-2, 1.5e-2, 2e-2,3e-2, 4e-2, 5e-2, 6e-2, 7e-2, 8e-2, 9e-2, 1e-1,2e-1]
for lr in lr_list:
    print(f"lr = {lr}")
    train(
        algorithm="PullDiag_GT",
        lr=lr,
        A=A,
        B=B,# 实际没用用到
        dataset_name="MNIST",
        batch_size=128,
        num_epochs=400,
        remark=f"MG={k}, RING 16, HIGH HETERO",
    )

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n=16
A,B = get_matrixs_from_exp_graph(n=n,seed=42)
k = 1
A = np.linalg.matrix_power(A, k)
show_row(A)
print(A.shape)
lr_list = [1e-2, 1.5e-2, 2e-2,3e-2, 4e-2, 5e-2, 6e-2, 7e-2, 8e-2, 9e-2, 1e-1,2e-1]
for lr in lr_list:
    print(f"lr = {lr}")
    train(
        algorithm="PullDiag_GT",
        lr=lr,
        A=A,
        B=B,# 实际没用用到
        dataset_name="MNIST",
        batch_size=128,
        num_epochs=400,
        remark=f"MG={k}, RING 16, HIGH HETERO",
    )

# 在 Python 程序的最后执行 shell 命令
os.chdir("/root/GanLuo/ICML2025_project")
os.system("source /etc/network_turbo")
os.system("git add .")
os.system('git commit -m "add all this"')
os.system("git push")