import sys
import os
import numpy as np
import pandas as pd
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


from training import train
from network_utils import get_matrixs_from_exp_graph, generate_grid_matrices, generate_ring_matrices
from utils import show_row, show_col

lr = 5e-3
num_epochs = 500
bs = 128
remark="grid"
device = "cuda:1"

# n = 4
# A, B = get_matrixs_from_exp_graph(n = n, seed=48)
# print(B)
# show_col(B)

# -----------------------------
# n = 1
# A = np.full((1,1), 1)
# B = np.full((1,1), 1)
# df = train(
#     algorithm="PushPull",
#     lr=lr,
#     A=A,
#     B=B,
#     dataset_name="CIFAR10",
#     batch_size=bs,
#     num_epochs=num_epochs,
#     remark=remark,
# )
# -----------------------------
n = 4
A, B = generate_grid_matrices(n = n, seed=42)
df = train(
    algorithm="PushPull",
    lr=lr,
    A=A,
    B=B,
    dataset_name="CIFAR10",
    batch_size=bs,
    num_epochs=num_epochs,
    remark=remark,
    device=device,
)
# -----------------------------
n = 9
A, B = generate_grid_matrices(n = n, seed=42)
df = train(
    algorithm="PushPull",
    lr=lr,
    A=A,
    B=B,
    dataset_name="CIFAR10",
    batch_size=bs,
    num_epochs=num_epochs,
    remark=remark,
    device=device,
)
# -----------------------------
n = 16
A, B = generate_grid_matrices(n = n, seed=42)
df = train(
    algorithm="PushPull",
    lr=lr,
    A=A,
    B=B,
    dataset_name="CIFAR10",
    batch_size=bs,
    num_epochs=num_epochs,
    remark=remark,
    device=device,
)
# -----------------------------