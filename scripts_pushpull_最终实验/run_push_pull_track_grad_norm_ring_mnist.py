import sys
import os
import numpy as np
import pandas as pd
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


from training import train, train_per_iteration, train_just_per_batch_loss
from training import train_track_grad_norm, train_track_grad_norm_with_hetero
import torch
from utils import ring1, show_row 
from network_utils import get_matrixs_from_exp_graph, generate_grid_matrices, generate_ring_matrices

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


lr = 5e-3
num_epochs = 100
bs = 128
alpha = 0.9
use_hetero=True
remark="Grid_异质性",
device = "cuda:1"
root = "/home/lg/ICML2025_project/PUSHPULL_PROJECT/最终的实验/ring_mnist"

# n=4
# A, B = generate_ring_matrices(n = n, seed=42)
# show_row(A)
# print(A.shape)
# train_track_grad_norm_with_hetero(
#     algorithm="PushPull",
#     lr=lr,
#     A=A,
#     B=B,
#     dataset_name="MNIST",
#     batch_size=bs,
#     num_epochs=300,
#     remark=remark,
#     alpha = alpha,
#     root = root,
#     use_hetero=use_hetero,
#     device=device
# )

n=8
A, B = generate_ring_matrices(n = n, seed=42)
show_row(A)
print(A.shape)
train_track_grad_norm_with_hetero(
    algorithm="PushPull",
    lr=lr,
    A=A,
    B=B,
    dataset_name="MNIST",
    batch_size=bs,
    num_epochs=1500,
    remark=remark,
    alpha = alpha,
    root = root,
    use_hetero=use_hetero,
    device=device
)

# n=12
# A, B = generate_ring_matrices(n = n, seed=42)
# show_row(A)
# print(A.shape)
# train_track_grad_norm_with_hetero(
#     algorithm="PushPull",
#     lr=lr,
#     A=A,
#     B=B,
#     dataset_name="MNIST",
#     batch_size=bs,
#     num_epochs=1500,
#     remark=remark,
#     alpha = alpha,
#     root = root,
#     use_hetero=use_hetero,
#     device=device
# )

# n=16
# A, B = generate_ring_matrices(n = n, seed=42)
# show_row(A)
# print(A.shape)
# train_track_grad_norm_with_hetero(
#     algorithm="PushPull",
#     lr=lr,
#     A=A,
#     B=B,
#     dataset_name="MNIST",
#     batch_size=bs,
#     num_epochs=1500,
#     remark=remark,
#     alpha = alpha,
#     root = root,
#     use_hetero=use_hetero,
#     device=device
# )