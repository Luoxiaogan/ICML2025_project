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

lr = 3e-3
num_epochs = 100
bs = 128
alpha = 0.9
use_hetero=True
remark="new"
device = "cuda:0"
root = "/home/lg/ICML2025_project/PUSHPULL_PROJECT/最终的实验/grid_mnist_tmp"

n=25
A, B = generate_grid_matrices(n = n, seed=48)
show_row(A)
print(A.shape)
for i in range(20):
    df = train_track_grad_norm_with_hetero(
        algorithm="PushPull",
        lr=lr,
        A=A,
        B=B,
        dataset_name="MNIST",
        batch_size=bs,
        num_epochs=5000,
        remark=remark,
        alpha = alpha,
        root = root,
        use_hetero=use_hetero,
        device=device,
        seed = i+2
    )
     
    if i == 0:
        df_sum = df
        sum = 1
    else:
        df_sum = df_sum+df
        sum = sum + 1
    df_output = df_sum/sum
    df_output.to_csv(f"/home/lg/ICML2025_project/PUSHPULL_PROJECT/最终的实验/grid_mnist_repeat/new_brand_new_for_draw_exp_n={n}_lr={lr}.csv")

# n=9
# A, B = generate_grid_matrices(n = n, seed=48)
# show_row(A)
# print(A.shape)
# train_track_grad_norm_with_hetero(
#     algorithm="PushPull",
#     lr=lr,
#     A=A,
#     B=B,
#     dataset_name="MNIST",
#     batch_size=bs,
#     num_epochs=400,
#     remark=remark,
#     alpha = alpha,
#     root = "/home/lg/ICML2025_project/PUSHPULL_PROJECT/最终的实验/grid_mnist",
#     use_hetero=use_hetero,
# )

# n=16
# A, B = generate_grid_matrices(n = n, seed=48)
# show_row(A)
# print(A.shape)
# train_track_grad_norm_with_hetero(
#     algorithm="PushPull",
#     lr=lr,
#     A=A,
#     B=B,
#     dataset_name="MNIST",
#     batch_size=bs,
#     num_epochs=1000,
#     remark=remark,
#     alpha = alpha,
#     root = "/home/lg/ICML2025_project/PUSHPULL_PROJECT/最终的实验/grid_mnist",
#     use_hetero=use_hetero,
# )

# n=25
# A, B = generate_grid_matrices(n = n, seed=48)
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
#     root = "/home/lg/ICML2025_project/PUSHPULL_PROJECT/最终的实验/grid_mnist",
#     use_hetero=use_hetero,
# )