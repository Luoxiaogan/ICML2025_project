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

from training import train, train_per_iteration, train_just_per_batch_loss
from training import train_track_grad_norm
import torch
from utils import ring1, show_row 
from network_utils import get_matrixs_from_exp_graph

import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


lr = 1e-2
num_epochs = 50
bs = 128


# n=4
# A, B = get_matrixs_from_exp_graph(n = n, seed=48)
# show_row(A)
# print(A.shape)
# train_track_grad_norm(
#     algorithm="PushPull",
#     lr=lr,
#     A=A,
#     B=B,
#     dataset_name="MNIST",
#     batch_size=bs,
#     num_epochs=50,
#     remark=f"Exp_test",
# )

# n=8
# A, B = get_matrixs_from_exp_graph(n = n, seed=48)
# show_row(A)
# print(A.shape)
# train_track_grad_norm(
#     algorithm="PushPull",
#     lr=lr,
#     A=A,
#     B=B,
#     dataset_name="MNIST",
#     batch_size=bs,
#     num_epochs=100,
#     remark=f"Exp_test",
# )

# n=16
# A, B = get_matrixs_from_exp_graph(n = n, seed=48)
# show_row(A)
# print(A.shape)
# train_track_grad_norm(
#     algorithm="PushPull",
#     lr=lr,
#     A=A,
#     B=B,
#     dataset_name="MNIST",
#     batch_size=bs,
#     num_epochs=200,
#     remark=f"Exp_test",
# )

n=32
A, B = get_matrixs_from_exp_graph(n = n, seed=48)
show_row(A)
print(A.shape)
train_track_grad_norm(
    algorithm="PushPull",
    lr=lr,
    A=A,
    B=B,
    dataset_name="MNIST",
    batch_size=bs,
    num_epochs=400,
    remark=f"test_Exp_test",
)

# n=64
# A, B = get_matrixs_from_exp_graph(n = n, seed=48)
# show_row(A)
# print(A.shape)
# train_track_grad_norm(
#     algorithm="PushPull",
#     lr=5e-2,
#     A=A,
#     B=B,
#     dataset_name="MNIST",
#     batch_size=bs,
#     num_epochs=500,
#     remark=f"Exp_test",
# )

# n=4
# A, B = get_matrixs_from_exp_graph(n = n, seed=48)
# show_row(A)
# print(A.shape)
# train_track_grad_norm(
#     algorithm="PushPull",
#     lr=1e-3,
#     A=A,
#     B=B,
#     dataset_name="MNIST",
#     batch_size=128,
#     num_epochs=60,
#     remark=f"Exp_test",
# )

# n=8
# A, B = get_matrixs_from_exp_graph(n = n, seed=48)
# show_row(A)
# print(A.shape)
# train_track_grad_norm(
#     algorithm="PushPull",
#     lr=1e-3,
#     A=A,
#     B=B,
#     dataset_name="MNIST",
#     batch_size=128,
#     num_epochs=120,
#     remark=f"Exp_test",
# )

# n=16
# A, B = get_matrixs_from_exp_graph(n = n, seed=48)
# show_row(A)
# print(A.shape)
# train_track_grad_norm(
#     algorithm="PushPull",
#     lr=1e-3,
#     A=A,
#     B=B,
#     dataset_name="MNIST",
#     batch_size=128,
#     num_epochs=240,
#     remark=f"Exp_test",
# )