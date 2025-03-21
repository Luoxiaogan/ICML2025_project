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
import torch
from utils import ring1, show_row 
from network_utils import get_matrixs_from_exp_graph

import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n=16
#A, B = ring1(n)
#A = generate_row_stochastic_matrix_with_self_loops(seed=48)
A, B = get_matrixs_from_exp_graph(n = n, seed=48)
show_row(A)
print(A.shape)

train_just_per_batch_loss(
    algorithm="PushPull",
    lr=1e-2,
    A=A,
    B=B,
    dataset_name="MNIST",
    batch_size=16,
    num_epochs=50,
    remark=f"Exp_test",
)


n=8
#A, B = ring1(n)
#A = generate_row_stochastic_matrix_with_self_loops(seed=48)
A, B = get_matrixs_from_exp_graph(n = n, seed=48)
show_row(A)
print(A.shape)

train_just_per_batch_loss(
    algorithm="PushPull",
    lr=1e-2,
    A=A,
    B=B,
    dataset_name="MNIST",
    batch_size=16,
    num_epochs=50,
    remark=f"Exp_test",
)


n=4
#A, B = ring1(n)
#A = generate_row_stochastic_matrix_with_self_loops(seed=48)
A, B = get_matrixs_from_exp_graph(n = n, seed=48)
show_row(A)
print(A.shape)

train_just_per_batch_loss(
    algorithm="PushPull",
    lr=1e-2,
    A=A,
    B=B,
    dataset_name="MNIST",
    batch_size=16,
    num_epochs=50,
    remark=f"Exp_test",
)


n=2
#A, B = ring1(n)
#A = generate_row_stochastic_matrix_with_self_loops(seed=48)
A, B = get_matrixs_from_exp_graph(n = n, seed=48)
show_row(A)
print(A.shape)

train_just_per_batch_loss(
    algorithm="PushPull",
    lr=1e-2,
    A=A,
    B=B,
    dataset_name="MNIST",
    batch_size=16,
    num_epochs=50,
    remark=f"Exp_test",
)


n=32
#A, B = ring1(n)
#A = generate_row_stochastic_matrix_with_self_loops(seed=48)
A, B = get_matrixs_from_exp_graph(n = n, seed=48)
show_row(A)
print(A.shape)

train_just_per_batch_loss(
    algorithm="PushPull",
    lr=1e-2,
    A=A,
    B=B,
    dataset_name="MNIST",
    batch_size=16,
    num_epochs=50,
    remark=f"Exp_test",
)