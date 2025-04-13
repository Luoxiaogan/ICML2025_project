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

def average_dfs(df_list):
    """
    计算 DataFrame 列表中每个对应列的每个对应元素的平均值。

    Args:
        df_list: 一个包含结构相同的 Pandas DataFrame 的列表。

    Returns:
        一个新的 Pandas DataFrame，其中包含每个对应元素的平均值。
        如果 df_list 为空，则返回一个空的 DataFrame。
        如果 DataFrame 结构不一致，可能会引发错误。
    """
    if not df_list:
        return pd.DataFrame()

    # 将所有的 DataFrame 合并到一个大的 NumPy 数组中
    all_data = np.array([df.values for df in df_list])

    # 计算每个对应元素的平均值
    averaged_data = np.mean(all_data, axis=0)

    # 创建新的 DataFrame，使用第一个 DataFrame 的列名
    averaged_df = pd.DataFrame(averaged_data, columns=df_list[0].columns)

    return averaged_df

lr = 5e-3
num_epochs = 100
bs = 128
alpha = 0.9
use_hetero=True
remark="Exp_异质性",
device = "cuda:0"
root = "/home/lg/ICML2025_project/PUSHPULL_PROJECT/最终的实验/exp_mnist_tmp"

n=4
A, B = get_matrixs_from_exp_graph(n = n, seed=48)
show_row(A)
print(A.shape)
df_list =[]
for i in range(10):
    df = train_track_grad_norm_with_hetero(
        algorithm="PushPull",
        lr=lr,
        A=A,
        B=B,
        dataset_name="MNIST",
        batch_size=bs,
        num_epochs=200,
        remark=remark,
        alpha = alpha,
        root = root,
        use_hetero=use_hetero,
        device=device
    )
    df_list.append(df)
df_new = average_dfs(df_list)
df_new.to_csv(f"/home/lg/ICML2025_project/PUSHPULL_PROJECT/最终的实验/exp_mnist_10_repeat/exp_n={n}_lr={lr}")

n=8
A, B = get_matrixs_from_exp_graph(n = n, seed=48)
show_row(A)
print(A.shape)
df_list =[]
for i in range(10):
    df = train_track_grad_norm_with_hetero(
        algorithm="PushPull",
        lr=lr,
        A=A,
        B=B,
        dataset_name="MNIST",
        batch_size=bs,
        num_epochs=400,
        remark=remark,
        alpha = alpha,
        root = root,
        use_hetero=use_hetero,
        device=device
    )
    df_list.append(df)
df_new = average_dfs(df_list)
df_new.to_csv(f"/home/lg/ICML2025_project/PUSHPULL_PROJECT/最终的实验/exp_mnist_10_repeat/exp_n={n}_lr={lr}")


n=16
A, B = get_matrixs_from_exp_graph(n = n, seed=48)
show_row(A)
print(A.shape)
df_list =[]
for i in range(10):
    df = train_track_grad_norm_with_hetero(
        algorithm="PushPull",
        lr=lr,
        A=A,
        B=B,
        dataset_name="MNIST",
        batch_size=bs,
        num_epochs=1600,
        remark=remark,
        alpha = alpha,
        root = root,
        use_hetero=use_hetero,
        device=device
    )
    df_list.append(df)
df_new = average_dfs(df_list)
df_new.to_csv(f"/home/lg/ICML2025_project/PUSHPULL_PROJECT/最终的实验/exp_mnist_10_repeat/exp_n={n}_lr={lr}")


n=32
A, B = get_matrixs_from_exp_graph(n = n, seed=48)
show_row(A)
print(A.shape)
df_list =[]
for i in range(10):
    df = train_track_grad_norm_with_hetero(
        algorithm="PushPull",
        lr=lr,
        A=A,
        B=B,
        dataset_name="MNIST",
        batch_size=bs,
        num_epochs=2000,
        remark=remark,
        alpha = alpha,
        root = root,
        use_hetero=use_hetero,
        device=device
    )
    df_list.append(df)
df_new = average_dfs(df_list)
df_new.to_csv(f"/home/lg/ICML2025_project/PUSHPULL_PROJECT/最终的实验/exp_mnist_10_repeat/exp_n={n}_lr={lr}")