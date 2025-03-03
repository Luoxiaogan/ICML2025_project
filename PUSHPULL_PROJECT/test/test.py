import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy
from useful_functions import *
from opt_function import *
import os

def ring1(n=10):  # 生成稀疏环状图。也可以取n=5
    A, B = np.eye(n) / 2, np.eye(n) / 2
    m = int(n / 2)
    for i in range(n - 1):
        A[i][i + 1] = 0.5
        B[i][i + 1] = 0.5
    A[n - 1][0] = 0.5
    B[n - 1][0] = 0.5
    A[0][m] = 1 / 3
    A[m - 1][m] = 1 / 3
    A[m][m] = 1 / 3
    B[0][0] = 1 / 3
    B[0][1] = 1 / 3
    B[0][m] = 1 / 3
    return A.T, B.T  # A.T是行随机，B.T是列随机矩阵，

n=10
d=10
L=100
i=1

A, B = ring1(n=n)

init_x=init_x_func(n=n,d=d,seed=42)
h,y,x_opt,x_star=init_data(n=n,d=d,L=L,seed=4,sigma_h=0.1)

# 生成多个 sigma_n 值
sigma_n_values = [1e-2 / (2**i) for i in range(10) if 1e-2 / (2**i) > 1e-6]

# 循环运行实验并保存结果
output_dir = '/Users/luogan/Code/ICML2025_project/PUSHPULL_PROJECT/test/output'
os.makedirs(output_dir, exist_ok=True)

def calculate_variance(biglist):
    num_times = len(biglist[0])
    variance_over_time = []
    for T in range(num_times):
        matrices_at_time_T = [sublist[T] for sublist in biglist]
        mean_matrix = np.mean(matrices_at_time_T, axis=0)
        squared_norms = [np.linalg.norm(matrix - mean_matrix, 'fro') ** 2 for matrix in matrices_at_time_T]
        variance = np.mean(squared_norms)
        variance_over_time.append(variance)
    return variance_over_time

for sigma_n in sigma_n_values:
    big_y_lists = []
    big_pi_B_ones_T_g_lists = []
    for _ in range(10):
        df, y_list, pi_B_ones_T_g_list = PushPull(
            A=A,
            B=B,
            init_x=init_x,
            h_data=h,
            y_data=y,
            grad_func=grad,
            rho=0.1,
            lr=1e-2,
            sigma_n=sigma_n,  # 修改为当前 sigma_n
            max_it=20000
        )
        big_y_lists.append(y_list)
        big_pi_B_ones_T_g_lists.append(pi_B_ones_T_g_list)
        print(f"completed {_ + 1}th run for sigma_n={sigma_n}")
    
    # 计算方差
    variance_y = calculate_variance(big_y_lists)
    variance_pi_B_ones_T_g = calculate_variance(big_pi_B_ones_T_g_lists)
    # 存储到 CSV 文件
    df_variances = pd.DataFrame({
        'variance_y': variance_y,
        'variance_pi_B_ones_T_g': variance_pi_B_ones_T_g
    })
    filename = f"sigma_n_{sigma_n}.csv"
    df_variances.to_csv(os.path.join(output_dir, filename), index=False)