import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy
from useful_functions_with_batch import *
from opt_function_with_batch import *
from network_utils import *

d=10
L_total=204800

h_global, y_global, x_opt = init_global_data(d=d, L_total=L_total, seed=42)
print("h:",h_global.shape)
print("y:",y_global.shape)

n=1
h_tilde, y_tilde = distribute_data(h=h_global, y=y_global, n=n)
x_star = generate_x_star(n=n, d=d, x_opt=x_opt, sigma_h=10)
init_x = init_x_func(n=n, d=d, seed=42)
A,B = get_matrixs_from_exp_graph(n=n, seed=42)
print("h_tilde:",h_tilde.shape,'\n')
L_list = []
for i in range(20):
    L1 = PushPull_with_batch(
        A=A,
        B=B,
        init_x=init_x,
        h_data=h_tilde,
        y_data=y_tilde,
        grad_func=grad_with_batch,
        rho=1e-2,
        lr=5e-2,
        sigma_n=0,
        max_it=5000,
        batch_size=200
    )
    print(f"finish, n={n}, i={i}/20")
    L_list.append(L1)
L1_avg = pd.concat(L_list).groupby(level=0).mean()
L1_avg.to_csv(f"./output/L_avg_n={n}.csv")
print(f"finish, n={n}")

n=2
h_tilde, y_tilde = distribute_data(h=h_global, y=y_global, n=n)
x_star = generate_x_star(n=n, d=d, x_opt=x_opt, sigma_h=10)
init_x = init_x_func(n=n, d=d, seed=42)
A,B = get_matrixs_from_exp_graph(n=n, seed=42)
print("h_tilde:",h_tilde.shape,'\n')
L_list = []
for i in range(20):
    L1 = PushPull_with_batch(
        A=A,
        B=B,
        init_x=init_x,
        h_data=h_tilde,
        y_data=y_tilde,
        grad_func=grad_with_batch,
        rho=1e-2,
        lr=5e-2,
        sigma_n=0,
        max_it=5000,
        batch_size=200
    )
    print(f"finish, n={n}, i={i}/20")
    L_list.append(L1)
L1_avg = pd.concat(L_list).groupby(level=0).mean()
L1_avg.to_csv(f"./output/L_avg_n={n}.csv")
print(f"finish, n={n}")

n=8
h_tilde, y_tilde = distribute_data(h=h_global, y=y_global, n=n)
x_star = generate_x_star(n=n, d=d, x_opt=x_opt, sigma_h=10)
init_x = init_x_func(n=n, d=d, seed=42)
A,B = get_matrixs_from_exp_graph(n=n, seed=42)
print("h_tilde:",h_tilde.shape,'\n')
L_list = []
for i in range(20):
    L1 = PushPull_with_batch(
        A=A,
        B=B,
        init_x=init_x,
        h_data=h_tilde,
        y_data=y_tilde,
        grad_func=grad_with_batch,
        rho=1e-2,
        lr=5e-2,
        sigma_n=0,
        max_it=5000,
        batch_size=200
    )
    print(f"finish, n={n}, i={i}/20")
    L_list.append(L1)
L1_avg = pd.concat(L_list).groupby(level=0).mean()
L1_avg.to_csv(f"./output/L_avg_n={n}.csv")
print(f"finish, n={n}")

n=16
h_tilde, y_tilde = distribute_data(h=h_global, y=y_global, n=n)
x_star = generate_x_star(n=n, d=d, x_opt=x_opt, sigma_h=10)
init_x = init_x_func(n=n, d=d, seed=42)
A,B = get_matrixs_from_exp_graph(n=n, seed=42)
print("h_tilde:",h_tilde.shape,'\n')
L_list = []
for i in range(20):
    L1 = PushPull_with_batch(
        A=A,
        B=B,
        init_x=init_x,
        h_data=h_tilde,
        y_data=y_tilde,
        grad_func=grad_with_batch,
        rho=1e-2,
        lr=5e-2,
        sigma_n=0,
        max_it=5000,
        batch_size=200
    )
    print(f"finish, n={n}, i={i}/20")
    L_list.append(L1)
L1_avg = pd.concat(L_list).groupby(level=0).mean()
L1_avg.to_csv(f"./output/L_avg_n={n}.csv")
print(f"finish, n={n}")

n=128
h_tilde, y_tilde = distribute_data(h=h_global, y=y_global, n=n)
x_star = generate_x_star(n=n, d=d, x_opt=x_opt, sigma_h=10)
init_x = init_x_func(n=n, d=d, seed=42)
A,B = get_matrixs_from_exp_graph(n=n, seed=42)
print("h_tilde:",h_tilde.shape,'\n')
L_list = []
for i in range(20):
    L1 = PushPull_with_batch(
        A=A,
        B=B,
        init_x=init_x,
        h_data=h_tilde,
        y_data=y_tilde,
        grad_func=grad_with_batch,
        rho=1e-2,
        lr=5e-2,
        sigma_n=0,
        max_it=5000,
        batch_size=200
    )
    print(f"finish, n={n}, i={i}/20")
    L_list.append(L1)
L1_avg = pd.concat(L_list).groupby(level=0).mean()
L1_avg.to_csv(f"./output/L_avg_n={n}.csv")
print(f"finish, n={n}")

n=512
h_tilde, y_tilde = distribute_data(h=h_global, y=y_global, n=n)
x_star = generate_x_star(n=n, d=d, x_opt=x_opt, sigma_h=10)
init_x = init_x_func(n=n, d=d, seed=42)
A,B = get_matrixs_from_exp_graph(n=n, seed=42)
print("h_tilde:",h_tilde.shape,'\n')
L_list = []
for i in range(20):
    L1 = PushPull_with_batch(
        A=A,
        B=B,
        init_x=init_x,
        h_data=h_tilde,
        y_data=y_tilde,
        grad_func=grad_with_batch,
        rho=1e-2,
        lr=5e-2,
        sigma_n=0,
        max_it=5000,
        batch_size=200
    )
    print(f"finish, n={n}, i={i}/20")
    L_list.append(L1)
L1_avg = pd.concat(L_list).groupby(level=0).mean()
L1_avg.to_csv(f"./output/L_avg_n={n}.csv")
print(f"finish, n={n}")