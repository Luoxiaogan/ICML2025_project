import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy
from useful_functions_with_batch import *
from opt_function_with_batch import *
from network_utils import *

d=10
L_total=1040000
h_global, y_global, x_opt = init_global_data(d=d, L_total=L_total, seed=42)
print("h:",h_global.shape)
print("y:",y_global.shape)

n=16
h_tilde, y_tilde = distribute_data(h=h_global, y=y_global, n=n)
x_star = generate_x_star(n=n, d=d, x_opt=x_opt, sigma_h=10)
init_x = init_x_func(n=n, d=d, seed=42)
A,B = get_matrixs_from_exp_graph(n=n, seed=42)
print("h_tilde:",h_tilde.shape,'\n')
print("y_tilde:",y_tilde.shape,'\n')

print("starting experiment with n =", n)

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
        max_it=100,
        batch_size=200
    )
L1.to_csv(f"/home/lg/ICML2025_project/push_pull_linear/EXP_out/EXP_n={n}.csv")