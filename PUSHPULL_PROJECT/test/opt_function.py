import numpy as np
import pandas as pd
from useful_functions import *
import copy

def PushPull(
    A,
    B,
    init_x,
    h_data,
    y_data,
    grad_func,
    rho=0.1,
    lr=0.1,
    sigma_n=0.1,
    max_it=200,
):
    """PushPull"""

    h_data, y_data = copy.deepcopy(h_data), copy.deepcopy(y_data)
    x = copy.deepcopy(init_x)
    n, d = x.shape
    g = grad_func(x, y_data, h_data, rho=rho).reshape(x.shape) + sigma_n * np.random.normal(
        size=(n, d)
    )
    y = copy.deepcopy(g)# gradient tracking

    pi_B = get_left_perron(B).reshape((n, 1)) # 列向量pi_B
    ones = np.ones((1, n))  # 行向量
    pi_B_ones_T = np.dot(pi_B, ones)
    print(pi_B_ones_T.shape)

    # 需要记录的参数
    x_mean = np.mean(x, axis=0, keepdims=True)# 1*d
    x_mean_expand = np.broadcast_to(x_mean, (n, d))#n*d
    gradient_x_mean = grad_func(x_mean_expand, y_data, h_data, rho=rho).reshape(x.shape)
    mean_gradient_x_mean = np.mean(gradient_x_mean, axis=0, keepdims=True)#1*d
    gradient_history = [np.linalg.norm(mean_gradient_x_mean)]
    x_history = [np.linalg.norm(x_mean)]
    y_norm = [np.linalg.norm(y)]
    g_norm = [np.linalg.norm(g)]
    pi_B_ones_T_g = [np.linalg.norm(pi_B_ones_T @ g)]

    # 需要记录的矩阵
    y_list = [y]
    pi_B_ones_T_g_list = [pi_B_ones_T @ g]

    for _ in range(max_it):
        # 更新
        x = A @ x - lr * y
        g_new = grad_func(x, y_data, h_data, rho=rho).reshape(x.shape) + sigma_n * np.random.normal(
            size=(n, d)
        )
        y = B @ y + g_new - g
        g = g_new

        # 记录
        x_mean = np.mean(x, axis=0, keepdims=True)# 1*d
        x_mean_expand = np.broadcast_to(x_mean, (n, d))#n*d
        gradient_x_mean = grad_func(x_mean_expand, y_data, h_data, rho=rho).reshape(x.shape)
        mean_gradient_x_mean = np.mean(gradient_x_mean, axis=0, keepdims=True)#1*d
        gradient_history.append(np.linalg.norm(mean_gradient_x_mean))
        x_history.append(np.linalg.norm(x_mean))
        y_norm.append(np.linalg.norm(y))
        g_norm.append(np.linalg.norm(g))
        pi_B_ones_T_g.append(np.linalg.norm(pi_B_ones_T @ g))

        # 记录矩阵
        y_list.append(y)
        pi_B_ones_T_g_list.append(pi_B_ones_T @ g)

    result_df = pd.DataFrame({
        "gradient_norm": gradient_history, 
        "x_mean_norm": x_history, 
        "y_norm": y_norm, 
        "g_norm": g_norm,
        "pi_B_ones_T_g": pi_B_ones_T_g})
    
    return result_df, y_list, pi_B_ones_T_g_list

def PullDiag_GD(
    A,
    init_x,
    h_data,
    y_data,
    grad_func,
    rho=0.1,
    lr=0.1,
    sigma_n=0.1,
    max_it=200,
):
    """PullDiag_GD"""
    h, y = copy.deepcopy(h_data), copy.deepcopy(y_data)
    x = copy.deepcopy(init_x)
    n, d = x.shape
    W = np.eye(n)
    g = grad_func(x, y, h, rho=rho).reshape(x.shape) + sigma_n * np.random.normal(
        size=(n, d)
    )
    x_mean = np.mean(x, axis=0, keepdims=True)# 1*d
    x_mean_expand = np.broadcast_to(x_mean, (n, d))#n*d
    gradient_x_mean = grad_func(x_mean_expand, y, h, rho=rho).reshape(x.shape)
    mean_gradient_x_mean = np.mean(gradient_x_mean, axis=0, keepdims=True)#1*d
    gradient_history = [np.linalg.norm(mean_gradient_x_mean)]
    x_history = [np.linalg.norm(x_mean)]

    for _ in range(max_it):
        W = A @ W
        Diag_W_inv = np.diag(1 / np.diag(W))
        x_new = x - lr * Diag_W_inv @ g
        x = A @ x_new
        g = grad_func(x, y, h, rho=rho).reshape(x.shape) + sigma_n * np.random.normal(
            size=(n, d)
        )
        x_mean = np.mean(x, axis=0, keepdims=True)# 1*d
        x_mean_expand = np.broadcast_to(x_mean, (n, d))#n*d
        gradient_x_mean = grad_func(x_mean_expand, y, h, rho=rho).reshape(x.shape)
        mean_gradient_x_mean = np.mean(gradient_x_mean, axis=0, keepdims=True)#1*d
        gradient_history.append(np.linalg.norm(mean_gradient_x_mean))
        x_history.append(np.linalg.norm(x_mean))

    result_df = pd.DataFrame({"gradient_norm": gradient_history, "x_mean_norm": x_history})
    return result_df


def PullDiag_GT(
    A,
    init_x,
    h_data,
    y_data,
    grad_func,
    rho=0.1,
    lr=0.1,
    sigma_n=0.1,
    max_it=200,
):
    """PullDiag_GT"""
    h, y = copy.deepcopy(h_data), copy.deepcopy(y_data)
    x = copy.deepcopy(init_x)
    n, d = x.shape
    W = np.eye(n)
    g = grad_func(x, y, h, rho=rho).reshape(x.shape) + sigma_n * np.random.normal(
        size=(n, d)
    )
    w = np.diag(1 / np.diag(W)) @ g
    v = copy.deepcopy(g)

    v_history = [np.linalg.norm(v)]
    x_mean = np.mean(x, axis=0, keepdims=True)# 1*d
    x_mean_expand = np.broadcast_to(x_mean, (n, d))#n*d
    gradient_x_mean = grad_func(x_mean_expand, y, h, rho=rho).reshape(x.shape)
    mean_gradient_x_mean = np.mean(gradient_x_mean, axis=0, keepdims=True)#1*d
    gradient_history = [np.linalg.norm(mean_gradient_x_mean)]
    x_history = [np.linalg.norm(x_mean)]

    for _ in range(max_it):
        W = A @ W
        x = A @ x - lr * v
        g = grad_func(x, y, h, rho=rho).reshape(x.shape) + sigma_n * np.random.normal(
            size=(n, d)
        )
        v = A @ v + np.linalg.inv(np.diag(np.diag(W))) @ g - w
        w = (
            np.linalg.inv(np.diag(np.diag(W))) @ g
        )  # 这一步计算的w是下一步用到的w，因此程序没有问题
        v_history.append(np.linalg.norm(v))

        x_mean = np.mean(x, axis=0, keepdims=True)# 1*d
        x_mean_expand = np.broadcast_to(x_mean, (n, d))
        gradient_x_mean = grad_func(x_mean_expand, y, h, rho=rho).reshape(x.shape)
        mean_gradient_x_mean = np.mean(gradient_x_mean, axis=0, keepdims=True)
        gradient_history.append(np.linalg.norm(mean_gradient_x_mean))
        x_history.append(np.linalg.norm(x_mean))

    result_df = pd.DataFrame(
        {
            "gradient_norm": gradient_history,
            "gradient_tracking_norm": v_history,
            "x_mean": x_history,
        }
    )
    return result_df
