import numpy as np
import pandas as pd
from useful_functions_with_batch import *
import copy

# def loss_compute(x, y, h, rho=0.001):
#     """
#     计算所有节点的损失函数之和加上正则项，即 sum_{i=1}^n f_i(x_i) + 正则项。
#     注意：真正的损失应该是 loss / n。
    
#     Parameters:
#         x (np.ndarray): 输入参数，形状为 (n * d,) 或 (n, d)。
#         y (np.ndarray): 标签数据，形状为 (n, L)。
#         h (np.ndarray): 模型参数，形状为 (n, L, d)。
#         rho (float): 正则化系数，默认为 0.001。
    
#     Returns:
#         float: 损失值。
#     """

#     print("在loss函数中")
#     print("x.shape", x.shape)
#     print("y.shape", y.shape)
#     print("h.shape", h.shape)


#     n, L, d = h.shape
#     x = x.reshape(-1)  # 确保 x 是 (n * d,) 的一维向量
    
#     # 计算 h_dot_x: h 和 x 的点积，形状为 (n, L)
#     h_dot_x = np.einsum('ijk,k->ij', h, x, optimize=True)  # 使用 optimize=True 加速
    
#     # 计算 stable_log_exp: log(1 + exp(-y * h_dot_x))，形状为 (n, L)
#     log_exp_term = stable_log_exp(-y * h_dot_x)
    
#     # 计算 term1: 所有节点和样本的损失平均值，标量
#     term1 = np.sum(log_exp_term) / L
    
#     # 计算正则化项: rho * x^2 / (1 + x^2)，标量
#     x_squared = x**2
#     term2 = np.sum(rho * x_squared / (1 + x_squared))
    
#     return term1 + term2

import numpy as np

def stable_log_exp(z):
    # 假设这是一个稳定的 log(1 + exp(z)) 实现
    return np.log1p(np.exp(np.minimum(z, 0))) + np.maximum(z, 0)

def loss_compute(x, y, h, rho=0.001):
    """
    计算所有节点的损失函数之和加上正则项，即 sum_{i=1}^n f_i(x_i) + 正则项。
    注意：真正的损失应该是 loss / n。
    
    Parameters:
        x (np.ndarray): 输入参数，形状为 (n * d,) 或 (n, d)。
        y (np.ndarray): 标签数据，形状为 (n, L)。
        h (np.ndarray): 模型参数，形状为 (n, L, d)。
        rho (float): 正则化系数，默认为 0.001。
    
    Returns:
        float: 损失值。
    """

    # print("在loss函数中")
    # print("x.shape", x.shape)
    # print("y.shape", y.shape)
    # print("h.shape", h.shape)

    n, L, d = h.shape
    
    # 如果 x 是 (n * d,)，重塑为 (n, d)
    if x.ndim == 1:
        x = x.reshape(n, d)
    # 现在 x 的形状应该是 (n, d)，即 (2, 10)

    # 计算 h_dot_x: h 和 x 的点积，形状为 (n, L)
    h_dot_x = np.einsum('ijk,ik->ij', h, x, optimize=True)  # 修正为 'ijk,ik->ij'

    # 计算 stable_log_exp: log(1 + exp(-y * h_dot_x))，形状为 (n, L)
    log_exp_term = stable_log_exp(-y * h_dot_x)
    
    # 计算 term1: 所有节点和样本的损失平均值，标量
    term1 = np.sum(log_exp_term) / L
    
    # 计算正则化项: rho * x^2 / (1 + x^2)，标量
    x_squared = x**2
    term2 = np.sum(rho * x_squared / (1 + x_squared))
    
    return term1 #+ term2

# # 测试代码
# n, L, d = 2, 4000, 10
# x = np.random.randn(n, d)  # (2, 10)
# y = np.random.randn(n, L)  # (2, 4000)
# h = np.random.randn(n, L, d)  # (2, 4000, 10)
# loss = loss_compute(x, y, h)
# print("Loss:", loss)

def PushPull_with_batch_consensus(
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
    batch_size=None,
):
    """
    Push-Pull算法
    支持批次训练的分布式梯度下降
    """
    h_data, y_data = copy.deepcopy(h_data), copy.deepcopy(y_data)
    x = copy.deepcopy(init_x)
    n, d = x.shape

    # 初始化梯度（使用批次）
    g = grad_func(x, y_data, h_data, rho=rho, batch_size=batch_size) + sigma_n * np.random.normal(
        size=(n, d)
    )
    y = copy.deepcopy(g) # gradient tracking

    # 记录训练过程
    gradient_history_onfull = []
    consensus_history = []
    new_metric = []
    new_metric_onfull = []

    x_mean = np.mean(x, axis=0, keepdims=True)
    x_mean_expand = np.broadcast_to(x_mean, (n, d))
    _grad = grad_func(x_mean_expand, y_data, h_data, rho=rho, batch_size=None).reshape(
        x.shape
    )
    # 在整个训练数据集上计算梯度
    mean_grad = np.mean(_grad, axis=0, keepdims=True)
    gradient_history_onfull.append(np.linalg.norm(mean_grad))
    consensus_history.append(np.linalg.norm(x - x_mean_expand))

    # print("x.shape", x.shape)
    # print("y_data.shape", y_data.shape)
    # print("h_data.shape", h_data.shape)

    new_metric.append(loss_compute(x, y_data, h_data, rho=rho)/n)
    new_metric_onfull.append(loss_compute(x_mean_expand, y_data, h_data, rho=rho)/n)

    for _ in range(max_it):
        x = A @ x - lr * y
        g_new = grad_func(
            x, y_data, h_data, rho=rho, batch_size=batch_size
        ) + sigma_n * np.random.normal(size=(n, d))
        y = B @ y + g_new - g
        g = g_new

        # 记录平均梯度范数和参数范数
        x_mean = np.mean(x, axis=0, keepdims=True)
        x_mean_expand = np.broadcast_to(x_mean, (n, d))
        _grad = grad_func(x_mean_expand, y_data, h_data, rho=rho, batch_size=None).reshape(
            x.shape
        )
        # 在整个训练数据集上计算梯度
        mean_grad = np.mean(_grad, axis=0, keepdims=True)
        gradient_history_onfull.append(np.linalg.norm(mean_grad))
        consensus_history.append(np.linalg.norm(x - x_mean_expand))
        new_metric.append(loss_compute(x, y_data, h_data, rho=rho)/n)
        new_metric_onfull.append(loss_compute(x_mean_expand, y_data, h_data, rho=rho)/n)

    return pd.DataFrame(
        {
            "gradient_norm_on_full_trainset": gradient_history_onfull,
            "consensus_norm": consensus_history,
            "new_metric": new_metric,
            "new_metric_on_full_trainset": new_metric_onfull
        }
    )