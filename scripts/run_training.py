# scripts/run_training.py

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

from training import train, train_per_iteration
import torch
from utils import ring1, show_row 

# import numpy as np
# import networkx as nx

# def generate_row_stochastic_matrix(seed, G):
#     np.random.seed(seed)
#     n = len(G.nodes)
#     A = np.zeros((n, n))

#     # Assign values to the adjacency matrix based on graph edges
#     for i, j in G.edges():
#         A[i, j] = np.random.choice([1, 2])  # Assign random values 1 or 2

#     # Ensure self-loops are positive and random
#     for i in range(n):
#         A[i, i] = np.random.choice([1, 2])

#     # Normalize each row to make it row stochastic (row sums to 1)
#     row_sums = A.sum(axis=1, keepdims=True)
#     A = A / row_sums

#     return A

# # Define the graph structure from the previous example
# positions = {
#     0: (0, 0),  1: (1, 2),  2: (3, 1),  3: (5, 4),
#     4: (7, 3),  5: (8, 7),  6: (2, 6),  7: (4, 5),
#     8: (6, 6),  9: (9, 1), 10: (11, 3), 11: (13, 2),
#     12: (2, 9), 13: (6, 9), 14: (8, 10), 15: (12, 8)
# }

# # Create a directed graph
# G = nx.DiGraph()

# # Add nodes
# for node in positions:
#     G.add_node(node)

# # Define k-nearest neighbors (k=3)
# k = 3

# # Compute edges based on Euclidean distance
# for node in G.nodes():
#     distances = {n: np.linalg.norm(
#         [positions[node][0] - positions[n][0], positions[node][1] - positions[n][1]]
#     ) for n in G.nodes() if n != node}
    
#     # Get k nearest neighbors
#     nearest_neighbors = sorted(distances, key=distances.get)[:k]
    
#     # Add bidirectional edges and self-loops
#     for neighbor in nearest_neighbors:
#         G.add_edge(node, neighbor)
#         G.add_edge(neighbor, node)
#     G.add_edge(node, node)  # Self-loop


# # Generate the row-stochastic matrix with a given seed
# seed_value = 42


#几何图
import numpy as np
import networkx as nx

def generate_row_stochastic_matrix_with_self_loops(seed=42):
    np.random.seed(seed)
    
    # 定义16个点的坐标
    positions = np.array([
        [1,2], [3,5], [5,1], [6,6], [8,3], [2,8], [9,9], [4,4],
        [7,2], [1,7], [6,9], [3,1], [9,5], [5,5], [7,7], [2,3]
    ])

    # 创建图
    G = nx.Graph()
    for i in range(len(positions)):
        G.add_node(i, pos=positions[i])

    # 连接阈值距离内的点
    threshold = 3
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            dist = np.linalg.norm(positions[i] - positions[j])
            if dist <= threshold:
                G.add_edge(i, j)

    # 初始化邻接矩阵
    A = np.zeros((len(positions), len(positions)))

    # 赋值非负值 1 或 2 给边
    for i, j in G.edges():
        value = np.random.choice([1, 2])
        A[i, j] = value
        A[j, i] = value  # 确保矩阵对称

    # 为对角线元素赋值1或2，确保自环
    for i in range(len(positions)):
        A[i, i] = np.random.choice([1, 2])

    # 归一化每行，使其成为行随机矩阵
    row_sums = A.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # 防止除零错误
    A /= row_sums

    return A


# import numpy as np

# def Row(matrix):
#     """行归一化函数，确保每行和为1"""
#     M = matrix.copy()
#     row_sums = np.sum(M, axis=1)
#     for i in range(M.shape[0]):
#         M[i, :] /= row_sums[i]
#     return M

# def grid_16(seed=None):
#     """
#     生成4x4网格图的行随机矩阵，包含自环和随机权重。
#     参数:
#         seed (int): 控制随机权重生成的种子
#     返回:
#         A (np.ndarray): 行随机矩阵，shape=(16,16)
#     """
#     np.random.seed(seed)
#     n = 16
#     A = np.zeros((n, n))
    
#     # 生成4x4网格的拓扑结构（含自环）
#     for row in range(4):
#         for col in range(4):
#             node = row * 4 + col  # 当前节点编号（0~15）
            
#             # 添加自环权重（1~10）
#             A[node, node] = np.random.randint(1, 11)
            
#             # 横向连接（右邻居）
#             if col < 3:
#                 right_neighbor = node + 1
#                 A[node, right_neighbor] = np.random.randint(1, 11)
#                 A[right_neighbor, node] = np.random.randint(1, 11)  # 反向连接
            
#             # 纵向连接（下邻居）
#             if row < 3:
#                 down_neighbor = node + 4
#                 A[node, down_neighbor] = np.random.randint(1, 11)
#                 A[down_neighbor, node] = np.random.randint(1, 11)  # 反向连接
    
#     # 行归一化
#     A = Row(A)
#     return A

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n=16
#A, B = ring1(n)
A = generate_row_stochastic_matrix_with_self_loops(seed=48)
k = 10
A = np.linalg.matrix_power(A, k)
show_row(A)
print(A.shape)

train(
    algorithm="PullDiag_GT",
    lr=7e-2,
    A=A,
    dataset_name="CIFAR10",
    batch_size=128,
    num_epochs=200,
    remark=f"MG={k}, 几何图16",
)