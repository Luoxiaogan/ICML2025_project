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

import numpy as np
import networkx as nx

def generate_row_stochastic_matrix(seed, G):
    np.random.seed(seed)
    n = len(G.nodes)
    A = np.zeros((n, n))

    # Assign values to the adjacency matrix based on graph edges
    for i, j in G.edges():
        A[i, j] = np.random.choice([1, 2])  # Assign random values 1 or 2

    # Ensure self-loops are positive and random
    for i in range(n):
        A[i, i] = np.random.choice([1, 2])

    # Normalize each row to make it row stochastic (row sums to 1)
    row_sums = A.sum(axis=1, keepdims=True)
    A = A / row_sums

    return A

# Define the graph structure from the previous example
positions = {
    0: (0, 0),  1: (1, 2),  2: (3, 1),  3: (5, 4),
    4: (7, 3),  5: (8, 7),  6: (2, 6),  7: (4, 5),
    8: (6, 6),  9: (9, 1), 10: (11, 3), 11: (13, 2),
    12: (2, 9), 13: (6, 9), 14: (8, 10), 15: (12, 8)
}

# Create a directed graph
G = nx.DiGraph()

# Add nodes
for node in positions:
    G.add_node(node)

# Define k-nearest neighbors (k=3)
k = 3

# Compute edges based on Euclidean distance
for node in G.nodes():
    distances = {n: np.linalg.norm(
        [positions[node][0] - positions[n][0], positions[node][1] - positions[n][1]]
    ) for n in G.nodes() if n != node}
    
    # Get k nearest neighbors
    nearest_neighbors = sorted(distances, key=distances.get)[:k]
    
    # Add bidirectional edges and self-loops
    for neighbor in nearest_neighbors:
        G.add_edge(node, neighbor)
        G.add_edge(neighbor, node)
    G.add_edge(node, node)  # Self-loop

# Generate the row-stochastic matrix with a given seed
seed_value = 42

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n=16
A = generate_row_stochastic_matrix(seed_value, G)
k = 25
A = np.linalg.matrix_power(A, k)
show_row(A)
print(A.shape)

train(
    algorithm="PullDiag_GT",
    lr=7e-2,
    A=A,
    dataset_name="CIFAR10",
    batch_size=128,
    num_epochs=100,
    remark=f"MG={k}, 临近图16",
)