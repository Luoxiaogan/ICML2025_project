import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# @Lg 难点在于训练时间和跑收敛。建议现在先把MNIST的跑起来，然后写CIFAR的代码。如果Cifar-10 实验在10个节点无法收敛，可以考虑构造5个节点的稀疏图比如di_ring(n=5)和 Row(get_xinmeng_matrix(n=5))
# 一、 4层神经网络训练MNIST数据集：
# 【1】异质性观察
# 图1：pulldiag在di_ring(n=5)+三种不同异质性的表现(注：现在只弄了两种异质性，均匀分布和完全异质分布。能不能弄一个稍微混合一点的数据分布？比如在完全异质分布条件下，让1，2号节点数据混合一下。）
# 图2：pullsum在di_ring(n=5)+三种不同异质性的表现
# 图3：pulldiag和pullsum都在di_ring(n=5)或者di_ring(n=10)+强异质性下的对比表现

# 【2】拓扑影响
# 图1：在row_and_col_mat(n=10, p=0.5）+强异质性条件下比较pulldiag, pullsum, frsd, frozen
# 图2：在row_and_col_mat(n=10, p=0.2）+强异质性条件下比较pulldiag, pullsum, frsd, frozen
# 图3：只看pullsum, 在row_and_col_mat(n=10, p=0.5），row_and_col_mat(n=10, p=0.2），di_ring(n=10)，grid_10()上的表现。


# 二、Resnet训练CIFAR-10数据

# 图1：di_ring(n=10)+强异质性条件下比较 pulldiag, pullsum, frsd, frozen
# 图2：grid_10()+强异质性条件下比较 pulldiag, pullsum, frsd, frozen


# @cxy: 合成数据的实验难点在于怎么让强异质性真正发挥作用。我的初步建议是提升问题的维度，比如令d=20，在此基础上提高sigma_h。如果不行，要等吴百濠学长的异质性代码。
# 一、【确定性情况+强异质性。对比pullsum 和其他算法（pulldiag, frsd, frozen）】

# 图1：n=20, row_and_col_mat(n=20, p=0.5）
# 图2：n=20, row_and_col_mat(n=20, p=0.2）
# 图3:  n=20, di_ring(n=20)
# 图4：三角网格图 或者我提供的 grid_20()


# 二、【噪声取1e-3, 1e-2, 1e-1 ，强异质性。对比pullsum 和其他算法（pulldiag, frsd, frozen）】

# 图1：n=20, row_and_col_mat(n=20, p=0.5）
# 图2：n=20, row_and_col_mat(n=20, p=0.2）
# 图3:  n=20, di_ring(n=20)
# 图4：三角网格图 或者我提供的 grid_20()


# 三、【噪声取1e-4，强异质性，只看pullsum】
# 图1：pull sum 在row_and_col_mat(n=20, p=0.5), row_and_col_mat(n=20, p=0.2）,di_ring(n=20),grid_20() 这些拓扑下的表现画在一张图内


# 下列所有代码会返回一个行随机矩阵和一个列随机矩阵。它们的对角元可以达到$2^{-n}$量级，$\kappa_\pi$为$n$的量级。


def row_and_col_mat(
    n=10, p=0.3, seed=None, show_graph=None
):  # 生成p-随机图，一般比较好。
    # 设置随机种子
    if seed is not None:
        np.random.seed(seed)

    # 生成强连通的随机有向图
    G = nx.gnp_random_graph(n, p, seed=seed, directed=True)

    # 确保图是强连通的
    while not nx.is_strongly_connected(G):
        G = nx.gnp_random_graph(n, p, directed=True)

    # 计算每个节点的入度和出度
    in_degrees = dict(G.in_degree())
    out_degrees = dict(G.out_degree())

    # 初始化权重矩阵
    A = np.zeros((n, n))
    B = np.zeros((n, n))
    # 给每条边赋予权重 1 / (目标节点的入度 + 1)
    for i, j in G.edges():
        A[i][j] = 1 / (in_degrees[j] + 1)
        B[i][j] = 1 / (out_degrees[i] + 1)
    # 为每个节点添加自环，并计算自环权重 1 / (入度 + 1)
    for j in range(n):
        A[j][j] = 1 / (in_degrees[j] + 1)
        B[j][j] = 1 / (out_degrees[j] + 1)
    if show_graph is not None:
        return A.T, B.T, G
    return A.T, B.T  # A.T是行随机，B.T是列随机矩阵，


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


def grid_10():  # 生成10节点稀疏网格图。
    A = np.eye(10) / 2
    A[0][1] = 1 / 2
    A[3][2] = 1 / 3
    A[2][2] = 1 / 3
    A[1][2] = 1 / 3
    A[2][7] = 1 / 2
    A[7][8] = 1 / 2
    A[8][9] = 1 / 2
    A[9][0] = 1 / 2
    A[7][6] = 1 / 2
    A[6][5] = 1 / 2
    A[4][3] = 1 / 2
    A[5][4] = 1 / 2
    B = np.eye(10) / 2
    B[0][1] = 1 / 2
    B[1][2] = 1 / 2
    B[2][7] = 1 / 2
    B[3][2] = 1 / 2
    B[4][3] = 1 / 2
    B[5][4] = 1 / 2
    B[6][5] = 1 / 2
    B[7][6] = 1 / 3
    B[7][7] = 1 / 3
    B[7][8] = 1 / 3
    B[8][9] = 1 / 2
    B[9][0] = 1 / 2
    return A.T, B.T  # A.T是行随机，B.T是列随机矩阵，


def ring2():  # 生成20节点稀疏网格图
    A = np.eye(20) / 2
    # 定义A矩阵的非对角线元素
    A[0][1] = 1 / 2
    A[1][2] = 1 / 3
    A[2][3] = 1 / 3
    A[3][4] = 1 / 3
    A[4][5] = 1 / 2
    A[5][6] = 1 / 2
    A[6][7] = 1 / 2
    A[7][8] = 1 / 2
    A[8][9] = 1 / 2
    A[9][10] = 1 / 2
    A[10][11] = 1 / 2
    A[11][12] = 1 / 2
    A[12][13] = 1 / 2
    A[13][14] = 1 / 2
    A[14][15] = 1 / 2
    A[15][16] = 1 / 2
    A[16][17] = 1 / 2
    A[17][18] = 1 / 2
    A[18][19] = 1 / 2
    A[19][0] = 1 / 2

    # 再添加一些其他稀疏的连接
    A[2][10] = 1 / 2
    A[10][15] = 1 / 2
    # A[15][5] = 1/2

    B = np.eye(20) / 2
    # 定义B矩阵的非对角线元素
    B[0][1] = 1 / 2
    B[1][2] = 1 / 2
    B[2][3] = 1 / 2
    B[3][4] = 1 / 2
    B[4][5] = 1 / 2
    B[5][6] = 1 / 2
    B[6][7] = 1 / 2
    B[7][8] = 1 / 3
    B[8][9] = 1 / 3
    B[9][10] = 1 / 3
    B[10][11] = 1 / 2
    B[11][12] = 1 / 2
    B[12][13] = 1 / 2
    B[13][14] = 1 / 2
    B[14][15] = 1 / 2
    B[15][16] = 1 / 2
    B[16][17] = 1 / 2
    B[17][18] = 1 / 2
    B[18][19] = 1 / 2
    B[19][0] = 1 / 2

    # 再添加一些其他稀疏的连接
    B[5][10] = 1 / 3
    B[10][15] = 1 / 3
    B[15][5] = 1 / 3

    return Row(A.T), Col(B.T)


def ring3():  # 生成20节点稀疏网格图
    A = np.eye(20) / 2
    # 定义A矩阵的非对角线元素
    A[0][1] = 1 / 2
    A[1][2] = 1 / 3
    A[2][3] = 1 / 3
    A[3][4] = 1 / 3
    A[4][5] = 1 / 2
    A[5][6] = 1 / 2
    A[6][7] = 1 / 2
    A[7][8] = 1 / 2
    A[8][9] = 1 / 2
    A[9][10] = 1 / 2
    A[10][11] = 1 / 2
    A[11][12] = 1 / 2
    A[12][13] = 1 / 2
    A[13][14] = 1 / 2
    A[14][15] = 1 / 2
    A[15][16] = 1 / 2
    A[16][17] = 1 / 2
    A[17][18] = 1 / 2
    A[18][19] = 1 / 2
    A[19][0] = 1 / 2

    # 再添加一些其他稀疏的连接
    A[2][10] = 1 / 2
    A[10][15] = 1 / 2
    A[15][2] = 1 / 2

    B = np.eye(20) / 2
    # 定义B矩阵的非对角线元素
    B[0][1] = 1 / 2
    B[1][2] = 1 / 2
    B[2][3] = 1 / 2
    B[3][4] = 1 / 2
    B[4][5] = 1 / 2
    B[5][6] = 1 / 2
    B[6][7] = 1 / 2
    B[7][8] = 1 / 3
    B[8][9] = 1 / 3
    B[9][10] = 1 / 3
    B[10][11] = 1 / 2
    B[11][12] = 1 / 2
    B[12][13] = 1 / 2
    B[13][14] = 1 / 2
    B[14][15] = 1 / 2
    B[15][16] = 1 / 2
    B[16][17] = 1 / 2
    B[17][18] = 1 / 2
    B[18][19] = 1 / 2
    B[19][0] = 1 / 2

    # 再添加一些其他稀疏的连接
    B[5][10] = 1 / 3
    B[10][15] = 1 / 3
    B[15][5] = 1 / 3

    return Row(A.T), Col(B.T)


def ring4():  # 生成20节点稀疏网格图
    A = np.eye(20) / 2
    # 定义A矩阵的非对角线元素
    A[0][1] = 1 / 2
    A[1][2] = 1 / 3
    A[2][3] = 1 / 3
    A[3][4] = 1 / 3
    A[4][5] = 1 / 2
    A[5][6] = 1 / 2
    A[6][7] = 1 / 2
    A[7][8] = 1 / 2
    A[8][9] = 1 / 2
    A[9][10] = 1 / 2
    A[10][11] = 1 / 2
    A[11][12] = 1 / 2
    A[12][13] = 1 / 2
    A[13][14] = 1 / 2
    A[14][15] = 1 / 2
    A[15][16] = 1 / 2
    A[16][17] = 1 / 2
    A[17][18] = 1 / 2
    A[18][19] = 1 / 2
    A[19][0] = 1 / 2

    # 再添加一些其他稀疏的连接
    A[0][5] = 1 / 2
    A[10][15] = 1 / 2
    A[5][10] = 1 / 2
    A[15][0] = 1 / 3

    B = np.eye(20) / 2
    # 定义B矩阵的非对角线元素
    B[0][1] = 1 / 2
    B[1][2] = 1 / 2
    B[2][3] = 1 / 2
    B[3][4] = 1 / 2
    B[4][5] = 1 / 2
    B[5][6] = 1 / 2
    B[6][7] = 1 / 2
    B[7][8] = 1 / 3
    B[8][9] = 1 / 3
    B[9][10] = 1 / 3
    B[10][11] = 1 / 2
    B[11][12] = 1 / 2
    B[12][13] = 1 / 2
    B[13][14] = 1 / 2
    B[14][15] = 1 / 2
    B[15][16] = 1 / 2
    B[16][17] = 1 / 2
    B[17][18] = 1 / 2
    B[18][19] = 1 / 2
    B[19][0] = 1 / 2

    # 再添加一些其他稀疏的连接
    B[5][10] = 1 / 3
    B[10][15] = 1 / 3
    B[15][5] = 1 / 3

    return Row(A.T), Col(B.T)


def Row(matrix):
    # 计算每一行的和
    M = matrix.copy()
    row_sums = np.sum(M, axis=1)

    # 将每一行除以该行的和
    for i in range(M.shape[0]):
        M[i, :] /= row_sums[i]

    return M


def Col(matrix):
    W = matrix.copy()
    # 计算每一列的和
    col_sums = np.sum(W, axis=0)

    # 将每一列除以该列的和
    for i in range(W.shape[0]):
        W[:, i] /= col_sums[i]

    return W


def get_xinmeng_matrix(n=5):
    M = np.zeros((n, n))

    # 主对角线上的元素
    M[np.diag_indices(n)] = 1 / 3 * np.ones(n)
    M[n - 1, n - 1] = M[n - 1, n - 1] + 1 / 3

    # 次对角线上的元素
    for i in range(n - 1):
        M[i + 1, i] = M[i + 1, i] + 1 / 3

    # 第一行上的元素
    M[0, :] = M[0, :] + 1 / 3

    return M


import numpy as np
#======================== n=10 稀疏环图 ========================#
def ring4_node_10():
    n = 10
    A = np.eye(n) / 2  # 初始化对角线为1/2
    
    # 基础环状连接 (部分权重不同)
    A[0][1] = 1/2
    A[1][2] = 1/3
    A[2][3] = 1/3
    A[3][4] = 1/3
    A[4][5] = 1/2
    
    # 后续节点保持1/2权重
    for i in range(5, n-1):
        A[i][i+1] = 1/2
    A[n-1][0] = 1/2  # 闭环
    
    # 添加2条跨步连接 (步长=5)
    A[0][5] = 1/2
    A[5][0] = 1/2
    
    return Row(A.T)

#======================== n=100 稀疏环图 ========================#
def ring4_node_100():
    n = 100
    A = np.eye(n) / 2  # 初始化对角线为1/2
    
    # 基础环状连接 (部分权重不同)
    A[0][1] = 1/2
    A[1][2] = 1/3
    A[2][3] = 1/3
    A[3][4] = 1/3
    A[4][5] = 1/2
    
    # 后续节点保持1/2权重
    for i in range(5, n):
        next_node = (i + 1) % n
        A[i][next_node] = 1/2
    
    # 添加25条跨步连接 (步长=4)
    step = 4
    for i in range(0, n, step):
        j = (i + step) % n
        A[i][j] = 1/2
    
    return Row(A.T)


import numpy as np

def generate_exponential_weight_matrix(n):
    """
    生成静态指数图的权重矩阵。

    参数：
    n (int): 图中节点的数量。

    返回：
    numpy.ndarray: 形状为(n, n)的权重矩阵，元素类型为float。
    """
    if n < 1:
        raise ValueError("n必须为正整数")
    
    # 计算分母：|log2(n)| + 1
    denominator = np.abs(np.log2(n)) + 1
    
    # 创建索引网格
    i_indices, j_indices = np.indices((n, n))
    
    # 计算mod_val = (j - i) mod n
    mod_vals = (j_indices - i_indices) % n
    
    # 判断是否为2的幂
    is_power_of_two = (mod_vals != 0) & ((mod_vals & (mod_vals - 1)) == 0)
    
    # 判断是否为自环或mod_val是2的幂
    mask = (i_indices == j_indices) | is_power_of_two
    
    # 生成权重矩阵
    weight_matrix = np.where(mask, 1.0 / denominator, 0.0)
    
    return weight_matrix

def get_matrixs_from_exp_graph(n, seed=42):

    original_matrix = generate_exponential_weight_matrix(n)
    np.random.seed(seed)
    random_matrix = np.where(original_matrix != 0, np.random.randint(1, 10, size=original_matrix.shape), 0)

    random_matrix = np.array(random_matrix)

    M = random_matrix.copy().astype(float)
    
    A = Row(M)
    B = Col(M)
    
    return A, B


# 几何图

import numpy as np
import networkx as nx

def generate_geometric_graph(seed=48):

    """
    A = generate_geometric_graph(seed=48)
    是我们使用的几何图

    B矩阵就随机生成一个就好了, 反正没有用
    """
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

    B = np.eye(16)

    return A,B



import numpy as np
import networkx as nx

def generate_nearest_nightbor_graph(seed=42):
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

    B = np.eye(16)

    return A,B


def grid_16(seed=1):
    """
    生成4x4网格图的行随机矩阵，包含自环和随机权重。
    参数:
        seed (int): 控制随机权重生成的种子
    返回:
        A (np.ndarray): 行随机矩阵，shape=(16,16)
    """
    np.random.seed(seed)
    n = 16
    A = np.zeros((n, n))
    
    # 生成4x4网格的拓扑结构（含自环）
    for row in range(4):
        for col in range(4):
            node = row * 4 + col  # 当前节点编号（0~15）
            
            # 添加自环权重（1~10）
            A[node, node] = np.random.randint(1, 11)
            
            # 横向连接（右邻居）
            if col < 3:
                right_neighbor = node + 1
                A[node, right_neighbor] = np.random.randint(1, 11)
                A[right_neighbor, node] = np.random.randint(1, 11)  # 反向连接
            
            # 纵向连接（下邻居）
            if row < 3:
                down_neighbor = node + 4
                A[node, down_neighbor] = np.random.randint(1, 11)
                A[down_neighbor, node] = np.random.randint(1, 11)  # 反向连接
    
    # 行归一化
    A = Row(A)

    B = np.eye(16)  # 列随机矩阵，单位矩阵
    return A,B