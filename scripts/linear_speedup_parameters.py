# scripts/run_training.py

import sys
import os

# 将项目根目录添加到 sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from training import train, train_per_iteration
import torch
from utils import ring1, ring2, ring3, ring4, show_row

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 代码示例

n=5
A, B = ring1(n=n)
show_row(A)

train_per_iteration(
    algorithm="PullDiag_GT",
    lr=1e-2,
    A=A,
    dataset_name="MNIST",
    batch_size=500,
    num_epochs=10,
    remark="linear_speedup",
)

# 首先，可以根据train函数来得到在取定的n, bs, A下最优的lr
# 然后，可以根据train_per_iteration函数来得到逐iteration的结果

# 训练数据集有50000
# n=5,10,20,40
# bs=500
# 此时, 每个节点per_epoch执行的batch次数是50000//(n*bs)=100//n, 即 20,10,5,2

# 如果 bs = 500
# n = 2,4,8,16,32
# 此时, 每个节点per_epoch执行的batch次数是50000//(n*bs)=100//n, 即 50,25,12,6,3 

# 难绷, 如果是为了训练的方便, 可以考虑先把50000张图片重复8次, 得到400000张图片
# n = 2,4,8,16,32
# 此时, 每个节点per_epoch执行的batch次数是400000//(n*bs)=800//n, 即 400,200,100,50,25

# 最优的lr
# n=5, bs=500, opt_lr=1e-1
# n=10, bs=500, opt_lr=8e-2
# n=20, bs=500, opt_lr=1e-2(小于1e-2即可)

# 可能对于linear_speedup的实验来说，其实最优的学习率并不是在最小的iteration数目下达到最高的测试集正确率.
# 此时学习率其实并不重要, 重要的是需要 grad_norm 可以收敛到噪声的.