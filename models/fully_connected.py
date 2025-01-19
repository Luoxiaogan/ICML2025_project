# models/fully_connected.py

import torch.nn as nn
import torch.nn.functional as F

class FullyConnectedMNIST(nn.Module):
    def __init__(self, p=0.5):
        super(FullyConnectedMNIST, self).__init__()
        self.fc1 = nn.Linear(784, 256)           # 增加第一层神经元数量
        self.bn1 = nn.BatchNorm1d(256)           # 批归一化
        self.fc2 = nn.Linear(256, 128)           # 增加第二层
        self.bn2 = nn.BatchNorm1d(128)           # 批归一化
        self.fc3 = nn.Linear(128, 64)            # 增加第三层
        self.bn3 = nn.BatchNorm1d(64)            # 批归一化
        self.fc4 = nn.Linear(64, 10)             # 输出层
        self.dropout = nn.Dropout(p=p)           # Dropout 层

    def forward(self, x):
        x = x.view(x.size(0), -1)                # 展平成一维向量
        x = F.relu(self.bn1(self.fc1(x)))        # 第一层
        x = self.dropout(x)                      # Dropout
        x = F.relu(self.bn2(self.fc2(x)))        # 第二层
        x = self.dropout(x)                      # Dropout
        x = F.relu(self.bn3(self.fc3(x)))        # 第三层
        x = self.dropout(x)                      # Dropout
        x = self.fc4(x)                          # 输出层
        return x

class two_layer_fc(nn.Module):
    def __init__(self, p=0.5):
        super(two_layer_fc, self).__init__()
        self.fc1 = nn.Linear(784, 128)           # 第一层
        self.bn1 = nn.BatchNorm1d(128)           # 批归一化
        self.fc2 = nn.Linear(128, 64)            # 第二层
        self.bn2 = nn.BatchNorm1d(64)            # 批归一化
        self.fc3 = nn.Linear(64, 10)             # 输出层
        self.dropout = nn.Dropout(p=p)           # Dropout 层

    def forward(self, x):
        x = x.view(x.size(0), -1)                # 展平成一维向量
        x = F.relu(self.bn1(self.fc1(x)))        # 第一层
        x = self.dropout(x)                      # Dropout
        x = F.relu(self.bn2(self.fc2(x)))        # 第二层
        x = self.dropout(x)                      # Dropout
        x = self.fc3(x)                          # 输出层
        return x