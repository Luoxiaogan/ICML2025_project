# training/training_loop.py

import torch
import os
import torch.nn as nn
import pandas as pd
from datasets.prepare_data import get_dataloaders
from utils.train_utils import get_first_batch, compute_loss_and_accuracy
from training.optimizer import PullDiag_GT, PullDiag_GD
from models.cnn import new_ResNet18
from models.fully_connected import FullyConnectedMNIST, two_layer_fc
from tqdm import tqdm
from datetime import datetime
from typing import Tuple

def train(
    algorithm: str,
    lr: float,  
    A: torch.Tensor,  
    dataset_name: str,
    batch_size: int,
    num_epochs: int = 10,
    remark: str = "",
)-> Tuple[list, list, list]:
    """
    执行训练过程。

    Args:
        algorithm (str): 算法名称 ('PullDiag_GT' 或 'PullDiag_GD')
        lr (float): 学习率
        model_list (list): 模型列表
        A (torch.Tensor): 混合矩阵
        dataloaders (list): 训练数据加载器列表
        test_dataloader (DataLoader): 测试数据加载器
        num_epochs (int): 训练轮数
        remark (str): 备注
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    n = A.shape[0]
    A.to(device)

    if dataset_name == "CIFAR10":
        model_list = [new_ResNet18().to(device) for _ in range(n)]
        trainloader_list, testloader, full_trainloader = get_dataloaders(
            n, dataset_name, batch_size
        )
        model_class = new_ResNet18
        output_root = "/root/GanLuo/ICML2025_project/outputs/logs/CIFAR10"
    elif dataset_name == "MNIST":
        model_list = [FullyConnectedMNIST().to(device) for _ in range(n)]
        trainloader_list, testloader, full_trainloader = get_dataloaders(
            n, dataset_name, batch_size
        )
        model_class = FullyConnectedMNIST
        #output_root = "/root/GanLuo/ICML2025_project/outputs/logs/MNIST"
        output_root = "/root/GanLuo/ICML2025_project/outputs/Multi_Gossip_test"
    
    torch.backends.cudnn.benchmark = True

    h_data_train, y_data_train = get_first_batch(trainloader_list)
    h_data_train = [
        tensor.to(device, non_blocking=True) for tensor in h_data_train
    ]  # [tensor.to(device) for tensor in h_data_train]
    y_data_train = [
        tensor.to(device, non_blocking=True) for tensor in y_data_train
    ]  # [tensor.to(device) for tensor in y_data_train]

    def closure():
        total_loss = 0
        for i, model in enumerate(model_list):
            for param in model.parameters():
                param.requires_grad = True
            model.zero_grad()
            output = model(h_data_train[i])
            loss = criterion(output, y_data_train[i])
            loss.backward()
            total_loss += loss.item()
        return total_loss / len(model_list)
    
    # 初始化优化器
    if algorithm == "PullDiag_GT":
        optimizer = PullDiag_GT(model_list, lr=lr, A=A, closure=closure)
    elif algorithm == "PullDiag_GD":
        optimizer = PullDiag_GD(model_list, lr=lr, A=A, closure=closure)
    else:   
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    
    print("optimizer初始化成功!")

    train_loss_history = []
    train_average_loss_history = []
    train_average_accuracy_history = []
    test_average_loss_history = []
    test_average_accuracy_history = []

    progress_bar = tqdm(range(num_epochs), desc="Training Progress")

    for epoch in progress_bar:
        train_loss = 0.0

        for batch_idx, batch in enumerate(zip(*trainloader_list)):
            inputs = [
                data[0].to(device, non_blocking=True) for data in batch
            ]  # [data[0] for data in batch]
            labels = [
                data[1].to(device, non_blocking=True) for data in batch
            ]  # [data[1] for data in batch]
            h_data_train = inputs  # [tensor.to(device) for tensor in inputs]
            y_data_train = labels  # [tensor.to(device) for tensor in labels]
            loss = optimizer.step(closure=closure, lr=lr)
            train_loss += loss
        train_loss = train_loss / len(trainloader_list[0])
        train_loss_history.append(train_loss)

        train_average_loss, train_accuracy, test_average_loss, test_accuracy = compute_loss_and_accuracy(
            model_class=model_class, model_list=model_list, testloader=testloader, full_trainloader=full_trainloader
        )
        train_average_loss_history.append(train_average_loss)
        train_average_accuracy_history.append(train_accuracy)
        test_average_loss_history.append(test_average_loss)
        test_average_accuracy_history.append(test_accuracy)

        progress_bar.set_postfix(
            epoch=epoch + 1,
            train_loss=f"{train_loss_history[-1]:.4f}",
            train_average_accuracy=f"{100 * train_average_accuracy_history[-1]:.4f}%",
            test_loss=f"{test_average_loss_history[-1]:.4f}",
            test_accuracy=f"{100 * test_average_accuracy_history[-1]:.4f}%",
        )

        today_date = datetime.now().strftime("%Y-%m-%d")
        
        # 在每个 epoch 结束后保存数据到 CSV
        df = pd.DataFrame({
            "epoch": range(1, epoch + 2),  # epoch 从 1 开始
            "train_loss(total)": train_loss_history,
            "train_loss(average)": train_average_loss_history,
            "train_accuracy(average)": train_average_accuracy_history,
            "test_loss(average)": test_average_loss_history,
            "test_accuracy(average)": test_average_accuracy_history,
        })
        csv_filename = f"{remark}, {algorithm}, lr={lr}, n_nodes={n}, batch_size={batch_size}, {today_date}.csv"
        #csv_filename = f"{algorithm}, lr={lr}, n_nodes={n}, batch_size={batch_size}, {today_date}.csv"
        csv_path = os.path.join(output_root, csv_filename)
        df.to_csv(csv_path, index=False)

    return df