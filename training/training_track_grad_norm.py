# training/training_loop.py

import torch
import os
import torch.nn as nn
import pandas as pd
from datasets.prepare_data import get_dataloaders
from datasets.prepare_data import get_dataloaders_high_hetero_fixed_batch, get_dataloaders_fixed_batch
from utils.train_utils import get_first_batch, compute_loss_and_accuracy
from utils.train_utils import simple_compute_loss_and_accuracy
from training.optimizer import PullDiag_GT, PullDiag_GD, PushPull
from training.optimizer_push_pull_grad_norm_track import PushPull_grad_norm_track
from models.cnn import new_ResNet18
from models.fully_connected import FullyConnectedMNIST, SimpleFCN
from tqdm import tqdm
from datetime import datetime

def compute_normalized_avg_gradient_norm(model_list):
    norms = []
    for model in model_list:
        total_norm = 0
        num_params = 0
        for param in model.parameters():
            if param.grad is not None:
                total_norm += param.grad.data.norm(2) ** 2
                num_params += param.numel()
        total_norm = (total_norm ** 0.5) / (num_params ** 0.5)  # 归一化
        norms.append(total_norm.item())
    return sum(norms) / len(norms)

def compute_avg_gradient_matrix_norm(model_list):
    # 假设所有模型结构相同，取第一个模型的总参数量
    num_params = sum(p.numel() for p in model_list[0].parameters() if p.grad is not None)
    num_models = len(model_list)
    
    # 初始化一个张量来存储所有模型的梯度向量
    all_grads = torch.zeros(num_models, num_params)
    
    # 对每个模型拼接梯度向量
    for i, model in enumerate(model_list):
        grads = [param.grad.view(-1) for param in model.parameters() if param.grad is not None]
        if grads:  # 确保有梯度
            grad_vector = torch.cat(grads)
            all_grads[i] = grad_vector
    
    # 计算平均梯度向量
    avg_grad = all_grads.mean(dim=0)  # 按模型维度取平均
    
    # 计算平均梯度向量的 norm 并归一化
    avg_norm = avg_grad.norm(2) / (num_params ** 0.5)  # 归一化
    
    return avg_norm.item()

def train_track_grad_norm(
    algorithm: str,
    lr: float,  
    A: torch.Tensor,
    B: torch.Tensor,
    dataset_name: str,
    batch_size: int,
    num_epochs: int = 10,
    remark: str = "",
)-> pd.DataFrame:
    """
    执行训练过程。

    Args:
        algorithm (str): 算法名称 ("PushPull")
        lr (float): 学习率
        model_list (list): 模型列表
        A (torch.Tensor): 混合矩阵
        B (torch.Tensor): 混合矩阵
        dataloaders (list): 训练数据加载器列表
        test_dataloader (DataLoader): 测试数据加载器
        num_epochs (int): 训练轮数
        remark (str): 备注
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    n = A.shape[0]
    A = torch.from_numpy(A).float().to(device)
    B = torch.from_numpy(B).float().to(device)

    if dataset_name == "CIFAR10":
        model_list = [new_ResNet18().to(device) for _ in range(n)]
        trainloader_list, testloader, full_trainloader = get_dataloaders(
            n, dataset_name, batch_size, repeat=1
        )
        model_class = new_ResNet18
        #output_root = "/root/GanLuo/ICML2025_project/outputs/logs/CIFAR10_Multi_Gossip"
        output_root = "/home/lg/ICML2025_project/PUSHPULL_PROJECT/real_data_track_grad_norm/cifar10"
    elif dataset_name == "MNIST":
        model_list = [FullyConnectedMNIST().to(device) for _ in range(n)]
        trainloader_list, testloader, full_trainloader = get_dataloaders(
            n, dataset_name, batch_size, repeat=1
        )
        model_class = FullyConnectedMNIST
        #output_root = "/root/GanLuo/ICML2025_project/outputs/logs/MNIST"
        output_root = "/home/lg/ICML2025_project/PUSHPULL_PROJECT/real_data_track_grad_norm/new_mnist"
    
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
        grad_norm = compute_normalized_avg_gradient_norm(model_list)
        avg_grad_norm = compute_avg_gradient_matrix_norm(model_list)
        return total_loss / len(model_list), grad_norm, avg_grad_norm
    
    # 初始化优化器
    if algorithm == "PushPull":
        print("使用 PushPull 算法, 这里只记录grad_nrom")
        optimizer = PushPull_grad_norm_track(model_list, lr=lr, A=A, B=B, closure=closure)

        print("直接从第一次 closure() 调用中获取初始值")
        initial_loss, initial_grad_norm, initial_avg_grad_norm = closure()
        print(f"Initial grad norm: {initial_grad_norm}, Initial avg grad norm: {initial_avg_grad_norm}")
    else:   
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    
    print("optimizer初始化成功!")

    train_loss_history = []
    train_average_loss_history = []
    train_average_accuracy_history = []
    test_average_loss_history = []
    test_average_accuracy_history = []

    grad_norm_history = [initial_grad_norm]
    grad_norm_avg_history = [initial_avg_grad_norm]

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
            loss, grad_norm, avg_grad_norm = optimizer.step(closure=closure, lr=lr)
            train_loss += loss

            grad_norm_history.append(grad_norm)
            grad_norm_avg_history.append(avg_grad_norm)

        train_loss = train_loss / len(trainloader_list[0])
        train_loss_history.append(train_loss)

        # train_average_loss, train_accuracy, test_average_loss, test_accuracy, global_gradient_norm = compute_loss_and_accuracy(
        #     model_class=model_class, model_list=model_list, testloader=testloader, full_trainloader=full_trainloader
        # )
        test_average_loss, test_accuracy = simple_compute_loss_and_accuracy(model_class=model_class, model_list=model_list, testloader=testloader)
        # train_average_loss_history.append(train_average_loss)
        # train_average_accuracy_history.append(train_accuracy)
        test_average_loss_history.append(test_average_loss)
        test_average_accuracy_history.append(test_accuracy)
        # grad_norm_history.append(global_gradient_norm)

        progress_bar.set_postfix(
            epoch=epoch + 1,
            train_loss=f"{train_loss_history[-1]:.4f}",
            # train_average_accuracy=f"{100 * train_average_accuracy_history[-1]:.4f}%",
            test_loss=f"{test_average_loss_history[-1]:.4f}",
            test_accuracy=f"{100 * test_average_accuracy_history[-1]:.4f}%",
            # grad_norm=f"{global_gradient_norm:.4f}",
        )

        today_date = datetime.now().strftime("%Y-%m-%d")
        
        # 在每个 epoch 结束后保存数据到 CSV
        df = pd.DataFrame({
            "epoch": range(1, epoch + 2),  # epoch 从 1 开始
            "train_loss(total)": train_loss_history,
            # "train_loss(average)": train_average_loss_history,
            # "train_accuracy(average)": train_average_accuracy_history,
            "test_loss(average)": test_average_loss_history,
            "test_accuracy(average)": test_average_accuracy_history,
            # "global_gradient_norm(average)": grad_norm_history,
        })
        csv_filename = f"{remark}, {algorithm}, lr={lr}, n_nodes={n}, batch_size={batch_size}, {today_date}.csv"
        #csv_filename = f"{algorithm}, lr={lr}, n_nodes={n}, batch_size={batch_size}, {today_date}.csv"
        csv_path = os.path.join(output_root, csv_filename)
        df.to_csv(csv_path, index=False)

        df = pd.DataFrame({
            "grad_norm": grad_norm_history,
            "avg_grad_norm": grad_norm_avg_history,
        })
        csv_filename = f"grad_norm_{remark}, {algorithm}, lr={lr}, n_nodes={n}, batch_size={batch_size}, {today_date}.csv"
        csv_path = os.path.join(output_root, csv_filename)
        df.to_csv(csv_path, index=False)

    return df




def train_track_grad_norm_with_hetero(
    algorithm: str,
    lr: float,  
    A: torch.Tensor,
    B: torch.Tensor,
    dataset_name: str,
    batch_size: int,
    num_epochs: int = 10,
    remark: str = "",
    alpha: float = 0.5,
    root: str = None,
    use_hetero: bool = True,
    device = "cuda:0",
)-> pd.DataFrame:
    """
    执行训练过程。

    Lower alpha means higher heterogeneity

    Args:
        algorithm (str): 算法名称 ("PushPull")
        lr (float): 学习率
        model_list (list): 模型列表
        A (torch.Tensor): 混合矩阵
        B (torch.Tensor): 混合矩阵
        dataloaders (list): 训练数据加载器列表
        test_dataloader (DataLoader): 测试数据加载器
        num_epochs (int): 训练轮数
        remark (str): 备注
        alpha (float): 训练数据集的异质性参数
    """

    device = device
    criterion = nn.CrossEntropyLoss()
    n = A.shape[0]
    A = torch.from_numpy(A).float().to(device)
    B = torch.from_numpy(B).float().to(device)

    if use_hetero:
        print("使用异质性数据集")
        # 这里的 alpha 是异质性参数，值越小，异质性越高

        if dataset_name == "CIFAR10":
            model_list = [new_ResNet18().to(device) for _ in range(n)]
            trainloader_list, testloader, full_trainloader = get_dataloaders_high_hetero_fixed_batch(
                n, dataset_name, batch_size, repeat=1, alpha = alpha
            )
            model_class = new_ResNet18
            #output_root = "/root/GanLuo/ICML2025_project/outputs/logs/CIFAR10_Multi_Gossip"
            output_root = "/home/lg/ICML2025_project/PUSHPULL_PROJECT/real_data_track_grad_norm/cifar10"
            if root is not None:
                output_root = root
                print(f"root: {root}")
        elif dataset_name == "MNIST":
            model_list = [SimpleFCN().to(device) for _ in range(n)]
            trainloader_list, testloader, full_trainloader = get_dataloaders_high_hetero_fixed_batch(
                n, dataset_name, batch_size, repeat=1, alpha = alpha
            )
            model_class = SimpleFCN
            #output_root = "/root/GanLuo/ICML2025_project/outputs/logs/MNIST"
            output_root = "/home/lg/ICML2025_project/PUSHPULL_PROJECT/real_data_track_grad_norm/new_mnist"
            if root is not None:
                output_root = root
                print(f"root: {root}")
    else:
        print("使用同质性数据集")
        if dataset_name == "CIFAR10":
            model_list = [new_ResNet18().to(device) for _ in range(n)]
            trainloader_list, testloader, full_trainloader = get_dataloaders_fixed_batch(
                n, dataset_name, batch_size, repeat=1
            )
            model_class = new_ResNet18
            #output_root = "/root/GanLuo/ICML2025_project/outputs/logs/CIFAR10_Multi_Gossip"
            output_root = "/home/lg/ICML2025_project/PUSHPULL_PROJECT/real_data_track_grad_norm/cifar10"
            if root is not None:
                output_root = root
                print(f"root: {root}")
        elif dataset_name == "MNIST":
            model_list = [SimpleFCN().to(device) for _ in range(n)]
            trainloader_list, testloader, full_trainloader = get_dataloaders_fixed_batch(
                n, dataset_name, batch_size, repeat=1
            )
            model_class = SimpleFCN
            #output_root = "/root/GanLuo/ICML2025_project/outputs/logs/MNIST"
            output_root = "/home/lg/ICML2025_project/PUSHPULL_PROJECT/real_data_track_grad_norm/new_mnist"
            if root is not None:
                output_root = root
                print(f"root: {root}")
    
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
        grad_norm = compute_normalized_avg_gradient_norm(model_list)
        avg_grad_norm = compute_avg_gradient_matrix_norm(model_list)
        return total_loss / len(model_list), grad_norm, avg_grad_norm
    
    # 初始化优化器
    if algorithm == "PushPull":
        print("使用 PushPull 算法, 这里只记录grad_nrom")
        optimizer = PushPull_grad_norm_track(model_list, lr=lr, A=A, B=B, closure=closure)

        print("直接从第一次 closure() 调用中获取初始值")
        initial_loss, initial_grad_norm, initial_avg_grad_norm = closure()
        print(f"Initial grad norm: {initial_grad_norm}, Initial avg grad norm: {initial_avg_grad_norm}")
    else:   
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    
    print("optimizer初始化成功!")

    train_loss_history = []
    train_average_loss_history = []
    train_average_accuracy_history = []
    test_average_loss_history = []
    test_average_accuracy_history = []

    grad_norm_history = [initial_grad_norm]
    grad_norm_avg_history = [initial_avg_grad_norm]

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
            loss, grad_norm, avg_grad_norm = optimizer.step(closure=closure, lr=lr)
            train_loss += loss

            grad_norm_history.append(grad_norm)
            grad_norm_avg_history.append(avg_grad_norm)

        train_loss = train_loss / len(trainloader_list[0])
        train_loss_history.append(train_loss)

        # train_average_loss, train_accuracy, test_average_loss, test_accuracy, global_gradient_norm = compute_loss_and_accuracy(
        #     model_class=model_class, model_list=model_list, testloader=testloader, full_trainloader=full_trainloader
        # )
        test_average_loss, test_accuracy = simple_compute_loss_and_accuracy(model_class=model_class, model_list=model_list, testloader=testloader)
        # train_average_loss_history.append(train_average_loss)
        # train_average_accuracy_history.append(train_accuracy)
        test_average_loss_history.append(test_average_loss)
        test_average_accuracy_history.append(test_accuracy)
        # grad_norm_history.append(global_gradient_norm)

        progress_bar.set_postfix(
            epoch=epoch + 1,
            train_loss=f"{train_loss_history[-1]:.4f}",
            # train_average_accuracy=f"{100 * train_average_accuracy_history[-1]:.4f}%",
            test_loss=f"{test_average_loss_history[-1]:.4f}",
            test_accuracy=f"{100 * test_average_accuracy_history[-1]:.4f}%",
            # grad_norm=f"{global_gradient_norm:.4f}",
        )

        today_date = datetime.now().strftime("%Y-%m-%d")
        
        # 在每个 epoch 结束后保存数据到 CSV
        df = pd.DataFrame({
            "epoch": range(1, epoch + 2),  # epoch 从 1 开始
            "train_loss(total)": train_loss_history,
            # "train_loss(average)": train_average_loss_history,
            # "train_accuracy(average)": train_average_accuracy_history,
            "test_loss(average)": test_average_loss_history,
            "test_accuracy(average)": test_average_accuracy_history,
            # "global_gradient_norm(average)": grad_norm_history,
        })
        csv_filename = f"hetero={use_hetero}, alpha={alpha}, {algorithm}, lr={lr}, n_nodes={n}, batch_size={batch_size}, {today_date}.csv"
        #csv_filename = f"{algorithm}, lr={lr}, n_nodes={n}, batch_size={batch_size}, {today_date}.csv"
        csv_path = os.path.join(output_root, csv_filename)
        df.to_csv(csv_path, index=False)

        df = pd.DataFrame({
            "grad_norm": grad_norm_history,
            "avg_grad_norm": grad_norm_avg_history,
        })
        csv_filename = f"grad_norm,hetero={use_hetero}, alpha={alpha}, {algorithm}, lr={lr}, n_nodes={n}, batch_size={batch_size}, {today_date}.csv"
        csv_path = os.path.join(output_root, csv_filename)
        df.to_csv(csv_path, index=False)

    return df