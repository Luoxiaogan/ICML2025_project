# utils/train_utils.py

import torch
import torch.nn as nn
import itertools
from torch.cuda.amp import autocast
from typing import Tuple

def get_first_batch(trainloader_list: list):
    h_data_train = []
    y_data_train = []

    # 遍历每个 trainloader
    for trainloader in trainloader_list:
        # 使用 tee 复制迭代器，不改变原始的迭代器
        loader_copy, trainloader = itertools.tee(trainloader, 2)

        # 从复制的迭代器中取第一个批次的数据
        first_batch = next(iter(loader_copy))

        # 分别保存 X 和 y
        h_data_train.append(first_batch[0])  # inputs (X)
        y_data_train.append(first_batch[1])  # labels (y)

    return h_data_train, y_data_train

def compute_normalized_global_gradient_norm(model):
    total_norm = 0.0
    total_elements = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(2)  # 单参数梯度L2范数
            total_norm += param_norm.item() ** 2       # 平方累加
            total_elements += p.grad.numel()           # 累加元素数目
    total_norm = total_norm ** 0.5                     # 全局L2范数
    return total_norm

def compute_loss_and_accuracy(
    model_class, model_list, testloader, full_trainloader, use_amp=False, train_dataset=None
) -> Tuple[float, float, float, float]:
    # 使用 CrossEntropyLoss 作为默认损失函数
    criterion = nn.CrossEntropyLoss()

    # 确保模型在正确的设备上
    device = next(model_list[0].parameters()).device

    # Step 1: Compute the average of the parameters from all models
    avg_model = model_class().to(device)  # 创建新的模型实例，并将其移动到同一设备上
    avg_state_dict = avg_model.state_dict()  # 获取新模型的状态字典

    # 初始化 sum_state_dict
    sum_state_dict = {
        key: torch.zeros_like(param).to(device) for key, param in avg_state_dict.items()
    }

    # 汇总所有模型的参数
    for model in model_list:
        state_dict = model.state_dict()
        for key in sum_state_dict.keys():
            sum_state_dict[key] += state_dict[key].to(device)

    # 计算平均值
    num_models = len(model_list)
    avg_state_dict = {key: value / num_models for key, value in sum_state_dict.items()}

    # 将平均参数加载到新模型中
    avg_model.load_state_dict(avg_state_dict)

    # Step 2: Evaluate the new model's loss and accuracy on the training set
    avg_model.eval()
    train_correct = 0
    train_total = 0
    train_total_loss = 0.0

    with torch.no_grad():
        for inputs, labels in full_trainloader:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(
                device, non_blocking=True
            )

            with autocast(enabled=use_amp):
                # 前向传播
                outputs = avg_model(inputs)
                loss = criterion(outputs, labels)

            # 汇总损失
            train_total_loss += loss.item()

            # 计算准确率
            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)

    # 计算训练集的平均损失和准确率
    train_average_loss = train_total_loss / len(full_trainloader)
    train_accuracy = train_correct / train_total

    # Step 3: Evaluate the new model's loss and accuracy on the test set
    test_correct = 0
    test_total = 0
    test_total_loss = 0.0

    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(
                device, non_blocking=True
            )

            with autocast(enabled=use_amp):
                # 前向传播
                outputs = avg_model(inputs)
                loss = criterion(outputs, labels)

            # 汇总损失
            test_total_loss += loss.item()

            # 计算准确率
            _, predicted = torch.max(outputs, 1)
            test_correct += (predicted == labels).sum().item()
            test_total += labels.size(0)

    # 计算测试集的平均损失和准确率
    test_average_loss = test_total_loss / len(testloader)
    test_accuracy = test_correct / test_total

    grad_norm = compute_full_gradient_norm(avg_model, train_dataset, criterion, batch_size=None, device="cuda")

    return (
        train_average_loss,
        train_accuracy,
        test_average_loss,
        test_accuracy,
        grad_norm,
    )

import copy
import torch
from torch.utils.data import DataLoader

def compute_full_gradient_norm(model, train_dataset, criterion, batch_size=None, device="cuda"):
    """
    计算模型在整个训练集上的梯度F范数（所有参数梯度元素的平方和开根号）
    
    参数:
        model: 原始模型（不会被修改）
        train_dataset: 训练集数据集
        criterion: 损失函数
        batch_size: 如果为None则使用全量数据，否则使用指定batch_size（内存不足时使用）
        device: 计算设备
    
    返回:
        float: 全局梯度范数（Frobenius范数）
    """
    # 深拷贝模型以避免影响原始模型
    model_copy = copy.deepcopy(model).to(device)
    model_copy.train()
    
    # 自动确定batch_size（全量数据或用户指定）
    use_full_batch = batch_size is None
    effective_bs = len(train_dataset) if use_full_batch else batch_size
    
    # 创建数据加载器（全量数据时关闭shuffle和drop_last）
    loader = DataLoader(train_dataset,
                        batch_size=effective_bs,
                        shuffle=False,
                        drop_last=False,
                        num_workers=12,
                        pin_memory=True)
    
    # 初始化梯度缓冲区
    for param in model_copy.parameters():
        param.grad = torch.zeros_like(param.data)
    
    total_samples = 0  # 已处理样本计数
    for inputs, labels in loader:
        # 将数据转移到设备
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        # 展平图像（适配全连接网络）
        batch_size = inputs.shape[0]
        inputs = inputs.view(batch_size, -1)
        
        # 前向传播
        outputs = model_copy(inputs)
        loss = criterion(outputs, labels)
        
        # 反向传播（计算梯度并累加）
        loss.backward()
        
        # 记录已处理样本数
        total_samples += batch_size
    
    # 验证数据完整性
    assert total_samples == len(train_dataset), "数据存在缺失"
    
    # 计算全局梯度范数（所有参数梯度的平方和开根号）
    total_norm_sq = 0.0
    for param in model_copy.parameters():
        if param.grad is not None:
            grad = param.grad.data
            total_norm_sq += torch.sum(grad ** 2).item()
    
    return total_norm_sq ** 0.5