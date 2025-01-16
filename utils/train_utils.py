# utils/train_utils.py

import torch
import torch.nn as nn
import itertools
from torch.cuda.amp import autocast

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

def compute_test_loss_and_accuracy(model_class, model_list, testloader, use_amp=False):
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

    # Step 2: Evaluate the new model's loss and accuracy using test_loader
    avg_model.eval()
    correct = 0
    total = 0
    total_loss = 0.0

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
            total_loss += loss.item()

            # 计算准确率
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    # 计算最终的平均损失和准确率
    average_loss = total_loss / (len(testloader))  # 两次标准化
    accuracy = correct / total

    return average_loss, accuracy