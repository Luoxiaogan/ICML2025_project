# datasets/prepare_data.py

import torch
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import numpy as np
import random
from typing import Tuple

# MNIST transforms
MNIST_transform_train = transforms.Compose(
    [
        transforms.RandomCrop(28, padding=4),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ]
)

MNIST_transform_test = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ]
)

# CIFAR-10 transforms
cifar10_transform_train = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

cifar10_transform_test = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)


def get_dataloaders(
    n: int, dataset_name: str, batch_size: int
) -> Tuple[list, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    generator = torch.Generator()
    generator.manual_seed(seed)

    if dataset_name == "CIFAR10":
        transform_train, transform_test = (
            cifar10_transform_train,
            cifar10_transform_test,
        )
        trainset = torchvision.datasets.CIFAR10(
            root="/root/GanLuo/ICML2025_project/data/raw/CIFAR10",
            train=True,
            download=False,
            transform=transform_train,
        )
        testset = torchvision.datasets.CIFAR10(
            root="/root/GanLuo/ICML2025_project/data/raw/CIFAR10",
            train=False,
            download=False,
            transform=transform_test,
        )
    elif dataset_name == "MNIST":
        transform_train, transform_test = MNIST_transform_train, MNIST_transform_test
        trainset = torchvision.datasets.MNIST(
            root="/root/GanLuo/ICML2025_project/data/raw/MNIST",
            train=True,
            download=False,
            transform=transform_train,
        )
        testset = torchvision.datasets.MNIST(
            root="/root/GanLuo/ICML2025_project/data/raw/MNIST",
            train=False,
            download=False,
            transform=transform_test,
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    total_train_size = len(trainset)
    subset_sizes = [
        total_train_size // n + (1 if i < total_train_size % n else 0) for i in range(n)
    ]

    subsets = torch.utils.data.random_split(trainset, subset_sizes, generator=generator)

    trainloader_list = [
        torch.utils.data.DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=12,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True,
            generator=generator,
        )
        for subset in subsets
    ]

    # Create a DataLoader for the full training set
    full_trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=12,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
        generator=generator,
    )

    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=100,
        shuffle=False,
        num_workers=12,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
        generator=generator,
    )

    return trainloader_list, testloader, full_trainloader