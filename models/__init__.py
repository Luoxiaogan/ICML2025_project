# models/__init__.py

from .cnn import new_ResNet18
from .fully_connected import FullyConnectedMNIST

__all__ = [
    'new_ResNet18',
    'FullyConnectedMNIST',
]
