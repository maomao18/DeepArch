# -*- coding: utf-8 -*-
"""
Random Seed Utilities / 随机种子工具
=====================================
Functions for setting random seeds and ensuring reproducibility.
设置随机种子和确保可复现性的函数。
"""

import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int = 618) -> None:
    """
    Set random seed for reproducibility.
    设置随机种子以确保可复现性。

    Args:
        seed: Random seed value / 随机种子值
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # For full determinism (slower) / 完全确定性（较慢）
    # Uncomment if needed / 如需要请取消注释:
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id: int) -> None:
    """
    Initialize worker seed for DataLoader.
    初始化DataLoader工作进程的种子。

    Use this function as the worker_init_fn parameter in DataLoader
    to ensure reproducibility in multi-worker data loading.

    将此函数作为DataLoader的worker_init_fn参数使用，
    以确保多工作进程数据加载的可复现性。

    Args:
        worker_id: Worker ID / 工作进程ID
    """
    base_seed = torch.initial_seed() % 2**32
    np.random.seed(base_seed + worker_id)
    random.seed(base_seed + worker_id)