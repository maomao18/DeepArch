# -*- coding: utf-8 -*-
"""
I/O Utilities / 输入输出工具
============================
Functions for loading and saving models and scalers.
加载和保存模型和标准化器的函数。
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import joblib
import torch
import torch.nn as nn


def load_model(
    model_path: str,
    device: str = "cuda",
    model_class: Optional[type] = None
) -> Tuple[nn.Module, Any, Dict[str, Any]]:
    """
    Load trained model from checkpoint.
    从检查点加载训练好的模型。

    Args:
        model_path: Path to model checkpoint / 模型检查点路径
        device: Device to load model on / 加载模型的设备
        model_class: Model class to instantiate (if None, uses BucklingPredictor)
                     要实例化的模型类（如果为None，使用BucklingPredictor）

    Returns:
        Tuple of (model, config, checkpoint) / (模型, 配置, 检查点)元组
    """
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Get config from checkpoint / 从检查点获取配置
    config = checkpoint.get('config', None)
    if config is None:
        raise ValueError(
            "Config not found in checkpoint / 检查点中未找到配置"
        )

    # Import model class if not provided / 如果未提供则导入模型类
    if model_class is None:
        from models.predictor import BucklingPredictor
        model_class = BucklingPredictor

    # Create model from config / 根据配置创建模型
    model = model_class(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    logging.info(f"Model loaded from {model_path} / 模型已从{model_path}加载")

    return model, config, checkpoint


def load_scalers(
    scalers_path: str,
    checkpoint: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Load scalers from file or checkpoint.
    从文件或检查点加载标准化器。

    Args:
        scalers_path: Path to scalers file / 标准化器文件路径
        checkpoint: Optional checkpoint containing scalers / 可选的包含标准化器的检查点

    Returns:
        Dictionary of scalers / 标准化器字典
    """
    if checkpoint is not None and 'scalers' in checkpoint:
        return checkpoint['scalers']

    return joblib.load(scalers_path)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epoch: int,
    config: Any,
    scalers: Dict[str, Any],
    save_path: str,
    monitor_value: Optional[float] = None
) -> None:
    """
    Save training checkpoint.
    保存训练检查点。

    Args:
        model: Model to save / 要保存的模型
        optimizer: Optimizer to save / 要保存的优化器
        scheduler: Learning rate scheduler / 学习率调度器
        epoch: Current epoch / 当前epoch
        config: Training configuration / 训练配置
        scalers: Data scalers / 数据标准化器
        save_path: Path to save checkpoint / 保存检查点的路径
        monitor_value: Optional monitor value (e.g., validation loss) / 可选的监控值（如验证损失）
    """
    # Ensure directory exists / 确保目录存在
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'config': config,
        'scalers': scalers
    }

    if monitor_value is not None:
        checkpoint['monitor_value'] = monitor_value

    torch.save(checkpoint, save_path)
    logging.info(f"Checkpoint saved -> {save_path} / 检查点已保存 -> {save_path}")


def save_model_weights(
    model: nn.Module,
    save_path: str
) -> None:
    """
    Save model weights only.
    仅保存模型权重。

    Args:
        model: Model to save / 要保存的模型
        save_path: Path to save weights / 保存权重的路径
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    logging.info(f"Model weights saved -> {save_path} / 模型权重已保存 -> {save_path}")


def load_model_weights(
    model: nn.Module,
    weights_path: str,
    device: str = "cuda"
) -> nn.Module:
    """
    Load model weights into existing model.
    将模型权重加载到现有模型中。

    Args:
        model: Model to load weights into / 要加载权重的模型
        weights_path: Path to weights file / 权重文件路径
        device: Device to load weights on / 加载权重的设备

    Returns:
        Model with loaded weights / 加载了权重的模型
    """
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    return model