# -*- coding: utf-8 -*-
"""
Scaler Utilities / 标准化器工具
================================
Functions for handling StandardScaler objects.
处理StandardScaler对象的函数。
"""

import logging
from typing import Dict

import numpy as np
from sklearn.preprocessing import StandardScaler


def clamp_scaler_std_inplace(
    scaler: StandardScaler,
    eps: float = 1e-8,
    name: str = ""
) -> None:
    """
    Clamp scaler standard deviation to avoid division by near-zero values.
    限制标准化器的标准差以避免除以接近零的值。

    Modifies scaler in-place to replace very small scale values with 1.0.

    原地修改标准化器，将非常小的scale值替换为1.0。

    Args:
        scaler: StandardScaler object / StandardScaler对象
        eps: Threshold below which to clamp / 限制阈值
        name: Name for logging / 用于日志的名称
    """
    if not hasattr(scaler, "scale_") or scaler.scale_ is None:
        return

    mask = scaler.scale_ < eps
    if np.any(mask):
        scaler.scale_[mask] = 1.0
        if hasattr(scaler, "var_") and scaler.var_ is not None:
            scaler.var_[mask] = 1.0
        logging.info(f"[Scaler clamp] {name}: indices {np.where(mask)[0].tolist()}")


def safe_transform_np(
    scaler: StandardScaler,
    x: np.ndarray,
    eps: float = 1e-8
) -> np.ndarray:
    """
    Safely transform data using scaler, handling near-zero scales.
    使用标准化器安全地转换数据，处理接近零的scale值。

    Args:
        scaler: StandardScaler object / StandardScaler对象
        x: Data to transform / 要转换的数据
        eps: Threshold for safe division / 安全除法阈值

    Returns:
        Transformed data / 转换后的数据
    """
    scale = getattr(scaler, "scale_", None)
    mean = getattr(scaler, "mean_", None)

    if scale is None or mean is None:
        return x.astype(np.float32)

    scale_eff = np.where(scale < eps, 1.0, scale)
    return ((x - mean) / scale_eff).astype(np.float32)


def denormalize_abs(
    abs_norm: np.ndarray,
    scalers: Dict[str, StandardScaler],
    eps: float = 1e-8
) -> np.ndarray:
    """
    Denormalize absolute values from normalized space to physical space.
    将绝对值从标准化空间反归一化到物理空间。

    Args:
        abs_norm: Normalized absolute values / 标准化的绝对值
        scalers: Dictionary of scalers / 标准化器字典
        eps: Threshold for safe division / 安全除法阈值

    Returns:
        Denormalized values in physical space / 物理空间中的反归一化值
    """
    mean = scalers['abs'].mean_
    scale = scalers['abs'].scale_
    scale_eff = np.where(scale < eps, 1.0, scale)
    return abs_norm * scale_eff + mean