# -*- coding: utf-8 -*-
"""
Buckling Extraction Utilities / 屈曲提取工具
=============================================
Functions for extracting buckling response from load-displacement curves.
从荷载-位移曲线中提取屈曲响应的函数。
"""

from typing import Tuple, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F


def moving_average_1d_torch(x: torch.Tensor, window: int) -> torch.Tensor:
    """
    Apply moving average to 1D tensor (PyTorch version).
    对1D张量应用移动平均（PyTorch版本）。

    Args:
        x: Input tensor of shape [B, T] / 输入张量，形状为[B, T]
        window: Window size (will be made odd if even) / 窗口大小（如果是偶数会被调整为奇数）

    Returns:
        Smoothed tensor of shape [B, T] / 平滑后的张量，形状为[B, T]
    """
    if window <= 1:
        return x
    if window % 2 == 0:
        window = window + 1  # Enforce odd for symmetric padding / 强制奇数以实现对称填充

    B, T = x.shape
    pad = window // 2

    # Reflect padding / 反射填充
    x_pad = F.pad(x.unsqueeze(1), (pad, pad), mode="reflect").squeeze(1)  # [B, T+2pad]
    kernel = torch.ones(1, 1, window, device=x.device, dtype=x.dtype) / window
    y = F.conv1d(x_pad.unsqueeze(1), kernel).squeeze(1)

    return y


def moving_average_1d_numpy(x: np.ndarray, window: int) -> np.ndarray:
    """
    Apply moving average to 1D array (NumPy version).
    对1D数组应用移动平均（NumPy版本）。

    Args:
        x: Input array of shape [T] / 输入数组，形状为[T]
        window: Window size / 窗口大小

    Returns:
        Smoothed array of shape [T] / 平滑后的数组，形状为[T]
    """
    if window <= 1:
        return x
    return np.convolve(x, np.ones(window) / window, mode='same')


def extract_buckling_index_numpy(
    load_seq: np.ndarray,
    disp_seq: np.ndarray,
    slope_threshold: float = 0.0,
    min_index: int = 5,
    eps: float = 1e-9,
    method: str = "peak_load"
) -> int:
    """
    Extract buckling index from load-displacement curve (NumPy version).
    从荷载-位移曲线中提取屈曲索引（NumPy版本）。

    Args:
        load_seq: Load sequence [T] / 荷载序列 [T]
        disp_seq: Displacement sequence [T] / 位移序列 [T]
        slope_threshold: Slope threshold for stiffness drop (only used if method='stiffness') /
                        刚度下降的斜率阈值（仅当method='stiffness'时使用）
        min_index: Minimum index to consider / 考虑的最小索引
        eps: Small value to avoid division by zero / 避免除以零的小值
        method: Detection method - 'peak_load' or 'stiffness' /
               检测方法 - 'peak_load'（峰值荷载法）或 'stiffness'（刚度法）

    Returns:
        Buckling index / 屈曲索引
    """
    if method == "peak_load":
        # Peak load method: buckling point = maximum load / 峰值荷载法：屈曲点 = 最大荷载点
        # This is reliable when post-buckling is truncated / 当后屈曲部分已截断时此方法可靠
        return int(np.argmax(load_seq))

    # Stiffness drop method / 刚度下降法
    dload = load_seq[1:] - load_seq[:-1]
    ddisp = disp_seq[1:] - disp_seq[:-1]

    denom = np.sign(ddisp) * np.maximum(np.abs(ddisp), eps)
    stiff = dload / denom
    stiff = np.nan_to_num(stiff, nan=1e9, posinf=1e9, neginf=-1e9)

    # Find stiffness drop / 查找刚度下降
    mask = stiff <= slope_threshold
    if min_index > 0 and len(mask) > min_index:
        mask[:min_index] = False

    true_indices = np.where(mask)[0]
    if len(true_indices) > 0:
        return int(true_indices[0] + 1)

    # Fallback to peak load / 回退到峰值荷载
    return int(np.argmax(load_seq))


def extract_buckling_index_torch(
    load_seq: torch.Tensor,
    disp_seq: torch.Tensor,
    slope_threshold: float = 0.0,
    min_index: int = 5,
    eps: float = 1e-9,
    method: str = "peak_load"
) -> torch.Tensor:
    """
    Extract buckling index from load-displacement curve (PyTorch version).
    从荷载-位移曲线中提取屈曲索引（PyTorch版本）。

    Args:
        load_seq: Load sequence [B, T] / 荷载序列 [B, T]
        disp_seq: Displacement sequence [B, T] / 位移序列 [B, T]
        slope_threshold: Slope threshold for stiffness drop (only used if method='stiffness') /
                        刚度下降的斜率阈值（仅当method='stiffness'时使用）
        min_index: Minimum index to consider / 考虑的最小索引
        eps: Small value to avoid division by zero / 避免除以零的小值
        method: Detection method - 'peak_load' or 'stiffness' /
               检测方法 - 'peak_load'（峰值荷载法）或 'stiffness'（刚度法）

    Returns:
        Buckling indices [B] / 屈曲索引 [B]
    """
    if method == "peak_load":
        # Peak load method: buckling point = maximum load / 峰值荷载法
        return load_seq.argmax(dim=1)

    # Stiffness drop method / 刚度下降法
    dload = load_seq[:, 1:] - load_seq[:, :-1]
    ddisp = disp_seq[:, 1:] - disp_seq[:, :-1]

    denom = ddisp.sign() * ddisp.abs().clamp_min(eps)
    stiff = dload / denom  # [B, T-1]
    stiff = torch.nan_to_num(stiff, nan=1e9, posinf=1e9, neginf=-1e9)

    mask = stiff <= slope_threshold

    # Ignore early indices / 忽略早期索引
    if min_index > 0 and stiff.size(1) > min_index:
        mask[:, :min_index] = False

    # First true index + 1 / 第一个True索引 + 1
    idx = torch.where(mask.any(dim=1), mask.float().argmax(dim=1) + 1, load_seq.argmax(dim=1))

    return idx.long()


def extract_buckling_response_numpy(
    abs_phys: np.ndarray,
    load_dim: int,
    buckling_disp_index: int,
    slope_threshold: float = 0.0,
    min_index: int = 5,
    eps: float = 1e-9,
    smooth: bool = True,
    smooth_window: int = 7,
    method: str = "peak_load"
) -> Tuple[float, float]:
    """
    Extract buckling load and displacement from absolute sequence (NumPy version).
    从绝对序列中提取屈曲荷载和位移（NumPy版本）。

    Args:
        abs_phys: Absolute physical sequence [T, D] / 绝对物理序列 [T, D]
        load_dim: Load dimension index (usually 0) / 荷载维度索引（通常为0）
        buckling_disp_index: Buckling displacement index / 屈曲位移索引
        slope_threshold: Slope threshold for stiffness drop (only used if method='stiffness') /
                        刚度下降的斜率阈值（仅当method='stiffness'时使用）
        min_index: Minimum index to consider / 考虑的最小索引
        eps: Small value to avoid division by zero / 避免除以零的小值
        smooth: Whether to apply smoothing / 是否应用平滑
        smooth_window: Smoothing window size / 平滑窗口大小
        method: Detection method - 'peak_load' or 'stiffness' /
               检测方法 - 'peak_load'（峰值荷载法）或 'stiffness'（刚度法）

    Returns:
        Tuple of (buckling_load, buckling_disp) / (屈曲荷载, 屈曲位移)元组
    """
    load_seq = abs_phys[:, 0]
    disp_seq = abs_phys[:, load_dim + buckling_disp_index]

    # Apply smoothing if enabled / 如果启用则应用平滑
    if smooth and smooth_window > 1:
        load_sm = moving_average_1d_numpy(load_seq, smooth_window)
        disp_sm = moving_average_1d_numpy(disp_seq, smooth_window)
    else:
        load_sm, disp_sm = load_seq, disp_seq

    idx = extract_buckling_index_numpy(
        load_sm, disp_sm,
        slope_threshold=slope_threshold,
        min_index=min_index,
        eps=eps,
        method=method
    )

    return float(load_seq[idx]), float(disp_seq[idx])


def extract_buckling_response_torch(
    abs_phys: torch.Tensor,
    load_dim: int,
    buckling_disp_index: int,
    slope_threshold: float = 0.0,
    min_index: int = 5,
    eps: float = 1e-9,
    smooth: bool = True,
    smooth_window: int = 7,
    method: str = "peak_load"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract buckling load and displacement from absolute sequence (PyTorch version).
    从绝对序列中提取屈曲荷载和位移（PyTorch版本）。

    Args:
        abs_phys: Absolute physical sequence [B, T, D] / 绝对物理序列 [B, T, D]
        load_dim: Load dimension index (usually 0) / 荷载维度索引（通常为0）
        buckling_disp_index: Buckling displacement index / 屈曲位移索引
        slope_threshold: Slope threshold for stiffness drop (only used if method='stiffness') /
                        刚度下降的斜率阈值（仅当method='stiffness'时使用）
        min_index: Minimum index to consider / 考虑的最小索引
        eps: Small value to avoid division by zero / 避免除以零的小值
        smooth: Whether to apply smoothing / 是否应用平滑
        smooth_window: Smoothing window size / 平滑窗口大小
        method: Detection method - 'peak_load' or 'stiffness' /
               检测方法 - 'peak_load'（峰值荷载法）或 'stiffness'（刚度法）

    Returns:
        Tuple of (buckling_load [B], buckling_disp [B]) / (屈曲荷载 [B], 屈曲位移 [B])元组
    """
    load_seq = abs_phys[:, :, 0]  # [B, T]
    disp_col = load_dim + buckling_disp_index
    disp_seq = abs_phys[:, :, disp_col]  # [B, T]

    # Apply smoothing if enabled / 如果启用则应用平滑
    if smooth and smooth_window > 1:
        load_sm = moving_average_1d_torch(load_seq, smooth_window)
        disp_sm = moving_average_1d_torch(disp_seq, smooth_window)
    else:
        load_sm, disp_sm = load_seq, disp_seq

    idx = extract_buckling_index_torch(
        load_sm, disp_sm,
        slope_threshold=slope_threshold,
        min_index=min_index,
        eps=eps,
        method=method
    )

    b = torch.arange(load_seq.size(0), device=abs_phys.device)
    buckling_load = load_seq[b, idx]
    buckling_disp = disp_seq[b, idx]

    return buckling_load, buckling_disp