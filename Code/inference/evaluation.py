# -*- coding: utf-8 -*-
"""
Model Evaluation / 模型评估
============================
Evaluation functions for trained models.
训练好的模型的评估函数。
"""

from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.buckling import extract_buckling_response_numpy


# =============================================================================
# Metric Functions / 指标函数
# =============================================================================
def calculate_mae(pred: np.ndarray, target: np.ndarray) -> float:
    """Calculate Mean Absolute Error / 计算平均绝对误差"""
    return float(np.mean(np.abs(pred - target)))


def calculate_mse(pred: np.ndarray, target: np.ndarray) -> float:
    """Calculate Mean Squared Error / 计算均方误差"""
    return float(np.mean((pred - target) ** 2))


def calculate_rmse(pred: np.ndarray, target: np.ndarray) -> float:
    """Calculate Root Mean Squared Error / 计算均方根误差"""
    return float(np.sqrt(calculate_mse(pred, target)))


def calculate_r2(pred: np.ndarray, target: np.ndarray) -> float:
    """Calculate R-squared (Coefficient of Determination) / 计算R²（决定系数）"""
    ss_res = np.sum((target - pred) ** 2)
    ss_tot = np.sum((target - np.mean(target)) ** 2)
    if ss_tot < 1e-12:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def calculate_mape(pred: np.ndarray, target: np.ndarray, eps: float = 1e-8) -> float:
    """
    Calculate Mean Absolute Percentage Error / 计算平均绝对百分比误差

    Note: Excludes points where target is near zero to avoid division issues
    注意：排除目标值接近零的点以避免除法问题
    """
    mask = np.abs(target) > eps
    if np.sum(mask) == 0:
        return float('nan')
    return float(np.mean(np.abs((target[mask] - pred[mask]) / target[mask])) * 100)


def calculate_pcc(pred: np.ndarray, target: np.ndarray) -> float:
    """Calculate Pearson Correlation Coefficient / 计算皮尔逊相关系数"""
    pred_centered = pred - np.mean(pred)
    target_centered = target - np.mean(target)

    numerator = np.sum(pred_centered * target_centered)
    denominator = np.sqrt(np.sum(pred_centered ** 2) * np.sum(target_centered ** 2))

    if denominator < 1e-12:
        return 0.0
    return float(numerator / denominator)


def calculate_rae(pred: np.ndarray, target: np.ndarray) -> float:
    """Calculate Relative Absolute Error / 计算相对绝对误差"""
    numerator = np.sum(np.abs(target - pred))
    denominator = np.sum(np.abs(target - np.mean(target)))

    if denominator < 1e-12:
        return 0.0
    return float(numerator / denominator)


def calculate_nrmse(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Calculate Normalized Root Mean Squared Error / 计算归一化均方根误差

    Normalized by the range of target values
    通过目标值的范围进行归一化
    """
    rmse = calculate_rmse(pred, target)
    target_range = np.max(target) - np.min(target)

    if target_range < 1e-12:
        return 0.0
    return float(rmse / target_range)


# =============================================================================
# Prediction Functions / 预测函数
# =============================================================================
@torch.no_grad()
def predict_sequence(
    model: nn.Module,
    fixed_norm: torch.Tensor,
    abs_norm: torch.Tensor,
    scalers: Dict[str, Any],
    device: torch.device
) -> np.ndarray:
    """
    Predict sequence using autoregressive mode.
    使用自回归模式预测序列。

    Args:
        model: Trained model / 训练好的模型
        fixed_norm: Normalized fixed features [16] / 标准化的固定特征 [16]
        abs_norm: Normalized absolute states [T, 43] / 标准化的绝对状态 [T, 43]
        scalers: Scalers for denormalization / 用于反标准化的标准化器
        device: Device for computation / 计算设备

    Returns:
        Predicted absolute sequence [T, 43] in physical space / 物理空间中的预测绝对序列 [T, 43]
    """
    model.eval()

    # Move to device / 移动到设备
    fixed = fixed_norm.unsqueeze(0).to(device)  # [1, 16]
    abs0 = abs_norm[0:1].to(device)  # [1, 43]

    T = abs_norm.shape[0]
    steps = T - 1

    # Get scalers / 获取标准化器
    abs_mean = torch.tensor(scalers['abs'].mean_, dtype=torch.float32, device=device)
    abs_std = torch.tensor(np.maximum(scalers['abs'].scale_, 1e-8), dtype=torch.float32, device=device)
    delta_mean = torch.tensor(scalers['delta'].mean_, dtype=torch.float32, device=device)
    delta_std = torch.tensor(np.maximum(scalers['delta'].scale_, 1e-8), dtype=torch.float32, device=device)

    # Transformation coefficients / 变换系数
    A = (delta_std / abs_std).view(1, -1)  # [1, 43]
    B = (delta_mean / abs_std).view(1, -1)  # [1, 43]

    # Initial state in physical space / 物理空间中的初始状态
    abs0_phys = abs0 * abs_std + abs_mean  # [1, 43]
    predictions = [abs0_phys.squeeze(0).cpu().numpy()]  # [43]

    h = None
    prev = abs0  # [1, 43]

    for _ in range(steps):
        delta, h = model.step(fixed, prev, h)  # delta: [1, 43]
        next_abs_norm = prev + A * delta + B  # [1, 43]
        next_abs_phys = (next_abs_norm * abs_std + abs_mean).squeeze(0).cpu().numpy()  # [43]
        predictions.append(next_abs_phys)
        prev = next_abs_norm  # [1, 43]

    return np.array(predictions)


# =============================================================================
# Evaluation Functions / 评估函数
# =============================================================================
def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    scalers: Dict[str, Any],
    config: Any,
    device: torch.device
) -> Dict[str, float]:
    """
    Evaluate model with comprehensive metrics.
    使用全面指标评估模型。

    Args:
        model: Trained model / 训练好的模型
        test_loader: Test data loader / 测试数据加载器
        scalers: Scalers for denormalization / 用于反标准化的标准化器
        config: Evaluation configuration / 评估配置
        device: Device for computation / 计算设备

    Returns:
        Dictionary of evaluation metrics / 评估指标字典
    """
    model.eval()

    # Storage for predictions and targets / 存储预测和目标
    all_pred_sequences = []
    all_true_sequences = []
    all_pred_loads = []
    all_true_loads = []
    all_pred_disps = []
    all_true_disps = []
    all_pred_buckling_loads = []
    all_true_buckling_loads = []
    all_pred_buckling_disps = []
    all_true_buckling_disps = []

    for fixed_norm, abs_norm, delta_norm in tqdm(test_loader, desc="Evaluating / 评估中"):
        for i in range(fixed_norm.shape[0]):
            # Predict sequence / 预测序列
            pred_seq = predict_sequence(
                model, fixed_norm[i], abs_norm[i], scalers, device
            )

            # Get true sequence in physical space / 获取物理空间中的真实序列
            abs_mean = scalers['abs'].mean_
            abs_std = np.maximum(scalers['abs'].scale_, 1e-8)
            true_seq = (abs_norm[i].numpy() * abs_std + abs_mean)

            # Store sequences / 存储序列
            all_pred_sequences.append(pred_seq.flatten())
            all_true_sequences.append(true_seq.flatten())

            # Extract load and displacement / 提取荷载和位移
            all_pred_loads.extend(pred_seq[:, 0])
            all_true_loads.extend(true_seq[:, 0])

            disp_col = config.load_dim + config.buckling_disp_index
            all_pred_disps.extend(pred_seq[:, disp_col])
            all_true_disps.extend(true_seq[:, disp_col])

            # Extract buckling response / 提取屈曲响应
            pred_buckling_load, pred_buckling_disp = extract_buckling_response_numpy(
                pred_seq, config.load_dim, config.buckling_disp_index,
                slope_threshold=config.buckling_slope_threshold,
                min_index=config.buckling_min_index,
                eps=config.buckling_eps,
                smooth=config.buckling_smooth,
                smooth_window=config.buckling_smooth_window,
                method=config.buckling_method
            )
            true_buckling_load, true_buckling_disp = extract_buckling_response_numpy(
                true_seq, config.load_dim, config.buckling_disp_index,
                slope_threshold=config.buckling_slope_threshold,
                min_index=config.buckling_min_index,
                eps=config.buckling_eps,
                smooth=config.buckling_smooth,
                smooth_window=config.buckling_smooth_window,
                method=config.buckling_method
            )

            all_pred_buckling_loads.append(pred_buckling_load)
            all_true_buckling_loads.append(true_buckling_load)
            all_pred_buckling_disps.append(pred_buckling_disp)
            all_true_buckling_disps.append(true_buckling_disp)

    # Convert to arrays / 转换为数组
    pred_sequences = np.concatenate(all_pred_sequences)
    true_sequences = np.concatenate(all_true_sequences)
    pred_loads = np.array(all_pred_loads)
    true_loads = np.array(all_true_loads)
    pred_disps = np.array(all_pred_disps)
    true_disps = np.array(all_true_disps)
    pred_buckling_loads = np.array(all_pred_buckling_loads)
    true_buckling_loads = np.array(all_true_buckling_loads)
    pred_buckling_disps = np.array(all_pred_buckling_disps)
    true_buckling_disps = np.array(all_true_buckling_disps)

    # Calculate metrics / 计算指标
    metrics = {}

    # Full sequence metrics / 全序列指标
    metrics['Sequence_MAE'] = calculate_mae(pred_sequences, true_sequences)
    metrics['Sequence_MSE'] = calculate_mse(pred_sequences, true_sequences)
    metrics['Sequence_RMSE'] = calculate_rmse(pred_sequences, true_sequences)
    metrics['Sequence_R2'] = calculate_r2(pred_sequences, true_sequences)
    metrics['Sequence_MAPE'] = calculate_mape(pred_sequences, true_sequences)
    metrics['Sequence_PCC'] = calculate_pcc(pred_sequences, true_sequences)
    metrics['Sequence_RAE'] = calculate_rae(pred_sequences, true_sequences)
    metrics['Sequence_NRMSE'] = calculate_nrmse(pred_sequences, true_sequences)

    # Load channel metrics / 荷载通道指标
    metrics['Load_MAE'] = calculate_mae(pred_loads, true_loads)
    metrics['Load_MSE'] = calculate_mse(pred_loads, true_loads)
    metrics['Load_RMSE'] = calculate_rmse(pred_loads, true_loads)
    metrics['Load_R2'] = calculate_r2(pred_loads, true_loads)
    metrics['Load_MAPE'] = calculate_mape(pred_loads, true_loads)
    metrics['Load_PCC'] = calculate_pcc(pred_loads, true_loads)

    # Displacement metrics / 位移指标
    metrics['Disp_MAE'] = calculate_mae(pred_disps, true_disps)
    metrics['Disp_MSE'] = calculate_mse(pred_disps, true_disps)
    metrics['Disp_RMSE'] = calculate_rmse(pred_disps, true_disps)
    metrics['Disp_R2'] = calculate_r2(pred_disps, true_disps)
    metrics['Disp_MAPE'] = calculate_mape(pred_disps, true_disps)
    metrics['Disp_PCC'] = calculate_pcc(pred_disps, true_disps)

    # Buckling load metrics / 屈曲荷载指标
    metrics['BucklingLoad_MAE'] = calculate_mae(pred_buckling_loads, true_buckling_loads)
    metrics['BucklingLoad_MSE'] = calculate_mse(pred_buckling_loads, true_buckling_loads)
    metrics['BucklingLoad_RMSE'] = calculate_rmse(pred_buckling_loads, true_buckling_loads)
    metrics['BucklingLoad_R2'] = calculate_r2(pred_buckling_loads, true_buckling_loads)
    metrics['BucklingLoad_MAPE'] = calculate_mape(pred_buckling_loads, true_buckling_loads)
    metrics['BucklingLoad_PCC'] = calculate_pcc(pred_buckling_loads, true_buckling_loads)
    metrics['BucklingLoad_RAE'] = calculate_rae(pred_buckling_loads, true_buckling_loads)
    metrics['BucklingLoad_NRMSE'] = calculate_nrmse(pred_buckling_loads, true_buckling_loads)

    # Buckling displacement metrics / 屈曲位移指标
    metrics['BucklingDisp_MAE'] = calculate_mae(pred_buckling_disps, true_buckling_disps)
    metrics['BucklingDisp_MSE'] = calculate_mse(pred_buckling_disps, true_buckling_disps)
    metrics['BucklingDisp_RMSE'] = calculate_rmse(pred_buckling_disps, true_buckling_disps)
    metrics['BucklingDisp_R2'] = calculate_r2(pred_buckling_disps, true_buckling_disps)
    metrics['BucklingDisp_MAPE'] = calculate_mape(pred_buckling_disps, true_buckling_disps)
    metrics['BucklingDisp_PCC'] = calculate_pcc(pred_buckling_disps, true_buckling_disps)

    # Pack raw data for further analysis / 打包原始数据供后续分析
    raw_data = {
        'pred_buckling_loads': pred_buckling_loads,
        'true_buckling_loads': true_buckling_loads,
        'pred_buckling_disps': pred_buckling_disps,
        'true_buckling_disps': true_buckling_disps,
        'pred_loads': pred_loads,
        'true_loads': true_loads,
        'pred_disps': pred_disps,
        'true_disps': true_disps,
    }

    return metrics, raw_data