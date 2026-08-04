# -*- coding: utf-8 -*-
"""
Attention Weight Extraction Script / 注意力权重提取脚本
=========================================================
Extract attention pooling weights from trained FA-LSTM model
for physical interpretability analysis in scientific papers.
从训练好的FA-LSTM模型提取注意力池化权重，用于科研论文物理可解释性分析。

Output files:
    - attn_weights_detail.csv: Per-sample, per-timestep weights (N × 200 rows)
    - attn_weights_mean_over_time.csv: Mean weights per timestep (200 rows)
    - attn_weights_overall_mean.csv: Overall mean weights per feature (59 rows)

Usage:
    python Code/extract_attention_weights.py --model ./models/v2F/buckling_predictor_best.pth
    python Code/extract_attention_weights.py --model ./models/FA-LSTM/seed_618/buckling_predictor_best.pth --output ./results/FA-LSTM/attention
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
import logging
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

warnings.filterwarnings('ignore')

# Import project modules / 导入项目模块
from config.train import TrainingConfig
from models.predictor import BucklingPredictor
from data.scalers import safe_transform_np

# Ensure TrainingConfig is available in __main__ namespace for pickle
# 确保TrainingConfig在__main__命名空间中可用，用于pickle反序列化
sys.modules['__main__'].TrainingConfig = TrainingConfig


# =============================================================================
# Feature Names / 特征名称
# =============================================================================
FIXED_FEATURE_NAMES: Tuple[str, ...] = (
    "I0",      # Section quality / 截面质量
    "A11",     # Compression stiffness / 压缩刚度
    "B11",     # Compression-bending coupling / 压-弯耦合刚度
    "D11",     # Bending stiffness / 弯曲刚度
    "L",       # Span / 跨径
    "f",       # Arch height / 拱高
    "b",       # Section width / 拱截面宽
    "h",       # Section height / 拱截面高
    "S",       # Arch length / 拱长
    "lambda",  # Slenderness ratio / 长细比
    "KXL",     # Left X elastic support / 左X弹性支撑
    "KYL",     # Left Y elastic support / 左Y弹性支撑
    "KZL",     # Left rotation elastic support / 左转动弹性支撑
    "KXR",     # Right X elastic support / 右X弹性支撑
    "KYR",     # Right Y elastic support / 右Y弹性支撑
    "KZR",     # Right rotation elastic support / 右转动弹性支撑
)

DYNAMIC_FEATURE_NAMES: Tuple[str, ...] = (
    "load",    # Load / 荷载
    *[f"disp_{i}" for i in range(42)]  # 42 displacements / 42个位移
)

ALL_FEATURE_NAMES: Tuple[str, ...] = FIXED_FEATURE_NAMES + DYNAMIC_FEATURE_NAMES  # 59 features


# =============================================================================
# Configuration / 配置
# =============================================================================
@dataclass
class AttentionExtractionConfig:
    """Configuration for attention weight extraction / 注意力权重提取配置"""

    # Model path / 模型路径
    model_path: str = r"./models/v2F/buckling_predictor_best.pth"

    # Data path / 数据路径
    data_dir: str = r"./Data/Test"

    # Output path / 输出路径
    output_dir: str = r"./results/attention/v2F"

    # Device / 设备
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Feature dimensions (from config) / 特征维度（来自配置）
    fixed_features_dim: int = 16
    dynamic_features_dim: int = 43
    meta_cols: int = 1
    sequence_length: int = 200

    def __post_init__(self) -> None:
        """Post-initialization / 后初始化"""
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        # Compute derived / 计算派生值
        self.dynamic_col_start = self.meta_cols + self.fixed_features_dim


# =============================================================================
# Core Functions / 核心函数
# =============================================================================
def load_model(model_path: str, device: str) -> Tuple[BucklingPredictor, TrainingConfig, Dict]:
    """
    Load trained model from checkpoint.
    从检查点加载训练好的模型。

    Args:
        model_path: Path to model checkpoint / 模型检查点路径
        device: Device to load model on / 加载模型的设备

    Returns:
        Tuple of (model, config, scalers) / (模型, 配置, 标准化器)元组
    """
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    config = checkpoint.get('config', None)
    if config is None:
        raise ValueError("Config not found in checkpoint / 检查点中未找到配置")

    model = BucklingPredictor(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)  # Ensure model is on correct device / 确保模型在正确设备上
    model.eval()

    scalers = checkpoint.get('scalers', None)

    logging.info(f"Model loaded from {model_path} / 模型已从{model_path}加载")
    logging.info(f"Model has attention pooling: {model.pool is not None} / 模型是否有注意力池化: {model.pool is not None}")

    return model, config, scalers


def read_csv_sample(
    csv_path: Path,
    config: AttentionExtractionConfig
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read single CSV file and extract features.
    读取单个CSV文件并提取特征。

    Args:
        csv_path: CSV file path / CSV文件路径
        config: Extraction configuration / 提取配置

    Returns:
        Tuple of (fixed_features, dynamic_abs) / (固定特征, 动态绝对值)元组
    """
    df = pd.read_csv(csv_path)
    data = df.values.astype(np.float32)

    fixed_start = config.meta_cols
    fixed_end = fixed_start + config.fixed_features_dim
    dyn_start = config.dynamic_col_start
    dyn_end = dyn_start + config.dynamic_features_dim

    if data.shape[1] < dyn_end:
        raise ValueError(
            f"{csv_path.name}: columns < expected {dyn_end}, got {data.shape[1]} / "
            f"{csv_path.name}: 列数 < 期望值 {dyn_end}，实际 {data.shape[1]}"
        )

    fixed = data[0, fixed_start:fixed_end]
    dynamic_abs = data[:, dyn_start:dyn_end]

    # Truncate or pad to sequence_length / 截断或填充到序列长度
    L = dynamic_abs.shape[0]
    target_L = config.sequence_length
    if L >= target_L:
        dynamic_abs = dynamic_abs[:target_L]
    else:
        pad_n = target_L - L
        last = dynamic_abs[-1:].repeat(pad_n, axis=0)
        dynamic_abs = np.concatenate([dynamic_abs, last], axis=0)

    return fixed, dynamic_abs


@torch.no_grad()
def extract_attention_weights(
    model: BucklingPredictor,
    fixed_norm: torch.Tensor,
    abs_norm: torch.Tensor,
    device: torch.device
) -> np.ndarray:
    """
    Extract attention weights for all timesteps in a sample.
    提取样本中所有时间步的注意力权重。

    Args:
        model: BucklingPredictor model / BucklingPredictor模型
        fixed_norm: Normalized fixed features [B, Dfix] / 标准化的固定特征 [B, Dfix]
        abs_norm: Normalized absolute states [B, T, Ddyn] / 标准化的绝对状态 [B, T, Ddyn]
        device: Device for computation / 计算设备

    Returns:
        Attention weights array [T, 59] / 注意力权重数组 [T, 59]
    """
    B, T, D = abs_norm.shape
    num_features = model.config.fixed_features_dim + model.config.dynamic_features_dim  # 59

    weights_list = []

    for t in range(T):
        # Get current timestep state / 获取当前时间步状态
        abs_state = abs_norm[:, t, :]  # [B, Ddyn]

        # Generate tokens / 生成标记
        tokens = model._tokens_from_features(fixed_norm, abs_state)  # [B, 59, d_model]

        # Extract weights using forward_with_weights / 使用forward_with_weights提取权重
        if model.pool is not None:
            _, weights = model.pool.forward_with_weights(tokens)  # [B, 59]
        else:
            # No attention pooling: uniform weights / 无注意力池化：均匀权重
            weights = torch.ones(B, num_features, device=device) / num_features

        weights_list.append(weights.cpu().numpy())

    # Stack along timestep dimension: [T, B, 59] -> [B, T, 59] -> [T, 59] for single sample
    weights_array = np.stack(weights_list, axis=0)  # [T, B, 59]

    # For single sample (B=1), squeeze batch dimension / 对于单个样本，压缩批次维度
    if B == 1:
        weights_array = weights_array[:, 0, :]  # [T, 59]

    return weights_array


def process_all_samples(
    model: BucklingPredictor,
    scalers: Dict,
    csv_files: List[Path],
    config: AttentionExtractionConfig,
    device: torch.device
) -> Tuple[np.ndarray, List[str]]:
    """
    Process all samples and extract attention weights.
    处理所有样本并提取注意力权重。

    Args:
        model: BucklingPredictor model / BucklingPredictor模型
        scalers: Dictionary of scalers / 标准化器字典
        csv_files: List of CSV file paths / CSV文件路径列表
        config: Extraction configuration / 提取配置
        device: Device for computation / 计算设备

    Returns:
        Tuple of (all_weights, sample_names) / (所有权重, 样本名称)元组
        - all_weights: [N_samples, T, 59] / 所有权重数组
        - sample_names: List of sample filenames / 样本文件名列表
    """
    all_weights = []
    sample_names = []

    for csv_path in tqdm(csv_files, desc="Extracting weights / 提取权重"):
        try:
            # Read and normalize / 读取并标准化
            fixed_phys, abs_phys = read_csv_sample(csv_path, config)

            fixed_norm = safe_transform_np(scalers['fixed'], fixed_phys.reshape(1, -1))
            abs_norm = safe_transform_np(scalers['abs'], abs_phys)

            # Convert to tensors / 转换为张量
            fixed_tensor = torch.tensor(fixed_norm, dtype=torch.float32, device=device)
            abs_tensor = torch.tensor(abs_norm, dtype=torch.float32, device=device).unsqueeze(0)  # [1, T, D]

            # Extract weights / 提取权重
            weights = extract_attention_weights(model, fixed_tensor, abs_tensor, device)

            all_weights.append(weights)
            sample_names.append(csv_path.stem)

        except Exception as e:
            logging.warning(f"Skip {csv_path.name}: {e} / 跳过 {csv_path.name}: {e}")

    # Stack all samples / 堆叠所有样本
    all_weights_array = np.stack(all_weights, axis=0)  # [N, T, 59]

    logging.info(f"Extracted weights from {len(sample_names)} samples / 从{len(sample_names)}个样本提取了权重")
    logging.info(f"Weight array shape: {all_weights_array.shape} / 权重数组形状: {all_weights_array.shape}")

    return all_weights_array, sample_names


# =============================================================================
# Save Functions / 保存函数
# =============================================================================
def save_detail_weights(
    weights: np.ndarray,
    sample_names: List[str],
    output_dir: Path
) -> None:
    """
    Save detailed per-sample, per-timestep weights to CSV.
    保存详细的每样本、每时间步权重到CSV。

    Args:
        weights: Weight array [N, T, 59] / 权重数组 [N, T, 59]
        sample_names: List of sample names / 样本名称列表
        output_dir: Output directory / 输出目录
    """
    N, T, F = weights.shape

    # Build DataFrame / 构建DataFrame
    records = []
    for i, name in enumerate(sample_names):
        for t in range(T):
            row = {
                'sample': name,
                'timestep': t,
            }
            for f_idx, feat_name in enumerate(ALL_FEATURE_NAMES):
                row[feat_name] = weights[i, t, f_idx]
            records.append(row)

    df = pd.DataFrame(records)

    output_path = output_dir / "attn_weights_detail.csv"
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    logging.info(f"Detail weights saved to {output_path} / 详细权重已保存到{output_path}")
    logging.info(f"  Rows: {len(df)} (= {N} samples × {T} timesteps) / 行数: {len(df)}")


def save_mean_over_time(
    weights: np.ndarray,
    output_dir: Path
) -> None:
    """
    Save mean weights per timestep (averaged over all samples).
    保存每时间步的平均权重（所有样本平均）。

    Args:
        weights: Weight array [N, T, 59] / 权重数组 [N, T, 59]
        output_dir: Output directory / 输出目录
    """
    # Mean over samples: [T, 59] / 样本平均: [T, 59]
    mean_weights = weights.mean(axis=0)

    # Build DataFrame / 构建DataFrame
    records = []
    for t in range(mean_weights.shape[0]):
        row = {'timestep': t}
        for f_idx, feat_name in enumerate(ALL_FEATURE_NAMES):
            row[feat_name] = mean_weights[t, f_idx]
        records.append(row)

    df = pd.DataFrame(records)

    output_path = output_dir / "attn_weights_mean_over_time.csv"
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    logging.info(f"Mean weights over time saved to {output_path} / 时间平均权重已保存到{output_path}")
    logging.info(f"  Rows: {len(df)} (= {mean_weights.shape[0]} timesteps) / 行数: {len(df)}")


def save_overall_mean(
    weights: np.ndarray,
    output_dir: Path
) -> None:
    """
    Save overall mean weights per feature (averaged over all samples and timesteps).
    保存每个特征的整体平均权重（所有样本和时间步平均）。

    Args:
        weights: Weight array [N, T, 59] / 权重数组 [N, T, 59]
        output_dir: Output directory / 输出目录
    """
    # Mean over samples and timesteps: [59] / 样本和时间步平均: [59]
    overall_mean = weights.mean(axis=(0, 1))
    overall_std = weights.std(axis=(0, 1))

    # Build DataFrame / 构建DataFrame
    records = []
    for f_idx, feat_name in enumerate(ALL_FEATURE_NAMES):
        records.append({
            'feature': feat_name,
            'mean_weight': overall_mean[f_idx],
            'std_weight': overall_std[f_idx],
        })

    df = pd.DataFrame(records)

    output_path = output_dir / "attn_weights_overall_mean.csv"
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    logging.info(f"Overall mean weights saved to {output_path} / 整体平均权重已保存到{output_path}")
    logging.info(f"  Rows: {len(df)} (= {len(ALL_FEATURE_NAMES)} features) / 行数: {len(df)}")


def print_summary(weights: np.ndarray) -> None:
    """
    Print summary statistics of extracted weights.
    打印提取权重的摘要统计。
    """
    print("\n" + "=" * 70)
    print("Attention Weights Summary / 注意力权重摘要")
    print("=" * 70)

    N, T, F = weights.shape
    print(f"  Samples / 样本数: {N}")
    print(f"  Timesteps / 时间步数: {T}")
    print(f"  Features / 特征数: {F}")
    print("-" * 70)

    # Overall statistics / 整体统计
    overall_mean = weights.mean(axis=(0, 1))
    overall_std = weights.std(axis=(0, 1))

    # Fixed vs Dynamic / 固定 vs 动态
    fixed_mean = overall_mean[:16].sum()
    dynamic_mean = overall_mean[16:].sum()

    print(f"  Fixed features total weight / 固定特征总权重: {fixed_mean:.4f}")
    print(f"  Dynamic features total weight / 动态特征总权重: {dynamic_mean:.4f}")
    print("-" * 70)

    # Top 10 features by mean weight / 按平均权重的top 10特征
    print("  Top 10 features by mean weight / 按平均权重的top 10特征:")
    sorted_indices = np.argsort(overall_mean)[::-1]
    for i, idx in enumerate(sorted_indices[:10]):
        print(f"    {i+1}. {ALL_FEATURE_NAMES[idx]}: {overall_mean[idx]:.4f} ± {overall_std[idx]:.4f}")

    print("-" * 70)

    # Weight evolution over time (first vs last timestep) / 权重随时间演化（第一 vs 最后时间步）
    first_timestep_mean = weights.mean(axis=0)[0]
    last_timestep_mean = weights.mean(axis=0)[-1]

    print("  Weight change from first to last timestep / 第一到最后时间步权重变化:")
    for f_idx in [0, 16, 17, 18, 19, 20]:  # I0, load, disp_0-4
        change = last_timestep_mean[f_idx] - first_timestep_mean[f_idx]
        print(f"    {ALL_FEATURE_NAMES[f_idx]}: {first_timestep_mean[f_idx]:.4f} -> {last_timestep_mean[f_idx]:.4f} (Δ={change:+.4f})")


# =============================================================================
# Main Function / 主函数
# =============================================================================
def main() -> None:
    """Main function / 主函数"""
    parser = argparse.ArgumentParser(
        description="Extract attention weights from trained model / 从训练模型提取注意力权重"
    )

    parser.add_argument(
        "--model", type=str, default=r"./models/v2F/buckling_predictor_best.pth",
        help="Path to trained model / 训练好的模型路径"
    )
    parser.add_argument(
        "--data", type=str, default=r"./Data/Test",
        help="Path to data directory / 数据目录路径"
    )
    parser.add_argument(
        "--output", type=str, default=r"./results/attention",
        help="Output directory / 输出目录"
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for computation / 计算设备"
    )

    args = parser.parse_args()

    # Setup logging / 设置日志
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / f"extract_attention_{datetime.now():%Y%m%d_%H%M%S}.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler()
        ]
    )

    logging.info("=" * 60)
    logging.info("Attention Weight Extraction / 注意力权重提取")
    logging.info("=" * 60)

    # Initialize config / 初始化配置
    config = AttentionExtractionConfig(
        model_path=args.model,
        data_dir=args.data,
        output_dir=args.output,
        device=args.device
    )

    device = torch.device(config.device)

    # Load model / 加载模型
    logging.info(f"Loading model from {config.model_path} / 从{config.model_path}加载模型")
    model, model_config, scalers = load_model(config.model_path, config.device)

    if scalers is None:
        raise ValueError("Scalers not found in checkpoint / 检查点中未找到标准化器")

    # Discover CSV files / 发现CSV文件
    csv_files = sorted(Path(config.data_dir).glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {config.data_dir} / 在{config.data_dir}中未找到CSV文件")

    logging.info(f"Found {len(csv_files)} CSV files / 找到{len(csv_files)}个CSV文件")

    # Extract weights / 提取权重
    logging.info("Extracting attention weights / 提取注意力权重...")
    weights, sample_names = process_all_samples(model, scalers, csv_files, config, device)

    # Save results / 保存结果
    logging.info("Saving results / 保存结果...")
    save_detail_weights(weights, sample_names, output_dir)
    save_mean_over_time(weights, output_dir)
    save_overall_mean(weights, output_dir)

    # Print summary / 打印摘要
    print_summary(weights)

    logging.info("=" * 60)
    logging.info("Extraction completed / 提取完成")
    logging.info("=" * 60)


if __name__ == "__main__":
    main()