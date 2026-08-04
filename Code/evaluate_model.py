# -*- coding: utf-8 -*-
"""
Evaluation Entry Point / 评估入口点
=====================================
Main script for evaluating trained models on test set.
在测试集上评估训练好的模型的主脚本。

Usage:
    python Code/evaluate_model.py
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import logging
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

warnings.filterwarnings('ignore')

# Import from new modules / 从新模块导入
from config.evaluation import EvaluationConfig
from config.train import TrainingConfig
from models.predictor import BucklingPredictor
from data.dataset import BucklingDataset
from data.scalers import safe_transform_np
from inference.evaluation import evaluate_model, predict_sequence


class EvaluationDataset(torch.utils.data.Dataset):
    """Dataset for evaluation / 评估数据集"""

    def __init__(
        self,
        csv_files,
        config: EvaluationConfig,
        scalers
    ) -> None:
        self.csv_files = csv_files
        self.config = config
        self.scalers = scalers

    def __len__(self) -> int:
        return len(self.csv_files)

    def __getitem__(self, idx):
        import pandas as pd
        fp = self.csv_files[idx]

        df = pd.read_csv(fp)
        data = df.values.astype(np.float32)

        fixed_start = self.config.meta_cols
        fixed_end = fixed_start + self.config.fixed_features_dim
        dyn_start = self.config.dynamic_col_start
        dyn_end = dyn_start + self.config.dynamic_features_dim

        fixed = data[0, fixed_start:fixed_end]
        dynamic_abs = data[:, dyn_start:dyn_end]

        # Truncate or pad / 截断或填充
        L = dynamic_abs.shape[0]
        target_L = self.config.sequence_length
        if L >= target_L:
            dynamic_abs = dynamic_abs[:target_L]
        else:
            pad_n = target_L - L
            last = dynamic_abs[-1:].repeat(pad_n, axis=0)
            dynamic_abs = np.concatenate([dynamic_abs, last], axis=0)

        # Normalize / 标准化
        fixed_norm = safe_transform_np(
            self.scalers['fixed'],
            fixed.reshape(1, -1)
        ).flatten()
        abs_norm = safe_transform_np(self.scalers['abs'], dynamic_abs)
        delta_norm = safe_transform_np(
            self.scalers['delta'],
            (dynamic_abs[1:] - dynamic_abs[:-1]).astype(np.float32)
        )

        return (
            torch.tensor(fixed_norm, dtype=torch.float32),
            torch.tensor(abs_norm, dtype=torch.float32),
            torch.tensor(delta_norm, dtype=torch.float32)
        )

    @staticmethod
    def discover_files(data_dir: str):
        return sorted(Path(data_dir).glob("*.csv"))


def save_metrics(metrics: dict, config: EvaluationConfig) -> None:
    """Save metrics to CSV / 保存指标到CSV"""
    import pandas as pd

    df = pd.DataFrame([
        {"Metric": k, "Value": v} for k, v in metrics.items()
    ])

    def get_category(metric_name: str) -> str:
        if metric_name.startswith('Sequence_'):
            return 'Sequence'
        elif metric_name.startswith('Load_'):
            return 'Load Channel'
        elif metric_name.startswith('Disp_'):
            return 'Displacement'
        elif metric_name.startswith('BucklingLoad_'):
            return 'Buckling Load'
        elif metric_name.startswith('BucklingDisp_'):
            return 'Buckling Displacement'
        return 'Other'

    df['Category'] = df['Metric'].apply(get_category)
    df = df[['Category', 'Metric', 'Value']]
    df = df.sort_values(['Category', 'Metric'])

    output_path = Path(config.output_dir) / "evaluation_metrics.csv"
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    logging.info(f"Metrics saved to {output_path} / 指标已保存到{output_path}")


def save_buckling_data(raw_data: dict, config: EvaluationConfig) -> None:
    """
    Save buckling load predictions and ground truth to CSV for Origin plotting.
    保存屈曲荷载预测值和真实值到CSV，供Origin作图使用。
    """
    pred_loads = raw_data['pred_buckling_loads']
    true_loads = raw_data['true_buckling_loads']
    pred_disps = raw_data['pred_buckling_disps']
    true_disps = raw_data['true_buckling_disps']

    # Calculate errors / 计算误差
    errors = pred_loads - true_loads
    abs_errors = np.abs(errors)
    rel_errors = np.abs(errors / (true_loads + 1e-8)) * 100  # Percentage

    # Create DataFrame for buckling load / 创建屈曲荷载数据框
    df_load = pd.DataFrame({
        'Sample': np.arange(1, len(pred_loads) + 1),
        'True_Load': true_loads,
        'Pred_Load': pred_loads,
        'Error': errors,
        'AbsError': abs_errors,
        'RelError_Percent': rel_errors,
        'True_Disp': true_disps,
        'Pred_Disp': pred_disps,
    })

    output_path = Path(config.output_dir) / "buckling_load_comparison.csv"
    df_load.to_csv(output_path, index=False, encoding="utf-8-sig")
    logging.info(f"Buckling load data saved to {output_path} / 屈曲荷载数据已保存到{output_path}")

    return df_load


def plot_buckling_scatter(raw_data: dict, config: EvaluationConfig) -> None:
    """
    Plot scatter plot comparing predicted vs true buckling loads.
    绘制屈曲荷载预测值与真实值对比散点图。
    """
    pred_loads = raw_data['pred_buckling_loads']
    true_loads = raw_data['true_buckling_loads']

    # Set font for Chinese / 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. Scatter plot: Predicted vs True / 散点图：预测值 vs 真实值
    ax1 = axes[0]
    ax1.scatter(true_loads, pred_loads, alpha=0.6, edgecolors='k', linewidth=0.5)

    # Add diagonal line (perfect prediction) / 添加对角线（完美预测）
    min_val = min(true_loads.min(), pred_loads.min())
    max_val = max(true_loads.max(), pred_loads.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')

    ax1.set_xlabel('True Buckling Load / 真实屈曲荷载', fontsize=11)
    ax1.set_ylabel('Predicted Buckling Load / 预测屈曲荷载', fontsize=11)
    ax1.set_title('Buckling Load: Prediction vs Truth\n屈曲荷载：预测值与真实值对比', fontsize=12)
    ax1.legend()
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)

    # 2. Absolute error distribution histogram / 绝对误差分布直方图
    ax2 = axes[1]
    abs_rel_errors = np.abs((pred_loads - true_loads) / (true_loads + 1e-8)) * 100
    ax2.hist(abs_rel_errors, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    ax2.axvline(x=np.mean(abs_rel_errors), color='orange', linestyle='-', lw=2,
                label=f'Mean: {np.mean(abs_rel_errors):.2f}%')
    ax2.axvline(x=np.median(abs_rel_errors), color='green', linestyle='--', lw=2,
                label=f'Median: {np.median(abs_rel_errors):.2f}%')
    ax2.set_xlabel('Absolute Relative Error (%) / 绝对相对误差 (%)', fontsize=11)
    ax2.set_ylabel('Count / 频数', fontsize=11)
    ax2.set_title('Absolute Error Distribution\n绝对误差分布', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Error range statistics bar chart / 误差区间统计柱状图
    ax3 = axes[2]
    error_bins = [0, 1, 2, 3, 5, 10, 20, 50, 100]
    error_labels = ['0-1%', '1-2%', '2-3%', '3-5%', '5-10%', '10-20%', '20-50%', '50-100%', '>100%']
    counts = []
    for i in range(len(error_bins) - 1):
        mask = (abs_rel_errors >= error_bins[i]) & (abs_rel_errors < error_bins[i+1])
        counts.append(np.sum(mask))
    # Add last bin / 添加最后一个区间
    counts.append(np.sum(abs_rel_errors >= error_bins[-1]))

    colors = ['green', 'lightgreen', 'yellowgreen', 'yellow', 'orange', 'coral', 'red', 'darkred', 'purple']
    bars = ax3.bar(error_labels, counts, color=colors, edgecolor='black')

    # Add count labels on bars / 在柱子上添加计数标签
    for bar, count in zip(bars, counts):
        if count > 0:
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{int(count)}', ha='center', va='bottom', fontsize=10)

    ax3.set_xlabel('Error Range / 误差区间', fontsize=11)
    ax3.set_ylabel('Sample Count / 样本数量', fontsize=11)
    ax3.set_title('Error Distribution by Range\n误差区间分布统计', fontsize=12)
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    # Save figure / 保存图像
    output_path = Path(config.output_dir) / "buckling_load_scatter.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logging.info(f"Scatter plot saved to {output_path} / 散点图已保存到{output_path}")

    plt.close()


def analyze_buckling_errors(raw_data: dict, config: EvaluationConfig) -> dict:
    """
    Perform detailed error analysis on buckling load predictions.
    对屈曲荷载预测进行详细的误差分析。
    """
    pred_loads = raw_data['pred_buckling_loads']
    true_loads = raw_data['true_buckling_loads']

    errors = pred_loads - true_loads
    abs_errors = np.abs(errors)
    rel_errors = np.abs(errors / (true_loads + 1e-8)) * 100
    squared_errors = errors ** 2

    # Statistics / 统计信息
    error_stats = {
        'Error_Mean': float(np.mean(errors)),
        'Error_Std': float(np.std(errors)),
        'Error_Min': float(np.min(errors)),
        'Error_Max': float(np.max(errors)),
        'Error_Median': float(np.median(errors)),
        'AbsError_Mean': float(np.mean(abs_errors)),
        'AbsError_Std': float(np.std(abs_errors)),
        'AbsError_Median': float(np.median(abs_errors)),
        'RelError_Mean_Percent': float(np.mean(rel_errors)),
        'RelError_Std_Percent': float(np.std(rel_errors)),
        'RelError_Median_Percent': float(np.median(rel_errors)),
        'RelError_Max_Percent': float(np.max(rel_errors)),
        'RMSE': float(np.sqrt(np.mean(squared_errors))),
        'MAE': float(np.mean(abs_errors)),
        'N_samples': int(len(pred_loads)),
        'True_Load_Range': f"[{true_loads.min():.2f}, {true_loads.max():.2f}]",
        'Pred_Load_Range': f"[{pred_loads.min():.2f}, {pred_loads.max():.2f}]",
    }

    # Error range statistics / 误差区间统计
    error_bins = [0, 1, 2, 3, 5, 10, 20, 50, 100]
    error_labels = ['0-1%', '1-2%', '2-3%', '3-5%', '5-10%', '10-20%', '20-50%', '>50%']

    range_counts = []
    for i in range(len(error_bins) - 1):
        mask = (rel_errors >= error_bins[i]) & (rel_errors < error_bins[i+1])
        count = int(np.sum(mask))
        pct = count / len(rel_errors) * 100
        range_counts.append({
            'Range': error_labels[i],
            'Count': count,
            'Percentage': f'{pct:.1f}%'
        })
        error_stats[f'ErrorRange_{error_labels[i].replace("-", "_").replace("%", "")}'] = count

    # Last bin (>100%) / 最后一个区间
    count = int(np.sum(rel_errors >= error_bins[-1]))
    pct = count / len(rel_errors) * 100
    range_counts.append({
        'Range': '>50%',
        'Count': count,
        'Percentage': f'{pct:.1f}%'
    })

    # Save error range statistics to CSV / 保存误差区间统计到CSV
    df_range = pd.DataFrame(range_counts)
    output_path = Path(config.output_dir) / "error_range_statistics.csv"
    df_range.to_csv(output_path, index=False, encoding="utf-8-sig")

    # Print summary / 打印摘要
    print("\n" + "=" * 70)
    print("Buckling Load Error Analysis / 屈曲荷载误差分析")
    print("=" * 70)
    print(f"  Samples / 样本数: {error_stats['N_samples']}")
    print(f"  True Load Range / 真实荷载范围: {error_stats['True_Load_Range']}")
    print(f"  Pred Load Range / 预测荷载范围: {error_stats['Pred_Load_Range']}")
    print("-" * 70)
    print(f"  Mean Error / 平均误差: {error_stats['Error_Mean']:.4f}")
    print(f"  Std Error / 误差标准差: {error_stats['Error_Std']:.4f}")
    print(f"  Median Error / 误差中位数: {error_stats['Error_Median']:.4f}")
    print(f"  Error Range / 误差范围: [{error_stats['Error_Min']:.4f}, {error_stats['Error_Max']:.4f}]")
    print("-" * 70)
    print(f"  MAE / 平均绝对误差: {error_stats['MAE']:.4f}")
    print(f"  RMSE / 均方根误差: {error_stats['RMSE']:.4f}")
    print(f"  Mean Rel. Error / 平均相对误差: {error_stats['RelError_Mean_Percent']:.2f}%")
    print(f"  Max Rel. Error / 最大相对误差: {error_stats['RelError_Max_Percent']:.2f}%")
    print("-" * 70)
    print("  Error Range Distribution / 误差区间分布:")
    for rc in range_counts:
        print(f"    {rc['Range']:>8s}: {rc['Count']:4d} samples ({rc['Percentage']})")

    # Save error analysis to CSV / 保存误差分析到CSV
    df_error = pd.DataFrame([
        {"Statistic": k, "Value": v} for k, v in error_stats.items()
    ])
    output_path = Path(config.output_dir) / "buckling_error_analysis.csv"
    df_error.to_csv(output_path, index=False, encoding="utf-8-sig")
    logging.info(f"Error analysis saved to {output_path} / 误差分析已保存到{output_path}")

    return error_stats


def save_high_error_samples(
    raw_data: dict,
    test_files: list,
    config: EvaluationConfig,
    threshold_percent: float = 5.0
) -> pd.DataFrame:
    """
    Save samples with relative error > threshold to CSV for analysis.
    保存相对误差大于阈值的样本到CSV供分析。
    """
    pred_loads = raw_data['pred_buckling_loads']
    true_loads = raw_data['true_buckling_loads']
    pred_disps = raw_data['pred_buckling_disps']
    true_disps = raw_data['true_buckling_disps']

    errors = pred_loads - true_loads
    rel_errors = np.abs(errors / (true_loads + 1e-8)) * 100

    # Find high error samples / 找出高误差样本
    high_error_mask = rel_errors > threshold_percent
    high_error_indices = np.where(high_error_mask)[0]

    if len(high_error_indices) == 0:
        logging.info(f"No samples with relative error > {threshold_percent}%")
        return None

    # Build DataFrame / 构建数据框
    records = []
    for idx in high_error_indices:
        records.append({
            'Sample_Index': int(idx + 1),
            'Filename': Path(test_files[idx]).name,
            'Full_Path': str(test_files[idx]),
            'True_Load': float(true_loads[idx]),
            'Pred_Load': float(pred_loads[idx]),
            'Error': float(errors[idx]),
            'AbsError': float(np.abs(errors[idx])),
            'RelError_Percent': float(rel_errors[idx]),
            'True_Disp': float(true_disps[idx]),
            'Pred_Disp': float(pred_disps[idx]),
        })

    df = pd.DataFrame(records)
    df = df.sort_values('RelError_Percent', ascending=False)

    output_path = Path(config.output_dir) / f"high_error_samples_{threshold_percent}percent.csv"
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    logging.info(f"Found {len(high_error_indices)} samples with error > {threshold_percent}%")
    logging.info(f"High error samples saved to {output_path}")
    print(f"\n高误差样本 (> {threshold_percent}%): {len(high_error_indices)} 个")
    print(f"文件已保存: {output_path}")

    return df


def save_extreme_error_samples(
    raw_data: dict,
    test_files: list,
    config: EvaluationConfig,
    threshold_percent: float = 20.0
) -> pd.DataFrame:
    """
    Save samples with extremely high relative error for data quality inspection.
    保存绝对误差极大的样本供数据质量检查，可能需要剔除污染数据。

    Args:
        raw_data: Dictionary containing prediction results / 包含预测结果的字典
        test_files: List of test file paths / 测试文件路径列表
        config: Evaluation configuration / 评估配置
        threshold_percent: Error threshold for extreme cases / 极端误差阈值

    Returns:
        DataFrame with extreme error samples / 包含极端误差样本的数据框
    """
    pred_loads = raw_data['pred_buckling_loads']
    true_loads = raw_data['true_buckling_loads']
    pred_disps = raw_data['pred_buckling_disps']
    true_disps = raw_data['true_buckling_disps']

    errors = pred_loads - true_loads
    rel_errors = np.abs(errors / (true_loads + 1e-8)) * 100

    # Find extreme error samples / 找出极端误差样本
    extreme_mask = rel_errors > threshold_percent
    extreme_indices = np.where(extreme_mask)[0]

    if len(extreme_indices) == 0:
        logging.info(f"No samples with relative error > {threshold_percent}%")
        print(f"\n无极端误差样本 (> {threshold_percent}%)")
        return None

    # Build DataFrame / 构建数据框
    records = []
    for idx in extreme_indices:
        records.append({
            'Sample_Index': int(idx + 1),
            'Filename': Path(test_files[idx]).name,
            'Full_Path': str(test_files[idx]),
            'True_Load': float(true_loads[idx]),
            'Pred_Load': float(pred_loads[idx]),
            'Error': float(errors[idx]),
            'AbsError': float(np.abs(errors[idx])),
            'RelError_Percent': float(rel_errors[idx]),
            'True_Disp': float(true_disps[idx]),
            'Pred_Disp': float(pred_disps[idx]),
            'Note': 'Check for data quality issues / 检查数据质量问题'
        })

    df = pd.DataFrame(records)
    df = df.sort_values('RelError_Percent', ascending=False)

    output_path = Path(config.output_dir) / f"extreme_error_samples_{threshold_percent}percent.csv"
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    logging.info(f"Found {len(extreme_indices)} EXTREME error samples (> {threshold_percent}%)")
    logging.info(f"Extreme error samples saved to {output_path}")

    print("\n" + "=" * 70)
    print(f"[WARNING] Extreme error samples (> {threshold_percent}%): {len(extreme_indices)}")
    print("=" * 70)
    print(f"File saved: {output_path}")
    print("Recommend checking data quality for these samples, consider removing contaminated data.")

    # Print top 10 extreme errors / 打印前10个极端误差
    print("\nTop 10 samples with largest errors:")
    print("-" * 70)
    for i, row in df.head(10).iterrows():
        print(f"  {row['Filename']}: True={row['True_Load']:.2f}, Pred={row['Pred_Load']:.2f}, Error={row['RelError_Percent']:.1f}%")

    return df


def plot_high_error_curves(
    model,
    test_files: list,
    scalers: dict,
    config: EvaluationConfig,
    raw_data: dict,
    device: torch.device,
    threshold_percent: float = 5.0,
    max_plots: int = 10
) -> None:
    """
    Plot load-displacement curves for high error samples with buckling points marked.
    绘制高误差样本的荷载-位移曲线并标注屈曲点。
    """
    pred_loads = raw_data['pred_buckling_loads']
    true_loads = raw_data['true_buckling_loads']

    errors = pred_loads - true_loads
    rel_errors = np.abs(errors / (true_loads + 1e-8)) * 100
    high_error_mask = rel_errors > threshold_percent
    high_error_indices = np.where(high_error_mask)[0]

    if len(high_error_indices) == 0:
        return

    # Limit plots / 限制绘图数量
    plot_indices = high_error_indices[:max_plots]

    from utils.buckling import extract_buckling_index_numpy

    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    n_plots = len(plot_indices)
    fig, axes = plt.subplots(n_plots, 2, figsize=(14, 4 * n_plots))

    if n_plots == 1:
        axes = axes.reshape(1, -1)

    for row, idx in enumerate(plot_indices):
        # Load original data / 加载原始数据
        df_data = pd.read_csv(test_files[idx])
        data = df_data.values.astype(np.float32)

        dyn_start = config.dynamic_col_start
        dyn_end = dyn_start + config.dynamic_features_dim
        dynamic_abs = data[:, dyn_start:dyn_end]

        # Get load and displacement / 获取荷载和位移
        load_seq = dynamic_abs[:, 0]
        disp_col = config.load_dim + config.buckling_disp_index
        disp_seq = dynamic_abs[:, disp_col]

        # Predict sequence / 预测序列
        fixed = data[0, config.meta_cols:config.meta_cols + config.fixed_features_dim]
        fixed_norm = safe_transform_np(scalers['fixed'], fixed.reshape(1, -1)).flatten()
        abs_norm = safe_transform_np(scalers['abs'], dynamic_abs)

        fixed_tensor = torch.tensor(fixed_norm, dtype=torch.float32)
        abs_tensor = torch.tensor(abs_norm, dtype=torch.float32)

        pred_seq = predict_sequence(model, fixed_tensor, abs_tensor, scalers, device)
        pred_load_seq = pred_seq[:, 0]
        pred_disp_seq = pred_seq[:, disp_col]

        # Find buckling indices using peak_load method / 使用峰值荷载法查找屈曲索引
        true_buck_idx = extract_buckling_index_numpy(
            load_seq, disp_seq, method="peak_load"
        )
        pred_buck_idx = extract_buckling_index_numpy(
            pred_load_seq, pred_disp_seq, method="peak_load"
        )

        # Left plot: True curve / 左图：真实曲线
        ax1 = axes[row, 0]
        ax1.plot(disp_seq, load_seq, 'b-', lw=1.5, label='Load-Disp Curve')
        ax1.scatter([disp_seq[true_buck_idx]], [load_seq[true_buck_idx]],
                   c='red', s=200, marker='*', zorder=5, label=f'Buckling Point (Load={load_seq[true_buck_idx]:.2f})')
        ax1.axhline(y=load_seq[true_buck_idx], color='r', linestyle='--', alpha=0.5)
        ax1.set_xlabel(f'Displacement (Node {config.buckling_node}, {config.buckling_axis})', fontsize=10)
        ax1.set_ylabel('Load', fontsize=10)
        ax1.set_title(f'TRUE: {Path(test_files[idx]).name}\nBuckling Load = {load_seq[true_buck_idx]:.2f}', fontsize=11)
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)

        # Right plot: Predicted curve / 右图：预测曲线
        ax2 = axes[row, 1]
        ax2.plot(pred_disp_seq, pred_load_seq, 'b-', lw=1.5, label='Pred Load-Disp Curve')
        ax2.scatter([pred_disp_seq[pred_buck_idx]], [pred_load_seq[pred_buck_idx]],
                   c='red', s=200, marker='*', zorder=5, label=f'Pred Buckling (Load={pred_load_seq[pred_buck_idx]:.2f})')
        ax2.axhline(y=pred_load_seq[pred_buck_idx], color='r', linestyle='--', alpha=0.5)
        # Add true buckling load line for comparison / 添加真实屈曲荷载线用于对比
        ax2.axhline(y=load_seq[true_buck_idx], color='green', linestyle=':', alpha=0.7, label=f'True Buckling = {load_seq[true_buck_idx]:.2f}')
        ax2.set_xlabel(f'Displacement (Node {config.buckling_node}, {config.buckling_axis})', fontsize=10)
        ax2.set_ylabel('Load', fontsize=10)
        ax2.set_title(f'PREDICTED | Error: {rel_errors[idx]:.1f}%\nPred={pred_load_seq[pred_buck_idx]:.2f}, True={load_seq[true_buck_idx]:.2f}', fontsize=11)
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = Path(config.output_dir) / f"high_error_curves_{threshold_percent}percent.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logging.info(f"High error curves saved to {output_path}")
    plt.close()


def analyze_buckling_detection_params(
    model,
    test_files: list,
    scalers: dict,
    config: EvaluationConfig,
    raw_data: dict,
    device: torch.device,
    n_samples: int = 20
) -> None:
    """
    Analyze the effect of smoothing and slope threshold on buckling detection.
    分析平滑和斜率阈值对屈曲检测的影响。
    """
    from utils.buckling import extract_buckling_index_numpy, moving_average_1d_numpy

    np.random.seed(42)
    sample_indices = np.random.choice(len(test_files), min(n_samples, len(test_files)), replace=False)

    results = []

    for idx in sample_indices:
        # Load data / 加载数据
        df_data = pd.read_csv(test_files[idx])
        data = df_data.values.astype(np.float32)

        dyn_start = config.dynamic_col_start
        dyn_end = dyn_start + config.dynamic_features_dim
        dynamic_abs = data[:, dyn_start:dyn_end]

        load_seq = dynamic_abs[:, 0]
        disp_col = config.load_dim + config.buckling_disp_index
        disp_seq = dynamic_abs[:, disp_col]

        # Predict / 预测
        fixed = data[0, config.meta_cols:config.meta_cols + config.fixed_features_dim]
        fixed_norm = safe_transform_np(scalers['fixed'], fixed.reshape(1, -1)).flatten()
        abs_norm = safe_transform_np(scalers['abs'], dynamic_abs)

        fixed_tensor = torch.tensor(fixed_norm, dtype=torch.float32)
        abs_tensor = torch.tensor(abs_norm, dtype=torch.float32)

        pred_seq = predict_sequence(model, fixed_tensor, abs_tensor, scalers, device)
        pred_load_seq = pred_seq[:, 0]
        pred_disp_seq = pred_seq[:, disp_col]

        # True buckling load (peak_load method) / 真实屈曲荷载（峰值荷载法）
        true_idx = extract_buckling_index_numpy(load_seq, disp_seq, method="peak_load")
        true_load = load_seq[true_idx]

        # Test different methods / 测试不同方法
        methods = [
            ("peak_load", 0, 1),
            ("stiffness", 0.0, 1),
            ("stiffness", -0.5, 1),
            ("stiffness", -1.0, 1),
        ]

        for method, threshold, window in methods:
            if method == "peak_load":
                pred_idx = extract_buckling_index_numpy(
                    pred_load_seq, pred_disp_seq, method="peak_load"
                )
            else:
                pred_load_sm = moving_average_1d_numpy(pred_load_seq, window) if window > 1 else pred_load_seq
                pred_disp_sm = moving_average_1d_numpy(pred_disp_seq, window) if window > 1 else pred_disp_seq
                pred_idx = extract_buckling_index_numpy(
                    pred_load_sm, pred_disp_sm,
                    slope_threshold=threshold,
                    method="stiffness"
                )

            pred_load = pred_load_seq[pred_idx]
            error = abs(pred_load - true_load) / (true_load + 1e-8) * 100

            results.append({
                'Sample': Path(test_files[idx]).name,
                'Method': method,
                'Threshold': threshold,
                'Window': window,
                'True_Load': true_load,
                'Pred_Load': pred_load,
                'RelError_Percent': error
            })

    df = pd.DataFrame(results)

    # Summary by method / 按方法汇总
    summary = df.groupby(['Method', 'Threshold', 'Window']).agg({
        'RelError_Percent': ['mean', 'std', 'max']
    }).round(2)
    summary.columns = ['MeanError_%', 'StdError_%', 'MaxError_%']
    summary = summary.reset_index()

    # Save / 保存
    output_path = Path(config.output_dir) / "buckling_detection_methods_comparison.csv"
    summary.to_csv(output_path, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 70)
    print("屈曲检测方法对比 / Buckling Detection Methods Comparison")
    print("=" * 70)
    print(summary.to_string(index=False))
    print(f"\n详细结果已保存: {output_path}")

    return summary


def print_metrics(metrics: dict) -> None:
    """Print metrics summary / 打印指标摘要"""
    print("\n" + "=" * 80)
    print("Model Evaluation Results / 模型评估结果")
    print("=" * 80)

    categories = {
        'Sequence': 'Full Sequence Metrics / 全序列指标',
        'Load Channel': 'Load Channel Metrics / 荷载通道指标',
        'Displacement': 'Displacement Metrics / 位移指标',
        'Buckling Load': 'Buckling Load Metrics / 屈曲荷载指标',
        'Buckling Displacement': 'Buckling Displacement Metrics / 屈曲位移指标',
    }

    for cat, cat_name in categories.items():
        print(f"\n{cat_name}:")
        print("-" * 60)
        for k, v in metrics.items():
            if k.startswith(cat.replace(' ', '') + '_') or \
               (cat == 'Sequence' and k.startswith('Sequence_')):
                print(f"  {k}: {v:.6e}")


def load_model_and_config(model_path: str, device: str):
    """Load trained model and configuration / 加载训练好的模型和配置"""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    config = checkpoint.get('config', None)
    if config is None:
        raise ValueError("Config not found in checkpoint / 检查点中未找到配置")

    model = BucklingPredictor(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    scalers = checkpoint.get('scalers', None)

    return model, config, scalers


def main() -> None:
    """Main function / 主函数"""
    # Setup logging / 设置日志
    log_dir = Path("./results/evaluation")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"evaluate_model_{datetime.now():%Y%m%d_%H%M%S}.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler()
        ]
    )

    logging.info("=" * 60)
    logging.info("Model Evaluation / 模型评估")
    logging.info("=" * 60)

    # Initialize config / 初始化配置
    config = EvaluationConfig()
    device = torch.device(config.device)

    # Load model / 加载模型
    logging.info(f"Loading model from {config.model_path}")
    model, model_config, scalers = load_model_and_config(config.model_path, config.device)
    model = model.to(device)

    # Load scalers if not in checkpoint / 如果检查点中没有标准化器则加载
    if scalers is None:
        scalers = joblib.load(config.scalers_path)
        logging.info(f"Loaded scalers from {config.scalers_path}")

    # Discover and split files / 发现和划分文件
    all_files = EvaluationDataset.discover_files(config.data_dir)
    logging.info(f"Found {len(all_files)} CSV files / 找到{len(all_files)}个CSV文件")

    # Split with same seed as training / 使用与训练相同的种子划分
    rng = np.random.default_rng(config.seed)
    indices = np.arange(len(all_files))
    rng.shuffle(indices)

    n_total = len(indices)
    n_train = int(config.train_ratio * n_total)
    n_val = int(config.val_ratio * n_total)

    test_idx = indices[n_train + n_val:]
    test_files = [all_files[i] for i in test_idx]

    logging.info(f"Test set size: {len(test_files)} / 测试集大小: {len(test_files)}")

    # Create test dataset and loader / 创建测试数据集和加载器
    test_dataset = EvaluationDataset(test_files, config, scalers)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0
    )

    # Evaluate / 评估
    logging.info("Evaluating model on test set / 在测试集上评估模型...")
    metrics, raw_data = evaluate_model(model, test_loader, scalers, config, device)

    # Save results / 保存结果
    save_metrics(metrics, config)

    # Save buckling load data for Origin plotting / 保存屈曲荷载数据供Origin作图
    save_buckling_data(raw_data, config)

    # Plot scatter plots / 绘制散点图
    plot_buckling_scatter(raw_data, config)

    # Perform error analysis / 进行误差分析
    analyze_buckling_errors(raw_data, config)

    # Save high error samples (> 5%) / 保存高误差样本
    save_high_error_samples(raw_data, test_files, config, threshold_percent=5.0)

    # Save extreme error samples (> 20%) for data quality check / 保存极端误差样本供数据质量检查
    save_extreme_error_samples(raw_data, test_files, config, threshold_percent=15.0)

    # Plot high error curves / 绘制高误差样本曲线
    plot_high_error_curves(model, test_files, scalers, config, raw_data, device, threshold_percent=5.0)

    # Analyze buckling detection parameters / 分析屈曲检测参数影响
    analyze_buckling_detection_params(model, test_files, scalers, config, raw_data, device)

    # Print summary / 打印摘要
    print_metrics(metrics)

    logging.info("=" * 60)
    logging.info("Evaluation completed / 评估完成")
    logging.info("=" * 60)


if __name__ == "__main__":
    main()