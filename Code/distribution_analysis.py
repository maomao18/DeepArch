# -*- coding: utf-8 -*-
"""
Parameter Distribution Analysis Script / 参数分布分析脚本
===========================================================
Analyze distribution of 16 fixed parameters in the dataset.
分析数据集中16个固定参数的分布情况。

Functions:
    1. Load parameter statistics from CSV
       从CSV加载参数统计
    2. Calculate distribution statistics (mean, std, skewness, kurtosis, etc.)
       计算分布统计（均值、标准差、偏度、峰度等）
    3. Generate distribution plots (histograms, box plots, etc.)
       生成分布图（直方图、箱线图等）
    4. Export statistics to CSV for Origin plotting
       导出统计数据到CSV用于Origin精细绘图

Author: DeepArches Project
"""

import os
# Fix OpenMP conflict / 修复OpenMP冲突
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import logging
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from tqdm import tqdm

warnings.filterwarnings('ignore')

# Chinese font support / 中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# =============================================================================
# Configuration / 配置
# =============================================================================
@dataclass
class DistributionConfig:
    """Configuration for distribution analysis / 分布分析配置"""
    
    # Input/Output paths / 输入输出路径
    input_csv: str = r"./results/analysis/parameter_statistics.csv"
    output_dir: str = r"./results/analysis"
    
    # Variable names (16 parameters) / 变量名称（16个参数）
    param_names: Tuple[str, ...] = (
        "I0",      # 截面质量 / Section quality
        "A11",     # 压缩刚度 / Compression stiffness
        "B11",     # 压-弯耦合刚度 / Compression-bending coupling stiffness
        "D11",     # 弯曲刚度 / Bending stiffness
        "L",       # 跨径 / Span
        "f",       # 拱高 / Arch height
        "b",       # 拱截面宽 / Arch section width
        "h",       # 拱截面高 / Arch section height
        "S",       # 拱长 / Arch length
        "lambda",  # 长细比 / Slenderness ratio
        "KXL",     # 左X弹性支撑系数 / Left X elastic support coefficient
        "KYL",     # 左Y弹性支撑系数 / Left Y elastic support coefficient
        "KZL",     # 左转动弹性支撑系数 / Left rotation elastic support coefficient
        "KXR",     # 右X弹性支撑系数 / Right X elastic support coefficient
        "KYR",     # 右Y弹性支撑系数 / Right Y elastic support coefficient
        "KZR",     # 右转动弹性支撑系数 / Right rotation elastic support coefficient
    )
    
    # Chinese names for plotting / 中文标签用于绘图
    param_names_cn: Tuple[str, ...] = (
        "截面质量I0",
        "压缩刚度A11",
        "压-弯耦合刚度B11",
        "弯曲刚度D11",
        "跨径L",
        "拱高f",
        "拱截面宽b",
        "拱截面高h",
        "拱长S",
        "长细比λ",
        "KXL",
        "KYL",
        "KZL",
        "KXR",
        "KYR",
        "KZR",
    )
    
    # Plotting settings / 绘图设置
    figure_size_single: Tuple[int, int] = (8, 6)
    figure_size_combined: Tuple[int, int] = (20, 16)
    dpi: int = 150
    bins: int = 30
    
    def __post_init__(self) -> None:
        """Post-initialization / 后初始化"""
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)


# =============================================================================
# Distribution Statistics Functions / 分布统计函数
# =============================================================================
def load_data(config: DistributionConfig) -> pd.DataFrame:
    """
    Load parameter statistics from CSV / 从CSV加载参数统计
    
    Args:
        config: Distribution configuration / 分布配置
        
    Returns:
        DataFrame with parameters / 包含参数的DataFrame
    """
    input_path = Path(config.input_csv)
    
    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}\n"
            f"Please run analyze_data.py first to generate the parameter statistics.\n"
            f"输入文件未找到: {input_path}\n"
            f"请先运行analyze_data.py生成参数统计数据。"
        )
    
    df = pd.read_csv(input_path)
    
    # Only keep 16 fixed parameters / 仅保留16个固定参数
    param_cols = list(config.param_names)
    df = df[param_cols]
    
    logging.info(f"Loaded {len(df)} samples with {len(param_cols)} parameters / "
                 f"加载了{len(df)}个样本，{len(param_cols)}个参数")
    
    return df


def calculate_distribution_statistics(df: pd.DataFrame, config: DistributionConfig) -> pd.DataFrame:
    """
    Calculate comprehensive distribution statistics / 计算综合分布统计
    
    Args:
        df: DataFrame with parameters / 包含参数的DataFrame
        config: Distribution configuration / 分布配置
        
    Returns:
        DataFrame with distribution statistics / 包含分布统计的DataFrame
    """
    stats_list = []
    
    for col in df.columns:
        data = df[col].values
        
        # Basic statistics / 基本统计
        mean = np.mean(data)
        std = np.std(data)
        median = np.median(data)
        minimum = np.min(data)
        maximum = np.max(data)
        data_range = maximum - minimum
        
        # Quantiles / 分位数
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        
        # Shape statistics / 形状统计
        skewness = stats.skew(data)
        kurtosis = stats.kurtosis(data)
        
        # Coefficient of variation / 变异系数
        cv = std / mean if mean != 0 else np.nan
        
        # Normality test / 正态性检验
        if len(data) >= 20:
            stat_sw, p_sw = stats.shapiro(data[:5000] if len(data) > 5000 else data)
        else:
            stat_sw, p_sw = np.nan, np.nan
        
        stats_list.append({
            "Parameter": col,
            "Parameter_CN": config.param_names_cn[list(config.param_names).index(col)],
            "Count": len(data),
            "Mean": mean,
            "Std": std,
            "Median": median,
            "Min": minimum,
            "Max": maximum,
            "Range": data_range,
            "Q1": q1,
            "Q3": q3,
            "IQR": iqr,
            "Skewness": skewness,
            "Kurtosis": kurtosis,
            "CV": cv,
            "Shapiro_W": stat_sw,
            "Shapiro_p": p_sw,
        })
    
    stats_df = pd.DataFrame(stats_list)
    return stats_df


def save_statistics(stats_df: pd.DataFrame, config: DistributionConfig) -> None:
    """
    Save statistics to CSV / 保存统计到CSV
    
    Args:
        stats_df: Statistics DataFrame / 统计DataFrame
        config: Distribution configuration / 分布配置
    """
    output_path = Path(config.output_dir) / "distribution_statistics.csv"
    stats_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    logging.info(f"Distribution statistics saved to {output_path} / "
                 f"分布统计已保存到{output_path}")


def plot_histograms(df: pd.DataFrame, config: DistributionConfig, use_chinese: bool = True) -> None:
    """
    Plot histograms for all parameters / 绘制所有参数的直方图
    
    Args:
        df: DataFrame with parameters / 包含参数的DataFrame
        config: Distribution configuration / 分布配置
        use_chinese: Whether to use Chinese labels / 是否使用中文标签
    """
    n_params = len(df.columns)
    n_cols = 4
    n_rows = (n_params + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=config.figure_size_combined)
    axes = axes.flatten()
    
    for i, col in enumerate(df.columns):
        ax = axes[i]
        data = df[col].values
        
        # Choose label / 选择标签
        if use_chinese:
            label = config.param_names_cn[list(config.param_names).index(col)]
        else:
            label = col
        
        # Plot histogram with KDE / 绘制带KDE的直方图
        sns.histplot(data, bins=config.bins, kde=True, ax=ax, color='steelblue', edgecolor='white')
        
        ax.set_title(label, fontsize=11, fontweight='bold')
        ax.set_xlabel("Value / 值")
        ax.set_ylabel("Count / 频数")
        
        # Add statistics text / 添加统计文本
        mean = np.mean(data)
        std = np.std(data)
        ax.text(0.95, 0.95, f"μ={mean:.2e}\nσ={std:.2e}",
                transform=ax.transAxes, fontsize=8,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Hide unused axes / 隐藏未使用的轴
    for i in range(n_params, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle("Parameter Distribution Histograms / 参数分布直方图", 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_path = Path(config.output_dir) / "distribution_histograms.png"
    plt.savefig(output_path, dpi=config.dpi, bbox_inches='tight')
    logging.info(f"Histograms saved to {output_path} / 直方图已保存到{output_path}")
    
    plt.close()


def plot_box_plots(df: pd.DataFrame, config: DistributionConfig, use_chinese: bool = True) -> None:
    """
    Plot box plots for all parameters / 绘制所有参数的箱线图
    
    Args:
        df: DataFrame with parameters / 包含参数的DataFrame
        config: Distribution configuration / 分布配置
        use_chinese: Whether to use Chinese labels / 是否使用中文标签
    """
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Prepare data for box plot / 准备箱线图数据
    data_list = []
    labels = []
    
    for col in df.columns:
        data_list.append(df[col].values)
        if use_chinese:
            labels.append(config.param_names_cn[list(config.param_names).index(col)])
        else:
            labels.append(col)
    
    # Create box plot / 创建箱线图
    bp = ax.boxplot(data_list, patch_artist=True, labels=labels)
    
    # Color the boxes / 给箱体上色
    colors = plt.cm.Set3(np.linspace(0, 1, len(data_list)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax.set_ylabel("Value / 值")
    ax.set_title("Parameter Box Plots / 参数箱线图", fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    output_path = Path(config.output_dir) / "distribution_boxplots.png"
    plt.savefig(output_path, dpi=config.dpi, bbox_inches='tight')
    logging.info(f"Box plots saved to {output_path} / 箱线图已保存到{output_path}")
    
    plt.close()


def plot_normalized_comparison(df: pd.DataFrame, config: DistributionConfig, use_chinese: bool = True) -> None:
    """
    Plot normalized distribution comparison / 绘制归一化分布比较图
    
    Args:
        df: DataFrame with parameters / 包含参数的DataFrame
        config: Distribution configuration / 分布配置
        use_chinese: Whether to use Chinese labels / 是否使用中文标签
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Normalize data / 归一化数据
    df_normalized = (df - df.mean()) / df.std()
    
    # Plot KDE for each parameter / 绘制每个参数的KDE
    colors = plt.cm.tab20(np.linspace(0, 1, len(df.columns)))
    
    for i, col in enumerate(df.columns):
        data = df_normalized[col].values
        if use_chinese:
            label = config.param_names_cn[list(config.param_names).index(col)]
        else:
            label = col
        
        sns.kdeplot(data, ax=ax, label=label, color=colors[i], linewidth=1.5)
    
    ax.set_xlabel("Normalized Value (Z-score) / 归一化值（Z分数）")
    ax.set_ylabel("Density / 密度")
    ax.set_title("Normalized Parameter Distributions / 归一化参数分布", 
                 fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = Path(config.output_dir) / "distribution_normalized.png"
    plt.savefig(output_path, dpi=config.dpi, bbox_inches='tight')
    logging.info(f"Normalized distribution plot saved to {output_path} / "
                 f"归一化分布图已保存到{output_path}")
    
    plt.close()


def plot_pairwise_scatter(df: pd.DataFrame, config: DistributionConfig, use_chinese: bool = True) -> None:
    """
    Plot pairwise scatter matrix for selected parameters / 绘制选定参数的成对散点矩阵
    
    Args:
        df: DataFrame with parameters / 包含参数的DataFrame
        config: Distribution configuration / 分布配置
        use_chinese: Whether to use Chinese labels / 是否使用中文标签
    """
    # Select key parameters for pair plot (to avoid too large figure)
    # 选择关键参数进行配对图（避免图像过大）
    key_params = ["I0", "L", "f", "b", "h", "lambda", "Pcr"] if "Pcr" in df.columns else \
                 ["I0", "L", "f", "b", "h", "lambda"]
    
    # Filter to available columns / 过滤到可用列
    available_params = [p for p in key_params if p in df.columns]
    
    if len(available_params) < 2:
        logging.warning("Not enough parameters for pair plot / 参数不足，无法绘制配对图")
        return
    
    df_subset = df[available_params].copy()
    
    # Rename columns for display / 重命名列用于显示
    if use_chinese:
        rename_dict = {p: config.param_names_cn[list(config.param_names).index(p)] 
                       for p in available_params if p in config.param_names}
        df_subset = df_subset.rename(columns=rename_dict)
    
    # Create pair plot / 创建配对图
    g = sns.pairplot(df_subset, diag_kind='kde', corner=True, 
                     plot_kws={'alpha': 0.5, 's': 10}, 
                     diag_kws={'linewidth': 2})
    
    g.fig.suptitle("Pairwise Parameter Scatter Matrix / 参数成对散点矩阵", 
                   y=1.02, fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    output_path = Path(config.output_dir) / "distribution_pairplot.png"
    plt.savefig(output_path, dpi=config.dpi, bbox_inches='tight')
    logging.info(f"Pair plot saved to {output_path} / 配对图已保存到{output_path}")
    
    plt.close()


def print_distribution_summary(stats_df: pd.DataFrame) -> None:
    """
    Print distribution summary / 打印分布摘要
    
    Args:
        stats_df: Statistics DataFrame / 统计DataFrame
    """
    print("\n" + "=" * 100)
    print("Parameter Distribution Statistics Summary / 参数分布统计摘要")
    print("=" * 100)
    
    print("\n{:<15} {:<12} {:<12} {:<12} {:<12} {:<12} {:<12} {:<12}".format(
        "Parameter", "Mean", "Std", "Min", "Max", "Skewness", "Kurtosis", "CV"
    ))
    print("-" * 100)
    
    for _, row in stats_df.iterrows():
        print("{:<15} {:<12.4e} {:<12.4e} {:<12.4e} {:<12.4e} {:<12.4f} {:<12.4f} {:<12.4f}".format(
            row["Parameter"],
            row["Mean"],
            row["Std"],
            row["Min"],
            row["Max"],
            row["Skewness"],
            row["Kurtosis"],
            row["CV"]
        ))
    
    print("\n" + "-" * 100)
    print("Interpretation Guidelines / 解释指南:")
    print("  Skewness: |s| < 0.5 (symmetric), 0.5-1 (moderate), > 1 (highly skewed)")
    print("            偏度: |s| < 0.5 (对称), 0.5-1 (中等), > 1 (高度偏斜)")
    print("  Kurtosis: k > 0 (leptokurtic/peaked), k < 0 (platykurtic/flat)")
    print("            峰度: k > 0 (尖峰), k < 0 (扁峰)")
    print("  CV: < 0.1 (low variability), 0.1-0.3 (moderate), > 0.3 (high variability)")
    print("      变异系数: < 0.1 (低变异性), 0.1-0.3 (中等), > 0.3 (高变异性)")


# =============================================================================
# Main Function / 主函数
# =============================================================================
def main() -> None:
    """Main function / 主函数"""
    # Setup logging / 设置日志
    log_dir = Path("./results/analysis")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"distribution_analysis_{datetime.now():%Y%m%d_%H%M%S}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler()
        ]
    )
    
    logging.info("=" * 60)
    logging.info("Parameter Distribution Analysis / 参数分布分析")
    logging.info("=" * 60)
    
    # Initialize config / 初始化配置
    config = DistributionConfig()
    
    # Load data / 加载数据
    df = load_data(config)
    
    # Calculate statistics / 计算统计
    logging.info("Calculating distribution statistics / 计算分布统计...")
    stats_df = calculate_distribution_statistics(df, config)
    
    # Save statistics / 保存统计
    save_statistics(stats_df, config)
    
    # Generate plots / 生成图像
    logging.info("Generating distribution plots / 生成分布图...")
    plot_histograms(df, config)
    plot_box_plots(df, config)
    plot_normalized_comparison(df, config)
    
    # Print summary / 打印摘要
    print_distribution_summary(stats_df)
    
    logging.info("=" * 60)
    logging.info("Analysis completed / 分析完成")
    logging.info("=" * 60)


if __name__ == "__main__":
    main()