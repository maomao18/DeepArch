# -*- coding: utf-8 -*-
"""
Pearson Correlation Analysis Script / 皮尔逊相关系数分析脚本
=============================================================
Calculate and visualize Pearson correlation coefficients for 17 variables.
计算并可视化17个变量的皮尔逊相关系数。

Functions:
    1. Load parameter statistics table from CSV
       从CSV加载参数统计表格
    2. Calculate Pearson correlation coefficients for 17 variables
       计算17个变量的皮尔逊相关系数
    3. Generate correlation heatmap for intuitive analysis
       生成相关系数热力图用于直观分析
    4. Export correlation matrix to CSV for Origin plotting
       导出相关系数矩阵到CSV用于Origin精细绘图

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
class CorrelationConfig:
    """Configuration for correlation analysis / 相关系数分析配置"""
    
    # Input/Output paths / 输入输出路径
    input_csv: str = r"./results/analysis/parameter_statistics.csv"
    output_dir: str = r"./results/analysis"
    
    # Variable names (16 parameters + buckling load) / 变量名称（16个参数 + 屈曲荷载）
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
        "Pcr",     # 屈曲荷载 / Buckling load
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
        "屈曲荷载Pcr",
    )
    
    # Plotting settings / 绘图设置
    figure_size: Tuple[int, int] = (14, 12)
    dpi: int = 150
    cmap: str = "RdBu_r"  # Red-Blue reversed / 红蓝反转
    
    def __post_init__(self) -> None:
        """Post-initialization / 后初始化"""
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)


# =============================================================================
# Correlation Analysis Functions / 相关系数分析函数
# =============================================================================
def load_data(config: CorrelationConfig) -> pd.DataFrame:
    """
    Load parameter statistics from CSV / 从CSV加载参数统计
    
    Args:
        config: Correlation configuration / 相关系数配置
        
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
    logging.info(f"Loaded {len(df)} samples from {input_path} / 从{input_path}加载了{len(df)}个样本")
    
    return df


def calculate_pearson_correlation(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate Pearson correlation coefficients / 计算皮尔逊相关系数
    
    Args:
        df: DataFrame with parameters / 包含参数的DataFrame
        
    Returns:
        Tuple of (correlation_matrix, p_value_matrix) / (相关系数矩阵, p值矩阵)元组
    """
    n_vars = len(df.columns)
    corr_matrix = np.zeros((n_vars, n_vars))
    p_matrix = np.zeros((n_vars, n_vars))
    
    columns = df.columns.tolist()
    
    for i, col1 in enumerate(columns):
        for j, col2 in enumerate(columns):
            x = df[col1].values
            y = df[col2].values
            
            # Calculate Pearson correlation / 计算皮尔逊相关系数
            corr, p_value = stats.pearsonr(x, y)
            corr_matrix[i, j] = corr
            p_matrix[i, j] = p_value
    
    return corr_matrix, p_matrix


def create_correlation_dataframe(
    corr_matrix: np.ndarray,
    columns: List[str]
) -> pd.DataFrame:
    """
    Create correlation DataFrame / 创建相关系数DataFrame
    
    Args:
        corr_matrix: Correlation matrix / 相关系数矩阵
        columns: Column names / 列名
        
    Returns:
        DataFrame with correlation coefficients / 包含相关系数的DataFrame
    """
    return pd.DataFrame(corr_matrix, index=columns, columns=columns)


def save_correlation_matrix(
    corr_df: pd.DataFrame,
    p_matrix: np.ndarray,
    config: CorrelationConfig
) -> None:
    """
    Save correlation matrix to CSV / 保存相关系数矩阵到CSV
    
    Args:
        corr_df: Correlation DataFrame / 相关系数DataFrame
        p_matrix: P-value matrix / P值矩阵
        config: Correlation configuration / 相关系数配置
    """
    output_dir = Path(config.output_dir)
    
    # Save correlation matrix / 保存相关系数矩阵
    corr_path = output_dir / "correlation_matrix.csv"
    corr_df.to_csv(corr_path, encoding="utf-8-sig")
    logging.info(f"Correlation matrix saved to {corr_path} / 相关系数矩阵已保存到{corr_path}")
    
    # Save p-value matrix / 保存P值矩阵
    p_df = pd.DataFrame(p_matrix, index=corr_df.index, columns=corr_df.columns)
    p_path = output_dir / "correlation_pvalues.csv"
    p_df.to_csv(p_path, encoding="utf-8-sig")
    logging.info(f"P-value matrix saved to {p_path} / P值矩阵已保存到{p_path}")
    
    # Save correlation with Pcr (last column) / 保存与Pcr的相关系数（最后一列）
    pcr_corr = corr_df.iloc[:, -1].copy()
    pcr_pvalues = p_df.iloc[:, -1].copy()
    
    pcr_df = pd.DataFrame({
        "Variable": corr_df.index.tolist(),
        "Correlation_with_Pcr": pcr_corr.values,
        "P_value": pcr_pvalues.values
    })
    pcr_path = output_dir / "correlation_with_Pcr.csv"
    pcr_df.to_csv(pcr_path, index=False, encoding="utf-8-sig")
    logging.info(f"Correlation with Pcr saved to {pcr_path} / 与Pcr的相关系数已保存到{pcr_path}")


def plot_correlation_heatmap(
    corr_matrix: np.ndarray,
    config: CorrelationConfig,
    use_chinese: bool = True
) -> None:
    """
    Plot correlation heatmap / 绘制相关系数热力图
    
    Args:
        corr_matrix: Correlation matrix / 相关系数矩阵
        config: Correlation configuration / 相关系数配置
        use_chinese: Whether to use Chinese labels / 是否使用中文标签
    """
    fig, ax = plt.subplots(figsize=config.figure_size)
    
    # Choose labels / 选择标签
    labels = config.param_names_cn if use_chinese else config.param_names
    
    # Create heatmap / 创建热力图
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)  # Upper triangle mask / 上三角遮罩
    
    sns.heatmap(
        corr_matrix,
        mask=None,  # Show full matrix / 显示完整矩阵
        annot=True,
        fmt=".2f",
        cmap=config.cmap,
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8, "label": "Pearson Correlation Coefficient / 皮尔逊相关系数"},
        xticklabels=labels,
        yticklabels=labels,
        ax=ax
    )
    
    ax.set_title(
        "Pearson Correlation Matrix (17 Variables)\n皮尔逊相关系数矩阵（17个变量）",
        fontsize=14,
        fontweight="bold"
    )
    
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Save figure / 保存图像
    output_path = Path(config.output_dir) / "correlation_heatmap.png"
    plt.savefig(output_path, dpi=config.dpi, bbox_inches="tight")
    logging.info(f"Correlation heatmap saved to {output_path} / 相关系数热力图已保存到{output_path}")
    
    plt.close()


def plot_pcr_correlation_bar(
    corr_df: pd.DataFrame,
    config: CorrelationConfig,
    use_chinese: bool = True
) -> None:
    """
    Plot bar chart of correlation with Pcr / 绘制与Pcr相关系数的条形图
    
    Args:
        corr_df: Correlation DataFrame / 相关系数DataFrame
        config: Correlation configuration / 相关系数配置
        use_chinese: Whether to use Chinese labels / 是否使用中文标签
    """
    # Get correlation with Pcr (excluding Pcr itself) / 获取与Pcr的相关系数（排除Pcr本身）
    pcr_corr = corr_df.iloc[:-1, -1]  # Exclude last row (Pcr with itself)
    
    # Sort by absolute value / 按绝对值排序
    pcr_corr_sorted = pcr_corr.abs().sort_values(ascending=True)
    pcr_corr_plot = pcr_corr[pcr_corr_sorted.index]
    
    # Choose labels / 选择标签
    if use_chinese:
        labels = [config.param_names_cn[list(config.param_names).index(name)] 
                  for name in pcr_corr_plot.index]
    else:
        labels = pcr_corr_plot.index.tolist()
    
    # Create bar chart / 创建条形图
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = ['#d7191c' if v < 0 else '#2c7bb6' for v in pcr_corr_plot.values]
    
    bars = ax.barh(range(len(pcr_corr_plot)), pcr_corr_plot.values, color=colors)
    
    ax.set_yticks(range(len(pcr_corr_plot)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Pearson Correlation Coefficient / 皮尔逊相关系数")
    ax.set_title(
        "Correlation with Buckling Load (Pcr)\n与屈曲荷载的相关系数",
        fontsize=12,
        fontweight="bold"
    )
    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.set_xlim(-1, 1)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels / 添加数值标签
    for i, (bar, val) in enumerate(zip(bars, pcr_corr_plot.values)):
        ax.text(val + 0.02, i, f"{val:.3f}", va="center", fontsize=9)
    
    plt.tight_layout()
    
    # Save figure / 保存图像
    output_path = Path(config.output_dir) / "correlation_with_Pcr_bar.png"
    plt.savefig(output_path, dpi=config.dpi, bbox_inches="tight")
    logging.info(f"Correlation bar chart saved to {output_path} / 相关系数条形图已保存到{output_path}")
    
    plt.close()


def print_correlation_summary(corr_df: pd.DataFrame) -> None:
    """
    Print correlation summary / 打印相关系数摘要
    
    Args:
        corr_df: Correlation DataFrame / 相关系数DataFrame
    """
    print("\n" + "=" * 80)
    print("Correlation Analysis Summary / 相关系数分析摘要")
    print("=" * 80)
    
    # Print correlation with Pcr / 打印与Pcr的相关系数
    print("\nCorrelation with Buckling Load (Pcr) / 与屈曲荷载的相关系数:")
    print("-" * 50)
    pcr_corr = corr_df.iloc[:-1, -1]  # Exclude Pcr itself
    for name, corr in pcr_corr.items():
        strength = ""
        abs_corr = abs(corr)
        if abs_corr >= 0.8:
            strength = "(Very Strong / 很强)"
        elif abs_corr >= 0.6:
            strength = "(Strong / 强)"
        elif abs_corr >= 0.4:
            strength = "(Moderate / 中等)"
        elif abs_corr >= 0.2:
            strength = "(Weak / 弱)"
        else:
            strength = "(Very Weak / 很弱)"
        
        direction = "Positive / 正相关" if corr > 0 else "Negative / 负相关"
        print(f"  {name:10s}: {corr:+.4f} {direction} {strength}")
    
    # Find highest correlations (excluding diagonal) / 找出最高相关系数（排除对角线）
    print("\n" + "-" * 50)
    print("Top 10 Strongest Correlations (excluding diagonal) / 最强的10个相关系数（排除对角线）:")
    print("-" * 50)
    
    corr_values = []
    for i, col1 in enumerate(corr_df.columns):
        for j, col2 in enumerate(corr_df.columns):
            if i < j:  # Upper triangle only / 仅上三角
                corr_values.append((col1, col2, corr_df.iloc[i, j]))
    
    corr_values.sort(key=lambda x: abs(x[2]), reverse=True)
    
    for col1, col2, corr in corr_values[:10]:
        print(f"  {col1} <-> {col2}: {corr:+.4f}")


# =============================================================================
# Main Function / 主函数
# =============================================================================
def main() -> None:
    """Main function / 主函数"""
    # Setup logging / 设置日志
    log_dir = Path("./results/analysis")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"correlation_analysis_{datetime.now():%Y%m%d_%H%M%S}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler()
        ]
    )
    
    logging.info("=" * 60)
    logging.info("Pearson Correlation Analysis / 皮尔逊相关系数分析")
    logging.info("=" * 60)
    
    # Initialize config / 初始化配置
    config = CorrelationConfig()
    
    # Load data / 加载数据
    df = load_data(config)
    
    # Calculate correlations / 计算相关系数
    logging.info("Calculating Pearson correlation coefficients / 计算皮尔逊相关系数...")
    corr_matrix, p_matrix = calculate_pearson_correlation(df)
    
    # Create DataFrame / 创建DataFrame
    corr_df = create_correlation_dataframe(corr_matrix, df.columns.tolist())
    
    # Save results / 保存结果
    save_correlation_matrix(corr_df, p_matrix, config)
    
    # Generate plots / 生成图像
    logging.info("Generating plots / 生成图像...")
    plot_correlation_heatmap(corr_matrix, config)
    plot_pcr_correlation_bar(corr_df, config)
    
    # Print summary / 打印摘要
    print_correlation_summary(corr_df)
    
    logging.info("=" * 60)
    logging.info("Analysis completed / 分析完成")
    logging.info("=" * 60)


if __name__ == "__main__":
    main()