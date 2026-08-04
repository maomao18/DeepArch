# -*- coding: utf-8 -*-
"""
Dataset Statistics Analysis Script / 数据集统计分析脚本
========================================================
Extract buckling load from training data and create parameter table.
从训练数据中提取屈曲荷载并创建参数表格。

Functions:
    1. Read all CSV files in training data directory
       读取训练数据目录中的所有CSV文件
    2. Extract buckling load using stiffness drop method (same as Train_MultiSeed.py)
       使用刚度下降法提取屈曲荷载（与Train_MultiSeed.py相同）
    3. Create parameter table (16 fixed parameters + buckling load)
       创建参数表格（16个固定参数 + 屈曲荷载）
    4. Export to CSV for further analysis
       导出CSV用于后续分析

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
from scipy.ndimage import uniform_filter1d
from tqdm import tqdm

warnings.filterwarnings('ignore')

# Chinese font support / 中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# =============================================================================
# Configuration / 配置
# =============================================================================
@dataclass
class AnalysisConfig:
    """Configuration for dataset analysis / 数据集分析配置"""
    
    # Input/Output paths / 输入输出路径
    data_dir: str = r"./Data/Train"
    output_dir: str = r"./results/analysis"
    
    # CSV column layout (same as Train_MultiSeed.py) / CSV列布局（与Train_MultiSeed.py相同）
    meta_cols: int = 1  # 时间步列 / Time step column
    fixed_features_dim: int = 16  # 固定特征维度 / Fixed features dimension
    dynamic_features_dim: int = 43  # 动态特征维度 / Dynamic features dimension
    load_dim: int = 1  # 荷载维度 / Load dimension
    dynamic_col_start: int = 17  # 动态列起始位置 / Dynamic column start (meta_cols + fixed_features_dim)
    
    # Buckling extraction parameters / 屈曲提取参数
    buckling_node: int = 11  # 屈曲分析节点 / Buckling analysis node
    buckling_axis: str = "y"  # 屈曲分析轴 ("x" or "y") / Buckling analysis axis
    buckling_slope_threshold: float = 0.0  # 刚度阈值 / Stiffness threshold
    buckling_eps: float = 1e-9  # 数值稳定性 / Numerical stability
    buckling_min_index: int = 5  # 最小索引（忽略早期点）/ Minimum index (ignore early points)
    buckling_smooth: bool = True  # 是否平滑 / Whether to smooth
    buckling_smooth_window: int = 7  # 平滑窗口大小 / Smoothing window size
    
    # Derived parameters (set in __post_init__) / 派生参数（在__post_init__中设置）
    buckling_disp_index: int = 21  # 屈曲位移索引 / Buckling displacement index
    disp_dim: int = 42  # 位移维度 / Displacement dimension
    
    # Fixed parameter names / 固定参数名称
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
    
    def __post_init__(self) -> None:
        """Post-initialization / 后初始化"""
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Calculate buckling displacement index / 计算屈曲位移索引
        axis = self.buckling_axis.lower()
        if axis not in ("x", "y"):
            raise ValueError("buckling_axis must be 'x' or 'y' / buckling_axis必须是'x'或'y'")
        axis_idx = 0 if axis == "x" else 1
        object.__setattr__(self, 'buckling_disp_index', (self.buckling_node - 1) * 2 + axis_idx)
        object.__setattr__(self, 'disp_dim', self.dynamic_features_dim - self.load_dim)


# =============================================================================
# Buckling Extraction Functions / 屈曲提取函数
# =============================================================================
def moving_average_1d(x: np.ndarray, window: int) -> np.ndarray:
    """
    Apply moving average to 1D array / 对一维数组应用移动平均
    
    Args:
        x: Input array / 输入数组
        window: Window size / 窗口大小
        
    Returns:
        Smoothed array / 平滑后的数组
    """
    if window <= 1:
        return x
    return uniform_filter1d(x, size=window, mode='reflect')


def find_buckling_index_stiffness_drop(
    load_seq: np.ndarray,
    disp_seq: np.ndarray,
    slope_threshold: float = 0.0,
    min_index: int = 5,
    eps: float = 1e-9
) -> int:
    """
    Find buckling index using stiffness drop method / 使用刚度下降法查找屈曲索引
    
    Args:
        load_seq: Load sequence / 荷载序列
        disp_seq: Displacement sequence / 位移序列
        slope_threshold: Stiffness threshold / 刚度阈值
        min_index: Minimum index to consider / 考虑的最小索引
        eps: Numerical stability / 数值稳定性
        
    Returns:
        Buckling index / 屈曲索引
    """
    # Compute stiffness / 计算刚度
    dload = load_seq[1:] - load_seq[:-1]
    ddisp = disp_seq[1:] - disp_seq[:-1]
    
    # Avoid division by zero / 避免除零
    denom = np.sign(ddisp) * np.maximum(np.abs(ddisp), eps)
    stiff = dload / denom
    
    # Replace NaN/Inf / 替换NaN/Inf
    stiff = np.nan_to_num(stiff, nan=1e9, posinf=1e9, neginf=-1e9)
    
    # Find where stiffness drops below threshold / 查找刚度下降到阈值以下的位置
    mask = stiff <= slope_threshold
    
    # Ignore early indices / 忽略早期索引
    if min_index > 0 and len(mask) > min_index:
        mask[:min_index] = False
    
    # Find first True index / 查找第一个True索引
    true_indices = np.where(mask)[0]
    if len(true_indices) > 0:
        return true_indices[0] + 1  # +1 because we used differences
    
    # Fallback to peak load / 回退到峰值荷载
    return int(np.argmax(load_seq))


def extract_buckling_response(
    abs_phys: np.ndarray,
    config: AnalysisConfig
) -> Tuple[float, float]:
    """
    Extract buckling load and displacement / 提取屈曲荷载和位移
    
    Args:
        abs_phys: Absolute physical values [T, D] / 绝对物理值 [T, D]
        config: Analysis configuration / 分析配置
        
    Returns:
        Tuple of (buckling_load, buckling_disp) / (屈曲荷载, 屈曲位移)元组
    """
    load_seq = abs_phys[:, 0]
    disp_seq = abs_phys[:, config.load_dim + config.buckling_disp_index]
    
    # Apply smoothing if enabled / 如果启用则应用平滑
    if config.buckling_smooth and config.buckling_smooth_window > 1:
        load_sm = moving_average_1d(load_seq, config.buckling_smooth_window)
        disp_sm = moving_average_1d(disp_seq, config.buckling_smooth_window)
    else:
        load_sm = load_seq
        disp_sm = disp_seq
    
    # Find buckling index using stiffness drop / 使用刚度下降查找屈曲索引
    idx = find_buckling_index_stiffness_drop(
        load_sm, disp_sm,
        slope_threshold=config.buckling_slope_threshold,
        min_index=config.buckling_min_index,
        eps=config.buckling_eps
    )
    
    return load_seq[idx], disp_seq[idx]


# =============================================================================
# Data Processing Functions / 数据处理函数
# =============================================================================
def read_single_csv(
    csv_path: Path,
    config: AnalysisConfig
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read single CSV file / 读取单个CSV文件
    
    Args:
        csv_path: Path to CSV file / CSV文件路径
        config: Analysis configuration / 分析配置
        
    Returns:
        Tuple of (fixed_params, dynamic_data) / (固定参数, 动态数据)元组
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
            f"{csv_path.name}: 列数 < 期望值 {dyn_end}, 实际为 {data.shape[1]}"
        )
    
    fixed = data[0, fixed_start:fixed_end]
    dynamic = data[:, dyn_start:dyn_end]
    
    return fixed, dynamic


def process_all_files(
    config: AnalysisConfig
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Process all CSV files and extract parameters / 处理所有CSV文件并提取参数
    
    Args:
        config: Analysis configuration / 分析配置
        
    Returns:
        Tuple of (param_table, buckling_loads, buckling_disps) / (参数表格, 屈曲荷载, 屈曲位移)元组
    """
    csv_files = sorted(Path(config.data_dir).glob("*.csv"))
    
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {config.data_dir} / "
            f"在{config.data_dir}中未找到CSV文件"
        )
    
    logging.info(f"Found {len(csv_files)} CSV files / 找到{len(csv_files)}个CSV文件")
    
    all_params = []
    all_buckling_loads = []
    all_buckling_disps = []
    
    for csv_path in tqdm(csv_files, desc="Processing files / 处理文件"):
        try:
            fixed, dynamic = read_single_csv(csv_path, config)
            buckling_load, buckling_disp = extract_buckling_response(dynamic, config)
            
            all_params.append(fixed)
            all_buckling_loads.append(buckling_load)
            all_buckling_disps.append(buckling_disp)
        except Exception as e:
            logging.warning(f"Skip {csv_path.name}: {e} / 跳过 {csv_path.name}: {e}")
    
    # Convert to numpy arrays / 转换为numpy数组
    param_table = np.array(all_params)
    buckling_loads = np.array(all_buckling_loads)
    buckling_disps = np.array(all_buckling_disps)
    
    logging.info(
        f"Successfully processed {len(param_table)} samples / "
        f"成功处理{len(param_table)}个样本"
    )
    
    return param_table, buckling_loads, buckling_disps


def create_parameter_dataframe(
    param_table: np.ndarray,
    buckling_loads: np.ndarray,
    config: AnalysisConfig
) -> pd.DataFrame:
    """
    Create parameter DataFrame / 创建参数DataFrame
    
    Args:
        param_table: Parameter table [N, 16] / 参数表格 [N, 16]
        buckling_loads: Buckling loads [N] / 屈曲荷载 [N]
        config: Analysis configuration / 分析配置
        
    Returns:
        DataFrame with parameters and buckling load / 包含参数和屈曲荷载的DataFrame
    """
    # Create column names / 创建列名
    columns = list(config.param_names) + ["Pcr"]  # Pcr = critical buckling load
    
    # Combine parameters and buckling loads / 合并参数和屈曲荷载
    data = np.column_stack([param_table, buckling_loads])
    
    df = pd.DataFrame(data, columns=columns)
    
    return df


def save_results(
    df: pd.DataFrame,
    config: AnalysisConfig
) -> None:
    """
    Save results to CSV / 保存结果到CSV
    
    Args:
        df: Parameter DataFrame / 参数DataFrame
        config: Analysis configuration / 分析配置
    """
    output_path = Path(config.output_dir) / "parameter_statistics.csv"
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    logging.info(f"Results saved to {output_path} / 结果已保存到{output_path}")


def print_statistics(df: pd.DataFrame) -> None:
    """
    Print statistics summary / 打印统计摘要
    
    Args:
        df: Parameter DataFrame / 参数DataFrame
    """
    print("\n" + "=" * 80)
    print("Dataset Statistics Summary / 数据集统计摘要")
    print("=" * 80)
    
    stats_df = df.describe()
    print(stats_df.to_string())
    
    print("\n" + "-" * 80)
    print("Additional Statistics / 附加统计")
    print("-" * 80)
    
    for col in df.columns:
        data = df[col].values
        print(f"\n{col}:")
        print(f"  Mean / 均值: {np.mean(data):.6e}")
        print(f"  Std / 标准差: {np.std(data):.6e}")
        print(f"  Min / 最小值: {np.min(data):.6e}")
        print(f"  Max / 最大值: {np.max(data):.6e}")
        print(f"  Range / 范围: {np.max(data) - np.min(data):.6e}")


# =============================================================================
# Main Function / 主函数
# =============================================================================
def main() -> None:
    """Main function / 主函数"""
    # Setup logging / 设置日志
    log_dir = Path("./results/analysis")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"analyze_data_{datetime.now():%Y%m%d_%H%M%S}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler()
        ]
    )
    
    logging.info("=" * 60)
    logging.info("Dataset Statistics Analysis / 数据集统计分析")
    logging.info("=" * 60)
    
    # Initialize config / 初始化配置
    config = AnalysisConfig()
    
    # Process all files / 处理所有文件
    param_table, buckling_loads, buckling_disps = process_all_files(config)
    
    # Create DataFrame / 创建DataFrame
    df = create_parameter_dataframe(param_table, buckling_loads, config)
    
    # Save results / 保存结果
    save_results(df, config)
    
    # Print statistics / 打印统计
    print_statistics(df)
    
    logging.info("=" * 60)
    logging.info("Analysis completed / 分析完成")
    logging.info("=" * 60)


if __name__ == "__main__":
    main()