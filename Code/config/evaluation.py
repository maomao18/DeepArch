# -*- coding: utf-8 -*-
"""
Evaluation Configuration / 评估配置
====================================
Configuration for model evaluation on test set.
测试集上的模型评估配置。
"""

from dataclasses import dataclass
from pathlib import Path

from .base import BaseConfig


@dataclass
class EvaluationConfig(BaseConfig):
    """
    Configuration for model evaluation.
    模型评估配置。

    Includes:
        - Model and data paths
        - Data split ratios (same as training)
        - Data dimensions
        - Buckling extraction parameters
        - Evaluation settings
    """

    # =========================================================================
    # Paths / 路径
    # =========================================================================
    model_path: str = r"./models/v2F/buckling_predictor_best.pth"
    data_dir: str = r"./Data/Train"
    output_dir: str = r"./results/v2F/evaluation"
    scalers_path: str = r"./models/v2F/scalers.pkl"

    # =========================================================================
    # Data Split (same as training) / 数据划分（与训练相同）
    # =========================================================================
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # =========================================================================
    # Data Dimensions / 数据维度
    # =========================================================================
    meta_cols: int = 1
    fixed_features_dim: int = 16
    dynamic_features_dim: int = 43
    load_dim: int = 1
    dynamic_col_start: int = 17
    sequence_length: int = 200

    # =========================================================================
    # Buckling Extraction / 屈曲提取
    # =========================================================================
    buckling_node: int = 11
    buckling_axis: str = "y"
    buckling_method: str = "peak_load"  # 'peak_load' or 'stiffness' / 峰值荷载法或刚度法
    buckling_slope_threshold: float = 0.0
    buckling_eps: float = 1e-9
    buckling_min_index: int = 5
    buckling_smooth: bool = False  # No need for smoothing with peak_load method
    buckling_smooth_window: int = 7

    # =========================================================================
    # Evaluation Settings / 评估设置
    # =========================================================================
    batch_size: int = 16

    # Derived / 派生参数
    disp_dim: int = 42
    buckling_disp_index: int = 21

    def __post_init__(self) -> None:
        """Post-initialization / 后初始化"""
        super().__post_init__()

        # Create output directory / 创建输出目录
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        # Set derived dimension / 设置派生维度
        object.__setattr__(self, 'disp_dim', self.dynamic_features_dim - self.load_dim)

        # Calculate buckling displacement index / 计算屈曲位移索引
        axis = self.buckling_axis.lower()
        axis_idx = 0 if axis == "x" else 1
        object.__setattr__(
            self,
            'buckling_disp_index',
            (self.buckling_node - 1) * 2 + axis_idx
        )