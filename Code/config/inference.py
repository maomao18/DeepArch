# -*- coding: utf-8 -*-
"""
Inference Configuration / 推理配置
===================================
Configuration for model inference and prediction.
模型推理和预测配置。
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

from .base import BaseConfig


@dataclass
class InferenceConfig(BaseConfig):
    """
    Configuration for inference.
    推理配置。

    Includes:
        - Model paths
        - Data dimensions
        - Buckling extraction parameters
        - Plotting settings
    """

    # =========================================================================
    # Model Paths / 模型路径
    # =========================================================================
    model_path: str = r"./models/v2F/buckling_predictor_best.pth"
    scalers_path: str = r"./models/v2F/scalers.pkl"
    output_dir: str = r"./results/inference"

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
    buckling_slope_threshold: float = 0.0
    buckling_eps: float = 1e-9
    buckling_min_index: int = 5
    buckling_smooth: bool = True
    buckling_smooth_window: int = 7

    # =========================================================================
    # Plotting / 绘图
    # =========================================================================
    plot_nodes: Tuple[int, ...] = (1, 6, 11, 16, 21)

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