# -*- coding: utf-8 -*-
"""
Training Configuration / 训练配置
=================================
Configuration for model training with all hyperparameters.
模型训练配置，包含所有超参数。
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .base import BaseConfig


@dataclass
class TrainingConfig(BaseConfig):
    """
    Configuration for model training.
    模型训练配置。

    Includes:
        - I/O paths
        - Data split ratios
        - Model hyperparameters
        - Training hyperparameters
        - Loss weights
        - Buckling extraction parameters
    """

    # =========================================================================
    # I/O Paths / 输入输出路径
    # =========================================================================
    data_dir: str = r"./Data/Train"
    model_save_path: str = r"./models/FA-LSTM/buckling_predictor.pth"
    scaler_save_path: str = r"./models/FA-LSTM/scalers.pkl"
    log_dir: str = r"./logs/FA-LSTM"
    metrics_csv_path: str = r"./logs/FA-LSTM/training_metrics.csv"

    # Optional external test folder / 可选的外部测试文件夹
    external_test_dir: Optional[str] = r"./Data/Test"
    external_test_out_dir: str = r"./logs/FA-LSTM/test_predictions"
    external_test_plot: bool = True
    external_test_plot_node: int = 11
    external_test_plot_axis: str = "y"  # x/y

    # =========================================================================
    # Data Split / 数据划分
    # =========================================================================
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # Caps for HPO speed-up / 用于HPO加速的上限
    max_train_files: Optional[int] = None
    max_val_files: Optional[int] = None
    max_test_files: Optional[int] = None

    # =========================================================================
    # DataLoader / 数据加载器
    # =========================================================================
    batch_size: int = 16
    num_workers: int = 0  # Windows: start with 0, increase after stability
    pin_memory: bool = True
    persistent_workers: bool = False
    prefetch_factor: int = 4

    # =========================================================================
    # Feature Dimensions & CSV Layout / 特征维度和CSV布局
    # =========================================================================
    fixed_features_dim: int = 16
    dynamic_features_dim: int = 43  # load + displacements
    load_dim: int = 1
    meta_cols: int = 1
    dynamic_col_start: Optional[int] = None
    sequence_length: int = 200  # enforce length by trunc/pad

    # =========================================================================
    # Model Hyperparameters / 模型超参数
    # =========================================================================
    hidden_size: int = 384
    num_layers: int = 3
    num_heads: int = 8
    dropout: float = 0.2

    # Toggles / 开关
    use_feature_mhsa: bool = True
    use_attn_pool: bool = True
    use_lstm_prior: bool = True
    use_multihead_outputs: bool = True
    use_film: bool = True
    film_hidden: int = 128
    use_state_residual: bool = True
    state_residual_hidden: int = 128

    # =========================================================================
    # Training Hyperparameters / 训练超参数
    # =========================================================================
    epochs: int = 120
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    early_stopping_patience: int = 50

    # Scheduled sampling stages / 调度采样阶段
    teacher_forcing_epochs: int = 40
    transition_epochs: int = 40

    # =========================================================================
    # Loss Weights / 损失权重
    # =========================================================================
    load_weight: float = 0.5
    displacement_weight: float = 0.5
    w_delta: float = 0.5
    w_abs: float = 0.5

    # Monotonic load penalty / 单调荷载惩罚
    use_monotonic_load_loss: bool = False
    w_monotonic_load: float = 0.05

    # Stiffness loss / 刚度损失
    use_stiffness_loss: bool = False
    w_stiffness: float = 0.05
    use_huber_for_stiffness: bool = True
    huber_stiffness: float = 1.0

    # Early-step weighting / 早期步骤加权
    early_steps_span: int = 10
    early_step_max_boost: float = 5.0
    early_step_decay_power: float = 1.0
    use_step_weights_in_val: bool = True

    # =========================================================================
    # AMP & Compile / AMP和编译
    # =========================================================================
    use_amp: bool = True
    use_huber_for_delta: bool = True
    huber_delta: float = 1.0
    use_torch_compile: bool = False

    # =========================================================================
    # Validation Policy / 验证策略
    # =========================================================================
    validate_match_train: bool = True
    validate_always_ar: bool = True
    best_monitor: str = "val_ar_total_loss"  # or "val_match_total_loss"
    export_metrics_every_epoch: bool = True

    # =========================================================================
    # LR Factors per Phase / 各阶段学习率因子
    # =========================================================================
    lr_factor_tf: float = 1.0
    lr_factor_mixed: float = 0.2
    lr_factor_ar: float = 0.1

    # =========================================================================
    # Buckling Extraction / 屈曲提取
    # =========================================================================
    buckling_node: int = 11
    buckling_axis: str = "y"  # "x" or "y"
    buckling_disp_index: Optional[int] = None
    buckling_slope_threshold: float = 0.0  # for stiffness_drop
    buckling_eps: float = 1e-9
    buckling_method: str = "peak_load"  # "peak_load" | "stiffness_drop" | "combined"
    buckling_min_index: int = 5  # ignore first few points for stiffness_drop
    buckling_smooth: bool = True
    buckling_smooth_window: int = 7  # moving average window (odd recommended)

    # Derived / 派生参数
    disp_dim: int = field(init=False)

    def __post_init__(self) -> None:
        """Post-initialization validation / 后初始化验证"""
        # Call parent __post_init__ first
        super().__post_init__()

        # Create directories / 创建目录
        Path(self.model_save_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.scaler_save_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        Path(self.metrics_csv_path).parent.mkdir(parents=True, exist_ok=True)
        if self.external_test_dir is not None:
            Path(self.external_test_out_dir).mkdir(parents=True, exist_ok=True)

        # Set derived dimension / 设置派生维度
        object.__setattr__(self, 'disp_dim', self.dynamic_features_dim - self.load_dim)
        if self.disp_dim <= 0:
            raise ValueError("dynamic_features_dim must be > load_dim. / dynamic_features_dim必须大于load_dim")

        # Set dynamic column start if not provided / 如果未提供则设置动态列起始位置
        if self.dynamic_col_start is None:
            object.__setattr__(self, 'dynamic_col_start', self.meta_cols + self.fixed_features_dim)

        # Compute buckling_disp_index if not set / 如果未设置则计算buckling_disp_index
        if self.buckling_disp_index is None:
            axis = self.buckling_axis.lower()
            if axis not in ("x", "y"):
                raise ValueError("buckling_axis must be 'x' or 'y'. / buckling_axis必须是'x'或'y'")
            axis_idx = 0 if axis == "x" else 1
            idx = (self.buckling_node - 1) * 2 + axis_idx
            object.__setattr__(self, 'buckling_disp_index', idx)

        # Validate buckling_disp_index / 验证buckling_disp_index
        if not (0 <= self.buckling_disp_index < self.disp_dim):
            raise ValueError(
                f"buckling_disp_index exceeds displacement dimension. / "
                f"buckling_disp_index超出位移维度"
            )

        # Validate sequence_length / 验证序列长度
        if self.sequence_length < 2:
            raise ValueError("sequence_length must be >= 2. / sequence_length必须>=2")

        # Validate num_heads divisibility / 验证num_heads可除性
        if self.num_heads > 0 and (self.hidden_size % self.num_heads != 0):
            raise ValueError(
                f"hidden_size({self.hidden_size}) must be divisible by "
                f"num_heads({self.num_heads}). / hidden_size必须能被num_heads整除"
            )

        # Validate smooth window / 验证平滑窗口
        if self.buckling_smooth_window < 1:
            raise ValueError(
                "buckling_smooth_window must be >= 1. / buckling_smooth_window必须>=1"
            )

        # Validate best_monitor / 验证best_monitor
        if self.best_monitor not in ("val_ar_total_loss", "val_match_total_loss"):
            raise ValueError(
                "best_monitor must be val_ar_total_loss or val_match_total_loss. / "
                "best_monitor必须是val_ar_total_loss或val_match_total_loss"
            )