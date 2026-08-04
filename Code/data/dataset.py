# -*- coding: utf-8 -*-
"""
Buckling Dataset / 屈曲数据集
==============================
PyTorch Dataset for buckling prediction training and evaluation.
用于屈曲预测训练和评估的PyTorch数据集。
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from tqdm import tqdm

from .scalers import clamp_scaler_std_inplace, safe_transform_np

if TYPE_CHECKING:
    from config.train import TrainingConfig


class BucklingDataset(Dataset):
    """
    Dataset for buckling prediction.
    屈曲预测数据集。

    Reads CSV files containing structural parameters and load-displacement sequences.
    读取包含结构参数和荷载-位移序列的CSV文件。
    """

    def __init__(
        self,
        csv_files: List[Path],
        config: "TrainingConfig",
        scalers: Optional[Dict[str, StandardScaler]],
        normalize: bool = True
    ) -> None:
        """
        Initialize BucklingDataset.
        初始化BucklingDataset。

        Args:
            csv_files: List of CSV file paths / CSV文件路径列表
            config: Training configuration / 训练配置
            scalers: Dictionary of scalers for normalization / 用于标准化的标准化器字典
            normalize: Whether to normalize data / 是否标准化数据
        """
        self.csv_files = csv_files
        self.config = config
        self.scalers = scalers
        self.normalize = normalize

    @staticmethod
    def discover_files(data_dir: str) -> List[Path]:
        """
        Discover CSV files in a directory.
        发现目录中的CSV文件。

        Args:
            data_dir: Directory path / 目录路径

        Returns:
            Sorted list of CSV file paths / 排序后的CSV文件路径列表
        """
        return sorted(Path(data_dir).glob("*.csv"))

    def _read_file(self, fp: Path) -> Tuple[np.ndarray, np.ndarray]:
        """
        Read single CSV file and extract features.
        读取单个CSV文件并提取特征。

        Args:
            fp: CSV file path / CSV文件路径

        Returns:
            Tuple of (fixed_features, dynamic_abs) / (固定特征, 动态绝对值)元组

        Raises:
            ValueError: If file has insufficient columns / 如果文件列数不足
        """
        df = pd.read_csv(fp)
        data = df.values.astype(np.float32)

        fixed_start = self.config.meta_cols
        fixed_end = fixed_start + self.config.fixed_features_dim
        dyn_start = self.config.dynamic_col_start
        dyn_end = dyn_start + self.config.dynamic_features_dim

        if data.shape[1] < dyn_end:
            raise ValueError(
                f"{fp.name}: columns < expected {dyn_end}. got {data.shape[1]} / "
                f"{fp.name}: 列数 < 期望值 {dyn_end}，实际 {data.shape[1]}"
            )

        fixed = data[0, fixed_start:fixed_end]
        dynamic_abs = data[:, dyn_start:dyn_end]

        # Enforce length (trunc/pad) / 强制长度（截断/填充）
        L = dynamic_abs.shape[0]
        target_L = self.config.sequence_length
        if L >= target_L:
            dynamic_abs = dynamic_abs[:target_L]
        else:
            # Pad with last row repeat / 用最后一行重复填充
            pad_n = target_L - L
            last = dynamic_abs[-1:].repeat(pad_n, axis=0)
            dynamic_abs = np.concatenate([dynamic_abs, last], axis=0)

        return fixed, dynamic_abs

    @classmethod
    def fit_scalers_on_files(
        cls,
        csv_files: List[Path],
        config: "TrainingConfig"
    ) -> Dict[str, StandardScaler]:
        """
        Fit scalers on training files.
        在训练文件上拟合标准化器。

        Args:
            csv_files: List of CSV file paths / CSV文件路径列表
            config: Training configuration / 训练配置

        Returns:
            Dictionary of fitted scalers / 拟合好的标准化器字典

        Raises:
            ValueError: If no data is available for fitting / 如果没有可用于拟合的数据
        """
        all_fixed, all_abs, all_delta = [], [], []
        dummy = cls([], config, None)

        for fp in tqdm(csv_files, desc="Fitting scalers / 拟合标准化器"):
            try:
                fixed, dynamic_abs = dummy._read_file(fp)
                if dynamic_abs.shape[0] < 2:
                    continue
                delta = dynamic_abs[1:] - dynamic_abs[:-1]
                all_fixed.append(fixed.reshape(1, -1))
                all_abs.append(dynamic_abs)
                all_delta.append(delta)
            except Exception as e:
                import logging
                logging.warning(f"Skip {fp.name}: {e}")

        if not all_fixed:
            raise ValueError("No data for scaler fitting. / 没有用于拟合标准化器的数据")

        scalers = {
            'fixed': StandardScaler().fit(np.vstack(all_fixed)),
            'abs': StandardScaler().fit(np.vstack(all_abs)),
            'delta': StandardScaler().fit(np.vstack(all_delta))
        }

        clamp_scaler_std_inplace(scalers['fixed'], name="fixed")
        clamp_scaler_std_inplace(scalers['abs'], name="abs")
        clamp_scaler_std_inplace(scalers['delta'], name="delta")

        return scalers

    def __len__(self) -> int:
        """
        Get dataset length.
        获取数据集长度。

        Returns:
            Number of samples / 样本数量
        """
        return len(self.csv_files)

    def __getitem__(
        self,
        idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a single sample.
        获取单个样本。

        Args:
            idx: Sample index / 样本索引

        Returns:
            Tuple of (fixed_norm, abs_norm, delta_norm) tensors
            (固定特征, 绝对状态, delta)张量元组

        Raises:
            ValueError: If sequence is too short / 如果序列太短
        """
        fixed, dynamic_abs = self._read_file(self.csv_files[idx])

        if dynamic_abs.shape[0] < 2:
            raise ValueError(f"{self.csv_files[idx].name} length<2. / 长度<2")

        if self.normalize and self.scalers is not None:
            fixed_norm = safe_transform_np(
                self.scalers['fixed'],
                fixed.reshape(1, -1)
            ).flatten()
            abs_norm = safe_transform_np(self.scalers['abs'], dynamic_abs)
            delta_norm = safe_transform_np(
                self.scalers['delta'],
                (dynamic_abs[1:] - dynamic_abs[:-1]).astype(np.float32)
            )
        else:
            fixed_norm = fixed
            abs_norm = dynamic_abs
            delta_norm = dynamic_abs[1:] - dynamic_abs[:-1]

        return (
            torch.tensor(fixed_norm, dtype=torch.float32),
            torch.tensor(abs_norm, dtype=torch.float32),
            torch.tensor(delta_norm, dtype=torch.float32)
        )