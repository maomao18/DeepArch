# -*- coding: utf-8 -*-
"""
Buckling Predictor Inference / 屈曲预测器推理
==============================================
Inference class for making predictions with trained models.
使用训练好的模型进行预测的推理类。
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from config.train import TrainingConfig
from models.predictor import BucklingPredictor
from data.scalers import safe_transform_np
from utils.buckling import extract_buckling_response_numpy

# Ensure TrainingConfig is available in __main__ namespace for pickle deserialization
# This handles cases where models were saved with TrainingConfig referenced from __main__
sys.modules['__main__'].TrainingConfig = TrainingConfig


class BucklingPredictorInference:
    """
    Inference class for BucklingPredictor.
    BucklingPredictor的推理类。

    Provides methods for:
        - Loading trained models
        - Making predictions from CSV files
        - Batch prediction on folders
    """

    def __init__(self, model_path: str, device: str = "cuda") -> None:
        """
        Initialize BucklingPredictorInference.
        初始化BucklingPredictorInference。

        Args:
            model_path: Path to model checkpoint / 模型检查点路径
            device: Device to run inference on / 运行推理的设备
        """
        self.device = torch.device(device)
        ckpt = torch.load(model_path, map_location=self.device, weights_only=False)
        self.config: TrainingConfig = ckpt['config']
        self.scalers: Dict[str, Any] = ckpt['scalers']

        self.model = BucklingPredictor(self.config).to(self.device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.eval()

        # Setup delta->abs transformation / 设置delta到abs变换
        abs_scale_np = np.where(
            self.scalers['abs'].scale_.copy() < 1e-8, 1.0, self.scalers['abs'].scale_.copy()
        )
        delta_mean_np = self.scalers['delta'].mean_.copy()
        delta_scale_np = np.where(
            self.scalers['delta'].scale_.copy() < 1e-8, 1.0, self.scalers['delta'].scale_.copy()
        )

        self.abs_mean = torch.tensor(
            self.scalers['abs'].mean_.copy(), dtype=torch.float32, device=self.device
        ).view(1, -1)
        self.abs_std = torch.tensor(
            abs_scale_np, dtype=torch.float32, device=self.device
        ).view(1, -1)
        self.A = torch.tensor(
            delta_scale_np / abs_scale_np, dtype=torch.float32, device=self.device
        ).view(1, -1)
        self.B = torch.tensor(
            delta_mean_np / abs_scale_np, dtype=torch.float32, device=self.device
        ).view(1, -1)

    def _read_csv(self, fp: Path) -> Tuple[np.ndarray, np.ndarray]:
        """
        Read CSV file and extract features.
        读取CSV文件并提取特征。

        Args:
            fp: CSV file path / CSV文件路径

        Returns:
            Tuple of (fixed_features, dynamic_abs) / (固定特征, 动态绝对值)元组
        """
        df = pd.read_csv(fp)
        data = df.values.astype(np.float32)

        fixed_start = self.config.meta_cols
        fixed_end = fixed_start + self.config.fixed_features_dim
        dyn_start = self.config.dynamic_col_start
        dyn_end = dyn_start + self.config.dynamic_features_dim

        if data.shape[1] < dyn_end:
            raise ValueError(
                f"{fp.name}: columns < expected {dyn_end}. got {data.shape[1]}"
            )

        fixed = data[0, fixed_start:fixed_end]
        dynamic_abs = data[:, dyn_start:dyn_end]

        # Align length for comparison/plot / 对齐长度用于比较/绘图
        if dynamic_abs.shape[0] > self.config.sequence_length:
            dynamic_abs = dynamic_abs[:self.config.sequence_length]

        return fixed, dynamic_abs

    @torch.no_grad()
    def predict_from_csv(self, csv_path: str) -> Dict[str, np.ndarray]:
        """
        Make prediction from CSV file.
        从CSV文件进行预测。

        Args:
            csv_path: Path to CSV file / CSV文件路径

        Returns:
            Dictionary containing:
                - loads_pred: Predicted loads / 预测的荷载
                - disps_pred: Predicted displacements / 预测的位移
                - loads_true: True loads / 真实荷载
                - disps_true: True displacements / 真实位移
        """
        fp = Path(csv_path)
        fixed_phys, abs_phys_true = self._read_csv(fp)
        T = abs_phys_true.shape[0]
        steps = T - 1

        # Normalize inputs / 标准化输入
        fixed_norm = safe_transform_np(self.scalers['fixed'], fixed_phys.reshape(1, -1))
        fixed = torch.tensor(fixed_norm, dtype=torch.float32, device=self.device)

        abs0_phys = abs_phys_true[0:1]  # [1, D]
        abs0_norm = (
            torch.tensor(abs0_phys, dtype=torch.float32, device=self.device) - self.abs_mean
        ) / self.abs_std

        # Initialize prediction lists / 初始化预测列表
        loads_pred = [float(abs0_phys[0, 0])]
        disps_pred = [abs0_phys[0, self.config.load_dim:].copy()]

        h = None
        prev = abs0_norm

        for _ in range(steps):
            delta, h = self.model.step(fixed, prev, h)
            next_abs_norm = prev + self.A * delta + self.B
            next_abs_phys = next_abs_norm * self.abs_std + self.abs_mean
            next_np = next_abs_phys.squeeze(0).cpu().numpy()
            loads_pred.append(float(next_np[0]))
            disps_pred.append(next_np[self.config.load_dim:].copy())
            prev = next_abs_norm

        loads_true = abs_phys_true[:, 0].astype(np.float32)
        disps_true = abs_phys_true[:, self.config.load_dim:].astype(np.float32)

        return dict(
            loads_pred=np.array(loads_pred, dtype=np.float32),
            disps_pred=np.array(disps_pred, dtype=np.float32),
            loads_true=loads_true,
            disps_true=disps_true
        )

    @torch.no_grad()
    def predict_on_folder(
        self,
        test_dir: str,
        out_dir: str,
        plot: bool = True,
        plot_node: int = 11,
        plot_axis: str = "y"
    ) -> None:
        """
        Run batch prediction on a folder of CSV files.
        对CSV文件文件夹运行批量预测。

        Args:
            test_dir: Test folder path / 测试文件夹路径
            out_dir: Output folder path / 输出文件夹路径
            plot: Whether to generate plots / 是否生成图表
            plot_node: Node to plot / 要绘制的节点
            plot_axis: Axis to plot ('x' or 'y') / 要绘制的轴
        """
        import matplotlib.pyplot as plt

        out_dir_p = Path(out_dir)
        out_dir_p.mkdir(parents=True, exist_ok=True)

        files = sorted(Path(test_dir).glob("*.csv"))
        if not files:
            logging.warning(f"No csv found in external test dir: {test_dir}")
            return

        axis = plot_axis.lower()
        axis_idx = 0 if axis == "x" else 1
        disp_idx = (plot_node - 1) * 2 + axis_idx

        for fp in tqdm(files, desc="ExternalTestPredict"):
            try:
                res = self.predict_from_csv(str(fp))
                loads_p, disps_p = res['loads_pred'], res['disps_pred']
                loads_t, disps_t = res['loads_true'], res['disps_true']

                # Save comparison CSV / 保存对比CSV
                data = {
                    "step": np.arange(len(loads_t), dtype=np.int32),
                    "load_true": loads_t,
                    "load_pred": loads_p[:len(loads_t)]
                }
                D = disps_t.shape[1]
                for i in range(D):
                    data[f"disp_true_{i}"] = disps_t[:, i]
                    data[f"disp_pred_{i}"] = disps_p[:len(loads_t), i]

                df = pd.DataFrame(data)
                df.to_csv(out_dir_p / f"{fp.stem}_pred_vs_true.csv", index=False, encoding="utf-8-sig")

                if plot:
                    import matplotlib
                    matplotlib.rcParams['font.sans-serif'] = ['SimHei']
                    matplotlib.rcParams['axes.unicode_minus'] = False

                    plt.figure(figsize=(9, 6))
                    idx_plot = disp_idx if disp_idx < D else 0
                    plt.plot(disps_t[:, idx_plot], loads_t, label="True", lw=1.6)
                    plt.plot(disps_p[:len(loads_t), idx_plot], loads_p[:len(loads_t)], label="Pred", lw=1.6)
                    plt.xlabel(f"Node{plot_node}-{axis} disp (idx={idx_plot})")
                    plt.ylabel("Load")
                    plt.title(fp.name)
                    plt.grid(True, alpha=0.3)
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(out_dir_p / f"{fp.stem}_curve.png", dpi=150)
                    plt.close()

            except Exception as e:
                logging.warning(f"Skip {fp.name}: {e}")