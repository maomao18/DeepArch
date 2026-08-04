# -*- coding: utf-8 -*-
"""
Metrics / 指标
==============
Metric computation utilities for training and evaluation.
用于训练和评估的指标计算工具。
"""

import math
from typing import Dict, List

import torch


class SequenceMetricMeter:
    """
    Vector-wide metrics in physical space.
    物理空间中的向量级指标。
    """

    def __init__(self, eps: float = 1e-12) -> None:
        """
        Initialize SequenceMetricMeter.
        初始化SequenceMetricMeter。

        Args:
            eps: Small value for numerical stability / 数值稳定性的小值
        """
        self.eps = eps
        self.reset()

    def reset(self) -> None:
        """Reset all metrics. / 重置所有指标。"""
        self.count = 0
        self.sse = 0.0
        self.sae = 0.0
        self.sum_y = 0.0
        self.sum_y2 = 0.0

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        """
        Update metrics with new predictions.
        用新预测更新指标。

        Args:
            pred: Predictions / 预测值
            target: Targets / 目标值
        """
        diff = pred - target
        self.sse += diff.square().sum().item()
        self.sae += diff.abs().sum().item()
        self.sum_y += target.sum().item()
        self.sum_y2 += target.square().sum().item()
        self.count += diff.numel()

    def compute(self) -> Dict[str, float]:
        """
        Compute final metrics.
        计算最终指标。

        Returns:
            Dictionary of metrics (mse, rmse, mae, r2)
            指标字典（mse, rmse, mae, r2）
        """
        if self.count == 0:
            return dict(mse=0.0, rmse=0.0, mae=0.0, r2=0.0)

        mse = self.sse / self.count
        rmse = math.sqrt(max(mse, 0.0))
        mae = self.sae / self.count
        mean_y = self.sum_y / self.count
        sst = self.sum_y2 - self.count * (mean_y ** 2)
        r2 = 0.0 if sst <= self.eps else (1.0 - self.sse / sst)

        return dict(mse=mse, rmse=rmse, mae=mae, r2=r2)


class ScalarMetricMeter:
    """
    Scalar metrics for a selected channel.
    选定通道的标量指标。
    """

    def __init__(self, eps: float = 1e-12) -> None:
        """
        Initialize ScalarMetricMeter.
        初始化ScalarMetricMeter。

        Args:
            eps: Small value for numerical stability / 数值稳定性的小值
        """
        self.eps = eps
        self.reset()

    def reset(self) -> None:
        """Reset all metrics. / 重置所有指标。"""
        self.count = 0
        self.sse = 0.0
        self.sae = 0.0
        self.sum_y = 0.0
        self.sum_y2 = 0.0

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        """
        Update metrics with new predictions.
        用新预测更新指标。

        Args:
            pred: Predictions / 预测值
            target: Targets / 目标值
        """
        diff = (pred - target).reshape(-1)
        y = target.reshape(-1)
        self.sse += diff.square().sum().item()
        self.sae += diff.abs().sum().item()
        self.sum_y += y.sum().item()
        self.sum_y2 += y.square().sum().item()
        self.count += diff.numel()

    def compute(self) -> Dict[str, float]:
        """
        Compute final metrics.
        计算最终指标。

        Returns:
            Dictionary of metrics (mse, rmse, mae, r2)
            指标字典（mse, rmse, mae, r2）
        """
        if self.count == 0:
            return dict(mse=0.0, rmse=0.0, mae=0.0, r2=0.0)

        mse = self.sse / self.count
        rmse = math.sqrt(max(mse, 0.0))
        mae = self.sae / self.count
        mean_y = self.sum_y / self.count
        sst = self.sum_y2 - self.count * (mean_y ** 2)
        r2 = 0.0 if sst <= self.eps else (1.0 - self.sse / sst)

        return dict(mse=mse, rmse=rmse, mae=mae, r2=r2)


class BucklingMetricMeter:
    """
    Comprehensive metrics for buckling point predictions.
    屈曲点预测的综合指标。

    Computes: MAE, RMSE, R², MAPE, NRMSE, PCC for both buckling load and displacement.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.count = 0
        self.load_preds: List[float] = []
        self.load_trues: List[float] = []
        self.disp_preds: List[float] = []
        self.disp_trues: List[float] = []

    def update(
        self,
        load_pred: torch.Tensor,
        load_true: torch.Tensor,
        disp_pred: torch.Tensor,
        disp_true: torch.Tensor
    ) -> None:
        self.load_preds.extend(load_pred.detach().cpu().tolist())
        self.load_trues.extend(load_true.detach().cpu().tolist())
        self.disp_preds.extend(disp_pred.detach().cpu().tolist())
        self.disp_trues.extend(disp_true.detach().cpu().tolist())
        self.count += load_pred.numel()

    @staticmethod
    def _calc(p: List[float], t: List[float], eps: float = 1e-8) -> Dict[str, float]:
        """Compute MAE, RMSE, R², MAPE, NRMSE, PCC for a (pred, true) pair."""
        import numpy as np
        p, t = np.asarray(p, dtype=np.float64), np.asarray(t, dtype=np.float64)
        if len(p) == 0:
            return dict(MAE=0.0, RMSE=0.0, R2=0.0, MAPE=0.0, NRMSE=0.0, PCC=0.0)

        mae = float(np.abs(p - t).mean())
        rmse = float(np.sqrt(((p - t) ** 2).mean()))
        ss_res = ((t - p) ** 2).sum()
        ss_tot = ((t - t.mean()) ** 2).sum()
        r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0
        mape = float((np.abs((t - p) / (np.abs(t) + eps)) * 100.0).mean())
        trange = t.max() - t.min()
        nrmse = float(rmse / trange) if trange > 1e-12 else 0.0
        p_mean, t_mean = p.mean(), t.mean()
        p_std, t_std = p.std(ddof=0), t.std(ddof=0)
        if p_std > 1e-12 and t_std > 1e-12:
            pcc = float(((p - p_mean) * (t - t_mean)).mean() / (p_std * t_std))
        else:
            pcc = 0.0
        return dict(MAE=mae, RMSE=rmse, R2=r2, MAPE=mape, NRMSE=nrmse, PCC=pcc)

    def compute(self) -> Dict[str, float]:
        if self.count == 0:
            return dict(
                buckling_load_mae=0.0, buckling_load_rmse=0.0, buckling_load_r2=0.0,
                buckling_load_mape=0.0, buckling_load_nrmse=0.0, buckling_load_pcc=0.0,
                buckling_disp_mae=0.0, buckling_disp_rmse=0.0, buckling_disp_r2=0.0,
            )

        bl = self._calc(self.load_preds, self.load_trues)
        bd = self._calc(self.disp_preds, self.disp_trues)
        return dict(
            buckling_load_mae=bl["MAE"], buckling_load_rmse=bl["RMSE"],
            buckling_load_r2=bl["R2"], buckling_load_mape=bl["MAPE"],
            buckling_load_nrmse=bl["NRMSE"], buckling_load_pcc=bl["PCC"],
            buckling_disp_mae=bd["MAE"], buckling_disp_rmse=bd["RMSE"],
            buckling_disp_r2=bd["R2"],
        )