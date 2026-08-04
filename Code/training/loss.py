# -*- coding: utf-8 -*-
"""
Loss Functions / 损失函数
==========================
Custom loss functions for buckling prediction.
用于屈曲预测的自定义损失函数。
"""

from typing import Tuple, TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from config.train import TrainingConfig


class DeltaAbsoluteLoss(nn.Module):
    """
    Combined delta and absolute loss for sequence prediction.
    用于序列预测的组合delta和绝对损失。

    Computes a weighted combination of:
    - Delta prediction loss (Huber or MSE)
    - Absolute prediction loss (MSE)
    - Optional stiffness loss
    - Optional monotonic load penalty

    计算以下内容的加权组合：
    - Delta预测损失（Huber或MSE）
    - 绝对预测损失（MSE）
    - 可选的刚度损失
    - 可选的单调荷载惩罚
    """

    def __init__(self, config: "TrainingConfig") -> None:
        """
        Initialize DeltaAbsoluteLoss.
        初始化DeltaAbsoluteLoss。

        Args:
            config: Training configuration / 训练配置
        """
        super().__init__()
        self.config = config

    def _compute_stiffness(self, abs_seq: torch.Tensor) -> torch.Tensor:
        """
        Compute stiffness (dload/ddisp) from absolute sequence.
        从绝对序列计算刚度（dload/ddisp）。

        Args:
            abs_seq: Absolute sequence [B, T, D] / 绝对序列 [B, T, D]

        Returns:
            Stiffness values [B, T-1] / 刚度值 [B, T-1]
        """
        load = abs_seq[:, :, 0]  # [B, T]
        disp = abs_seq[:, :, self.config.load_dim + self.config.buckling_disp_index]  # [B, T]

        dload = load[:, 1:] - load[:, :-1]
        ddisp = disp[:, 1:] - disp[:, :-1]

        eps = self.config.buckling_eps
        denom = ddisp.sign() * ddisp.abs().clamp_min(eps)
        stiff = dload / denom
        stiff = torch.nan_to_num(stiff, nan=0.0, posinf=1e6, neginf=-1e6)

        return stiff

    def forward(
        self,
        delta_pred: torch.Tensor,
        delta_target: torch.Tensor,
        abs_pred: torch.Tensor,
        abs_target: torch.Tensor,
        step_weights: torch.Tensor = None
    ) -> Tuple[torch.Tensor, Tuple]:
        """
        Forward pass.
        前向传播。

        Args:
            delta_pred: Predicted delta [B, T-1, D] / 预测的delta [B, T-1, D]
            delta_target: Target delta [B, T-1, D] / 目标delta [B, T-1, D]
            abs_pred: Predicted absolute [B, T-1, D] / 预测的绝对值 [B, T-1, D]
            abs_target: Target absolute [B, T-1, D] / 目标绝对值 [B, T-1, D]
            step_weights: Optional step weights [B, T-1] / 可选的步骤权重 [B, T-1]

        Returns:
            Tuple of (total_loss, (delta_disp_loss, delta_load_loss, abs_disp_loss,
                                  abs_load_loss, stiffness_loss))
            (总损失, (delta位移损失, delta荷载损失, 绝对位移损失, 绝对荷载损失, 刚度损失))元组
        """
        load_dim = self.config.load_dim
        disp_slice = slice(load_dim, load_dim + self.config.disp_dim)

        # Loss functions / 损失函数
        if self.config.use_huber_for_delta:
            def delta_loss_fn(a, b):
                return F.huber_loss(a, b, reduction='none', delta=self.config.huber_delta)
        else:
            def delta_loss_fn(a, b):
                return F.mse_loss(a, b, reduction='none')

        def abs_loss_fn(a, b):
            return F.mse_loss(a, b, reduction='none')

        # Split predictions / 分割预测
        dp_load, dt_load = delta_pred[:, :, :load_dim], delta_target[:, :, :load_dim]
        dp_disp, dt_disp = delta_pred[:, :, disp_slice], delta_target[:, :, disp_slice]
        ap_load, at_load = abs_pred[:, :, :load_dim], abs_target[:, :, :load_dim]
        ap_disp, at_disp = abs_pred[:, :, disp_slice], abs_target[:, :, disp_slice]

        # Compute losses / 计算损失
        delta_load_loss = delta_loss_fn(dp_load, dt_load)
        delta_disp_loss = delta_loss_fn(dp_disp, dt_disp)
        abs_load_loss = abs_loss_fn(ap_load, at_load)
        abs_disp_loss = abs_loss_fn(ap_disp, at_disp)

        # Apply step weights / 应用步骤权重
        if step_weights is not None:
            sw = step_weights.unsqueeze(-1)  # [B, T, 1]
            delta_load_loss *= sw
            delta_disp_loss *= sw
            abs_load_loss *= sw
            abs_disp_loss *= sw

        # Average losses / 平均损失
        delta_load_loss = delta_load_loss.mean()
        delta_disp_loss = delta_disp_loss.mean()
        abs_load_loss = abs_load_loss.mean()
        abs_disp_loss = abs_disp_loss.mean()

        # Combine losses / 组合损失
        L_delta = (self.config.load_weight * delta_load_loss +
                   self.config.displacement_weight * delta_disp_loss)
        L_abs = (self.config.load_weight * abs_load_loss +
                 self.config.displacement_weight * abs_disp_loss)

        total = self.config.w_delta * L_delta + self.config.w_abs * L_abs

        # Stiffness loss / 刚度损失
        stiffness_loss = torch.tensor(0.0, device=abs_pred.device)
        if self.config.use_stiffness_loss and abs_pred.size(1) > 1:
            stiff_pred = self._compute_stiffness(abs_pred)
            stiff_true = self._compute_stiffness(abs_target)
            if self.config.use_huber_for_stiffness:
                stiff_loss = F.huber_loss(
                    stiff_pred, stiff_true,
                    reduction='none',
                    delta=self.config.huber_stiffness
                )
            else:
                stiff_loss = F.mse_loss(stiff_pred, stiff_true, reduction='none')

            if step_weights is not None and step_weights.size(1) > 1:
                stiff_loss *= step_weights[:, 1:]

            stiffness_loss = stiff_loss.mean()
            total = total + self.config.w_stiffness * stiffness_loss

        # Monotonic load penalty / 单调荷载惩罚
        if self.config.use_monotonic_load_loss:
            load_pred = abs_pred[:, :, 0]  # [B, T]
            dload = load_pred[:, 1:] - load_pred[:, :-1]
            mono_loss = F.relu(-dload).mean()
            total = total + self.config.w_monotonic_load * mono_loss

        return total, (delta_disp_loss, delta_load_loss, abs_disp_loss, abs_load_loss, stiffness_loss)