# -*- coding: utf-8 -*-
"""
Buckling Trainer / 屈曲训练器
==============================
Trainer class for training the BucklingPredictor model.
用于训练BucklingPredictor模型的训练器类。
"""

import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config.train import TrainingConfig
from data.dataset import BucklingDataset
from data.scalers import clamp_scaler_std_inplace
from models.predictor import BucklingPredictor, ARWrapper
from training.loss import DeltaAbsoluteLoss
from training.metrics import SequenceMetricMeter, ScalarMetricMeter, BucklingMetricMeter
from utils.seed import set_seed, worker_init_fn
from utils.buckling import moving_average_1d_torch, extract_buckling_response_torch

if TYPE_CHECKING:
    pass

warnings.filterwarnings('ignore')


class BucklingTrainer:
    """
    Trainer for the BucklingPredictor model.
    BucklingPredictor模型的训练器。

    Handles:
        - Data loading and splitting
        - Model training with scheduled sampling
        - Validation and early stopping
        - Checkpoint saving
        - TensorBoard logging
    """

    def __init__(self, config: TrainingConfig) -> None:
        """
        Initialize BucklingTrainer.
        初始化BucklingTrainer。

        Args:
            config: Training configuration / 训练配置
        """
        self.config = config
        self.device = torch.device(config.device)
        set_seed(config.seed)
        self._setup_logging()

        self.metrics_records: List[Dict[str, Any]] = []

        # Discover files / 发现文件
        all_files = BucklingDataset.discover_files(config.data_dir)
        if not all_files:
            raise FileNotFoundError(f"No CSV found in {config.data_dir}")

        # Split files / 划分文件
        rng = np.random.default_rng(config.seed)
        indices = np.arange(len(all_files))
        rng.shuffle(indices)

        n_total = len(indices)
        n_train = int(config.train_ratio * n_total)
        n_val = int(config.val_ratio * n_total)

        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]

        train_files = [all_files[i] for i in train_idx]
        val_files = [all_files[i] for i in val_idx]
        test_files = [all_files[i] for i in test_idx]

        # Apply caps for HPO / 应用于HPO的上限
        if config.max_train_files is not None:
            train_files = train_files[:config.max_train_files]
        if config.max_val_files is not None:
            val_files = val_files[:config.max_val_files]
        if config.max_test_files is not None:
            test_files = test_files[:config.max_test_files]

        self.train_files, self.val_files, self.test_files = train_files, val_files, test_files

        # Load or fit scalers / 加载或拟合标准化器
        scalers_loaded = False
        try:
            self.scalers = joblib.load(self.config.scaler_save_path)
            scalers_loaded = True
            logging.info("Loaded existing scalers. / 已加载现有标准化器")
        except Exception:
            self.scalers = BucklingDataset.fit_scalers_on_files(self.train_files, self.config)
            joblib.dump(self.scalers, self.config.scaler_save_path)
            logging.info("Fitted scalers on train set. / 在训练集上拟合标准化器")

        clamp_scaler_std_inplace(self.scalers['fixed'], name="fixed(post)")
        clamp_scaler_std_inplace(self.scalers['abs'], name="abs(post)")
        clamp_scaler_std_inplace(self.scalers['delta'], name="delta(post)")
        if scalers_loaded:
            joblib.dump(self.scalers, self.config.scaler_save_path)

        # Create datasets / 创建数据集
        self.train_dataset = BucklingDataset(self.train_files, self.config, self.scalers, normalize=True)
        self.val_dataset = BucklingDataset(self.val_files, self.config, self.scalers, normalize=True)
        self.test_dataset = BucklingDataset(self.test_files, self.config, self.scalers, normalize=True)

        # Create data loaders / 创建数据加载器
        pin = (self.config.pin_memory and self.device.type == "cuda")
        loader_kwargs = dict(
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            pin_memory=pin,
            persistent_workers=(self.config.persistent_workers and self.config.num_workers > 0),
            prefetch_factor=(self.config.prefetch_factor if self.config.num_workers > 0 else None),
            worker_init_fn=(worker_init_fn if self.config.num_workers > 0 else None),
        )

        self.train_loader = DataLoader(
            self.train_dataset, shuffle=True, drop_last=True, **loader_kwargs
        )
        self.val_loader = DataLoader(
            self.val_dataset, shuffle=False, **loader_kwargs
        )
        self.test_loader = DataLoader(
            self.test_dataset, shuffle=False, **loader_kwargs
        )

        # Create model / 创建模型
        self.model = BucklingPredictor(self.config).to(self.device)

        # Create optimizer and scheduler / 创建优化器和调度器
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2, eta_min=1e-8
        )

        # Create loss function / 创建损失函数
        self.criterion = DeltaAbsoluteLoss(self.config)

        # Setup AMP / 设置AMP
        self.amp_enabled = (self.config.use_amp and torch.cuda.is_available())
        try:
            self.scaler_amp = torch.amp.GradScaler('cuda', enabled=self.amp_enabled)
        except Exception:
            self.scaler_amp = torch.cuda.amp.GradScaler(enabled=self.amp_enabled)

        # Setup TensorBoard / 设置TensorBoard
        self.writer = SummaryWriter(log_dir=self.config.log_dir)

        # Early stopping / 早停
        self.best_score = float('inf')
        self.early_stop_counter = 0

        # Delta->abs transformation buffers / Delta到abs变换缓冲区
        abs_scale_np = np.where(self.scalers['abs'].scale_.copy() < 1e-9, 1.0, self.scalers['abs'].scale_.copy())
        delta_mean_np = self.scalers['delta'].mean_.copy()
        delta_scale_np = np.where(self.scalers['delta'].scale_.copy() < 1e-9, 1.0, self.scalers['delta'].scale_.copy())

        self.abs_mean = torch.tensor(
            self.scalers['abs'].mean_.copy(), dtype=torch.float32, device=self.device
        ).view(1, -1)
        self.abs_std = torch.tensor(
            abs_scale_np, dtype=torch.float32, device=self.device
        ).view(1, -1)

        self.A_vec = torch.tensor(
            delta_scale_np / abs_scale_np, dtype=torch.float32, device=self.device
        )
        self.B_vec = torch.tensor(
            delta_mean_np / abs_scale_np, dtype=torch.float32, device=self.device
        )
        self.A_broadcast = self.A_vec.view(1, 1, -1)
        self.B_broadcast = self.B_vec.view(1, 1, -1)

        # AR module / AR模块
        self.ar_module = ARWrapper(self.model, self.A_vec, self.B_vec).to(self.device)
        self.ar = self.ar_module

        if hasattr(torch, "compile") and self.config.use_torch_compile:
            try:
                self.ar = torch.compile(
                    self.ar_module, backend="inductor", mode="max-autotune",
                    fullgraph=False, dynamic=True
                )
                logging.info("torch.compile enabled for AR. / AR已启用torch.compile")
            except Exception as e:
                logging.warning(f"torch.compile failed: {e}. fallback eager. / torch.compile失败，回退到eager模式")
                self.ar = self.ar_module

        logging.info(
            f"Dataset split -> Train:{len(self.train_dataset)} "
            f"Val:{len(self.val_dataset)} Test:{len(self.test_dataset)}"
        )

    def _setup_logging(self) -> None:
        """Setup logging. / 设置日志。"""
        log_file = Path(self.config.log_dir) / f"training_{datetime.now():%Y%m%d_%H%M%S}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.FileHandler(log_file, encoding="utf-8"), logging.StreamHandler()]
        )

    # =========================================================================
    # Schedule Helpers / 调度辅助函数
    # =========================================================================
    def _get_teacher_forcing_ratio(self, epoch: int) -> float:
        """
        Get teacher forcing ratio for current epoch.
        获取当前epoch的教师强制比率。

        Args:
            epoch: Current epoch / 当前epoch

        Returns:
            Teacher forcing ratio / 教师强制比率
        """
        if epoch < self.config.teacher_forcing_epochs:
            return 1.0

        transition = self.config.transition_epochs
        if transition > 0 and epoch < self.config.teacher_forcing_epochs + transition:
            progress = (epoch - self.config.teacher_forcing_epochs) / transition
            return max(0.0, 1.0 - progress)

        return 0.0

    def _apply_phase_lr(self, tf_ratio: float) -> Tuple[float, float]:
        """
        Apply learning rate factor based on training phase.
        根据训练阶段应用学习率因子。

        Args:
            tf_ratio: Current teacher forcing ratio / 当前教师强制比率

        Returns:
            Tuple of (new_lr, factor) / (新学习率, 因子)元组
        """
        base_lr = self.scheduler.get_last_lr()[0]

        if tf_ratio >= 0.999:
            factor = self.config.lr_factor_tf
        elif tf_ratio <= 0.001:
            factor = self.config.lr_factor_ar
        else:
            factor = self.config.lr_factor_mixed

        new_lr = base_lr * factor
        for g in self.optimizer.param_groups:
            g['lr'] = new_lr

        return new_lr, factor

    def _create_step_weights(self, seq_len: int, batch_size: int) -> torch.Tensor:
        """
        Create step weights for early-step emphasis.
        创建早期步骤加权的步骤权重。

        Args:
            seq_len: Sequence length / 序列长度
            batch_size: Batch size / 批量大小

        Returns:
            Step weights [B, T] / 步骤权重 [B, T]
        """
        if seq_len <= 0 or self.config.early_steps_span <= 0 or self.config.early_step_max_boost <= 1.0:
            return torch.ones(batch_size, seq_len, device=self.device)

        span = min(self.config.early_steps_span, seq_len)
        idx = torch.arange(seq_len, device=self.device, dtype=torch.float32)
        weights = torch.ones(seq_len, device=self.device)

        if span > 1:
            portion = idx[:span] / (span - 1)
            boost = torch.pow(1.0 - portion, self.config.early_step_decay_power)
        else:
            boost = torch.ones(1, device=self.device)

        weights[:span] = 1.0 + (self.config.early_step_max_boost - 1.0) * boost

        return weights.unsqueeze(0).expand(batch_size, -1).contiguous()

    def _roll_abs_norm_from_delta(
        self,
        abs0_norm: torch.Tensor,
        delta_norm_seq: torch.Tensor
    ) -> torch.Tensor:
        """
        Convert delta predictions to absolute values.
        将delta预测转换为绝对值。

        Args:
            abs0_norm: Initial absolute state [B, D] / 初始绝对状态 [B, D]
            delta_norm_seq: Delta sequence [B, T, D] / Delta序列 [B, T, D]

        Returns:
            Absolute sequence [B, T, D] / 绝对序列 [B, T, D]
        """
        inc = delta_norm_seq * self.A_broadcast + self.B_broadcast
        return abs0_norm.unsqueeze(1) + torch.cumsum(inc, dim=1)

    def denorm_abs(self, abs_norm: torch.Tensor) -> torch.Tensor:
        """
        Denormalize absolute values to physical space.
        将绝对值反归一化到物理空间。

        Args:
            abs_norm: Normalized absolute values / 标准化的绝对值

        Returns:
            Physical space values / 物理空间值
        """
        return abs_norm * self.abs_std + self.abs_mean

    # =========================================================================
    # Core Prediction Methods / 核心预测方法
    # =========================================================================
    def _predict_sequence(
        self,
        fixed_norm: torch.Tensor,
        abs_norm: torch.Tensor,
        mode: str,
        tf_ratio: float,
        epoch_seed: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict sequence with specified mode.
        使用指定模式预测序列。

        Args:
            fixed_norm: Normalized fixed features [B, Dfix] / 标准化的固定特征 [B, Dfix]
            abs_norm: Normalized absolute states [B, T, D] / 标准化的绝对状态 [B, T, D]
            mode: Prediction mode ('tf', 'ar', 'match_train') / 预测模式
            tf_ratio: Teacher forcing ratio / 教师强制比率
            epoch_seed: Random seed for this epoch / 此epoch的随机种子

        Returns:
            Tuple of (delta_pred, abs_pred) / (delta预测, abs预测)元组
        """
        B, T, _ = abs_norm.shape
        x_seq = abs_norm[:, :-1, :]   # [B, T-1, D]
        steps = T - 1
        mode = mode.lower()

        if mode == "tf":
            delta_pred, _ = self.model(fixed_norm, x_seq)
            abs_pred = self._roll_abs_norm_from_delta(x_seq[:, 0, :], delta_pred)
            return delta_pred, abs_pred

        if mode == "ar":
            try:
                delta_pred, abs_pred = self.ar(fixed_norm, x_seq[:, 0, :], steps)
            except Exception as e:
                logging.warning(f"AR compiled path failed: {e}. fallback eager. / AR编译路径失败，回退到eager模式")
                delta_pred, abs_pred = self.ar_module(fixed_norm, x_seq[:, 0, :], steps)
            return delta_pred, abs_pred

        if mode == "match_train":
            if tf_ratio >= 0.999:
                return self._predict_sequence(fixed_norm, abs_norm, mode="tf", tf_ratio=tf_ratio, epoch_seed=epoch_seed)
            if tf_ratio <= 0.001:
                return self._predict_sequence(fixed_norm, abs_norm, mode="ar", tf_ratio=tf_ratio, epoch_seed=epoch_seed)

            # Deterministic per-sample mixture / 确定性的每样本混合
            g = torch.Generator(device=fixed_norm.device)
            g.manual_seed(epoch_seed)
            mask_tf = torch.bernoulli(
                torch.full((B,), tf_ratio, device=fixed_norm.device), generator=g
            ).bool()
            idx_tf = mask_tf.nonzero(as_tuple=False).squeeze(-1)
            idx_ar = (~mask_tf).nonzero(as_tuple=False).squeeze(-1)

            out_dtype = torch.float32
            delta_pred = torch.zeros(
                B, steps, self.config.dynamic_features_dim,
                device=fixed_norm.device, dtype=out_dtype
            )
            abs_pred = torch.zeros_like(delta_pred)

            if idx_tf.numel() > 0:
                fixed_tf = fixed_norm.index_select(0, idx_tf)
                abs_tf = abs_norm.index_select(0, idx_tf)
                d_tf, a_tf = self._predict_sequence(
                    fixed_tf, abs_tf, mode="tf", tf_ratio=tf_ratio, epoch_seed=epoch_seed
                )
                delta_pred.index_copy_(0, idx_tf, d_tf.to(out_dtype))
                abs_pred.index_copy_(0, idx_tf, a_tf.to(out_dtype))

            if idx_ar.numel() > 0:
                fixed_ar = fixed_norm.index_select(0, idx_ar)
                abs_ar = abs_norm.index_select(0, idx_ar)
                d_ar, a_ar = self._predict_sequence(
                    fixed_ar, abs_ar, mode="ar", tf_ratio=tf_ratio, epoch_seed=epoch_seed
                )
                delta_pred.index_copy_(0, idx_ar, d_ar.to(out_dtype))
                abs_pred.index_copy_(0, idx_ar, a_ar.to(out_dtype))

            return delta_pred, abs_pred

        raise ValueError(f"Unknown mode: {mode}")

    # =========================================================================
    # Training Loop / 训练循环
    # =========================================================================
    def train_epoch(self, epoch: int, tf_ratio: float) -> Dict[str, float]:
        """
        Train for one epoch.
        训练一个epoch。

        Args:
            epoch: Current epoch / 当前epoch
            tf_ratio: Teacher forcing ratio / 教师强制比率

        Returns:
            Dictionary of training metrics / 训练指标字典
        """
        self.model.train()
        total_loss = 0.0
        metrics_sum = torch.zeros(5, dtype=torch.float64)
        total_samples = 0

        use_seq_tf = tf_ratio >= 0.999
        use_seq_ar = tf_ratio <= 0.001

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1} Train", leave=False)
        for fixed_norm, abs_norm, delta_norm in pbar:
            fixed_norm = fixed_norm.to(self.device)
            abs_norm = abs_norm.to(self.device)
            delta_norm = delta_norm.to(self.device)

            B, T, _ = abs_norm.shape
            x_seq = abs_norm[:, :-1, :]
            y_abs = abs_norm[:, 1:, :]
            y_delta = delta_norm

            step_weights = self._create_step_weights(T - 1, B)

            self.optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=self.device.type, enabled=self.amp_enabled):
                if use_seq_tf:
                    delta_pred, _ = self.model(fixed_norm, x_seq)
                    abs_pred = self._roll_abs_norm_from_delta(x_seq[:, 0, :], delta_pred)
                    loss, metrics = self.criterion(delta_pred, y_delta, abs_pred, y_abs, step_weights)

                elif use_seq_ar:
                    delta_pred, abs_pred = self._predict_sequence(
                        fixed_norm, abs_norm, mode="ar",
                        tf_ratio=tf_ratio, epoch_seed=self.config.seed + epoch
                    )
                    loss, metrics = self.criterion(delta_pred, y_delta, abs_pred, y_abs, step_weights)

                else:
                    # Sample-wise mixture / 样本级混合
                    g = torch.Generator(device=self.device)
                    g.manual_seed(self.config.seed + epoch)
                    mask_tf = torch.bernoulli(
                        torch.full((B,), tf_ratio, device=self.device), generator=g
                    ).bool()
                    idx_tf = mask_tf.nonzero(as_tuple=False).squeeze(-1)
                    idx_ar = (~mask_tf).nonzero(as_tuple=False).squeeze(-1)

                    accum_loss = 0.0
                    accum_metrics = torch.zeros(5, device=self.device)
                    total_bs = 0

                    if idx_tf.numel() > 0:
                        fixed_tf = fixed_norm.index_select(0, idx_tf)
                        x_seq_tf = x_seq.index_select(0, idx_tf)
                        y_abs_tf = y_abs.index_select(0, idx_tf)
                        y_delta_tf = y_delta.index_select(0, idx_tf)
                        sw_tf = step_weights.index_select(0, idx_tf)

                        delta_pred_tf, _ = self.model(fixed_tf, x_seq_tf)
                        abs_pred_tf = self._roll_abs_norm_from_delta(x_seq_tf[:, 0, :], delta_pred_tf)
                        loss_tf, metrics_tf = self.criterion(delta_pred_tf, y_delta_tf, abs_pred_tf, y_abs_tf, sw_tf)

                        bs_tf = fixed_tf.size(0)
                        accum_loss += loss_tf * bs_tf
                        accum_metrics += torch.stack(metrics_tf) * bs_tf
                        total_bs += bs_tf

                    if idx_ar.numel() > 0:
                        fixed_ar = fixed_norm.index_select(0, idx_ar)
                        abs_ar = abs_norm.index_select(0, idx_ar)
                        y_abs_ar = y_abs.index_select(0, idx_ar)
                        y_delta_ar = y_delta.index_select(0, idx_ar)
                        sw_ar = step_weights.index_select(0, idx_ar)

                        delta_pred_ar, abs_pred_ar = self._predict_sequence(
                            fixed_ar, abs_ar, mode="ar",
                            tf_ratio=tf_ratio, epoch_seed=self.config.seed + epoch
                        )
                        loss_ar, metrics_ar = self.criterion(delta_pred_ar, y_delta_ar, abs_pred_ar, y_abs_ar, sw_ar)

                        bs_ar = fixed_ar.size(0)
                        accum_loss += loss_ar * bs_ar
                        accum_metrics += torch.stack(metrics_ar) * bs_ar
                        total_bs += bs_ar

                    loss = accum_loss / max(1, total_bs)
                    metrics = tuple((accum_metrics / max(1, total_bs)).tolist())

            self.scaler_amp.scale(loss).backward()
            self.scaler_amp.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)
            self.scaler_amp.step(self.optimizer)
            self.scaler_amp.update()

            bs = fixed_norm.size(0)
            total_loss += loss.item() * bs

            if isinstance(metrics[0], torch.Tensor):
                md = torch.tensor([m.item() for m in metrics], dtype=torch.float64)
            else:
                md = torch.tensor(metrics, dtype=torch.float64)
            metrics_sum += md * bs
            total_samples += bs

            pbar.set_postfix(loss=f"{total_loss / total_samples:.6f}", tf=f"{tf_ratio:.2f}")

        avg = metrics_sum / max(1, total_samples)
        return dict(
            total_loss=total_loss / max(1, total_samples),
            delta_disp_loss=float(avg[0]),
            delta_load_loss=float(avg[1]),
            abs_disp_loss=float(avg[2]),
            abs_load_loss=float(avg[3]),
            stiffness_loss=float(avg[4]),
        )

    @torch.no_grad()
    def eval_loop(
        self,
        loader: DataLoader,
        mode_name: str,
        predict_mode: str,
        tf_ratio: float,
        epoch: int
    ) -> Dict[str, float]:
        """
        Evaluation loop.
        评估循环。

        Args:
            loader: Data loader / 数据加载器
            mode_name: Name for logging / 用于日志的名称
            predict_mode: Prediction mode / 预测模式
            tf_ratio: Teacher forcing ratio / 教师强制比率
            epoch: Current epoch / 当前epoch

        Returns:
            Dictionary of evaluation metrics / 评估指标字典
        """
        self.model.eval()

        total_loss = 0.0
        metrics_sum = torch.zeros(5, dtype=torch.float64)
        nsamples = 0

        vec_meter = SequenceMetricMeter()
        load_meter = ScalarMetricMeter()
        disp_meter = ScalarMetricMeter()
        buckling_meter = BucklingMetricMeter()

        for fixed_norm, abs_norm, delta_norm in tqdm(loader, desc=mode_name, leave=False):
            fixed_norm = fixed_norm.to(self.device)
            abs_norm = abs_norm.to(self.device)
            delta_norm = delta_norm.to(self.device)

            B, T, _ = abs_norm.shape
            y_abs = abs_norm[:, 1:, :]
            y_delta = delta_norm

            step_weights = None
            if self.config.use_step_weights_in_val:
                step_weights = self._create_step_weights(T - 1, B)

            with torch.amp.autocast(device_type=self.device.type, enabled=self.amp_enabled):
                delta_pred, abs_pred = self._predict_sequence(
                    fixed_norm, abs_norm, mode=predict_mode,
                    tf_ratio=tf_ratio, epoch_seed=self.config.seed + epoch
                )
                loss, metrics = self.criterion(delta_pred, y_delta, abs_pred, y_abs, step_weights)

            total_loss += loss.item() * B
            metrics_sum += torch.tensor([m.item() for m in metrics], dtype=torch.float64) * B
            nsamples += B

            # Physical metrics / 物理指标
            abs_pred_phys = self.denorm_abs(abs_pred)
            y_abs_phys = self.denorm_abs(y_abs)

            vec_meter.update(abs_pred_phys, y_abs_phys)
            load_meter.update(abs_pred_phys[..., 0], y_abs_phys[..., 0])

            disp_col = self.config.load_dim + self.config.buckling_disp_index
            disp_meter.update(abs_pred_phys[..., disp_col], y_abs_phys[..., disp_col])

            # Buckling metrics / 屈曲指标
            load_p, disp_p = extract_buckling_response_torch(
                abs_pred_phys,
                self.config.load_dim,
                self.config.buckling_disp_index,
                slope_threshold=self.config.buckling_slope_threshold,
                min_index=self.config.buckling_min_index,
                eps=self.config.buckling_eps,
                smooth=self.config.buckling_smooth,
                smooth_window=self.config.buckling_smooth_window
            )
            load_t, disp_t = extract_buckling_response_torch(
                y_abs_phys,
                self.config.load_dim,
                self.config.buckling_disp_index,
                slope_threshold=self.config.buckling_slope_threshold,
                min_index=self.config.buckling_min_index,
                eps=self.config.buckling_eps,
                smooth=self.config.buckling_smooth,
                smooth_window=self.config.buckling_smooth_window
            )
            buckling_meter.update(load_p, load_t, disp_p, disp_t)

        avg_loss = total_loss / max(1, nsamples)
        avg_metrics = (metrics_sum / max(1, nsamples)).tolist()
        vec_stats = vec_meter.compute()
        load_stats = load_meter.compute()
        disp_stats = disp_meter.compute()
        buck_stats = buckling_meter.compute()

        return dict(
            total_loss=avg_loss,
            delta_disp_loss=avg_metrics[0],
            delta_load_loss=avg_metrics[1],
            abs_disp_loss=avg_metrics[2],
            abs_load_loss=avg_metrics[3],
            stiffness_loss=avg_metrics[4],
            phys_mse=vec_stats['mse'],
            phys_rmse=vec_stats['rmse'],
            phys_mae=vec_stats['mae'],
            phys_r2=vec_stats['r2'],
            phys_load_mae=load_stats['mae'],
            phys_load_rmse=load_stats['rmse'],
            phys_disp_mae=disp_stats['mae'],
            phys_disp_rmse=disp_stats['rmse'],
            **buck_stats
        )

    # =========================================================================
    # Checkpoint & Metrics IO / 检查点和指标输入输出
    # =========================================================================
    def save_checkpoint(
        self,
        epoch: int,
        monitor_value: float,
        is_best: bool = False
    ) -> None:
        """
        Save training checkpoint.
        保存训练检查点。

        Args:
            epoch: Current epoch / 当前epoch
            monitor_value: Value being monitored / 监控值
            is_best: Whether this is the best model / 是否为最佳模型
        """
        ckpt = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'monitor_value': monitor_value,
            'config': self.config,
            'scalers': self.scalers
        }

        path = self.config.model_save_path
        if is_best:
            path = path.replace(".pth", "_best.pth")

        torch.save(ckpt, path)
        logging.info(f"Saved checkpoint -> {path}")

    def _record_metrics(self, epoch: int, phase: str, metrics: Dict[str, Any]) -> None:
        """
        Record metrics for CSV export.
        记录指标用于CSV导出。

        Args:
            epoch: Current epoch / 当前epoch
            phase: Phase name ('train', 'val_match', 'val_ar', 'test')
            metrics: Metrics dictionary / 指标字典
        """
        entry = dict(epoch=epoch + 1, phase=phase)
        entry.update(metrics)
        self.metrics_records.append(entry)

    def _dump_metrics_csv(self) -> None:
        """Export metrics to CSV. / 导出指标到CSV。"""
        if not self.metrics_records:
            return
        df = pd.DataFrame(self.metrics_records)
        df.to_csv(self.config.metrics_csv_path, index=False, encoding="utf-8-sig")
        logging.info(f"Metrics saved -> {self.config.metrics_csv_path}")

    # =========================================================================
    # Main Training Loop / 主训练循环
    # =========================================================================
    def train(self) -> None:
        """
        Main training loop.
        主训练循环。
        """
        logging.info("Start training ...")
        logging.info(f"Num params: {sum(p.numel() for p in self.model.parameters()):,}")

        for epoch in range(self.config.epochs):
            tf_ratio = self._get_teacher_forcing_ratio(epoch)
            lr_now, phase_factor = self._apply_phase_lr(tf_ratio)

            train_metrics = self.train_epoch(epoch, tf_ratio)

            val_match = None
            val_ar = None

            # ValMatch: only in TF/mixed phase (AR phase → redundant with ValAR)
            if self.config.validate_match_train and tf_ratio > 0.0:
                val_match = self.eval_loop(
                    self.val_loader,
                    mode_name="Val(match_train)",
                    predict_mode="match_train",
                    tf_ratio=tf_ratio,
                    epoch=epoch
                )
                self._record_metrics(epoch, "val_match", val_match)

            # ValAR: always (AR phase uses this for best-model monitoring)
            if self.config.validate_always_ar:
                # In AR phase, ValMatch is skipped — copy ValAR to ValMatch slot for CSV compatibility
                if tf_ratio == 0.0:
                    val_match = self.eval_loop(
                        self.val_loader,
                        mode_name="Val(AR)",
                        predict_mode="ar",
                        tf_ratio=0.0,
                        epoch=epoch
                    )
                    val_ar = val_match  # same result, avoid double compute
                    self._record_metrics(epoch, "val_match", val_match)
                    self._record_metrics(epoch, "val_ar", val_ar)
                else:
                    val_ar = self.eval_loop(
                        self.val_loader,
                        mode_name="Val(AR)",
                        predict_mode="ar",
                        tf_ratio=0.0,
                        epoch=epoch
                    )
                    self._record_metrics(epoch, "val_ar", val_ar)
                    if val_match is None:
                        val_match = self.eval_loop(
                            self.val_loader,
                            mode_name="Val(match_train)",
                            predict_mode="match_train",
                            tf_ratio=tf_ratio,
                            epoch=epoch
                        )
                        self._record_metrics(epoch, "val_match", val_match)

            self.scheduler.step()
            base_lr_next = self.scheduler.get_last_lr()[0]

            self._record_metrics(epoch, "train", dict(**train_metrics, lr=lr_now, tf_ratio=tf_ratio))

            # TensorBoard logging / TensorBoard日志
            step = epoch
            self.writer.add_scalar('LR/Current', lr_now, step)
            self.writer.add_scalar('LR/BaseNextEpoch', base_lr_next, step)
            self.writer.add_scalar('TF_Ratio', tf_ratio, step)
            self.writer.add_scalar('Loss/Train_Total', train_metrics['total_loss'], step)

            if val_match is not None:
                self.writer.add_scalar('Loss/ValMatch_Total', val_match['total_loss'], step)
                self.writer.add_scalar('Metrics/ValMatch_BucklingLoad_MAE', val_match['buckling_load_mae'], step)

            if val_ar is not None:
                self.writer.add_scalar('Loss/ValAR_Total', val_ar['total_loss'], step)
                self.writer.add_scalar('Metrics/ValAR_BucklingLoad_MAE', val_ar['buckling_load_mae'], step)

            # Console logging / 控制台日志
            logging.info(
                f"Epoch {epoch + 1}/{self.config.epochs} | tf_ratio={tf_ratio:.2f} | "
                f"phase_lr_factor={phase_factor:.2f} | lr={lr_now:.3e} | base_lr(next)={base_lr_next:.3e}"
            )
            logging.info(
                f"Train | total:{train_metrics['total_loss']:.6f} "
                f"Δdisp:{train_metrics['delta_disp_loss']:.6f} Δload:{train_metrics['delta_load_loss']:.6f} "
                f"abs_disp:{train_metrics['abs_disp_loss']:.6f} abs_load:{train_metrics['abs_load_loss']:.6f} "
                f"stiff:{train_metrics['stiffness_loss']:.6f}"
            )

            if val_match is not None:
                logging.info(
                    f"ValMatch | total:{val_match['total_loss']:.6f} "
                    f"MSE:{val_match['phys_mse']:.3e} RMSE:{val_match['phys_rmse']:.3e} "
                    f"MAE:{val_match['phys_mae']:.3e} R2:{val_match['phys_r2']:.4f} | "
                    f"LoadRMSE:{val_match['phys_load_rmse']:.3e} DispRMSE:{val_match['phys_disp_rmse']:.3e} | "
                    f"BuckLoadMAE:{val_match['buckling_load_mae']:.3e} BuckDispMAE:{val_match['buckling_disp_mae']:.3e}"
                )

            if val_ar is not None:
                logging.info(
                    f"ValAR    | total:{val_ar['total_loss']:.6f} "
                    f"MSE:{val_ar['phys_mse']:.3e} RMSE:{val_ar['phys_rmse']:.3e} "
                    f"MAE:{val_ar['phys_mae']:.3e} R2:{val_ar['phys_r2']:.4f} | "
                    f"LoadRMSE:{val_ar['phys_load_rmse']:.3e} DispRMSE:{val_ar['phys_disp_rmse']:.3e} | "
                    f"BuckLoadMAE:{val_ar['buckling_load_mae']:.3e} BuckDispMAE:{val_ar['buckling_disp_mae']:.3e}"
                )

            # Check best model / 检查最佳模型
            monitor_value = None
            if self.config.best_monitor == "val_ar_total_loss":
                monitor_value = float('inf') if val_ar is None else val_ar['total_loss']
            else:
                monitor_value = float('inf') if val_match is None else val_match['total_loss']

            if monitor_value < self.best_score:
                self.best_score = monitor_value
                self.early_stop_counter = 0
                self.save_checkpoint(epoch, monitor_value, is_best=True)
                logging.info(f"New best ({self.config.best_monitor}) = {monitor_value:.6f}")
            else:
                self.early_stop_counter += 1

            # Periodic checkpoint / 周期性检查点
            if (epoch + 1) % 40 == 0:
                self.save_checkpoint(epoch, monitor_value, is_best=False)

            # Export metrics / 导出指标
            if self.config.export_metrics_every_epoch:
                self._dump_metrics_csv()

            # Early stopping / 早停
            if self.early_stop_counter >= self.config.early_stopping_patience:
                logging.info(f"Early stopping: {self.early_stop_counter} epochs no improvement.")
                break

        # Test evaluation / 测试评估
        test_metrics = self.test()
        self._record_metrics(epoch, "test", test_metrics)
        self._dump_metrics_csv()
        self.writer.close()

        logging.info("Training finished.")

    def train_for_optuna(self, trial) -> float:
        """
        Training for Optuna hyperparameter optimization.
        用于Optuna超参数优化的训练。

        Args:
            trial: Optuna trial object / Optuna trial对象

        Returns:
            Best validation loss / 最佳验证损失
        """
        best = float('inf')
        patience = max(5, self.config.early_stopping_patience)
        bad = 0

        for epoch in range(self.config.epochs):
            tf_ratio = self._get_teacher_forcing_ratio(epoch)
            _lr_now, _ = self._apply_phase_lr(tf_ratio)

            _ = self.train_epoch(epoch, tf_ratio)
            val_ar = self.eval_loop(
                self.val_loader, "Val(AR)", predict_mode="ar", tf_ratio=0.0, epoch=epoch
            )
            score = val_ar['total_loss']
            self.scheduler.step()

            trial.report(score, step=epoch)
            if trial.should_prune():
                raise Exception("TrialPruned")

            if score < best:
                best = score
                bad = 0
            else:
                bad += 1
                if bad >= patience:
                    break

        return best

    @torch.no_grad()
    def test(self) -> Dict[str, float]:
        """
        Evaluate on test set.
        在测试集上评估。

        Returns:
            Dictionary of test metrics / 测试指标字典
        """
        best_model_path = self.config.model_save_path.replace('.pth', '_best.pth')
        if Path(best_model_path).exists():
            ckpt = torch.load(best_model_path, map_location=self.device)
            self.model.load_state_dict(ckpt['model_state_dict'])
            logging.info(f"Loaded best model for test: {best_model_path}")

        metrics = self.eval_loop(
            self.test_loader, "Test(AR)", predict_mode="ar", tf_ratio=0.0, epoch=10_000
        )
        logging.info(
            f"Test(AR) | "
            f"SeqRMSE:{metrics['phys_rmse']:.3e} SeqR2:{metrics['phys_r2']:.4f} | "
            f"BuckLoad MAE:{metrics['buckling_load_mae']:.3e} "
            f"RMSE:{metrics.get('buckling_load_rmse', 0):.3e} "
            f"R2:{metrics.get('buckling_load_r2', 0):.4f} "
            f"MAPE:{metrics.get('buckling_load_mape', 0):.2f}% "
            f"NRMSE:{metrics.get('buckling_load_nrmse', 0):.4f} | "
            f"BuckDisp MAE:{metrics['buckling_disp_mae']:.3e}"
        )
        return metrics