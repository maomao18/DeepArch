# -*- coding: utf-8 -*-
"""
Baseline Model Trainer / 基线模型训练器
========================================
Generic training loop for baseline models (LSTM, TCN, Transformer).
All baselines share this trainer — only the model architecture differs.

通用训练循环，所有基线模型共用此训练器。
- Teacher forcing only (no scheduled sampling)
- Direct absolute-state MSE loss
- Same data pipeline, scalers, metrics as FA-LSTM

Usage:
    from training.baseline_trainer import BaselineTrainer
    trainer = BaselineTrainer(config, model)
    trainer.train()
"""

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data.dataset import BucklingDataset
from data.scalers import clamp_scaler_std_inplace
from utils.buckling import extract_buckling_response_torch
from utils.seed import set_seed, worker_init_fn


class BaselineTrainer:
    """
    Trainer for baseline models with teacher forcing.
    基线模型的教师强制训练器。
    """

    def __init__(self, config, model: nn.Module, best_monitor: str = "val_loss"):
        """
        Args:
            config:       TrainingConfig (or compatible namespace)
            model:        Baseline model (LSTMBaseline / TCNBaseline / ...)
            best_monitor: "val_loss" (TF, default) or "val_ar_total_loss" (AR)
        """
        self.config = config
        self.device = torch.device(config.device)
        set_seed(config.seed)
        self._setup_logging()

        self.model = model.to(self.device)

        # ── Data pipeline / 数据管道 ──
        all_files = BucklingDataset.discover_files(config.data_dir)
        if not all_files:
            raise FileNotFoundError(f"No CSV files in {config.data_dir}")

        rng = np.random.default_rng(config.seed)
        indices = np.arange(len(all_files))
        rng.shuffle(indices)

        n_total = len(indices)
        n_train = int(config.train_ratio * n_total)
        n_val = int(config.val_ratio * n_total)
        train_files = [all_files[i] for i in indices[:n_train]]
        val_files = [all_files[i] for i in indices[n_train:n_train + n_val]]
        test_files = [all_files[i] for i in indices[n_train + n_val:]]

        if config.max_train_files is not None:
            train_files = train_files[:config.max_train_files]
        if config.max_val_files is not None:
            val_files = val_files[:config.max_val_files]
        if config.max_test_files is not None:
            test_files = test_files[:config.max_test_files]

        # Ensure derived attributes / 确保派生属性存在
        if not hasattr(config, 'dynamic_col_start') or config.dynamic_col_start is None:
            config.dynamic_col_start = config.meta_cols + config.fixed_features_dim
        if not hasattr(config, 'disp_dim'):
            config.disp_dim = config.dynamic_features_dim - config.load_dim

        # Fit scalers on training data / 在训练数据上拟合标准化器
        self.scalers = BucklingDataset.fit_scalers_on_files(train_files, config)
        clamp_scaler_std_inplace(self.scalers)
        logging.info("Fitted scalers on train set.")

        # Create datasets with normalization / 创建标准化数据集
        self.train_dataset = BucklingDataset(train_files, config, self.scalers, normalize=True)
        self.val_dataset = BucklingDataset(val_files, config, self.scalers, normalize=True)
        self.test_dataset = BucklingDataset(test_files, config, self.scalers, normalize=True)

        logging.info(
            f"Dataset split -> Train:{len(self.train_dataset)} "
            f"Val:{len(self.val_dataset)} Test:{len(self.test_dataset)}"
        )

        # DataLoaders
        pin = (config.pin_memory and self.device.type == "cuda")
        loader_kwargs = dict(
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            pin_memory=pin,
            persistent_workers=(config.persistent_workers and config.num_workers > 0),
            prefetch_factor=(config.prefetch_factor if config.num_workers > 0 else None),
            worker_init_fn=(worker_init_fn if config.num_workers > 0 else None),
        )
        self.train_loader = DataLoader(self.train_dataset, shuffle=True, drop_last=True, **loader_kwargs)
        self.val_loader = DataLoader(self.val_dataset, shuffle=False, **loader_kwargs)
        self.test_loader = DataLoader(self.test_dataset, shuffle=False, **loader_kwargs)

        # ── Optimizer & Scheduler / 优化器和调度器 ──
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2, eta_min=1e-8,
        )

        # ── Loss / 损失 ──
        self.criterion = nn.MSELoss()

        # ── AMP / 混合精度 ──
        self.amp_enabled = (config.use_amp and torch.cuda.is_available())
        try:
            self.amp_ctx = torch.amp.autocast('cuda', enabled=self.amp_enabled)
        except Exception:
            self.amp_ctx = torch.cuda.amp.autocast(enabled=self.amp_enabled)

        # ── TensorBoard / 日志 ──
        self.writer = SummaryWriter(log_dir=config.log_dir)

        # ── Early stopping / 早停 ──
        self.best_monitor = best_monitor
        self.best_val_loss = float('inf')
        self.early_stop_counter = 0

        # ── Metrics records / 指标记录 ──
        self.metrics_records: List[Dict[str, Any]] = []

        # ── Denorm helpers / 反标准化辅助 ──
        abs_scale = np.where(
            self.scalers['abs'].scale_ < 1e-9, 1.0,
            self.scalers['abs'].scale_
        )
        self.abs_mean = torch.tensor(
            self.scalers['abs'].mean_, dtype=torch.float32, device=self.device
        ).view(1, 1, -1)
        self.abs_std = torch.tensor(
            abs_scale, dtype=torch.float32, device=self.device
        ).view(1, 1, -1)

        # External test dir / 外部测试目录
        self.ext_test_files: List[str] = []
        if config.external_test_dir:
            ext_dir = Path(config.external_test_dir)
            if ext_dir.exists():
                self.ext_test_files = sorted([str(p) for p in ext_dir.glob("*.csv")])
                logging.info(f"External test files: {len(self.ext_test_files)}")

    # ── Logging / 日志 ──
    def _setup_logging(self):
        log_dir = Path(self.config.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"training_{datetime.now():%Y%m%d_%H%M%S}.log"
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        for h in logger.handlers[:]:
            logger.removeHandler(h)
        logger.addHandler(logging.FileHandler(log_file, encoding="utf-8"))
        logger.addHandler(logging.StreamHandler())

    # ── Denormalization / 反标准化 ──
    def _denorm_abs(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.abs_std + self.abs_mean

    # ── Validation / 验证 ──
    @torch.no_grad()
    def _validate(self, loader: DataLoader, desc: str = "Val") -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        nsamples = 0

        # Physical metric accumulators
        vec_mse = 0.0
        vec_mae = 0.0
        vec_n = 0
        load_preds, load_trues = [], []
        disp_preds, disp_trues = [], []
        buck_load_p, buck_load_t = [], []
        buck_disp_p, buck_disp_t = [], []

        cfg = self.config

        for fixed_norm, abs_norm, _delta_norm in tqdm(loader, desc=desc, leave=False):
            fixed_norm = fixed_norm.to(self.device)
            abs_norm = abs_norm.to(self.device)
            B = fixed_norm.shape[0]

            x0 = abs_norm[:, 0, :]         # initial state
            target = abs_norm[:, 1:, :]    # target sequence
            T_eff = target.shape[1]

            # --- Teacher forcing (for loss / validation metric) ---
            with self.amp_ctx:
                pred_tf = self.model(fixed_norm, x0, target=target)
                loss = self.criterion(pred_tf, target)

            total_loss += loss.item() * B
            nsamples += B

            # Denormalize
            pred_phys = self._denorm_abs(pred_tf)
            true_phys = self._denorm_abs(target)

            # Physical metrics
            vec_mse += ((pred_phys - true_phys) ** 2).sum().item()
            vec_mae += (pred_phys - true_phys).abs().sum().item()
            vec_n += B * T_eff * cfg.dynamic_features_dim

            # Per-channel
            load_preds.append(pred_phys[..., 0].reshape(-1).cpu())
            load_trues.append(true_phys[..., 0].reshape(-1).cpu())
            disp_col = cfg.load_dim + cfg.buckling_disp_index
            disp_preds.append(pred_phys[..., disp_col].reshape(-1).cpu())
            disp_trues.append(true_phys[..., disp_col].reshape(-1).cpu())

            # Buckling extraction
            load_p, disp_p = extract_buckling_response_torch(
                pred_phys, cfg.load_dim, cfg.buckling_disp_index,
                slope_threshold=cfg.buckling_slope_threshold,
                min_index=cfg.buckling_min_index, eps=cfg.buckling_eps,
                smooth=cfg.buckling_smooth, smooth_window=cfg.buckling_smooth_window,
            )
            load_t, disp_t = extract_buckling_response_torch(
                true_phys, cfg.load_dim, cfg.buckling_disp_index,
                slope_threshold=cfg.buckling_slope_threshold,
                min_index=cfg.buckling_min_index, eps=cfg.buckling_eps,
                smooth=cfg.buckling_smooth, smooth_window=cfg.buckling_smooth_window,
            )
            buck_load_p.append(load_p.cpu())
            buck_load_t.append(load_t.cpu())
            buck_disp_p.append(disp_p.cpu())
            buck_disp_t.append(disp_t.cpu())

        # Compute aggregate metrics
        avg_loss = total_loss / nsamples if nsamples > 0 else float('inf')
        phys_rmse = np.sqrt(vec_mse / vec_n) if vec_n > 0 else 0.0
        phys_mae = vec_mae / vec_n if vec_n > 0 else 0.0

        # R2 (computed on load channel as primary metric)
        load_p_all = np.concatenate([t.numpy() for t in load_preds])
        load_t_all = np.concatenate([t.numpy() for t in load_trues])
        ss_res = ((load_t_all - load_p_all) ** 2).sum()
        ss_tot = ((load_t_all - load_t_all.mean()) ** 2).sum()
        load_r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
        load_rmse = float(np.sqrt(ss_res / len(load_t_all)))

        # Overall physical R2 (approximate via load R2)
        phys_r2 = load_r2

        buck_load_p_all = np.concatenate([t.numpy() for t in buck_load_p])
        buck_load_t_all = np.concatenate([t.numpy() for t in buck_load_t])
        buck_load_mae = float(np.abs(buck_load_p_all - buck_load_t_all).mean())
        buck_disp_p_all = np.concatenate([t.numpy() for t in buck_disp_p])
        buck_disp_t_all = np.concatenate([t.numpy() for t in buck_disp_t])
        buck_disp_mae = float(np.abs(buck_disp_p_all - buck_disp_t_all).mean())

        return {
            "total_loss": avg_loss,
            "phys_rmse": phys_rmse,
            "phys_mae": phys_mae,
            "phys_r2": float(phys_r2),
            "load_r2": float(load_r2),
            "load_rmse": load_rmse,
            "buckling_load_mae": buck_load_mae,
            "buckling_disp_mae": buck_disp_mae,
        }

    # ── Training epoch / 训练轮 ──
    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        nbatches = 0

        for fixed_norm, abs_norm, _delta_norm in tqdm(self.train_loader, desc=f"Train {epoch}", leave=False):
            fixed_norm = fixed_norm.to(self.device)
            abs_norm = abs_norm.to(self.device)

            x0 = abs_norm[:, 0, :]
            target = abs_norm[:, 1:, :]

            self.optimizer.zero_grad()
            with self.amp_ctx:
                pred = self.model(fixed_norm, x0, target=target)
                loss = self.criterion(pred, target)

            # Backward with AMP
            if self.amp_enabled:
                try:
                    from torch.amp import GradScaler
                    # Use simple backward for now (GradScaler state management is complex)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)
                    self.optimizer.step()
                except Exception:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)
                    self.optimizer.step()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)
                self.optimizer.step()

            total_loss += loss.item()
            nbatches += 1

        self.scheduler.step()
        return {"train_loss": total_loss / nbatches if nbatches > 0 else float('inf')}

    # ── Save checkpoint / 保存检查点 ──
    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        ckpt = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scalers": self.scalers,
            "config": self.config,
            "model_class": type(self.model).__name__,
        }
        path = Path(self.config.model_save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(ckpt, path)
        if is_best:
            best_path = Path(str(path).replace(".pth", "_best.pth"))
            torch.save(ckpt, best_path)
            logging.info(f"Saved best checkpoint -> {best_path}")

    # ── AR evaluation (for final test) / AR评估（用于最终测试）──
    @torch.no_grad()
    def _evaluate_ar(self, loader: DataLoader, desc: str = "TestAR") -> Dict[str, float]:
        """
        Evaluate using autoregressive inference (no teacher forcing).
        Computes comprehensive metrics matching v2F evaluation standards.
        使用自回归推理评估，计算与 v2F 评估标准一致的全面指标。
        """
        self.model.eval()
        # Accumulators
        all_preds, all_trues = [], []           # full sequence [B*T*D]
        load_preds, load_trues = [], []          # load channel only [B*T]
        disp_preds, disp_trues = [], []          # buckling disp channel [B*T]
        buck_load_preds, buck_load_trues = [], []  # buckling-point load
        buck_disp_preds, buck_disp_trues = [], []  # buckling-point disp

        disp_col = self.config.load_dim + self.config.buckling_disp_index

        for fixed_norm, abs_norm, _ in tqdm(loader, desc=desc, leave=False):
            fixed_norm = fixed_norm.to(self.device)
            abs_norm = abs_norm.to(self.device)
            B, T, _ = abs_norm.shape

            x0 = abs_norm[:, 0, :]
            steps = T - 1

            pred_ar = self.model(fixed_norm, x0, steps=steps)
            pred_phys = self._denorm_abs(pred_ar)          # [B, T-1, D]
            true_phys = self._denorm_abs(abs_norm[:, 1:, :])  # [B, T-1, D]

            # Full sequence
            all_preds.append(pred_phys.reshape(-1).cpu())
            all_trues.append(true_phys.reshape(-1).cpu())
            # Per-channel (load & buckling displacement)
            load_preds.append(pred_phys[..., 0].reshape(-1).cpu())
            load_trues.append(true_phys[..., 0].reshape(-1).cpu())
            disp_preds.append(pred_phys[..., disp_col].reshape(-1).cpu())
            disp_trues.append(true_phys[..., disp_col].reshape(-1).cpu())

            # Buckling-point extraction
            load_p, disp_p = extract_buckling_response_torch(
                pred_phys, self.config.load_dim, self.config.buckling_disp_index,
                slope_threshold=self.config.buckling_slope_threshold,
                min_index=self.config.buckling_min_index, eps=self.config.buckling_eps,
                smooth=self.config.buckling_smooth, smooth_window=self.config.buckling_smooth_window,
            )
            load_t, disp_t = extract_buckling_response_torch(
                true_phys, self.config.load_dim, self.config.buckling_disp_index,
                slope_threshold=self.config.buckling_slope_threshold,
                min_index=self.config.buckling_min_index, eps=self.config.buckling_eps,
                smooth=self.config.buckling_smooth, smooth_window=self.config.buckling_smooth_window,
            )
            buck_load_preds.append(load_p.cpu())
            buck_load_trues.append(load_t.cpu())
            buck_disp_preds.append(disp_p.cpu())
            buck_disp_trues.append(disp_t.cpu())

        # ── Helper: compute all metrics for a (pred, true) pair ──
        def _calc_metrics(p, t, eps=1e-8):
            """Return dict of MAE, RMSE, R², MAPE, NRMSE, PCC."""
            p, t = np.asarray(p), np.asarray(t)
            mae = float(np.abs(p - t).mean())
            mse = float(((p - t) ** 2).mean())
            rmse = float(np.sqrt(mse))
            ss_res = ((t - p) ** 2).sum()
            ss_tot = ((t - t.mean()) ** 2).sum()
            r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0
            # MAPE with epsilon protection
            mape = float((np.abs((t - p) / (np.abs(t) + eps)) * 100.0).mean())
            # NRMSE: normalized by range
            trange = t.max() - t.min()
            nrmse = float(rmse / trange) if trange > 1e-12 else 0.0
            # PCC
            p_mean, t_mean = p.mean(), t.mean()
            p_std, t_std = p.std(ddof=0), t.std(ddof=0)
            if p_std > 1e-12 and t_std > 1e-12:
                pcc = float(((p - p_mean) * (t - t_mean)).mean() / (p_std * t_std))
            else:
                pcc = 0.0
            return dict(MAE=mae, RMSE=rmse, R2=r2, MAPE=mape, NRMSE=nrmse, PCC=pcc)

        # Compute all metric groups
        seq_p = torch.cat(all_preds).numpy()
        seq_t = torch.cat(all_trues).numpy()
        load_p = torch.cat(load_preds).numpy()
        load_t = torch.cat(load_trues).numpy()
        disp_p = torch.cat(disp_preds).numpy()
        disp_t = torch.cat(disp_trues).numpy()
        bl_p = torch.cat(buck_load_preds).numpy()
        bl_t = torch.cat(buck_load_trues).numpy()
        bd_p = torch.cat(buck_disp_preds).numpy()
        bd_t = torch.cat(buck_disp_trues).numpy()

        seq = _calc_metrics(seq_p, seq_t)
        load = _calc_metrics(load_p, load_t)
        disp = _calc_metrics(disp_p, disp_t)
        bl = _calc_metrics(bl_p, bl_t)
        bd = _calc_metrics(bd_p, bd_t)

        return {
            # Sequence-level
            "seq_mae": seq["MAE"], "seq_rmse": seq["RMSE"], "seq_r2": seq["R2"],
            "seq_mape": seq["MAPE"], "seq_nrmse": seq["NRMSE"],
            # Load channel
            "load_mae": load["MAE"], "load_rmse": load["RMSE"], "load_r2": load["R2"],
            # Displacement channel
            "disp_mae": disp["MAE"], "disp_rmse": disp["RMSE"], "disp_r2": disp["R2"],
            # Buckling load (primary paper metric)
            "buckling_load_mae": bl["MAE"], "buckling_load_rmse": bl["RMSE"],
            "buckling_load_r2": bl["R2"], "buckling_load_mape": bl["MAPE"],
            "buckling_load_nrmse": bl["NRMSE"], "buckling_load_pcc": bl["PCC"],
            # Buckling displacement
            "buckling_disp_mae": bd["MAE"], "buckling_disp_rmse": bd["RMSE"],
            "buckling_disp_r2": bd["R2"],
        }

    # ── Main training loop / 主训练循环 ──
    def train(self):
        cfg = self.config
        logging.info(f"Start training (teacher forcing only)...")
        logging.info(f"Num params: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        logging.info(f"Model: {type(self.model).__name__}")

        best_val_loss = float('inf')
        bad_epochs = 0

        for epoch in range(1, cfg.epochs + 1):
            t0 = time.time()

            # Train
            train_metrics = self._train_epoch(epoch)
            train_loss = train_metrics["train_loss"]

            # Validate (TF)
            val_metrics = self._validate(self.val_loader, desc=f"Val {epoch}")
            val_loss = val_metrics["total_loss"]

            lr = self.optimizer.param_groups[0]["lr"]

            # Log TF validation
            logging.info(
                f"Epoch {epoch:3d}/{cfg.epochs} | "
                f"lr={lr:.3e} | "
                f"TrainLoss={train_loss:.6f} | "
                f"ValLoss(TF)={val_loss:.6f} | "
                f"RMSE={val_metrics['phys_rmse']:.3e} | "
                f"BuckLoadMAE={val_metrics['buckling_load_mae']:.3e} | "
                f"BuckDispMAE={val_metrics['buckling_disp_mae']:.3e}"
            )

            # Periodic AR validation (every 5 epochs)
            ar_metrics = None
            if epoch % 5 == 0:
                ar_metrics = self._evaluate_ar(self.val_loader, desc=f"ValAR {epoch}")
                logging.info(
                    f"  Val(AR) | BuckLoad MAE={ar_metrics['buckling_load_mae']:.3e} "
                    f"R2={ar_metrics['buckling_load_r2']:.4f} "
                    f"MAPE={ar_metrics['buckling_load_mape']:.2f}% | "
                    f"Seq R2={ar_metrics['seq_r2']:.4f}"
                )

            # Checkpoint (TF or AR depending on best_monitor)
            if self.best_monitor == "val_ar_total_loss" and ar_metrics is not None:
                monitor_loss = ar_metrics.get("total_loss", float('inf'))
                monitor_name = "val_ar_total_loss"
            else:
                monitor_loss = val_loss
                monitor_name = "val_loss"

            if monitor_loss < best_val_loss:
                best_val_loss = monitor_loss
                bad_epochs = 0
                self._save_checkpoint(epoch, is_best=True)
                logging.info(f"  New best {monitor_name} = {monitor_loss:.6f}")
            else:
                bad_epochs += 1

            # Early stopping
            if bad_epochs >= cfg.early_stopping_patience:
                logging.info(f"Early stopping at epoch {epoch}")
                break

            # Always save latest
            self._save_checkpoint(epoch, is_best=False)

            # Record metrics
            record = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": lr,
            }
            record.update({f"val_{k}": v for k, v in val_metrics.items()})
            if ar_metrics is not None:
                record.update({f"val_ar_{k}": v for k, v in ar_metrics.items()})
            self.metrics_records.append(record)

            # TensorBoard
            self.writer.add_scalar("Loss/Train", train_loss, epoch)
            self.writer.add_scalar("Loss/Val", val_loss, epoch)
            self.writer.add_scalar("Metrics/Val_BuckLoadMAE", val_metrics["buckling_load_mae"], epoch)
            if ar_metrics is not None:
                self.writer.add_scalar("Metrics/ValAR_BuckLoadMAE", ar_metrics["buckling_load_mae"], epoch)
                self.writer.add_scalar("Metrics/ValAR_BuckLoadR2", ar_metrics["buckling_load_r2"], epoch)

            t1 = time.time()
            if epoch % 10 == 0:
                logging.info(f"  Epoch {epoch} took {t1 - t0:.1f}s")

        # ── Final: load best model, run AR test ──
        best_path = Path(str(cfg.model_save_path).replace(".pth", "_best.pth"))
        if best_path.exists():
            ckpt = torch.load(best_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(ckpt["model_state_dict"])
            logging.info(f"Loaded best model for final AR test.")

        test_metrics_ar = self._evaluate_ar(self.test_loader, desc="TestAR")
        logging.info(
            f"Test(AR) | "
            f"SeqRMSE={test_metrics_ar['seq_rmse']:.3e} SeqR2={test_metrics_ar['seq_r2']:.6f} | "
            f"BuckLoad MAE={test_metrics_ar['buckling_load_mae']:.3e} "
            f"RMSE={test_metrics_ar['buckling_load_rmse']:.3e} "
            f"R2={test_metrics_ar['buckling_load_r2']:.6f} "
            f"MAPE={test_metrics_ar['buckling_load_mape']:.2f}% "
            f"NRMSE={test_metrics_ar['buckling_load_nrmse']:.6f} "
            f"PCC={test_metrics_ar['buckling_load_pcc']:.6f} | "
            f"BuckDisp MAE={test_metrics_ar['buckling_disp_mae']:.3e}"
        )

        # External test (AR)
        if self.ext_test_files:
            ext_dataset = BucklingDataset(self.ext_test_files, self.config, self.scalers, normalize=True)
            ext_loader = DataLoader(ext_dataset, batch_size=self.config.batch_size, shuffle=False, num_workers=0)
            ext_metrics = self._evaluate_ar(ext_loader, desc="ExtTestAR")
            logging.info(
                f"External Test(AR) | "
                f"BuckLoad MAE={ext_metrics['buckling_load_mae']:.4f} "
                f"RMSE={ext_metrics['buckling_load_rmse']:.4f} "
                f"R2={ext_metrics['buckling_load_r2']:.6f} "
                f"MAPE={ext_metrics['buckling_load_mape']:.2f}%"
            )

        # Save metrics CSV (include AR test results)
        import pandas as pd
        csv_path = Path(cfg.log_dir) / "training_metrics.csv"
        df = pd.DataFrame(self.metrics_records)
        # Append AR test metrics as last row
        test_row = {"epoch": cfg.epochs + 1}
        test_row.update({f"test_ar_{k}": v for k, v in test_metrics_ar.items()})
        df = pd.concat([df, pd.DataFrame([test_row])], ignore_index=True)
        df.to_csv(csv_path, index=False)
        logging.info(f"Metrics saved -> {csv_path}")

        self.writer.close()
        logging.info("Training completed.")

        return test_metrics_ar
