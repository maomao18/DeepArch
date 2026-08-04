# -*- coding: utf-8 -*-
"""
GRU Baseline — Multi-Seed Training / LSTM基线多种子训练
=========================================================
Standard stacked GRU with fixed-feature-conditioned initial state.
Teacher forcing only.  No FA, AttnPooling, FiLM, or delta prediction.

Output: models/LSTM/seed_{SEED}/  and  logs/LSTM/seed_{SEED}/

Usage:  python Code/Train_Baseline_LSTM.py
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from models.baselines import GRUBaseline, count_parameters
from training.baseline_trainer import BaselineTrainer

# ──────────────────────────────────────────────────────────────────
# Model version & seeds
# ──────────────────────────────────────────────────────────────────
MODEL_VERSION = "GRU"
# SEEDS: List[int] = [618, 42, 123, 2024, 9999]
SEEDS: List[int] = [618, 42, 123, 2024, 9999]

# ──────────────────────────────────────────────────────────────────
# Shared training config (same as FA-LSTM for fair comparison)
# ──────────────────────────────────────────────────────────────────
SHARED_CONFIG = dict(
    device="cuda",
    # Data
    data_dir=r"./Data/Train",
    train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
    max_train_files=None, max_val_files=None, max_test_files=None,
    # DataLoader
    batch_size=16, num_workers=0, pin_memory=True,
    persistent_workers=False, prefetch_factor=4,
    # Training
    learning_rate=0.0003047769777136961,
    weight_decay=3.0038179351735502e-05,
    grad_clip_norm=3.2194295706771046,
    epochs=150, early_stopping_patience=50,
    use_amp=True,
    # Data dimensions
    fixed_features_dim=16, dynamic_features_dim=43,
    load_dim=1, meta_cols=1, sequence_length=200,
    # Buckling extraction (same as FA-LSTM)
    buckling_method="stiffness_drop", buckling_node=11,
    buckling_axis="y", buckling_disp_index=21,
    buckling_slope_threshold=0.0, buckling_min_index=5,
    buckling_eps=1e-9, buckling_smooth=True, buckling_smooth_window=7,
)

# ──────────────────────────────────────────────────────────────────
# LSTM-specific model config
# ──────────────────────────────────────────────────────────────────
GRU_MODEL_CONFIG = dict(
    hidden_size=512,
    num_layers=4,
    dropout=0.1,
)


class _ConfigEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Path): return str(obj)
        if isinstance(obj, (np.integer, np.floating)): return float(obj)
        return super().default(obj)


def _build_config(seed: int) -> type:
    """Build a simple namespace config for BaselineTrainer."""
    from types import SimpleNamespace
    ver = MODEL_VERSION
    cfg = SHARED_CONFIG.copy()
    cfg.update(
        seed=seed,
        model_save_path=f"./models/{ver}/seed_{seed}/buckling_predictor.pth",
        scaler_save_path=f"./models/{ver}/seed_{seed}/scalers.pkl",
        log_dir=f"./logs/{ver}/seed_{seed}",
        metrics_csv_path=f"./logs/{ver}/seed_{seed}/training_metrics.csv",
        external_test_dir=r"./Data/Test",
        external_test_out_dir=f"./logs/{ver}/seed_{seed}/test_predictions",
        external_test_plot=False,
    )
    return SimpleNamespace(**cfg)


def _extract_test_metrics(csv_path: str) -> Dict[str, Optional[float]]:
    """Extract AR test metrics from the last row of training_metrics.csv."""
    if not Path(csv_path).exists():
        return {}
    df = pd.read_csv(csv_path)
    if df.empty:
        return {}
    # AR test metrics are in columns starting with 'test_ar_'
    ar_cols = [c for c in df.columns if c.startswith("test_ar_")]
    if not ar_cols:
        return {}
    last = df.iloc[-1]
    return {c.replace("test_ar_", ""): float(last[c]) for c in ar_cols if pd.notna(last.get(c))}


def _print_summary(results: List[Dict], key_metrics: List[str]) -> None:
    print("\n" + "=" * 110)
    print(f"  {MODEL_VERSION} BASELINE — MULTI-SEED SUMMARY")
    print("=" * 110)
    header = f"{'Seed':>6} | " + " | ".join(f"{m:>18}" for m in key_metrics)
    print(header)
    print("-" * len(header))
    for r in results:
        vals = []
        for m in key_metrics:
            v = r["metrics"].get(m)
            vals.append(f"{v:18.6f}" if v is not None else f"{'N/A':>18}")
        print(f"{r['seed']:>6} | " + " | ".join(vals))
    print("-" * len(header))
    mean_vals, std_vals = [], []
    for m in key_metrics:
        arr = np.array([r["metrics"].get(m, np.nan) for r in results])
        arr = arr[~np.isnan(arr)]
        mean_vals.append(f"{arr.mean():18.6f}" if len(arr) > 0 else f"{'N/A':>18}")
        std_vals.append(f"{arr.std(ddof=1):18.6f}" if len(arr) > 1 else f"{'--':>18}")
    print(f"{'Mean':>6} | " + " | ".join(mean_vals))
    print(f"{'Std':>6} | " + " | ".join(std_vals))
    print("=" * 110)


def main() -> None:
    summary_csv = f"./logs/{MODEL_VERSION}/multi_seed_summary.csv"
    results: List[Dict] = []

    for i, seed in enumerate(SEEDS):
        print(f"\n{'#' * 80}")
        print(f"# {MODEL_VERSION} Baseline  Seed {seed}  ({i + 1}/{len(SEEDS)})")
        print(f"{'#' * 80}")

        config = _build_config(seed)
        Path(config.log_dir).mkdir(parents=True, exist_ok=True)

        # Save config snapshot
        with open(Path(config.log_dir) / "config_used.json", "w", encoding="utf-8") as f:
            save_cfg = {**SHARED_CONFIG, **GRU_MODEL_CONFIG, "seed": seed, "model": "GRUBaseline"}
            json.dump(save_cfg, f, cls=_ConfigEncoder, indent=2)

        model = GRUBaseline(**GRU_MODEL_CONFIG)
        print(f"Model params: {count_parameters(model):,}")

        trainer = BaselineTrainer(config, model)
        trainer.train()

        metrics = _extract_test_metrics(config.metrics_csv_path)
        results.append({"seed": seed, "metrics": metrics})
        print(f"Seed {seed} done.")

    key_metrics = [
        # Buckling load — primary
        "buckling_load_mae", "buckling_load_rmse", "buckling_load_r2",
        "buckling_load_mape", "buckling_load_nrmse", "buckling_load_pcc",
        # Buckling displacement
        "buckling_disp_mae", "buckling_disp_rmse", "buckling_disp_r2",
        # Sequence-level
        "seq_mae", "seq_rmse", "seq_r2", "seq_mape", "seq_nrmse",
    ]
    _print_summary(results, key_metrics)

    rows = []
    for r in results:
        rows.append({"seed": r["seed"], "stat": "value", **r["metrics"]})
    mean_row, std_row = {"seed": "", "stat": "mean"}, {"seed": "", "stat": "std"}
    all_keys = set()
    for r in results:
        all_keys.update(r["metrics"].keys())
    for k in sorted(all_keys):
        vals = np.array([r["metrics"].get(k, np.nan) for r in results])
        vals = vals[~np.isnan(vals)]
        mean_row[k] = float(vals.mean()) if len(vals) > 0 else ""
        std_row[k] = float(vals.std(ddof=1)) if len(vals) > 1 else ""
    rows.extend([mean_row, std_row])
    Path(summary_csv).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(summary_csv, index=False)
    print(f"\nSummary: {summary_csv}")


if __name__ == "__main__":
    main()
