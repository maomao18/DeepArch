# -*- coding: utf-8 -*-
"""
TCN Baseline — Multi-Seed Training / TCN基线多种子训练
=======================================================
Temporal Convolutional Network (Bai et al. 2018) with causal dilated convolutions.
Teacher forcing only.  No FA, AttnPooling, FiLM, or delta prediction.

Output: models/TCN/seed_{SEED}/  and  logs/TCN/seed_{SEED}/

Usage:  python Code/Train_Baseline_TCN.py
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

from models.baselines import TCNBaseline, count_parameters
from training.baseline_trainer import BaselineTrainer

# ──────────────────────────────────────────────────────────────────
MODEL_VERSION = "TCN"
# SEEDS: List[int] = [618, 42, 123, 2024, 9999]
SEEDS: List[int] = [2024, 9999]

SHARED_CONFIG = dict(
    device="cuda",
    data_dir=r"./Data/Train",
    train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
    max_train_files=None, max_val_files=None, max_test_files=None,
    batch_size=16, num_workers=0, pin_memory=True,
    persistent_workers=False, prefetch_factor=4,
    learning_rate=0.0003047769777136961,
    weight_decay=3.0038179351735502e-05,
    grad_clip_norm=3.2194295706771046,
    epochs=150, early_stopping_patience=50,
    use_amp=True,
    fixed_features_dim=16, dynamic_features_dim=43,
    load_dim=1, meta_cols=1, sequence_length=200,
    buckling_method="stiffness_drop", buckling_node=11,
    buckling_axis="y", buckling_disp_index=21,
    buckling_slope_threshold=0.0, buckling_min_index=5,
    buckling_eps=1e-9, buckling_smooth=True, buckling_smooth_window=7,
)

TCN_MODEL_CONFIG = dict(
    hidden_size=256,
    num_blocks=8,
    kernel_size=3,
    dropout=0.1,
)


class _ConfigEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Path): return str(obj)
        if isinstance(obj, (np.integer, np.floating)): return float(obj)
        return super().default(obj)


def _build_config(seed: int):
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
    if not Path(csv_path).exists():
        return {}
    df = pd.read_csv(csv_path)
    if df.empty:
        return {}
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

        with open(Path(config.log_dir) / "config_used.json", "w", encoding="utf-8") as f:
            save_cfg = {**SHARED_CONFIG, **TCN_MODEL_CONFIG, "seed": seed, "model": "TCNBaseline"}
            json.dump(save_cfg, f, cls=_ConfigEncoder, indent=2)

        model = TCNBaseline(**TCN_MODEL_CONFIG)
        print(f"Model params: {count_parameters(model):,}")

        trainer = BaselineTrainer(config, model)
        trainer.train()

        metrics = _extract_test_metrics(config.metrics_csv_path)
        results.append({"seed": seed, "metrics": metrics})
        print(f"Seed {seed} done.")

    key_metrics = [
        "buckling_load_mae", "buckling_load_rmse", "buckling_load_r2",
        "buckling_load_mape", "buckling_load_nrmse", "buckling_load_pcc",
        "buckling_disp_mae", "buckling_disp_rmse", "buckling_disp_r2",
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
