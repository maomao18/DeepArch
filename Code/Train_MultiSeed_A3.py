# -*- coding: utf-8 -*-
"""
Multi-Seed Training Script — FA-LSTM A3: w/o FiLM
==================================================
Ablation: use_film=False
Output: models/FA-LSTM_A3/seed_{SEED}/  and  logs/FA-LSTM_A3/seed_{SEED}/

Usage:  python Code/Train_MultiSeed_A3.py
"""

import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config.train import TrainingConfig
from training.trainer import BucklingTrainer

# =============================================================================
# Paper ablation A3: w/o FiLM conditioning
# 论文消融 A3: 去掉 FiLM，固定特征不调 LSTM 输入
# =============================================================================
MODEL_VERSION = "FA-LSTM_A3"
SEEDS: List[int] = [618, 42, 123, 2024, 9999]

V2F_CONFIG = dict(
    hidden_size=512, num_layers=4, num_heads=4,
    dropout=0.18550794013645205, film_hidden=128, state_residual_hidden=128,
    learning_rate=0.0003047769777136961,
    weight_decay=3.0038179351735502e-05,
    grad_clip_norm=3.2194295706771046,
    batch_size=16, epochs=100,
    teacher_forcing_epochs=100, transition_epochs=0,
    w_delta=0.43537859299214776, w_abs=0.5646214070078522,
    load_weight=0.20875198842657713, displacement_weight=0.7912480115734228,
    # ── A3: w/o FiLM ──
    use_feature_mhsa=True, use_attn_pool=True, use_lstm_prior=True,
    use_multihead_outputs=True, use_film=False, use_state_residual=True,
    use_stiffness_loss=False, use_monotonic_load_loss=False,
    buckling_method="stiffness_drop", buckling_node=11, buckling_axis="y",
    buckling_smooth=True, buckling_smooth_window=7,
    use_amp=True, use_huber_for_delta=True, huber_delta=1.0,
    early_stopping_patience=50,
)


class _ConfigEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Path): return str(obj)
        if isinstance(obj, (np.integer, np.floating)): return float(obj)
        return super().default(obj)


def _build_config(seed: int, data_dir: str, ext_test_dir: Optional[str]) -> TrainingConfig:
    ver = MODEL_VERSION
    return TrainingConfig(
        data_dir=data_dir,
        model_save_path=f"./models/{ver}/seed_{seed}/buckling_predictor.pth",
        scaler_save_path=f"./models/{ver}/seed_{seed}/scalers.pkl",
        log_dir=f"./logs/{ver}/seed_{seed}",
        metrics_csv_path=f"./logs/{ver}/seed_{seed}/training_metrics.csv",
        external_test_dir=ext_test_dir,
        external_test_out_dir=f"./logs/{ver}/seed_{seed}/test_predictions",
        external_test_plot=True, external_test_plot_node=11, external_test_plot_axis="y",
        seed=seed,
        **V2F_CONFIG,
    )


def _extract_test_metrics(csv_path: str) -> Dict[str, Optional[float]]:
    if not Path(csv_path).exists():
        return {}
    df = pd.read_csv(csv_path)
    test_rows = df[df["phase"] == "test"]
    if test_rows.empty:
        return {}
    last = test_rows.iloc[-1]

    RENAME = {
        "phys_mse": "seq_mse", "phys_rmse": "seq_rmse",
        "phys_mae": "seq_mae", "phys_r2": "seq_r2",
        "phys_load_mae": "load_mae", "phys_load_rmse": "load_rmse",
        "phys_disp_mae": "disp_mae", "phys_disp_rmse": "disp_rmse",
    }

    metrics = {}
    for col in df.columns:
        if col in ("epoch", "phase", "lr", "tf_ratio"):
            continue
        val = last.get(col)
        if pd.notna(val):
            name = RENAME.get(col, col)
            metrics[name] = float(val)
    return metrics


def _print_summary(results: List[Dict], key_metrics: List[str]) -> None:
    print("\n" + "=" * 110)
    print(f"  {MODEL_VERSION} MULTI-SEED SUMMARY")
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
    data_dir = r"./Data/Train"
    ext_test_dir = r"./Data/Test"
    summary_csv = f"./logs/{MODEL_VERSION}/multi_seed_summary.csv"

    results: List[Dict] = []

    for i, seed in enumerate(SEEDS):
        print(f"\n{'#' * 80}")
        print(f"# {MODEL_VERSION}  Seed {seed}  ({i + 1}/{len(SEEDS)})")
        print(f"{'#' * 80}")

        config = _build_config(seed, data_dir, ext_test_dir)
        config_dir = Path(config.log_dir)
        config_dir.mkdir(parents=True, exist_ok=True)
        with open(config_dir / "config_used.json", "w", encoding="utf-8") as f:
            json.dump(asdict(config), f, cls=_ConfigEncoder, indent=2)

        trainer = BucklingTrainer(config)
        trainer.train()

        metrics = _extract_test_metrics(config.metrics_csv_path)
        results.append({"seed": seed, "metrics": metrics})
        print(f"Seed {seed} done.")

    key_metrics = [
        "seq_rmse", "seq_mae", "seq_r2",
        "load_mae", "load_rmse",
        "disp_mae", "disp_rmse",
        "buckling_load_mae", "buckling_load_rmse", "buckling_load_r2",
        "buckling_load_mape", "buckling_load_nrmse", "buckling_load_pcc",
        "buckling_disp_mae", "buckling_disp_rmse", "buckling_disp_r2",
    ]
    _print_summary(results, key_metrics)
    rows = []
    for r in results:
        row = {"seed": r["seed"], "stat": "value", **r["metrics"]}
        rows.append(row)
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
