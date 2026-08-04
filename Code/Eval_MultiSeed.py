# -*- coding: utf-8 -*-
"""
Multi-Seed Model Evaluation Script / 多种子模型评估脚本
========================================================
Evaluates all trained seed models and computes mean ± std.
Uses the existing evaluate_model() from inference/evaluation.py.

评估所有已训练的种子模型并计算均值±标准差。
复用 inference/evaluation.py 中现有的 evaluate_model()。

Usage:
    python Code/Eval_MultiSeed.py

Assumes models from Train_MultiSeed.py exist at models/seed_{SEED}/
"""

import json
import logging
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from models import BucklingPredictor
from data.dataset import BucklingDataset
from inference.evaluation import evaluate_model


# =============================================================================
# Configuration / 配置
# =============================================================================

SEEDS = [618, 42, 123, 2024, 9999]  # Must match Train_MultiSeed.py
DATA_DIR = Path("./Data/Train")
EXTERNAL_TEST_DIR = Path("./Data/Test")
RESULT_DIR = Path("./results/multi_seed_evaluation")

# These buckling settings match v2F training
# 这些屈曲设置与v2F训练一致
BUCKLING_SETTINGS = dict(
    buckling_node=11,
    buckling_axis="y",
    buckling_disp_index=21,
    buckling_method="stiffness_drop",
    buckling_slope_threshold=0.0,
    buckling_min_index=5,
    buckling_eps=1e-9,
    buckling_smooth=True,
    buckling_smooth_window=7,
    load_dim=1,
    disp_dim=42,
    fixed_features_dim=16,
    dynamic_features_dim=43,
)


# =============================================================================
# Main / 主函数
# =============================================================================

def _load_predictor(seed: int, device: torch.device):
    """
    Load the trained BucklingPredictor (not ARWrapper) for a given seed.
    """
    model_dir = Path(f"./models/seed_{seed}")
    model_path = model_dir / "buckling_predictor_best.pth"
    scaler_path = model_dir / "scalers.pkl"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    ckpt = torch.load(str(model_path), map_location="cpu", weights_only=False)
    saved_config = ckpt.get("config")
    scalers = ckpt.get("scalers")
    if scalers is None and scaler_path.exists():
        scalers = joblib.load(str(scaler_path))

    # Extract model hyperparameters from saved config
    def _get(cfg, key, default):
        if cfg is None:
            return default
        if hasattr(cfg, '__dict__'):
            return getattr(cfg, key, default)
        return cfg.get(key, default)

    cfg = saved_config
    predictor = BucklingPredictor(
        fixed_features_dim=16,
        dynamic_features_dim=43,
        hidden_size=_get(cfg, "hidden_size", 512),
        num_layers=_get(cfg, "num_layers", 4),
        num_heads=_get(cfg, "num_heads", 4),
        dropout=_get(cfg, "dropout", 0.18550794013645205),
        use_feature_mhsa=_get(cfg, "use_feature_mhsa", True),
        use_attn_pool=_get(cfg, "use_attn_pool", True),
        use_lstm_prior=_get(cfg, "use_lstm_prior", True),
        use_multihead_outputs=_get(cfg, "use_multihead_outputs", True),
        use_film=_get(cfg, "use_film", True),
        film_hidden=_get(cfg, "film_hidden", 128),
        use_state_residual=_get(cfg, "use_state_residual", True),
        state_residual_hidden=_get(cfg, "state_residual_hidden", 128),
    )

    # Load state dict - handle both wrapped and unwrapped checkpoints
    state = ckpt["model_state_dict"]
    # Strip ARWrapper prefix if present
    state = {k.replace("core.", ""): v for k, v in state.items()}
    predictor.load_state_dict(state, strict=False)
    predictor = predictor.to(device)
    predictor.eval()

    return predictor, scalers


def _make_config_for_eval(seed: int, scalers: Dict) -> Any:
    """
    Create a simple namespace config for evaluate_model() compatibility.
    """
    from types import SimpleNamespace
    return SimpleNamespace(
        seed=seed,
        load_dim=BUCKLING_SETTINGS["load_dim"],
        disp_dim=BUCKLING_SETTINGS["disp_dim"],
        buckling_disp_index=BUCKLING_SETTINGS["buckling_disp_index"],
        buckling_method=BUCKLING_SETTINGS["buckling_method"],
        buckling_slope_threshold=BUCKLING_SETTINGS["buckling_slope_threshold"],
        buckling_min_index=BUCKLING_SETTINGS["buckling_min_index"],
        buckling_eps=BUCKLING_SETTINGS["buckling_eps"],
        buckling_smooth=BUCKLING_SETTINGS["buckling_smooth"],
        buckling_smooth_window=BUCKLING_SETTINGS["buckling_smooth_window"],
        batch_size=16,
    )


def _print_summary_table(
    results: List[Dict],
    title: str,
    metrics_list: List[str],
) -> None:
    """Pretty-print results with mean ± std."""
    print(f"\n{'─' * 110}")
    print(f"  {title}")
    print(f"{'─' * 110}")

    # Header
    header = f"{'Seed':>6} | " + " | ".join(f"{m:>22}" for m in metrics_list)
    print(header)
    print("─" * len(header))

    for r in results:
        m = r["metrics"]
        vals = []
        for k in metrics_list:
            v = m.get(k)
            vals.append(f"{v:22.6f}" if v is not None else f"{'N/A':>22}")
        print(f"{r['seed']:>6} | " + " | ".join(vals))

    print("─" * len(header))

    mean_vals, std_vals = [], []
    for k in metrics_list:
        arr = np.array([r["metrics"].get(k, float('nan')) for r in results])
        arr = arr[~np.isnan(arr)]
        mean_vals.append(f"{arr.mean():22.6f}" if len(arr) > 0 else f"{'N/A':>22}")
        std_vals.append(
            f"{arr.std(ddof=1):22.6f}" if len(arr) > 1 else f"{'—':>22}"
        )
    print(f"{'Mean':>6} | " + " | ".join(mean_vals))
    print(f"{'Std':>6} | " + " | ".join(std_vals))
    print("─" * 110)


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = RESULT_DIR / f"eval_multi_seed_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(str(log_file), encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )

    logging.info("=" * 60)
    logging.info("Multi-Seed Model Evaluation / 多种子模型评估")
    logging.info(f"Log: {log_file}")
    logging.info("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device}")

    internal_results: List[Dict] = []
    external_results: List[Dict] = []

    for seed in SEEDS:
        model_dir = Path(f"./models/seed_{seed}")
        if not (model_dir / "buckling_predictor_best.pth").exists():
            logging.warning(f"Model for seed {seed} not found, skipping.")
            continue

        logging.info(f"\n{'#' * 50}")
        logging.info(f"Seed {seed}")
        logging.info(f"{'#' * 50}")

        predictor, scalers = _load_predictor(seed, device)
        eval_cfg = _make_config_for_eval(seed, scalers)

        # ── 1. Internal test split (15%, seed-specific) ──
        logging.info("Evaluating on internal 15% test split...")
        all_files = BucklingDataset.discover_files(str(DATA_DIR))
        rng = np.random.default_rng(seed)
        indices = np.arange(len(all_files))
        rng.shuffle(indices)
        n_train = int(0.7 * len(indices))
        n_val = int(0.15 * len(indices))
        test_files = [all_files[i] for i in indices[n_train + n_val:]]

        test_dataset = BucklingDataset(test_files, BUCKLING_SETTINGS, scalers, normalize=True)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)
        logging.info(f"  Internal test samples: {len(test_dataset)}")

        metrics_int = evaluate_model(predictor, test_loader, scalers, eval_cfg, device)
        internal_results.append({"seed": seed, "metrics": metrics_int})
        logging.info(f"  BucklingLoad MAE={metrics_int.get('BucklingLoad_MAE', 'N/A'):.4f}, "
                     f"R²={metrics_int.get('BucklingLoad_R2', 'N/A'):.6f}")

        # ── 2. External test set (Data/Test/, same 61 files for all) ──
        ext_files = sorted(EXTERNAL_TEST_DIR.glob("*.csv"))
        if ext_files:
            logging.info("Evaluating on external Data/Test/...")
            ext_dataset = BucklingDataset(
                [str(f) for f in ext_files], BUCKLING_SETTINGS, scalers, normalize=True
            )
            ext_loader = DataLoader(ext_dataset, batch_size=16, shuffle=False, num_workers=0)
            logging.info(f"  External test samples: {len(ext_dataset)}")

            metrics_ext = evaluate_model(predictor, ext_loader, scalers, eval_cfg, device)
            external_results.append({"seed": seed, "metrics": metrics_ext})
            logging.info(f"  BucklingLoad MAE={metrics_ext.get('BucklingLoad_MAE', 'N/A'):.4f}, "
                         f"R²={metrics_ext.get('BucklingLoad_R2', 'N/A'):.6f}")
        else:
            logging.warning(f"No files in {EXTERNAL_TEST_DIR}")

    # ── Summaries ──
    KEY_BUCKLING = [
        "BucklingLoad_MAE", "BucklingLoad_RMSE", "BucklingLoad_R2",
        "BucklingLoad_NRMSE", "BucklingLoad_MAPE", "BucklingLoad_PCC",
        "BucklingDisp_MAE",
    ]

    if internal_results:
        _print_summary_table(internal_results, "Internal Test Split (15%, seed-specific)", KEY_BUCKLING)

    if external_results:
        _print_summary_table(external_results, "External Test Set (Data/Test/, 61 files)", KEY_BUCKLING)

    # ── Save CSV ──
    for name, results in [("internal_test", internal_results), ("external_test", external_results)]:
        if not results:
            continue
        rows = []
        for r in results:
            row = {"seed": r["seed"]}
            row.update(r["metrics"])
            rows.append(row)

        # Mean/std rows
        mean_row = {"seed": "Mean"}
        std_row = {"seed": "Std"}
        all_keys = set()
        for r in results:
            all_keys.update(r["metrics"].keys())
        for k in sorted(all_keys):
            vals = np.array([r["metrics"].get(k, float('nan')) for r in results])
            vals = vals[~np.isnan(vals)]
            mean_row[k] = float(vals.mean()) if len(vals) > 0 else ""
            std_row[k] = float(vals.std(ddof=1)) if len(vals) > 1 else ""
        rows.append(mean_row)
        rows.append(std_row)

        csv_path = RESULT_DIR / f"summary_{name}.csv"
        pd.DataFrame(rows).to_csv(str(csv_path), index=False)
        logging.info(f"Saved: {csv_path}")

    logging.info("\n" + "=" * 60)
    logging.info("Evaluation complete! / 评估完成！")
    logging.info(f"Results in: {RESULT_DIR}")
    logging.info("=" * 60)


if __name__ == "__main__":
    main()
