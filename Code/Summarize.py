# -*- coding: utf-8 -*-
"""
Cross-Model Summary Aggregator / 跨模型汇总聚合器
===================================================
Reads multi_seed_summary.csv from ALL trained models and produces a single
paper-ready comparison table with mean ± std.

读取所有已训练模型的 multi_seed_summary.csv，生成一张论文就绪的对比表。

Usage:  python Code/Summarize.py

Output:
  results/summary/comparison_table.csv       ← 论文对比表
  results/summary/comparison_table_buckling.csv  ← 屈曲荷载精简表
"""

import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────────
# Configuration / 配置
# ──────────────────────────────────────────────────────────────────
LOGS_DIR = Path("./logs")
OUTPUT_DIR = Path("./results/summary")

# Models to collect — add ablation variants here as they finish training
# 收集的模型列表 — 消融变体训练完成后添加到此
MODELS = [
    # Proposed model & baselines
    "FA-LSTM",
    "LSTM", "TCN", "Transformer",
    # Paper ablations (uncomment when done) / 论文消融:
    # "FA-LSTM_A1",  # w/o FeatureAttention
    # "FA-LSTM_A2",  # w/o AttnPooling
    # "FA-LSTM_A3",  # w/o FiLM
    # "FA-LSTM_A4",  # w/o Prior MLP
    # "FA-LSTM_A5",  # w/o State Residual
    # Additional ablations / 补充消融:
    # "FA-LSTM_A6",  # w/o Dual-head
    # "FA-LSTM_A7",  # w/o Increment loss
    # "FA-LSTM_A8",  # w/o Absolute-state loss
    # "FA-LSTM_A9",  # w/o Scheduled Sampling
]

# Paper-ready metric groups / 论文就绪的指标组
METRIC_GROUPS = {
    "Buckling Load — Primary": [
        "buckling_load_mae", "buckling_load_rmse", "buckling_load_r2",
        "buckling_load_mape", "buckling_load_nrmse", "buckling_load_pcc",
    ],
    "Buckling Displacement": [
        "buckling_disp_mae", "buckling_disp_rmse", "buckling_disp_r2",
    ],
    "Full Sequence": [
        "seq_mae", "seq_rmse", "seq_r2", "seq_mape", "seq_nrmse",
    ],
}

# ──────────────────────────────────────────────────────────────────
# Helpers / 辅助
# ──────────────────────────────────────────────────────────────────

def _read_mean_std(csv_path: Path) -> Optional[Dict[str, str]]:
    """Read the mean and std rows from a summary CSV."""
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    if "stat" not in df.columns:
        return None
    mean_row = df[df["stat"] == "mean"]
    std_row = df[df["stat"] == "std"]
    if mean_row.empty:
        return None

    result = {}
    for col in df.columns:
        if col in ("seed", "stat"):
            continue
        m = mean_row[col].values[0]
        s = std_row[col].values[0] if not std_row.empty else None
        if pd.notna(m):
            if pd.notna(s) and s != "":
                result[col] = f"{float(m):.4f} ± {float(s):.4f}"
            else:
                result[col] = f"{float(m):.4f}"
    return result


def _format_value(val_str: str, metric: str) -> str:
    """Format a mean±std value based on metric type."""
    if " ± " not in val_str:
        return val_str
    parts = val_str.split(" ± ")
    v, s = float(parts[0]), float(parts[1])

    # Choose decimal places based on metric magnitude
    if "r2" in metric or "pcc" in metric:
        return f"{v:.4f} ± {s:.4f}"
    elif "mape" in metric:
        return f"{v:.2f} ± {s:.2f}"
    elif "nrmse" in metric:
        return f"{v:.6f} ± {s:.6f}"
    elif v >= 10:
        return f"{v:.2f} ± {s:.2f}"
    elif v >= 1:
        return f"{v:.4f} ± {s:.4f}"
    else:
        return f"{v:.6f} ± {s:.6f}"


# ──────────────────────────────────────────────────────────────────
# Main / 主函数
# ──────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Collect data / 收集数据
    all_data: Dict[str, Dict[str, str]] = {}

    for model in MODELS:
        csv_path = LOGS_DIR / model / "multi_seed_summary.csv"
        metrics = _read_mean_std(csv_path)
        if metrics is None:
            print(f"  [SKIP] {model} — no summary CSV found")
            continue
        all_data[model] = metrics
        print(f"  [OK]   {model} — {len(metrics)} metrics")

    if not all_data:
        print("No data found. Train models first.")
        return

    print(f"\nCollected {len(all_data)} models.\n")

    # Flatten all metric keys / 获取所有指标键
    all_keys = set()
    for mets in all_data.values():
        all_keys.update(mets.keys())

    # ── Table 1: Buckling Load (primary paper table) ──
    primary_metrics = [
        "buckling_load_mae", "buckling_load_rmse", "buckling_load_r2",
        "buckling_load_mape", "buckling_load_nrmse", "buckling_load_pcc",
    ]
    _print_and_save("Buckling Load Prediction", primary_metrics, all_data, OUTPUT_DIR / "comparison_buckling_load.csv")

    # ── Table 2: Full comparison ──
    all_print_metrics = [
        "buckling_load_mae", "buckling_load_rmse", "buckling_load_r2",
        "buckling_load_mape", "buckling_load_nrmse", "buckling_load_pcc",
        "buckling_disp_mae", "buckling_disp_rmse", "buckling_disp_r2",
        "seq_mae", "seq_rmse", "seq_r2", "seq_mape", "seq_nrmse",
    ]
    _print_and_save("Full Comparison", all_print_metrics, all_data, OUTPUT_DIR / "comparison_full.csv")

    print(f"\nAll tables saved to: {OUTPUT_DIR}")


def _print_and_save(title, metric_list, all_data, csv_path):
    """Print a formatted table and save to CSV."""
    # Filter to metrics that exist in at least one model
    available = [m for m in metric_list if any(m in d for d in all_data.values())]
    if not available:
        return

    print(f"\n{'=' * 120}")
    print(f"  {title}")
    print(f"{'=' * 120}")
    print()

    # Header
    col_width = max(len(m) for m in available) + 1
    header = f"{'Model':<16}" + "".join(f"{m:>{col_width}}" for m in available)
    print(header)
    print("-" * len(header))

    # Data rows
    rows_for_csv = []
    for model, metrics in all_data.items():
        row_csv = {"Model": model}
        vals = []
        for m in available:
            v = metrics.get(m, "—")
            vals.append(f"{v:>{col_width}}")
            row_csv[m] = v
        print(f"{model:<16}" + "".join(vals))
        rows_for_csv.append(row_csv)

    print()

    # Save CSV
    pd.DataFrame(rows_for_csv).to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")


if __name__ == "__main__":
    main()
