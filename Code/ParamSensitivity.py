# -*- coding: utf-8 -*-
"""
Parameter Sensitivity Analysis — Physically Consistent / 参数敏感性分析
=======================================================================
Sweeps INDEPENDENT physical parameters while fixing the cross-section (b, h)
and letting derived quantities (S, λ) change naturally — consistent with
classical arch buckling literature (Pi & Bradford, etc.).

Key design principles (revised):
  1. Baseline = dataset MEDIANS of independent physical parameters
  2. Homogeneous section (B11 = 0) — avoids FGM complexity
  3. Cross-section (b, h) is FIXED when sweeping geometry (L, f/L)
     → λ = S / i_x  varies as a DERIVED consequence, not an independent control
  4. f/L sampled as 1/x with x uniform in [3, 13], matching V1 data distribution
  5. 100 sweep points for smooth curves
  6. No smoothing — raw max of load curve

Data source: V1_FGM (2550 training samples).
Dataset statistics from: results/analysis/distribution_statistics.csv

Usage:  python Code/ParamSensitivity.py
Output: results/sensitivity/
"""

import os, sys, math, warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
import torch

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from models.predictor import BucklingPredictor

# =============================================================================
# Configuration
# =============================================================================
MODEL_PATH = "./models/v2F/buckling_predictor_best.pth"
SCALER_PATH = "./models/v2F/scalers.pkl"
OUTPUT_DIR = "./results/sensitivity_v2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_POINTS = 100
DEVICE = torch.device("cpu")

# =============================================================================
# Arc-length of a parabola / 抛物线弧长
# =============================================================================
def arc_length(L: float, f: float) -> float:
    """Arc length of parabolic arch y = (4f/L²)(L²/4 - x²)."""
    a = 4.0 * f / (L * L)
    term = math.sqrt(1.0 + (a * L) ** 2)
    return (L / 2.0) * term + (1.0 / (2.0 * a)) * math.log(a * L + term)


# =============================================================================
# Baseline — independent physical parameters calibrated to dataset medians
# =============================================================================
# True independent parameters for a homogeneous parabolic arch:
#   Geometry:  L, f/L, h, b          (span, rise ratio, section height, width)
#   Material:  E, ρ, μ
#   Boundary:  KX, KY, KZ (left & right, symmetric baseline)
#
# All other quantities are DERIVED:
#   f = L · (f/L)
#   S = arc_length(L, f)
#   i_x = √(D11/A11) = h/√12    (homogeneous rectangular)
#   λ  = S / i_x = S·√12 / h    (physical slenderness)
#   I0 = ρ·b·h,  A11 = E·b·h,  B11 = 0,  D11 = E·b·h³/12
#
# Dataset medians (2550 V1_FGM samples, for reference):
#   L=1.75  f=0.213  b=0.0615  h=0.00614  S=1.849  λ=1078
#   A11=4.48e7  D11=136.1  I0=1.76
# =============================================================================
class Baseline:
    """Independent physical parameters → 16 standardised training features."""

    def __init__(self):
        # --- Geometry (independent) ---
        self.L      = 1.75       # Span (m)              — data median 1.75
        self.f_L    = 0.122      # Rise-to-span ratio f/L — → f≈0.214 (median 0.213)
        self.h      = 0.00614    # Section height (m)     — data median 0.00614
        self.b      = 0.0615     # Section width (m)      — data median 0.0615

        # --- Material (independent) ---
        self.E      = 119e9      # Young's modulus (Pa)   — → A11≈4.49e7 (median 4.48e7)
        self.rho    = 4660.0     # Density (kg/m³)        — → I0≈1.76  (median 1.76)
        self.mu     = 0.3        # Poisson's ratio        — midpoint of [0.2, 0.4]

        # --- Boundary springs (independent, symmetric) ---
        self.KX     = 7.0        # Translational X  (median ≈ 6.85)
        self.KY     = 7.0        # Translational Y  (median ≈ 7.08)
        self.KZ     = 465.0      # Rotational       (median ≈ 464.7)

    def compute_features(self) -> np.ndarray:
        """
        Compute 16 training features from independent physical parameters.
        All derived quantities (f, S, λ, I0, A11, B11, D11) are computed
        consistently from the current state of independent parameters.

        Returns
        -------
        np.ndarray  shape (16,)
            [I0, A11, B11, D11, L, f, b, h, S, λ,
             KXL, KYL, KZL, KXR, KYR, KZR]
        """
        L = self.L
        f = L * self.f_L
        S = arc_length(L, f)
        h = self.h
        b = self.b

        # Homogeneous sectional properties
        I0  = self.rho * b * h
        A11 = self.E * b * h
        B11 = 0.0                     # homogeneous → no coupling
        D11 = self.E * b * h**3 / 12.0

        # Physical slenderness λ = S / i_x   (derived, not independent)
        # For homogeneous rectangular: i_x = h / √12
        SQRT12 = math.sqrt(12.0)
        lam = S * SQRT12 / h

        return np.array([
            I0, A11, B11, D11,
            L, f, b, h, S, lam,
            self.KX, self.KY, self.KZ,    # left  supports
            self.KX, self.KY, self.KZ,    # right supports (symmetric)
        ], dtype=np.float64)


# =============================================================================
# Model loading
# =============================================================================
def load_model(device: torch.device) -> Tuple[BucklingPredictor, Dict, torch.Tensor, torch.Tensor]:
    ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    saved_config = ckpt.get("config")

    def _get(cfg, key, default):
        if cfg is None: return default
        if hasattr(cfg, "__dict__"): return getattr(cfg, key, default)
        return cfg.get(key, default)

    model_cfg = SimpleNamespace(
        hidden_size=_get(saved_config, "hidden_size", 512),
        num_layers=_get(saved_config, "num_layers", 4),
        num_heads=_get(saved_config, "num_heads", 4),
        dropout=_get(saved_config, "dropout", 0.1855),
        fixed_features_dim=16, dynamic_features_dim=43,
        film_hidden=_get(saved_config, "film_hidden", 128),
        state_residual_hidden=_get(saved_config, "state_residual_hidden", 128),
        disp_dim=42, load_dim=1,
        use_feature_mhsa=_get(saved_config, "use_feature_mhsa", True),
        use_attn_pool=_get(saved_config, "use_attn_pool", True),
        use_lstm_prior=_get(saved_config, "use_lstm_prior", True),
        use_multihead_outputs=_get(saved_config, "use_multihead_outputs", True),
        use_film=_get(saved_config, "use_film", True),
        use_state_residual=_get(saved_config, "use_state_residual", True),
    )
    model = BucklingPredictor(model_cfg)

    state = ckpt["model_state_dict"]
    state = {k.replace("core.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model = model.to(device).eval()

    s = ckpt.get("scalers")
    if s is None:
        s = joblib.load(SCALER_PATH)

    # A, B for delta→abs integration
    eps = 1e-8
    abs_scale = np.where(s['abs'].scale_ < eps, 1.0, s['abs'].scale_)
    delta_scale = np.where(s['delta'].scale_ < eps, 1.0, s['delta'].scale_)
    A = torch.tensor(delta_scale / abs_scale, dtype=torch.float32, device=device)
    B_coef = torch.tensor(s['delta'].mean_ / abs_scale, dtype=torch.float32, device=device)

    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} params")
    return model, s, A, B_coef


# =============================================================================
# Inference
# =============================================================================
@torch.no_grad()
def infer(model, fixed_t, x0, A, B_coef, steps=199):
    """AR inference → standardized abs sequence [T, 43]."""
    model.eval()
    h_state = None
    x = x0
    preds = [x.cpu().numpy()]
    for _ in range(steps):
        se = model._step_embed(fixed_t, x).unsqueeze(1)
        lstm_out, h_state = model.lstm(se, h_state if h_state is not None else
                                        model._init_lstm_state_from_fixed(fixed_t))
        delta = model._decode_delta(lstm_out).squeeze(1)
        if model.state_residual is not None:
            delta = delta + model.state_residual(x)
        x = x + A * delta + B_coef
        preds.append(x.cpu().numpy())
    return np.concatenate(preds, axis=0)


def extract_q_cr(pred_seq: np.ndarray) -> float:
    """Critical buckling load — raw max of dimensionless load (no smoothing)."""
    return float(np.max(pred_seq[:, 0]))


# =============================================================================
# Parameter sweep definitions
# =============================================================================
# Each sweep varies ONE independent physical parameter.
# Cross-section (b, h) is FIXED when sweeping geometry (L, f/L), so λ varies
# as a natural consequence — matching Pi & Bradford's methodology.
#
# Sweep order follows the physical parameter hierarchy:
#   Geometry → Section → Material → Boundary
# =============================================================================

# Pre-compute f/L values matching V1 sampling: f_L = 1/x, x ∈ [3, 13]
_fL_x = np.linspace(3.0, 13.0, N_POINTS)
_fL_values = 1.0 / _fL_x  # dense at low f/L, sparse at high f/L

SWEEPS = [
    # ---- Geometry sweeps (cross-section FIXED) ----
    {
        "name": "L",
        "label": "Span L (m)",
        "values": np.linspace(0.5, 3.0, N_POINTS),
        "apply": lambda bl, val: setattr(bl, "L", val),
        "note": "f/L, h, b fixed → f, S, λ vary",
    },
    {
        "name": "f_L",
        "label": "Rise-to-span ratio f/L",
        "values": _fL_values,
        "apply": lambda bl, val: setattr(bl, "f_L", val),
        "note": "L, h, b fixed → f, S, λ vary  |  sampled as 1/x, x~U(3,13) matching V1",
    },

    # ---- Section sweeps ----
    {
        "name": "h",
        "label": "Section height h (mm)",
        "values": np.linspace(0.0035, 0.042, N_POINTS),
        "apply": lambda bl, val: setattr(bl, "h", val),
        "note": "L, f/L, b fixed → I0, A11, D11, λ vary  |  λ = S·√12/h ∈ [~150, ~1800]",
    },
    {
        "name": "b_h",
        "label": "Width-to-height ratio b/h",
        "values": np.linspace(3.0, 20.0, N_POINTS),
        "apply": lambda bl, val: setattr(bl, "b", val * bl.h),
        "note": "L, f/L, h fixed → b, I0, A11, D11 vary  |  λ = S·√12/h FIXED",
    },

    # ---- Material sweep (sanity check: dimensionless Q_cr should be independent of E) ----
    {
        "name": "E",
        "label": "Young's modulus E (GPa)",
        "values": np.linspace(60.0, 210.0, N_POINTS),
        "apply": lambda bl, val: setattr(bl, "E", val * 1e9),
        "note": "All geometry fixed → only I0, A11, D11 scale",
    },

    # ---- Boundary spring sweeps ----
    {
        "name": "K_rot",
        "label": "Rotational spring coefficient KZ",
        "values": np.linspace(0.1, 1000.0, N_POINTS),
        "apply": lambda bl, val: setattr(bl, "KZ", val),
        "note": "All geometry & material fixed  |  KZL = KZR = KZ (symmetric)",
    },
    {
        "name": "K_trans",
        "label": "Translational spring coefficient KX, KY",
        "values": np.linspace(0.1, 10.0, N_POINTS),
        "apply": lambda bl, val: [setattr(bl, k, val) for k in ("KX", "KY")],
        "note": "All geometry & material fixed  |  KX=KY varied together (symmetric)",
    },
]


# =============================================================================
# Main
# =============================================================================
def run():
    model, s, A, B_coef = load_model(DEVICE)
    fixed_scaler = s["fixed"]
    abs_scaler = s["abs"]
    x0 = torch.zeros(1, 43, device=DEVICE)

    # --- Print baseline features for verification ---
    bl_ref = Baseline()
    feats_ref = bl_ref.compute_features()
    print("\nBaseline 16-D features (independent parameters → derived features):")
    names = ["I0","A11","B11","D11","L","f","b","h","S","λ",
             "KXL","KYL","KZL","KXR","KYR","KZR"]
    print(f"  {'Feature':>5s}  {'Value':>14s}    {'Source':>30s}")
    print(f"  {'-'*5}  {'-'*14}    {'-'*30}")
    independent = {"L", "f_L", "h", "b", "E", "rho", "mu", "KX", "KY", "KZ"}
    for nm, v in zip(names, feats_ref):
        src = "← independent" if nm in independent else "← derived"
        print(f"  {nm:>5s}  {v:>14.6g}    {src}")
    # Show key derived quantities
    print(f"\n  Derived: f = L·(f/L) = {bl_ref.L * bl_ref.f_L:.4g}")
    print(f"           S = arc_length(L,f) = {arc_length(bl_ref.L, bl_ref.L * bl_ref.f_L):.4g}")
    print(f"           λ = S·√12/h = {arc_length(bl_ref.L, bl_ref.L * bl_ref.f_L) * math.sqrt(12) / bl_ref.h:.4g}")

    all_results = {}

    for sweep in SWEEPS:
        name = sweep["name"]
        values = sweep["values"]
        note = sweep.get("note", "")
        q_cr_vals = []
        P_cr_vals = []    # dimensional buckling load (N)

        print(f"\n{'='*70}")
        print(f"Sweeping {name} ({sweep['label']})  N={len(values)}")
        print(f"  {note}")
        print(f"  Range: [{values[0]:.4g}, {values[-1]:.4g}]")
        print(f"{'='*70}")

        for val in values:
            bl = Baseline()
            sweep["apply"](bl, val)
            features = bl.compute_features()

            # Extract geometry & stiffness for dimensional conversion
            D11 = features[3]   # bending stiffness
            L_f = features[4]   # span
            b_f = features[6]   # section width

            features_std = fixed_scaler.transform(features.reshape(1, -1))
            fixed_t = torch.tensor(features_std, dtype=torch.float32, device=DEVICE)

            pred_std = infer(model, fixed_t, x0, A, B_coef)
            pred_phys = abs_scaler.inverse_transform(pred_std)

            # Dimensionless critical load
            q_cr = extract_q_cr(pred_phys)
            q_cr_vals.append(q_cr)

            # Dimensional critical load:  P_cr = q_cr * D11 / (b * L²)
            P_cr = q_cr * D11 / (b_f * L_f * L_f)
            P_cr_vals.append(P_cr)

        df = pd.DataFrame({name: values, "q_cr": q_cr_vals, "P_cr_N": P_cr_vals})
        csv_path = os.path.join(OUTPUT_DIR, f"sensitivity_{name}.csv")
        df.to_csv(csv_path, index=False)

        # --- Dimensionless summary ---
        q_min, q_max = min(q_cr_vals), max(q_cr_vals)
        q_sens = (q_max - q_min) / (q_max + 1e-9) * 100
        q_trend = "↑" if q_cr_vals[-1] > q_cr_vals[0] else "↓"

        # --- Dimensional summary ---
        P_min, P_max = min(P_cr_vals), max(P_cr_vals)
        P_sens = (P_max - P_min) / (P_max + 1e-9) * 100
        P_trend = "↑" if P_cr_vals[-1] > P_cr_vals[0] else "↓"

        print(f"  q_cr (dimensionless): {q_min:.4f} → {q_max:.4f}  "
              f"({q_trend} {q_sens:.1f}%)")
        print(f"  P_cr (N):             {P_min:.2f} → {P_max:.2f}  "
              f"({P_trend} {P_sens:.1f}%)")

        all_results[name] = {
            "label": sweep["label"],
            "range": f"[{values[0]:.4g}, {values[-1]:.4g}]",
            "q_cr_min": q_min, "q_cr_max": q_max,
            "q_sensitivity_pct": q_sens, "q_trend": q_trend,
            "P_cr_min_N": P_min, "P_cr_max_N": P_max,
            "P_sensitivity_pct": P_sens, "P_trend": P_trend,
        }

    # =========================================================================
    # Multi-f_L E sweep: same E sweep at different rise-to-span ratios
    #   f/L = 1/13 (shallow), 1/8 (median), 1/3 (deep)
    # Shows how the E–P_cr relationship depends on arch shallowness.
    # =========================================================================
    f_L_cases = [
        (1.0/13.0, "1/13", "shallow"),
        (1.0/8.0,  "1/8",  "median"),
        (1.0/3.0,  "1/3",  "deep"),
    ]
    E_values = np.linspace(60.0, 210.0, N_POINTS)

    print(f"\n{'='*70}")
    print(f"Multi-f/L E sweep: E sensitivity at 3 rise-to-span ratios")
    print(f"{'='*70}")

    multi_E_results = {}

    for f_L_val, f_L_label, f_L_desc in f_L_cases:
        q_cr_vals = []
        P_cr_vals = []

        print(f"\n  f/L = {f_L_label} ({f_L_desc}) — E = {E_values[0]:.0f} → {E_values[-1]:.0f} GPa")

        for E_val in E_values:
            bl = Baseline()
            bl.f_L = f_L_val
            bl.E = E_val * 1e9
            features = bl.compute_features()

            D11 = features[3]
            L_f = features[4]
            b_f = features[6]

            features_std = fixed_scaler.transform(features.reshape(1, -1))
            fixed_t = torch.tensor(features_std, dtype=torch.float32, device=DEVICE)

            pred_std = infer(model, fixed_t, x0, A, B_coef)
            pred_phys = abs_scaler.inverse_transform(pred_std)

            q_cr = extract_q_cr(pred_phys)
            q_cr_vals.append(q_cr)
            P_cr = q_cr * D11 / (b_f * L_f * L_f)
            P_cr_vals.append(P_cr)

        # Save CSV
        df = pd.DataFrame({
            "E_GPa": E_values,
            "q_cr": q_cr_vals,
            "P_cr_N": P_cr_vals,
        })
        csv_path = os.path.join(OUTPUT_DIR, f"sensitivity_E_fL_{f_L_label.replace('/', '_')}.csv")
        df.to_csv(csv_path, index=False)

        q_min, q_max = min(q_cr_vals), max(q_cr_vals)
        q_sens = (q_max - q_min) / (q_max + 1e-9) * 100
        q_trend = "↑" if q_cr_vals[-1] > q_cr_vals[0] else "↓"
        P_min, P_max = min(P_cr_vals), max(P_cr_vals)
        P_sens = (P_max - P_min) / (P_max + 1e-9) * 100
        P_trend = "↑" if P_cr_vals[-1] > P_cr_vals[0] else "↓"

        print(f"    q_cr: {q_min:.4f} → {q_max:.4f}  ({q_trend} {q_sens:.2f}%)")
        print(f"    P_cr: {P_min:.2f} → {P_max:.2f} N  ({P_trend} {P_sens:.1f}%)")

        multi_E_results[f"E_fL_{f_L_label}"] = {
            "f_L": f_L_val, "f_L_label": f_L_label, "description": f_L_desc,
            "q_cr_min": q_min, "q_cr_max": q_max,
            "q_sensitivity_pct": q_sens, "q_trend": q_trend,
            "P_cr_min_N": P_min, "P_cr_max_N": P_max,
            "P_sensitivity_pct": P_sens, "P_trend": P_trend,
        }

    # Multi-f/L E summary
    print(f"\n  --- Multi-f/L E sweep summary ---")
    print(f"  {'f/L':>8s}  {'P_cr range (N)':>30s}  {'Δ%':>8s}  {'q_cr Δ%':>8s}")
    print(f"  {'-'*8}  {'-'*30}  {'-'*8}  {'-'*8}")
    for key, r in multi_E_results.items():
        print(f"  {r['f_L_label']:>8s}  {r['P_cr_min_N']:>12.1f} → {r['P_cr_max_N']:<12.1f}  "
              f"{r['P_sensitivity_pct']:>7.1f}%  {r['q_sensitivity_pct']:>7.2f}%")

    # Save dimensionless summary
    summary_q = pd.DataFrame({
        "label": [v["label"] for v in all_results.values()],
        "range": [v["range"] for v in all_results.values()],
        "q_cr_min": [v["q_cr_min"] for v in all_results.values()],
        "q_cr_max": [v["q_cr_max"] for v in all_results.values()],
        "sensitivity_pct": [v["q_sensitivity_pct"] for v in all_results.values()],
        "trend": [v["q_trend"] for v in all_results.values()],
    }, index=list(all_results.keys()))
    summary_q.index.name = "parameter"
    summary_q = summary_q.sort_values("sensitivity_pct", ascending=False)
    summary_q.to_csv(os.path.join(OUTPUT_DIR, "sensitivity_summary_dimensionless.csv"))

    # Save dimensional summary
    summary_P = pd.DataFrame({
        "label": [v["label"] for v in all_results.values()],
        "range": [v["range"] for v in all_results.values()],
        "P_cr_min_N": [v["P_cr_min_N"] for v in all_results.values()],
        "P_cr_max_N": [v["P_cr_max_N"] for v in all_results.values()],
        "sensitivity_pct": [v["P_sensitivity_pct"] for v in all_results.values()],
        "trend": [v["P_trend"] for v in all_results.values()],
    }, index=list(all_results.keys()))
    summary_P.index.name = "parameter"
    summary_P = summary_P.sort_values("sensitivity_pct", ascending=False)
    summary_P.to_csv(os.path.join(OUTPUT_DIR, "sensitivity_summary_dimensional.csv"))

    print(f"\n{'='*70}")
    print(f"Sensitivity Ranking — Dimensionless q_cr")
    print(f"{'='*70}")
    print(summary_q[["label", "q_cr_min", "q_cr_max", "sensitivity_pct", "trend"]].to_string())
    print(f"\n{'='*70}")
    print(f"Sensitivity Ranking — Dimensional P_cr (N)")
    print(f"{'='*70}")
    print(summary_P[["label", "P_cr_min_N", "P_cr_max_N", "sensitivity_pct", "trend"]].to_string())
    print(f"\nAll CSVs saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    run()
