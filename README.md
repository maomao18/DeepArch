<p align="center">
  <h1 align="center">DeepArch</h1>
  <p align="center">
    <strong>FA-LSTM</strong> — Data-driven prediction of nonlinear in-plane buckling<br>
    of composite parabolic deep arches
  </p>
</p>

<p align="center">
  <a href="#quick-start"><img src="https://img.shields.io/badge/python-3.9+-blue.svg"></a>
  <a href="#quick-start"><img src="https://img.shields.io/badge/pytorch-2.0+-orange.svg"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green.svg"></a>
</p>

---

## Overview

Deep learning surrogate for the complete nonlinear in-plane buckling response of FGM (functionally graded material) parabolic arches. Uses a **Feature-Attention-enhanced LSTM (FA-LSTM)** trained on ANSYS MAPDL shell finite-element simulations. Predicts the full 200-step load–displacement path from 16 structural parameters.

> **Manuscript:** Hao Tang, Airong Liu, Jie Yang, Jialin Wang, Shimao Qin  
> *"Data-driven prediction of nonlinear in-plane buckling of composite parabolic deep arches using feature-attention-enhanced LSTM networks"*  
> Submitted to *Composite Structures*

---

## Quick Start

```bash
conda create -n deeparch python=3.9
conda activate deeparch
pip install torch numpy pandas scikit-learn matplotlib tqdm joblib
```

**Train the full model (5 seeds):**
```bash
python Code/Train_MultiSeed.py
```

**Run inference on a single sample:**
```bash
python Code/inference.py --sample ./Data/Test/sample.csv
```

**Run parameter sensitivity analysis:**
```bash
python Code/ParamSensitivity.py
```

---

## Repository Structure

```
DeepArch/
├── Code/
│   ├── Train_MultiSeed.py            Full model: FA-LSTM, 5 seeds
│   ├── Train_MultiSeed_A1.py         Ablation: w/o Feature Attention (MHSA)
│   ├── Train_MultiSeed_A2.py         Ablation: w/o Attentional Pooling
│   ├── Train_MultiSeed_A3.py         Ablation: w/o FiLM conditioning
│   ├── Train_MultiSeed_A4.py         Ablation: w/o Prior Knowledge
│   ├── Train_MultiSeed_A5.py         Ablation: w/o State Residual
│   ├── Train_MultiSeed_A6.py         Ablation: w/o Multi-Head Output
│   ├── Train_MultiSeed_A8.py         Ablation: TF-only training
│   │
│   ├── Train_Baseline_LSTM.py        Baseline: vanilla LSTM
│   ├── Train_Baseline_LSTM_AR.py     Baseline: LSTM + AR checkpoint
│   ├── Train_Baseline_GRU.py         Baseline: GRU
│   ├── Train_Baseline_TCN.py         Baseline: TCN
│   ├── Train_Baseline_Transformer.py Baseline: Transformer
│   ├── Train_Baseline_DARNN.py       Baseline: DA-RNN
│   │
│   ├── Eval_MultiSeed.py             Multi-seed evaluation (mean ± std)
│   ├── Summarize.py                  Cross-model comparison table
│   ├── compute_tables.py             Reproducible metric extraction
│   │
│   ├── ParamSensitivity.py           Parameter sensitivity (7 sweeps, with scatter)
│   ├── inference.py                  Single-sample inference
│   ├── extract_attention_weights.py  Attention weight extraction
│   │
│   ├── AnsysBatch_V1_FGM.py          Data generation: V1 random uniform, 10-layer FGM
│   ├── AnsysBatch_V2_FGM.py          Data generation: V2 LHS, independent λ
│   ├── AnsysMeshConvergence.py       Mesh convergence study
│   ├── DataPre.py                    Raw FEM → 200-step CSV preprocessing
│   │
│   ├── distribution_analysis.py      Parameter distribution statistics & plots
│   ├── correlation_analysis.py       Pearson correlation analysis
│   ├── analyze_data.py               Dataset feature statistics
│   │
│   ├── buckling_visualization.py     Batch load–displacement curve plots
│   ├── buckling_individual_plots.py  Per-sample curve plots
│   ├── buckling_result_analysis.py   Buckling error analysis
│   │
│   ├── config/                       Training/Evaluation/Inference configs
│   ├── models/                       BucklingPredictor & baseline model classes
│   ├── data/                         Dataset, Scalers
│   ├── training/                     Trainer, Loss, Metrics
│   ├── inference/                    Inference pipeline & evaluation
│   └── utils/                        Seed, I/O, buckling extraction helpers
│
├── Data/
│   ├── Train/                        2,508 training CSVs (200 steps each)
│   └── Test/                         61 test CSVs
│
├── docs/                             Design specifications
├── CLAUDE.md                         Project guidance for coding agents
├── README.md                         This file
└── requirements.txt                  Python dependencies
```

---

## Model Architecture

```
Fixed Features (16) ──→ FeatureTokenizer ──┐
                                           ├─→ MHSA ──→ FiLM ──→ AttnPool ──→ LSTM ──→ Dual-Head Decode
Dynamic State (43)  ──→ DynTokenizer    ──┘              ↑                     ↑
                                                     Fixed Features     Prior MLP (init state)
                                                                             ↑
                                                                     State Residual
```

| Component | Description |
|-----------|-------------|
| **MHSA** | Multi-Head Self-Attention over 59 feature tokens |
| **AttnPool** | Learnable weighted aggregation → LSTM input |
| **FiLM** | Fixed-feature-conditioned modulation of dynamic representations |
| **Prior MLP** | Maps fixed features → LSTM initial hidden/cell states |
| **State Residual** | Shortcut: current state → delta prediction |
| **Dual-Head Decoder** | Separate MLPs for load and displacement outputs |
| **Scheduled Sampling** | Curriculum: Teacher Forcing → Mixed → AR (40/40/70 epochs) |

Key dimensions: `hidden_size=512`, `num_layers=4`, `num_heads=4`, `sequence_length=200`

---

## Ablation Variants

| Code | Label | Removed Component |
|------|-------|-------------------|
| `FA-LSTM` | FA-LSTM (full) | — |
| `FA-LSTM_A1` | w/o FA | Feature Attention (MHSA) |
| `FA-LSTM_A2` | w/o AttnPool | Attentional Pooling → mean pooling |
| `FA-LSTM_A3` | w/o FiLM | FiLM → direct concat |
| `FA-LSTM_A4` | w/o Prior | Delta-prediction → absolute prediction |
| `FA-LSTM_A5` | w/o State Res | State residual connection |
| `FA-LSTM_A6` | w/o Multi-Head | Multi-head → single-head output |
| `FA-LSTM_A8` | TF-only | Scheduled sampling → pure teacher forcing |

---

## Data Format

**Generation:** V1 random uniform sampling → ANSYS MAPDL SHELL181 → arc-length nonlinear buckling → 200-step resampling via linear interpolation.

**Independent parameters (12-D):**

| Category | Parameter | Distribution |
|----------|-----------|-------------|
| Geometry | Span `L` | U(0.5, 3.0) m |
| | Rise-to-span `f/L` | 1/x, x~U(3,13) |
| | Width-to-height `b/h` | U(3, 20) |
| | Slenderness `λ` (= S/h) | U(65, 500) |
| Material | Young's modulus `E` | U(60, 210) GPa |
| | Density `ρ` | U(2.7, 8.0) g/cm³ |
| | Poisson ratio `μ` | U(0.2, 0.4) |
| Boundary | Translational springs `KX, KY` | U(0.1, 10) |
| | Rotational springs `KZ` | U(0.1, 1000) |
| FGM | Gradient coefficient `e₀` | U(0.1, 0.3) |
| | Distribution type | cos(z) or cos(z/2 + π/4) |

**Each CSV (60 columns × 200 rows):**

| Cols | Content |
|-----:|---------|
| 1 | Time step (1–200) |
| 2–17 | Fixed features: `I₀, A₁₁, B₁₁, D₁₁, L, f, b, h, S, λ, KXL, KYL, KZL, KXR, KYR, KZR` |
| 18 | Dimensionless load |
| 19–60 | Dimensionless displacements (21 nodes × UX, UY) |

---

## Parameter Sensitivity Analysis

7 independent physical parameters swept with homogeneous-section baseline calibrated to dataset medians. Both dimensionless `q_cr` and dimensional `P_cr` (N) reported.

```bash
python Code/ParamSensitivity.py
# Output: results/sensitivity_v2/
```

See `CLAUDE.md` for full usage and experiment details.

---

## Citation

```bibtex
@article{tang2026deeparch,
  title  = {Data-driven prediction of nonlinear in-plane buckling of composite
            parabolic deep arches using feature-attention-enhanced {LSTM} networks},
  author = {Tang, Hao and Liu, Airong and Yang, Jie and Wang, Jialin and Qin, Shimao},
  journal= {Composite Structures},
  year   = {2026},
  note   = {Under review}
}
```

## License

MIT
