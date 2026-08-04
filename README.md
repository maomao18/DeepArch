# DeepArch — FA-LSTM for Nonlinear Buckling Prediction of Parabolic Deep Arches

Deep learning framework for data-driven prediction of the complete nonlinear in-plane buckling response of composite parabolic deep arches. This repository accompanies the manuscript:

> **"Data-driven prediction of nonlinear in-plane buckling of composite parabolic deep arches using feature-attention-enhanced LSTM networks"**
>
> Hao Tang, Airong Liu, Jie Yang, Jialin Wang, Shimao Qin
>
> Submitted to *Composite Structures* (COMSTR-D-26-02580)

---

## Quick Reference

| Task | Command |
|------|---------|
| Train full model (5 seeds) | `python Code/Train_MultiSeed.py` |
| Train ablation variant | `python Code/Train_MultiSeed_A1.py` (A1–A6, A8) |
| Train baseline | `python Code/Train_Baseline_LSTM.py` (LSTM/GRU/TCN/Transformer/DA-RNN) |
| Evaluate all models | `python Code/Eval_MultiSeed.py` |
| Compute comparison tables | `python Code/compute_tables.py` |
| Data generation (FE, V1 random) | `python Code/AnsysBatch_V1_FGM.py` |
| Data generation (FE, V2 LHS) | `python Code/AnsysBatch_V2_FGM.py` |
| Mesh convergence study | `python Code/AnsysMeshConvergence.py` |
| Parameter sensitivity analysis | `python Code/ParamSensitivity.py` |
| Attention weight extraction | `python Code/extract_attention_weights.py` |
| Single-sample inference | `python Code/inference.py --sample <path>` |

---

## Environment

- Python 3.9+, PyTorch 2.0+, CUDA 11.8+
- ANSYS MAPDL 2021 R1 (data generation only)

```bash
conda create -n deeparch python=3.9
conda activate deeparch
pip install torch numpy pandas scikit-learn matplotlib tqdm joblib tensorboard
```

---

## Repository Structure

```
├── Code/
│   ├── Train_MultiSeed.py              # Full model (FA-LSTM), 5 seeds
│   ├── Train_MultiSeed_A1.py ~ A5.py, A8.py  # Ablation experiments
│   ├── Train_MultiSeed_A6.py               # Ablation: w/o Multi-Head Output
│   ├── Train_Baseline_LSTM.py              # Baseline: LSTM
│   ├── Train_Baseline_GRU.py           # Baseline: GRU
│   ├── Train_Baseline_TCN.py           # Baseline: TCN
│   ├── Train_Baseline_Transformer.py   # Baseline: Transformer
│   ├── Train_Baseline_LSTM_AR.py           # Baseline: LSTM with AR checkpoint
│   ├── Train_Baseline_DARNN.py             # Baseline: DA-RNN (reviewer-requested)
│   ├── Eval_MultiSeed.py                   # Unified multi-seed evaluation
│   ├── AnsysBatch_V1_FGM.py                  # V1 data generation: FGM/Compose/Homogeneous
│   ├── AnsysBatch_V2_FGM.py                  # V2 data generation: LHS, independent lambda
│   ├── LHS_AnsysBatch.py                   # LHS data generation (reference)
│   ├── AnsysMeshConvergence.py         # Mesh convergence study (modal)
│   ├── ParamSensitivity.py             # Parameter sensitivity sweep
│   ├── DataPre.py                      # Data preprocessing
│   ├── inference.py                    # Single-sample inference
│   ├── extract_attention_weights.py    # Attention weight extraction
│   ├── Summarize.py                    # Multi-model summary table
│   ├── analyze_data.py                 # Dataset statistics
│   ├── buckling_result_analysis.py     # Buckling error analysis
│   ├── buckling_visualization.py       # Load-displacement curve plotting
│   ├── buckling_individual_plots.py    # Per-sample curve plots
│   ├── correlation_analysis.py         # Pearson correlation analysis
│   ├── distribution_analysis.py        # Parameter distribution analysis
│   ├── evaluate_model.py               # Single-model evaluation
│   ├── config/                         # Configuration (TrainingConfig etc.)
│   ├── models/                         # Model definitions (BucklingPredictor, baselines)
│   ├── data/                           # Data pipeline (Dataset, Scalers)
│   ├── training/                       # Training loop (Trainer, Loss, Metrics)
│   ├── inference/                      # Inference & evaluation
│   └── utils/                          # Seed, buckling extraction, I/O
│
├── Data/
│   ├── Train/                          # Training data (2508 CSVs)
│   └── Test/                           # Test data
│
├── models/                             # Trained model weights
│   ├── FA-LSTM/                        # Full model, 5 seeds
│   ├── FA-LSTM_A8/                     # w/o Scheduled Sampling, 5 seeds
│   ├── LSTM/                           # LSTM baseline, 5 seeds
│   ├── GRU/                            # GRU baseline (pending)
│   ├── TCN/                            # TCN baseline, 5 seeds
│   ├── Transformer/                    # Transformer baseline, 5 seeds
│   └── v2F/                            # Paper original best model (seed 618)
│
├── logs/                               # Training logs & metrics
│   └── (mirrors models/ structure)
│
├── results/
│   ├── v2F/evaluation/                 # Full evaluation of paper model
│   ├── attention/                      # Attention weights (per seed + v2F)
│   ├── origin_ready/                   # Pre-processed CSV for Origin plotting
│   ├── sensitivity/                    # Parameter sensitivity sweep results
│   └── mesh_convergence/               # Mesh convergence study results
│
├── CLAUDE.md                           # Claude Code helper
└── README.md                           # This file
```

---

## Paper Experiments

### 1. Full Model (FA-LSTM) — Multi-Seed

```bash
python Code/Train_MultiSeed.py          # 5 seeds: 618, 42, 123, 2024, 9999
```

Uses the HPO-optimized config: `hidden_size=512, num_layers=4, num_heads=4`.

### 2. Ablation Studies

```bash
python Code/Train_MultiSeed_A1.py       # w/o Feature Attention (MHSA)
python Code/Train_MultiSeed_A2.py       # w/o Attention Pooling
python Code/Train_MultiSeed_A3.py       # w/o FiLM conditioning
python Code/Train_MultiSeed_A4.py       # w/o Prior-informed initialization
python Code/Train_MultiSeed_A5.py       # w/o State Residual connection
python Code/Train_MultiSeed_A8.py       # Supplementary: w/o Scheduled Sampling
```

Each ablation turns off exactly ONE component; all other settings identical to full model.

### 3. Baseline Comparison

```bash
python Code/Train_Baseline_LSTM.py
python Code/Train_Baseline_GRU.py
python Code/Train_Baseline_TCN.py
python Code/Train_Baseline_Transformer.py
```

Pure teacher-forcing baselines with matched parameter counts and training budget.

### 4. Evaluation

```bash
python Code/Eval_MultiSeed.py           # Computes mean ± std across seeds
python Code/Summarize.py                # Generates multi-model comparison table
```

---

## FE Data Generation & Validation

### Dataset Generation

```bash
python Code/AnsysBatch_V1_FGM.py        # V1: Random uniform, 500 samples, 10-layer FGM
python Code/AnsysBatch_V2_FGM.py        # V2: LHS, independent λ, 10-layer FGM
```

V1 parameters (independent, uniform):
Geometry: L ∈ [0.5,3], f/L = 1/x (x∈[3,13]), b/h ∈ [3,20], λ ∈ [65,500]
Material: E ∈ [60,210] GPa, ρ ∈ [2.7,8] g/cm³, μ ∈ [0.2,0.4]
Boundary: KX,KY ∈ [0.1,10], KZ ∈ [0.1,1000]

### Mesh Convergence Study

```bash
python Code/AnsysMeshConvergence.py
```

Runs modal analysis on a representative arch at 7 mesh densities (10–640 elements along curve), centered on the paper mesh (160). Output: `results/mesh_convergence/frequencies.csv`.

---

## Analysis & Visualization

### Parameter Sensitivity

```bash
python Code/ParamSensitivity.py
```

Uses trained FA-LSTM as surrogate to sweep 7 independent physical parameters.
Outputs both dimensionless q_cr and dimensional P_cr (N).
Output: `results/sensitivity_v2/sensitivity_*.csv`.

### Attention Weights

```bash
python Code/extract_attention_weights.py
```

Extracts attention pooling weights per timestep. Three output levels:
- `attn_weights_overall_mean.csv` — bar chart
- `attn_weights_mean_over_time.csv` — U-shape line plot
- `attn_weights_detail.csv` — per-sample

Pre-processed Origin-ready CSVs: `results/origin_ready/`.

---

## Data Format

Each CSV sample file has 60 columns:

| Col | Content |
|----:|---------|
| 1 | Time step (0–199) |
| 2–17 | Fixed features: I0, A11, B11, D11, L, f, b, h, S, λ, KXL, KYL, KZL, KXR, KYR, KZR |
| 18 | Non-dimensional load |
| 19–60 | Non-dimensional displacements (21 nodes × 2 directions: x, y) |

Fixed sequence length: 200 steps. Dataset: 2508 training + 61 test samples.

---

## Model Architecture

```
Fixed Features (16) ──→ FixedTokenizer ──┐
                                         ├──→ MHSA ──→ FiLM ──→ AttnPool ──→ LSTM ──→ Dual-Head Decode
Dynamic State (43)  ──→ DynTokenizer  ──┘              ↑                     ↑
                                                   Fixed Features     Prior MLP (init state)
                                                                           ↑
                                                                   State Residual
```

Key components:
- **Multi-Head Self-Attention (MHSA)**: feature-level attention across 59 tokens
- **Attention Pooling**: learnable weighted aggregation into LSTM input
- **FiLM**: fixed-feature-conditioned modulation of dynamic representations
- **Prior MLP**: maps fixed features to LSTM initial hidden/cell states
- **State Residual**: shortcut connection from current state to delta prediction
- **Dual-Head Decoder**: separate MLPs for load and displacement outputs
- **Scheduled Sampling**: curriculum from teacher forcing → autoregressive (40/40/70 epochs)

---

## Citation

If you use this code, please cite the corresponding paper:

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
