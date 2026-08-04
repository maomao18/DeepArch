# CLAUDE.md

Project guidance for Claude Code and other coding agents working in this repository.

## Project Overview

DeepArch — Deep learning framework for predicting the complete nonlinear in-plane buckling response of composite parabolic deep arches. Uses FA-LSTM (Feature-Attention-enhanced LSTM) with PyTorch. Data generated via ANSYS MAPDL shell finite element simulations.

## Commands

```bash
# Multi-seed training (current pipeline)
python Code/Train_MultiSeed.py                     # Full FA-LSTM (5 seeds)
python Code/Train_MultiSeed_A1.py                  # Ablation A1: w/o Feature Attention
python Code/Train_MultiSeed_A2.py                  # Ablation A2: w/o Attentional Pooling
python Code/Train_MultiSeed_A3.py                  # Ablation A3: w/o FiLM
python Code/Train_MultiSeed_A4.py                  # Ablation A4: w/o Prior Knowledge
python Code/Train_MultiSeed_A5.py                  # Ablation A5: w/o State Residual
python Code/Train_MultiSeed_A8.py                  # Ablation A8: TF-only training

# Baseline training
python Code/Train_Baseline_LSTM.py
python Code/Train_Baseline_LSTM_AR.py
python Code/Train_Baseline_GRU.py
python Code/Train_Baseline_TCN.py
python Code/Train_Baseline_Transformer.py
python Code/Train_Baseline_DARNN.py

# Evaluation & summarization
python Code/Eval_MultiSeed.py                      # Multi-seed evaluation
python Code/Summarize.py                           # Cross-model comparison table
python Code/compute_tables.py                      # Reproducible metric extraction

# Inference
python Code/inference.py --folder ./Data/Test
python Code/inference.py --sample ./Data/Test/sample.csv
python Code/inference.py --model ./models/FA-LSTM/seed_618/buckling_predictor_best.pth

# Data generation (FE simulation)
python Code/AnsysBatch.py                          # Original random-sampling script
python Code/AnsysBatch_V1_FGM.py                   # V1: random uniform, FGM 10-layer
python Code/AnsysBatch_V2_FGM.py                   # V2: LHS, independent lambda, FGM 10-layer

# Data preprocessing
python Code/DataPre.py

# Analysis utilities
python Code/ParamSensitivity.py                    # Parameter sensitivity
python Code/extract_attention_weights.py            # Attention weight extraction
python Code/distribution_analysis.py               # Parameter distribution statistics
python Code/correlation_analysis.py                # Pearson correlation analysis
```

## Architecture

### FA-LSTM Model (`Code/models/predictor.py`)

Sequence-to-sequence model predicting structural displacement under load:
- **Input**: 16 fixed features (structural parameters) + dynamic sequence (load + 42 displacements)
- **Components**: FeatureTokenizer → MultiHeadSelfAttention → FiLM → LSTM+Prior → AttnPool → MultiHead output
- **Training**: Staged scheduled sampling (Teacher Forcing → Mixed → Autoregressive), 150 epochs

### Ablation Variants

| Code | Paper Label | Component Removed |
|---|---|---|
| `FA-LSTM` | FA-LSTM (full) | — |
| `FA-LSTM_A1` | w/o FA | Feature Attention (MHSA) |
| `FA-LSTM_A2` | w/o AttnPool | Attentional Pooling → mean pooling |
| `FA-LSTM_A3` | w/o FiLM | FiLM conditioning → direct concat |
| `FA-LSTM_A4` | w/o Prior | Delta-prediction → absolute prediction |
| `FA-LSTM_A5` | w/o State Residual | State residual connection |
| `FA-LSTM_A6` | w/o Multi-Head | Multi-head → single-head output |
| `FA-LSTM_A8` | TF-only | Scheduled sampling → pure teacher forcing |

### Key Dimensions

- `fixed_features_dim`: 16
- `dynamic_features_dim`: 43 (1 load + 42 displacements)
- `sequence_length`: 200
- `hidden_size`: 512
- `num_layers`: 4
- `num_heads`: 4

### Directory Structure

```
├── Code/
│   ├── config/           # Configuration dataclasses
│   ├── models/           # Neural network modules
│   ├── data/             # Data handling (dataset, scalers)
│   ├── training/         # Training loops and loss/metrics
│   ├── inference/        # Inference and evaluation
│   └── utils/            # Helpers (seed, IO, buckling extraction)
├── Data/
│   ├── Train/            # Training CSVs (2508 files)
│   └── Test/             # Test CSVs (61 files)
├── models/               # Trained model weights (gitignored)
├── logs/                 # Training logs & metrics (gitignored)
├── results/              # Analysis outputs (gitignored except attention/sensitivity)
└── docs/                 # Design specs and documentation
```

## Data Format

CSV files per sample:
- Column 1: Time step (1–200)
- Columns 2–17: Fixed features (I0, A11, B11, D11, L, f, b, h, S, lambda, KXL, KYL, KZL, KXR, KYR, KZR)
- Column 18: Dimensionless load
- Columns 19–60: Dynamic features (21 nodes × 2 displacements: x, y)

## Dependencies

Core: `torch>=2.0`, `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `joblib`, `tqdm`

Optional: `ansys-mapdl-core` (ANSYS interface, data generation), `tensorboard` (training monitoring), `scipy` (LHS sampling), `seaborn` (visualization)

## Notes

- Windows platform; use raw strings for paths
- `num_workers=0` for DataLoader on Windows
- Bilingual codebase (Chinese/English); maintain consistency within each file
- Multi-seed training uses seeds: 42, 123, 618, 2024, 9999
- Model checkpoint saved on best AR validation loss
- Test metrics extracted from `phase='test'` row (FA-LSTM) or `test_ar_*` columns (baselines)
