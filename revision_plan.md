# Revision Data — COMSTR-D-26-02580

**Manuscript:** Data-driven prediction of nonlinear in-plane buckling of composite parabolic deep arches using feature-attention-enhanced LSTM networks
**Journal:** Composite Structures
**Data Date:** 2026-07-28
**Metric Source:** Test set (held-out 15% split). 5 seeds (42, 123, 618, 2024, 9999). All values mean ± std.

---

## 1. Ablation — Critical Buckling Load (Test Set)

| Configuration | N | R²_cr | MAE_cr (kN) | Δ MAE_cr | RMSE_cr (kN) | MAPE_cr |
|---|---|---|---|---|---|---|
| FA-LSTM (full) | 5 | 0.9919 ± 0.0074 | 1.275 ± 0.142 | — | 2.378 ± 0.986 | 3.12% ± 0.37% |
| w/o Feature Attention (A1) | 5 | 0.9866 ± 0.0069 | 1.791 ± 0.369 | +40.4% | 3.210 ± 0.914 | 3.99% ± 0.75% |
| w/o Attentional Pooling (A2) | 5 | 0.9770 ± 0.0071 | 2.214 ± 0.186 | +73.6% | 4.269 ± 0.883 | 4.95% ± 0.74% |
| w/o FiLM Conditioning (A3) | 5 | 0.9865 ± 0.0049 | 1.714 ± 0.387 | +34.4% | 3.239 ± 0.698 | 3.93% ± 0.91% |
| w/o Prior Knowledge (A4) | 5 | 0.9774 ± 0.0207 | 2.148 ± 0.398 | +68.5% | 3.952 ± 1.663 | 4.89% ± 0.87% |
| w/o State Residual (A5) | 5 | 0.9896 ± 0.0049 | 1.612 ± 0.358 | +26.4% | 2.847 ± 0.786 | 3.79% ± 0.85% |
| w/o Multi-Head Output (A6) | 5 | 0.9838 ± 0.0064 | 1.999 ± 0.382 | +56.8% | 3.567 ± 0.879 | 4.57% ± 0.68% |
| TF-only training (A8) | 5 | 0.9913 ± 0.0061 | 1.298 ± 0.259 | +1.8% | 2.528 ± 0.932 | 3.12% ± 0.52% |

## 2. Ablation — Sequence-Level (Test Set)

| Configuration | N | R²_seq | MAE_seq | Δ MAE_seq | RMSE_seq |
|---|---|---|---|---|---|
| FA-LSTM (full) | 5 | 0.9920 ± 0.0041 | 0.0500 ± 0.0173 | — | 0.438 ± 0.131 |
| w/o Feature Attention (A1) | 5 | 0.9849 ± 0.0115 | 0.0655 ± 0.0327 | +31.2% | 0.583 ± 0.235 |
| w/o Attentional Pooling (A2) | 5 | 0.9794 ± 0.0089 | 0.0740 ± 0.0235 | +48.2% | 0.699 ± 0.163 |
| w/o FiLM Conditioning (A3) | 5 | 0.9899 ± 0.0040 | 0.0537 ± 0.0154 | +7.4% | 0.498 ± 0.115 |
| w/o Prior Knowledge (A4) | 5 | 0.9767 ± 0.0104 | 0.0903 ± 0.0113 | +80.8% | 0.754 ± 0.147 |
| w/o State Residual (A5) | 5 | 0.9806 ± 0.0107 | 0.0823 ± 0.0300 | +64.7% | 0.686 ± 0.221 |
| w/o Multi-Head Output (A6) | 5 | 0.9885 ± 0.0014 | 0.0572 ± 0.0065 | +14.5% | 0.541 ± 0.055 |
| TF-only training (A8) | 5 | 0.9888 ± 0.0058 | 0.0636 ± 0.0234 | +27.4% | 0.519 ± 0.175 |

## 3. Baseline Comparison — Critical Buckling Load (Test Set)

| Model | R²_cr | MAE_cr (kN) | RMSE_cr (kN) | MAPE_cr |
|---|---|---|---|---|
| **FA-LSTM** | **0.9919 ± 0.0074** | **1.275 ± 0.142** | **2.378 ± 0.986** | **3.12% ± 0.37%** |
| LSTM | -0.0609 ± 0.0710 | 22.205 ± 1.033 | 29.058 ± 1.458 | 68.74% ± 6.81% |
| LSTM-AR | -0.0837 ± 0.0406 | 21.979 ± 1.151 | 29.406 ± 1.893 | 64.53% ± 11.05% |
| GRU | -0.1539 ± 0.0765 | 23.005 ± 1.141 | 30.307 ± 1.570 | 70.31% ± 9.71% |
| TCN | -0.5348 ± 0.0706 | 23.033 ± 2.048 | 35.039 ± 3.010 | 41.97% ± 1.45% |
| Transformer | -1.3431 ± 1.6665 | 22.358 ± 11.101 | 39.946 ± 18.075 | 41.39% ± 22.47% |
| DA-RNN | -0.0883 ± 0.1073 | 23.150 ± 1.391 | 29.407 ± 1.621 | 73.70% ± 11.68% |

## 4. Baseline Comparison — Sequence-Level (Test Set)

| Model | R²_seq | MAE_seq | RMSE_seq |
|---|---|---|---|
| **FA-LSTM** | **0.9920 ± 0.0041** | **0.0500 ± 0.0173** | **0.438 ± 0.131** |
| LSTM | 0.6562 ± 0.0182 | 0.3276 ± 0.0163 | 2.949 ± 0.136 |
| LSTM-AR | 0.6471 ± 0.0110 | 0.3245 ± 0.0143 | 2.992 ± 0.192 |
| GRU | 0.6155 ± 0.0210 | 0.3531 ± 0.0136 | 3.119 ± 0.123 |
| TCN | 0.5038 ± 0.0154 | 0.4025 ± 0.0160 | 3.547 ± 0.224 |
| Transformer | 0.4481 ± 0.2737 | 0.3376 ± 0.0954 | 3.629 ± 1.039 |
| DA-RNN | 0.6510 ± 0.0329 | 0.3389 ± 0.0205 | 2.970 ± 0.169 |

## 5. Per-Seed Detail — Test Set Buckling Load MAE (kN)

| Model | seed_42 | seed_123 | seed_618 | seed_2024 | seed_9999 | MEAN ± STD |
|---|---|---|---|---|---|---|
| FA-LSTM | 1.302 | 1.352 | 1.074 | 1.443 | 1.205 | 1.275 ± 0.142 |
| w/o FA (A1) | 2.240 | 2.032 | 1.379 | 1.849 | 1.456 | 1.791 ± 0.369 |
| w/o AttnPool (A2) | 2.380 | 2.338 | 2.300 | 1.933 | 2.121 | 2.214 ± 0.186 |
| w/o FiLM (A3) | 2.198 | 2.063 | 1.477 | 1.479 | 1.351 | 1.714 ± 0.387 |
| w/o Prior (A4) | 2.397 | 1.919 | 1.824 | 2.730 | 1.872 | 2.148 ± 0.398 |
| w/o StateRes (A5) | 2.079 | 1.575 | 1.234 | 1.313 | 1.857 | 1.612 ± 0.358 |
| w/o MultiHead (A6) | 2.639 | 2.008 | 1.634 | 1.868 | 1.848 | 1.999 ± 0.382 |
| TF-only (A8) | 1.291 | 1.493 | 0.864 | 1.347 | 1.494 | 1.298 ± 0.259 |
| LSTM | 22.993 | 23.580 | 21.748 | 21.570 | 21.134 | 22.205 ± 1.033 |
| LSTM-AR | 22.342 | 23.549 | 21.358 | 22.175 | 20.471 | 21.979 ± 1.151 |
| GRU | 23.968 | 23.427 | 23.753 | 21.143 | 22.733 | 23.005 ± 1.141 |
| TCN | 24.771 | 23.777 | 19.826 | 22.245 | 24.547 | 23.033 ± 2.048 |
| Transformer | 9.575 | 33.700 | 11.378 | 29.830 | 27.307 | 22.358 ± 11.101 |
| DA-RNN | 21.857 | 22.864 | 23.254 | 25.450 | 22.323 | 23.150 ± 1.391 |

## 6. Ablation Impact Ranking

| Rank | Component | Δ MAE_cr |
|:---:|---|---|
| 1 | Attentional Pooling | +73.6% |
| 2 | Prior Knowledge (Δ-formulation) | +68.5% |
| 3 | Multi-Head Output | +56.8% |
| 4 | Feature Attention | +40.4% |
| 5 | FiLM Conditioning | +34.4% |
| 6 | State Residual | +26.4% |
| — | TF-only training (A8) | +1.8% |

## 7. Model Code → Paper Label

| Code | Paper Label | Component Modified |
|---|---|---|
| FA-LSTM | FA-LSTM (full) | — |
| FA-LSTM_A1 | w/o FA | Feature Attention removed |
| FA-LSTM_A2 | w/o AttnPool | Attentional Pooling → mean pooling |
| FA-LSTM_A3 | w/o FiLM | FiLM conditioning → direct concat |
| FA-LSTM_A4 | w/o Prior | Δ-prediction → absolute prediction |
| FA-LSTM_A5 | w/o State Residual | State residual connection removed |
| FA-LSTM_A6 | w/o Multi-Head | Multi-head → single-head output |
| FA-LSTM_A8 | TF-only | Scheduled sampling → teacher forcing |

## 8. Training Configuration (FA-LSTM)

```
hidden_size: 512
num_layers: 4
num_heads: 4
dropout: 0.186
film_hidden: 128
state_residual_hidden: 128
learning_rate: 3.048e-04
weight_decay: 3.004e-05
grad_clip_norm: 3.219
batch_size: 16
epochs: 150 (TF: 40, transition: 40, AR: 70)
w_delta: 0.435
w_abs: 0.565
load_weight: 0.209
displacement_weight: 0.791
Features enabled: MHSA, AttnPool, LSTM Prior (Δ-formulation), MultiHead Outputs, FiLM, State Residual
```

## 9. Attention Weights — v2F (main paper figure)

v2F shows a U-shaped fixed-feature attention pattern: high at t=0 (49.1%), declines as displacement features dominate (t=40-60), rebounds mid-sequence (t=80-120, peaking at 11.8%), then declines again.

| t | fixed features | load | displacement |
|---:|:---:|:---:|:---:|
| 0 | 0.491 | 0.042 | 0.467 |
| 10 | 0.296 | 0.006 | 0.699 |
| 20 | 0.120 | 0.003 | 0.877 |
| 40 | 0.031 | 0.001 | 0.969 |
| 60 | 0.017 | 0.000 | 0.983 |
| 80 | 0.037 | 0.000 | 0.962 |
| 100 | 0.106 | 0.003 | 0.891 |
| 120 | 0.118 | 0.005 | 0.877 |
| 140 | 0.022 | 0.000 | 0.978 |
| 160 | 0.006 | 0.000 | 0.994 |
| 180 | 0.022 | 0.002 | 0.976 |
| 199 | 0.053 | 0.003 | 0.945 |

Mean fixed feature weight: 0.073. Top fixed features by mean attention: A11 (0.012), I0 (0.010), B11 (0.010), KXL (0.005), lambda (0.005), S (0.005).

## 10. Attention Weights — Seed Comparison (t=0 fixed feature weight)

| Seed | t=0 fixed | mean fixed | Pattern |
|---|---|---|---|
| **v2F (paper)** | **0.491** | 0.073 | U-shaped, interpretable |
| seed 9999 | 0.541 | 0.104 | High initial fixed, gradual decline |
| seed 123 | 0.032 | 0.030 | Moderate fixed, significant load attention (0.30 at t=10) |
| seed 42 | 0.008 | 0.013 | Low fixed, displacement-dominated |
| seed 618 | 0.000 | 0.002 | Near-zero fixed attention throughout |
| seed 2024 | 0.000 | 0.009 | Near-zero fixed attention |

## 11. Attention Weights — seed 9999 (supplementary)

Seed 9999 independently shows high initial fixed-feature attention (54.1% at t=0), confirming the interpretable pattern is reproducible across training seeds.

| t | fixed features | load | displacement |
|---:|:---:|:---:|:---:|
| 0 | 0.541 | 0.030 | 0.430 |
| 10 | 0.276 | 0.005 | 0.719 |
| 20 | 0.223 | 0.003 | 0.774 |
| 40 | 0.170 | 0.002 | 0.828 |
| 60 | 0.176 | 0.002 | 0.822 |
| 80 | 0.147 | 0.001 | 0.852 |
| 100 | 0.063 | 0.004 | 0.933 |
| 120 | 0.003 | 0.000 | 0.997 |
| 140 | 0.011 | 0.000 | 0.989 |
| 160 | 0.025 | 0.001 | 0.975 |
| 180 | 0.024 | 0.001 | 0.975 |
| 199 | 0.005 | 0.000 | 0.994 |

## 12. Attention Weights — seed 123 (supplementary)

Seed 123 shows an alternative pattern: moderate fixed attention (3% at t=0) with unusually high load attention (30% at t=10, persists through t=80). This represents an alternative but equally valid internal routing strategy.

| t | fixed features | load | displacement |
|---:|:---:|:---:|:---:|
| 0 | 0.032 | 0.000 | 0.968 |
| 10 | 0.025 | 0.301 | 0.673 |
| 20 | 0.035 | 0.359 | 0.607 |
| 40 | 0.042 | 0.359 | 0.600 |
| 60 | 0.045 | 0.292 | 0.663 |
| 80 | 0.044 | 0.111 | 0.845 |
| 100 | 0.056 | 0.051 | 0.893 |
| 120 | 0.035 | 0.012 | 0.953 |
| 140 | 0.003 | 0.001 | 0.996 |
| 160 | 0.006 | 0.001 | 0.994 |
| 180 | 0.008 | 0.000 | 0.992 |
| 199 | 0.008 | 0.004 | 0.988 |

## 13. Attention Weights — seeds 42, 618, 2024 (supplementary)

These seeds show near-zero fixed feature attention throughout. They achieve comparable predictive performance through alternative internal representations.

| t | seed_42 fixed | seed_618 fixed | seed_2024 fixed |
|---:|:---:|:---:|:---:|
| 0 | 0.007 | 0.000 | 0.000 |
| 10 | 0.019 | 0.000 | 0.008 |
| 20 | 0.029 | 0.001 | 0.021 |
| 40 | 0.020 | 0.001 | 0.021 |
| 60 | 0.018 | 0.001 | 0.012 |
| 80 | 0.003 | 0.000 | 0.016 |
| 100 | 0.013 | 0.000 | 0.014 |
| 120 | 0.003 | 0.010 | 0.003 |
| 140 | 0.004 | 0.003 | 0.001 |
| 160 | 0.005 | 0.000 | 0.004 |
| 180 | 0.005 | 0.000 | 0.002 |
| 199 | 0.005 | 0.001 | 0.001 |
| mean | 0.013 | 0.002 | 0.009 |

## 14. Parameter Sensitivity

Critical buckling load predicted by FA-LSTM when sweeping each parameter individually while holding others at median values.

| Parameter | Range | q_cr_min (kN) | q_cr_max (kN) | Sensitivity | Trend |
|---|---|---|---|---|---|
| Span L | [0.5, 3] m | 42.67 | 199.95 | 78.7% | ↓ longer span → lower load |
| Translational spring KX/KY | [0.1, 10] normalized | 48.27 | 135.93 | 64.5% | ↓ stiffer → lower load |
| Slenderness λ | [40, 500] | 43.22 | 95.28 | 54.6% | ↑ more slender → higher load |
| Rise-to-span f/L | [0.077, 0.333] | 52.03 | 100.40 | 48.2% | ↑ deeper arch → higher load |
| Rotational spring KZ | [1e-6, 1000] normalized | 65.52 | 103.56 | 36.7% | ↑ stiffer → higher load (plateaus after ~100) |
| Young's modulus E | [60, 210] GPa | 65.27 | 84.64 | 22.9% | ↑ stiffer material → higher load |
| Width/height b/h | [3, 20] | 60.10 | 72.52 | 17.1% | ↓ wider section → slightly lower load |
