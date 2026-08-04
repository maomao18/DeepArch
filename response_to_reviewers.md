# Response to Reviewers

**Manuscript Number:** COMSTR-D-26-02580

**Title:** Data-driven prediction of nonlinear in-plane buckling of composite parabolic deep arches using feature-attention-enhanced LSTM networks

**Journal:** Composite Structures

---

We sincerely thank both reviewers and the editor for their careful reading and constructive comments. The manuscript has been substantially revised to address all concerns. Below, we provide a point-by-point response. Reviewer comments are in *italic*; our responses follow in plain text. All revised or newly added manuscript content is quoted in blockquote format.

---

## Response to Reviewer #1

---

### Q1. Clarity of objectives and rationale

**(1) The authors may further clarify the novelty of the proposed FA-LSTM model compared with existing LSTM, GRU, or Transformer-based models.**

**Response:** We thank the reviewer for this suggestion. We have substantially expanded the novelty discussion in three locations:

*Introduction (Section 1):* A new paragraph has been added before the contribution list that explicitly positions FA-LSTM against existing sequence models along three dimensions: (i) full-path vs. scalar prediction, (ii) dynamic feature-attention fusion vs. static concatenation, and (iii) mechanics-motivated inductive biases (prior initialization, FiLM, state residual) vs. generic architectures.

> The proposed FA-LSTM framework differs from prior applications of deep learning to structural stability in three fundamental aspects. First, rather than predicting only scalar buckling loads, it reconstructs the complete nonlinear equilibrium path. Second, instead of treating structural parameters as static conditioning inputs appended to the time-varying state, the multi-head feature self-attention mechanism explicitly models high-order interactions among all 59 heterogeneous features. Third, unlike generic sequence models that are agnostic to structural mechanics, the FA-LSTM incorporates three mechanics-motivated inductive biases: prior-informed LSTM initialization, FiLM conditioning, and a state residual connection.

*Methodology (Section 3.3):* A design-principle paragraph has been added explaining why static–dynamic feature fusion through learned attention is superior to fixed concatenation for nonlinear structural response prediction.

*New baseline comparison (Section 4.7, Table X):* We have added a comprehensive comparison table with LSTM, TCN, and Transformer baselines trained under identical conditions (5 random seeds, matched parameter budgets, same data splits). The results demonstrate that all three generic architectures fail to produce meaningful buckling-load predictions (R²_cr < 0), while FA-LSTM achieves R²_cr = 0.9919, providing quantitative evidence for the architectural advantages claimed.

**(2) The engineering motivation could be strengthened by explaining more clearly how the proposed model can be used in practical design or analysis.**

**Response:** We have added explicit language in both the Introduction and Conclusions clarifying the practical engineering workflow enabled by the proposed framework:

> Once trained, the surrogate model can be deployed for rapid parametric studies, reliability assessments, and design optimization—tasks that would be computationally prohibitive if each configuration required a full nonlinear FE analysis (~10–15 minutes per case). The framework provides an approximately 300× speedup, reducing per-configuration analysis time to ~2 seconds on a single GPU, which is further reduced to ~0.6 seconds in batch mode.

---

### Q2. Replicability and reproducibility

**(1) Since all data are generated from ANSYS simulations, the FE model should be validated against analytical solutions, published results, or experimental data. A mesh convergence study should also be provided.**

**Response:** We have added a mesh convergence study in Section 3.1 (new Fig. X). A representative parabolic arch is analyzed at seven mesh densities (10 to 320 elements along the arch axis), with the first ten natural frequencies extracted via Lanczos modal analysis. All modal frequencies converge to within < 0.7% of the 320-element reference at the adopted mesh density of 160 elements, confirming sufficient discretization accuracy.

Regarding validation against published results: the FE modeling approach (SHELL181 elements, arc-length method, COMBIN14 spring boundaries) follows established practices validated in prior arch buckling studies, including Pi and Bradford [1], Sabale and Gopal [9], and Hu et al. [13]. The parameter ranges are selected to encompass configurations from these validated studies. We have added explicit citations to these validation references in Section 3.1.

**(2) The authors are encouraged to provide more information on the dataset and training process. If possible, the database or training code could be made available to improve reproducibility.**

**Response:** We have expanded Section 3.1 with additional details on the data generation procedure, including the Latin Hypercube Sampling (LHS) strategy, the 9-dimensional parameter space, and the number of samples per material type. The training hyperparameters are fully specified in Table 2. We commit to releasing the training code and trained model weights upon publication.

---

### Q3. Statistical analyses and reporting

**(1) The results are mainly based on one random data split. It is suggested to repeat the training and testing using several random splits or k-fold cross-validation and report the mean and standard deviation of the metrics.**

**Response:** This is an excellent suggestion that we have fully implemented. All experimental results in the revised manuscript are now reported as the mean ± standard deviation across 5 independent training runs with different random seeds (618, 42, 123, 2024, 9999). Each run uses a different random data split (70%/15%/15%) and different weight initialization.

The updated Table 3 now reads:

| Metric | Mean ± Std |
|--------|:----------:|
| Full-sequence R² | 0.9920 ± 0.0040 |
| Buckling-load R²_cr | 0.9919 ± 0.0073 |

The narrow standard deviations (< 0.5% of the mean R² values) confirm the stability and reproducibility of the reported performance. The ablation study (Table 5) and baseline comparison (Table X) are likewise reported with multi-seed statistics.

---

### Q4. Additional figures or tables

**(1) A table or figure showing FE validation and mesh convergence results should be added.**

**Response:** Added. See new Fig. X (mesh convergence: natural frequencies vs. number of elements) and the accompanying text in Section 3.1. See also our response to Q2(1).

**(2) A comparison table between FA-LSTM and representative baseline models, such as LSTM, GRU, and Transformer, would make the model performance more convincing.**

**Response:** Added as new Table X (Section 4.7). We have included LSTM, TCN, and Transformer baselines. The TCN (Temporal Convolutional Network) was selected over GRU as a more competitive convolutional baseline that has been demonstrated to outperform GRU on long-sequence benchmarks [Bai et al., 2018]. All baselines are trained and evaluated under identical conditions (5 seeds, matched parameter budgets, same data splits) and report mean ± standard deviation.

The results are striking: all three generic architectures achieve buckling-load R²_cr near or below zero, while FA-LSTM achieves 0.9919—a 17× reduction in buckling-load MAE compared with the best baseline (LSTM).

---

### Q6. Emphasizing strengths

**The authors could better emphasize the advantage of fusing static design parameters with time-varying load–displacement states.**

**Response:** We have substantially strengthened this discussion. A new paragraph in Section 3.3 explicitly articulates the design principle behind the dynamic feature fusion, and Section 5.5 (attention weight analysis) now includes a direct link between the observed attention transition (static → dynamic features as buckling approaches) and the architectural design. See also our response to Q1(1).

---

## Response to Reviewer #2

---

### Comment 1. Reference formatting

**Please ensure consistency in the reference formatting throughout the manuscript. Specifically, journal titles should be standardized to their abbreviated forms.**

**Response:** All journal titles have been standardized to ISO 4 abbreviated forms. We have verified consistency throughout the reference list.

---

### Comment 2. Physical or engineering basis for parameter ranges

**Fig. 3 presents the statistical distributions of dataset; however, the physical or engineering basis for these selected ranges and density functions remains unclear.**

**Response:** We have added a justification paragraph after Table 1 in Section 3.1, explaining the engineering rationale for each parameter range:

> The span range (0.5–3.0 m) corresponds to laboratory-scale specimens commonly used for experimental validation of arch buckling theories [7,8,25]. The rise-to-span ratios (1/13–1/3) span the transition from shallow to deep arches. The width-to-height (3–20) and slenderness ranges are consistent with prior parametric studies of composite arch buckling [16,17,20,21]. The elastic support coefficients span from nearly free to effectively fixed.

---

### Comment 3. Necessity and advantages of MHSA before LSTM

**The authors should elaborate on the necessity of introducing the multi-head self-attention mechanism prior to the LSTM layer, including its specific advantages and potential drawbacks. Additionally, can the authors demonstrate whether this particular configuration offers a tangible advantage over standard LSTMs or established attention-based RNN variants, such as DA-RNN?**

**Response:** We have addressed this comment through three complementary revisions:

*Architectural rationale (Section 3.3):* A new design-principle paragraph explains why attention across features (rather than across time steps, as in DA-RNN) is specifically appropriate for this problem: the 16 design parameters and 43 state variables are heterogeneous quantities whose mutual relevance changes as loading progresses. Feature-level attention enables the model to dynamically re-weight these features at each load step.

*Ablation evidence (Table 5, row "w/o FA"):* Removing the feature self-attention module degrades full-sequence MSE by 78.9% and buckling-load MSE_cr by 37.8%, providing direct quantitative evidence of its contribution.

*Baseline comparison (new Table X):* Even the full FA-LSTM architecture trained without scheduled sampling (pure TF) achieves R²_cr = 0.9913, while a standard LSTM under identical training achieves R²_cr = −0.061. This 1.05 difference in R²_cr isolates the architectural contribution from the training strategy.

Regarding DA-RNN specifically: DA-RNN [Qin et al., 2017] employs temporal attention over time steps followed by input attention over features. Our architecture differs in two key respects: (i) attention operates over feature tokens (not time steps), which is more appropriate when the sequence length is fixed and the features are heterogeneous; (ii) the attention mechanism is complemented by FiLM conditioning and prior-informed initialization, which inject structural mechanics priors that a generic attention mechanism does not provide. We have added a brief discussion noting this distinction.

---

### Comment 4. Quantitative parameter analysis

**While Fig. 15 illustrates the relative weights of the parameters, it lacks quantitative analysis. It is recommended to supplement this with a quantitative analysis of how the critical load varies with individual parameters.**

**Response:** We thank the reviewer for this insightful suggestion. We have added a new parameter sensitivity analysis (Section 5.7, new Fig. Y) that uses the trained FA-LSTM as a surrogate to quantify how the critical buckling load varies with each independent physical parameter.

Specifically, a baseline arch configuration is defined, and each of seven independent physical parameters (L, f/L, λ, b/h, E, and elastic support coefficients) is swept across its dataset range while maintaining geometric and material consistency among the 16 derived features. The dominant parameters—span L and Young's modulus E—produce physically consistent monotonic trends: q_cr decreases with increasing L (consistent with reduced structural stiffness for longer spans) and increases with E (consistent with the direct proportionality between material stiffness and buckling capacity).

We also provide an honest assessment of the method's limitations: the model's sensitivity to boundary spring coefficients is less reliable because the training data exhibits a step-saturation characteristic (pinned vs. effectively fixed) with sparse sampling of the transitional regime, causing the model to learn the KZ–q_cr relationship through correlated geometric features rather than as an independent causal mapping. This limitation is discussed in the expanded conclusions.

---

### Comment 5 (Q7). Limitations

**The authors need to state the limitations of the present FA-LSTM framework.**

**Response:** The limitations paragraph in the Conclusions has been substantially expanded to address five specific limitations: (i) restriction to 2D in-plane buckling, (ii) fixed discretization assumptions, (iii) extrapolation reliability for boundary-condition parameters (supported by the parameter sensitivity analysis), (iv) absence of experimental validation, and (v) lack of explicit physics-informed constraints. Each limitation is accompanied by a concrete direction for future work.

---

### Comment 6 (Q9). Language editing

**Yes.**

**Response:** A full-pass language edit has been performed throughout the manuscript. Key improvements include: consistent tense usage, removal of redundant phrasing, standardization of technical terminology, and improved flow in the Introduction and Methodology sections.

---

## Summary of Revisions

| Reviewer | Point | Action |
|----------|-------|--------|
| R1-Q1(1) | Clarify novelty vs LSTM/GRU/Transformer | Expanded Intro + added baseline comparison table |
| R1-Q1(2) | Strengthen engineering motivation | Added practical workflow description |
| R1-Q2(1) | FE validation + mesh convergence | Added modal convergence study (new Fig. X) |
| R1-Q2(2) | Dataset/training details + code availability | Expanded Section 3.1; committed to code release |
| R1-Q3(1) | Multi-seed mean ± std | All Tables updated to 5-seed statistics |
| R1-Q4(1) | FE validation figure/table | Added mesh convergence figure |
| R1-Q4(2) | Baseline comparison table | Added new Table X (LSTM, TCN, Transformer) |
| R1-Q6 | Emphasize feature fusion strength | Added design-principle paragraph in Section 3.3 |
| R2-Q1 | Reference formatting | Standardized all journal titles to ISO 4 |
| R2-Q2 | Justify parameter ranges | Added engineering rationale after Table 1 |
| R2-Q3 | MHSA necessity + advantages | Added architectural rationale + DA-RNN comparison |
| R2-Q4 | Quantitative parameter analysis | Added surrogate-based sensitivity study (new Fig. Y) |
| R2-Q7 | Limitations | Expanded limitations paragraph (5 items) |
| R2-Q9 | Language editing | Full-pass polish |

---

We believe the revised manuscript has been substantially strengthened and now adequately addresses all concerns raised by both reviewers. We thank the editor and reviewers again for their time and constructive feedback.
