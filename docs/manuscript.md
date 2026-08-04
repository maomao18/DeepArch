Hao Tang^a^, Airong Liu^a\*^, Jie Yang^b\*^, Jialin Wang^a^, Shimao Qin^c^

^a^ Research Center for Wind Engineering and Engineering Vibration, Guangzhou University, Guangzhou 510006, China

^b^ School of Engineering, RMIT University, PO Box 71, Bundoora, VIC 3083, Australia

^c^ Municipal Facilities Management Center of Guangzhou Development District, Guangzhou 510700, China

\*Corresponding author. liuar\@gzhu.edu.cn; j.yang\@rmit.edu.au

**Abstract:** This paper proposes a feature-attention-enhanced long short-term memory (FA-LSTM) framework for data-driven prediction of the complete nonlinear in-plane buckling response of parabolic deep arches made of homogeneous, laminated-composite, and functionally graded materials (FGMs). A unified design parameter vector integrating equivalent sectional stiffnesses, geometric parameters, and non-dimensional elastic support coefficients is formulated to represent the three material systems within a single feature space. A multi-head feature self-attention module is employed to fuse the static design parameters with the time-varying load--displacement state at each load step, enabling the network to adaptively emphasize the most informative features throughout the buckling process. A joint loss function combining stepwise-increment and absolute-state supervision, together with a dual-head decoder and a curriculum-based training strategy, is adopted to suppress error accumulation during autoregressive prediction. Trained and tested on 2513 nonlinear finite element (FE) simulations, the proposed model achieves an overall coefficient of determination (R²) above 0.99 and a buckling-load R² of 0.996, while reducing the computational cost by approximately two orders of magnitude compared with nonlinear FE analysis. The framework provides an efficient surrogate tool for the buckling analysis and design of parabolic deep arches.

**Keywords:** Parabolic deep arch; Nonlinear in-plane buckling; Composite material; Functionally graded material; Feature self-attention; LSTM; Data-driven prediction

**Highlights**

• A unified parameterization is proposed for homogeneous, composite, and FGM arches.

• An FA-LSTM architecture is proposed for full nonlinear buckling path prediction.

• A joint loss with curriculum training is proposed to suppress error accumulation.

• The model achieves R² \> 0.99 with approximately300× speedup over nonlinear FE analysis.

# Introduction

Arches are one of the most efficient structural forms, transferring loads primarily through axial compression, which results in reduced bending moments and enhanced spanning capability. Under in-plane loading, arches may undergo nonlinear limit-point (snap-through) or bifurcation buckling \[1--6\]. The parabolic profile closely approximates the funicular curve under full-span uniformly distributed vertical loads, thereby minimizing bending moments and providing a clear and efficient load-transfer path. Owing to these mechanical advantages, parabolic arches have been widely adopted in long-span roofs, bridges, and underground structures \[7,8\]. As the rise-to-span ratio increases, the structure enters a deep-arch regime in which geometric nonlinearity, compression--bending coupling, and large-deflection effects become dominant. Slender parabolic deep arches are susceptible to strongly nonlinear in-plane buckling, typically manifested as stiffness degradation and displacement jumps (snap-through/snap-back) in the load--displacement response, leading to a sudden loss of load-carrying capacity \[9,10\]. With the increasing use of laminated composites and functionally graded materials (FGMs) in modern arch structures \[11\], material heterogeneity introduces additional complexity, including axial--bending coupling and through-thickness property gradients, which further complicates the buckling behavior. Consequently, accurate and efficient prediction of the nonlinear buckling response and critical load of parabolic deep arches, for both homogeneous and composite systems, is essential for safety assessment and design optimization.

The in-plane buckling of parabolic arches has been extensively studied using both analytical and numerical approaches. Pi and Bradford \[1\] established a theoretical framework for arch stability, and subsequent studies extended the analysis to shallow parabolic arches under various loading and boundary conditions \[2--6\]. Bradford et al. \[7,8\] further investigated arches with horizontal spring supports through combined theoretical and experimental approaches. For deep parabolic arches, where geometric nonlinearity and large-deflection effects become dominant, using a geometrically exact beam formulation, Sabale and Gopal \[9\] demonstrated that equilibrium paths and buckling modes are highly sensitive to boundary conditions. Numerical studies based on geometrically nonlinear finite element (FE) models have further identified transitions between different buckling modes and multiple critical states under varying support stiffness \[12,13\]. In addition, several studies have examined more complex arch configurations, including concrete-filled steel tubular arches \[14\] and layered arches with interlayer slip \[15\], highlighting the influence of material and interfacial properties on buckling behavior. However, these investigations have primarily focused on homogeneous or quasi-homogeneous metallic or concrete arches, whereas systematic studies on composite parabolic arches remain limited.

With the growing application of advanced composites and FGMs in arch structures, the nonlinear buckling behavior of material-heterogeneous arches has attracted increasing attention. A comprehensive review was provided by Hurdoganoglu et al. \[11\]. Existing studies have mainly focused on laminated and functionally graded shallow arches with circular geometries. Analytical investigations have examined the effects of loading conditions, shear deformation, and material gradation on buckling behavior \[16--19\]. More advanced material systems, such as graphene-reinforced and carbon nanotube-reinforced composites, have also been explored, highlighting the significant influence of reinforcement distribution and material heterogeneity on structural stability \[20--24\]. In addition, experimental studies have provided benchmark data for arches with elastic restraints \[25\]. Despite these advances, current analytical and semi-analytical studies are largely restricted to circular shallow arches under idealized boundary conditions. Nonlinear buckling analyses of parabolic deep arches composed of laminated composites or FGMs, particularly under realistic elastic boundary restraints, remain limited. Furthermore, no unified framework has been established to simultaneously account for multiple material systems within a single predictive model. This gap constitutes the primary motivation of the present study.

The analytical and semi-analytical approaches reviewed above, while effective for slender shallow arches under idealized conditions, have limited applicability to deep arches with large rise-to-span ratios, where second-order geometric effects, shear deformation, and membrane action become significant. Bradford et al. \[10\] showed that shallow-arch approximations may substantially overestimate buckling loads when the rise-to-span ratio exceeds a moderate threshold, indicating that their extension to deep arches is nontrivial. Consequently, finite element analysis, typically employing arc-length methods with Newton--Raphson iterations and geometrically exact beam or shell formulations, has become the primary tool for investigating the buckling and postbuckling behavior of parabolic deep arches \[6,9,26--28\]. Although robust in tracing equilibrium paths, such methods are computationally expensive, limiting their applicability in large-scale parametric studies and real-time design. Moreover, existing studies predominantly focus on critical load prediction for specific configurations and rarely provide efficient tools for capturing the full load--displacement response, including stiffness degradation and displacement jumps throughout the buckling process. These limitations motivate the development of computationally efficient data-driven approaches.

In recent years, machine learning (ML) methods have emerged as promising surrogates for computationally expensive structural simulations \[29--33\]. Once trained, an ML model can produce predictions within seconds, in contrast to nonlinear FE analysis that requires a costly iterative arc-length procedure for each new configuration. This computational advantage enables large-scale parametric studies, reliability assessments, and design optimizations. Neural networks have been successfully applied to predict critical buckling loads of various structural components, including steel columns \[34\], CFST members \[35\], composite panels \[36\], and perforated steel beams \[37,38\], as well as composite laminates under mechanical and thermal loads \[39,40\]. However, these studies are limited to scalar load prediction and do not capture the full load--displacement evolution that characterizes deep arch buckling. The nonlinear buckling of deep arches is inherently a path-dependent sequence: the displacement response at each load step is governed by the entire history of applied loading and accumulated deformation. Recurrent architectures, particularly long short-term memory networks (LSTMs) \[41,42\] and Transformers \[43\], are specifically designed to capture such long-range temporal dependencies \[44,45\] , and recent studies have applied them to structural field reconstruction, defect identification, and nonlinear response prediction of structural members \[46--49\]. In addition, machine-learning-based approaches have been explored for stability analysis of composite arch structures \[50\]. Despite these advances, the application of sequence learning models to predict the complete nonlinear buckling path of arch structures---including stiffness degradation, load peaks, and displacement jumps---remains unexplored. It should be emphasized that training such a surrogate model requires a substantial number of high-fidelity FE simulations as an upfront computational investment; this cost, however, is incurred only once during offline data generation, yielding efficiency gains of approximately two orders of magnitude per prediction during deployment.

To address this gap, this paper proposes a feature-attention-enhanced LSTM (FA-LSTM) framework for data-driven prediction of the nonlinear in-plane buckling response of parabolic deep arches under uniformly distributed vertical loading. Given only the design parameters and an initial zero-displacement state, the model autoregressively reconstructs the full load--displacement history and identifies the critical buckling load. The main contributions are summarized as follows:

\(1\) A unified design parameter vector is formulated to represent homogeneous, laminated-composite, and FGM parabolic arches within a single feature space using equivalent sectional stiffnesses, geometric parameters, and non-dimensional elastic support coefficients.

\(2\) A feature-attention-enhanced LSTM architecture is developed to capture interactions between static design parameters and time-varying load--displacement states, integrating multi-head feature attention, prior-informed initialization, Feature-wise Linear Modulation (FiLM) \[51\] conditioning, and a dual-head decoder.

\(3\) A joint loss function combining increment and absolute-state supervision, together with a curriculum training strategy from teacher forcing \[52\] to fully autoregressive learning, is proposed to mitigate error accumulation in autoregressive prediction.

\(4\) Based on 2513 nonlinear FE simulations, the model achieves a full-sequence R² above 0.99 and a buckling-load R~cr~² of 0.996, with a computational speedup of approximately two orders of magnitude compared with nonlinear FE analysis.

# Composite parabolic deep arches

This section formulates the nonlinear in-plane buckling of parabolic deep arches as a discrete sequence prediction problem. A parabolic arch subjected to a full-span uniformly distributed vertical load is shown in Fig. 1(a). The arch has a span $L$, a rise $f$, and a rectangular cross-section with width $b$ and height $h$, and the material is assumed to be uniform along the arch axis. Under the applied load $q$, the in-plane displacement of a point on the arch axis is denoted by $(u,\ v)$ in the Cartesian coordinate system. The deformation is approximated using $N$ control points distributed along the arch axis, and the global response is subsequently represented by the response vector formed by the displacements at all control points.

To ensure generality, three material systems are considered, namely homogeneous, laminated-composite, and FGM arches, as shown in Fig. 1(b--d). Four equivalent cross-sectional quantities are introduced to represent these systems within a unified parameter space, including the equivalent sectional mass per unit length $I_{0}$, the axial stiffness $A_{11}$, the axial--bending coupling stiffness $B_{11}$, and the bending stiffness $D_{11}$.

For a homogeneous parabolic arch with density $\rho$, Young\'s modulus $E$, cross-sectional width $b$, and height $h$, the sectional parameters are obtained by direct integration over the rectangular cross-section:

![图示 AI 生成的内容可能不正确。](media/image2.tiff){width="5.905555555555556in" height="2.952777777777778in"}

Fig. 1. (a) Parabolic deep arch under a full-span uniformly distributed vertical load with elastic end restraints; (b) homogeneous rectangular cross-section; (c) laminated-composite cross-section; (d) functionally graded material (FGM) cross-section with through-thickness property gradient.

For a laminated-composite parabolic arch consisting of $n$ layers, the equivalent sectional parameters are obtained by summing the contributions from each layer:

where $\rho_{k}$, $E_{k}$, $h_{k}$, and $z_{k}$ denote the density, Young\'s modulus, thickness, and centroidal coordinate of the $k$-th layer, respectively.

For a functionally graded material (FGM) parabolic arch, the material properties vary continuously through the thickness. The equivalent sectional parameters are obtained by integration over the cross-section:

For homogeneous arches, the axial--bending coupling stiffness vanishes, $B_{11} = 0$, since the neutral axis coincides with the geometric centroid. In contrast, laminated composite and FGM arches generally exhibit nonzero $B_{11}$ due to material asymmetry, which introduces coupling between axial and bending deformations and significantly influences the buckling behavior.

The geometric properties of the arch are described by the span $L$, rise $f$, cross-sectional width $b$ and height $h$, arch length $S$, and slenderness ratio $\lambda$, defined as

where $\sqrt{\frac{D_{11}}{A_{11}}}$ represents the equivalent radius of gyration of the cross-section. The arc length $S$ is computed from the parabolic geometry as

The boundary conditions at the arch supports are described by six non-dimensional elastic support coefficients. For the left and right supports, the horizontal ($\eta_{L1}$, $\eta_{R1}$), vertical ($\eta_{L2}$, $\eta_{R2}$), and rotational ($\eta_{L3}$, $\eta_{R3}$) coefficients are defined as

where $k_{\text{Li}}$ and $k_{Ri}$ denote the corresponding spring stiffnesses. The normalization ensures that the coefficients reflect the relative stiffness of the elastic restraints with respect to the stiffness of the arch.

Based on the above definitions, each parabolic arch is represented by a 16-dimensional design parameter vector:

Under a uniformly distributed vertical load $q$, the structural state at a load step is defined as an $\left( 2N + 1 \right)$-dimensional vector:

where the tilde denotes non-dimensional quantities. As the load increases, the structural response evolves as a sequence $\begin{pmatrix}
\begin{matrix}
\mathbf{D}_{0} & \mathbf{D}_{1} \\
\end{matrix} & \begin{matrix}
\cdots & \mathbf{D}_{T} \\
\end{matrix} \\
\end{pmatrix}$, where $\mathbf{D}_{0}$ corresponds to the undeformed state. The nonlinear buckling process is thus formulated as a discrete sequential evolution problem, in which the state at step *t* is approximated by:

# Methodology

This section presents the methodology of the proposed FA-LSTM framework, including the generation of the FE dataset, the network architecture, the training strategy, and the evaluation procedure.

## FE Data Generation

A shell finite element model of the parabolic deep arch is established in ANSYS Mechanical APDL, as illustrated in Fig. 2. The arch geometry is defined by fitting a B-spline through key points on the parabolic axis and discretized using four-node SHELL181 elements with a layered section definition. Laminated-composite arches are modeled with five layers; while FGM arches are approximated using ten layers to represent the through-thickness property gradient, following established convergence practices \[20,21,24\]. The mesh consists of 160 elements along the arch axis and 4 segments across the width, which is verified to provide converged buckling loads. Boundary conditions at the two supports are implemented using translational and rotational springs (COMBIN14), with the stiffness values determined from the non-dimensional coefficients in Eq. (6). A uniformly distributed vertical load is applied, and only in-plane deformation is considered. Nonlinear buckling analysis is performed using the arc-length method to capture the complete equilibrium path, including limit points and snap-back behavior.

A total of 2513 parabolic arch models are generated through a parametric procedure, with material and geometric parameters randomly sampled within the ranges specified in Table 1. The dataset includes 97 homogeneous arches, 1516 laminated composite arches, and 900 FGM arches. For laminated composite arches, the material properties of each layer are obtained by scaling the base values with random coefficients uniformly distributed in \[0.85, 1.15\]. For FGM arches, the through-thickness property distributions are defined using prescribed functions applied to the base values. All models are automatically analyzed in ANSYS, and the load--displacement responses at 21 uniformly distributed control points along the arch axis are extracted.

Table 1. Parameter ranges for the parametric generation of parabolic arch FE models.

  FE model parameter                                                Range
  ----------------------------------------------------------------- ---------------------------------------------------------
  Young's modulus $E$ (GPa)                                         $$\left\lbrack 60,\ 210 \right\rbrack$$
  Density $\rho$ ($\frac{\text{kg}}{m^{3}}$)                        $$\left\lbrack 800,2500 \right\rbrack$$
  Poisson's ratio $\mu$                                             $$\left\lbrack 0.2,0.4 \right\rbrack$$
  Span $L$ (m)                                                      $$\left\lbrack 0.5,1 \right\rbrack$$
  Rise-to-span ratio $\frac{f}{L}$                                  $$\left\lbrack \frac{1}{13},\frac{1}{3} \right\rbrack$$
  Width-to-height ratio $\frac{b}{h}$                               $$\left\lbrack 3,20 \right\rbrack$$
  Slenderness ratio $\lambda$                                       $$\left\lbrack 150,1800 \right\rbrack$$
  Horizontal elastic support coefficients $\eta_{L1}$,$\eta_{R1}$   $$\left\lbrack 0.1,10 \right\rbrack$$
  Vertical elastic support coefficients $\eta_{L2}$,$\eta_{R2}$     $$\left\lbrack 0.1,10 \right\rbrack$$
  Rotational elastic support coefficients $\eta_{L3}$,$\eta_{R3}$   $$\left\lbrack 0,1000 \right\rbrack$$

![图示 AI 生成的内容可能不正确。](media/image11.tiff){width="3.435416666666667in" height="2.5833333333333335in"}

Fig. 2. Finite element model of the parabolic deep arch with SHELL181 elements.

The raw FE output sequences are resampled to a fixed length of $T = 200$ load steps using linear interpolation along the arc-length parameter, with the initial zero-displacement state retained as the first entry. The load and displacement components are non-dimensionalized as

where $D_{11}$ is the bending stiffness and $L$ is the span of the arch. The state vector $\mathbf{D}_{t}$ at each load step is thus expressed in non-dimensional form. The material and geometric parameters are mapped to the design vector $\mathbf{B}$, and the resulting dataset consists of 2513 paired sequences $\left\{ \left\lbrack \mathbf{B}^{\left( j \right)},\left( \mathbf{D}_{t}^{\left( j \right)} \right)_{t = 0}^{T} \right\rbrack \right\}_{j = 1}^{2513}$. The dataset is randomly divided into training, validation, and test sets in a ratio of 70%/15%/15% to prevent data leakage.

## Data Analysis

A statistical overview of the generated dataset is presented in Fig. 3, showing the distributions of the key design parameters and the corresponding nonlinear critical buckling loads for the 2513 parabolic deep arch models. The critical load ${\widetilde{q}}_{\text{cr}}$ is defined as the non-dimensional load at which the tangent stiffness of the mid-span load--displacement curve first reaches zero, corresponding to a limit-point instability.

The sampled parameters are consistent with the prescribed ranges in Table 1 and provide broad coverage of geometric configurations and material systems, including cases with significant axial--bending coupling.

![图表, 条形图, 直方图 AI 生成的内容可能不正确。](media/image14.tiff){width="5.905555555555556in" height="3.8006944444444444in"}

Fig. 3 Statistical distributions of the design parameters and critical buckling loads in the generated dataset of 2513 parabolic deep arches

The Pearson correlation coefficients between each component of $\mathbf{B}$ and ${\widetilde{q}}_{\text{cr}}$ are presented in Fig. 4. The span $L$ and the arch length $S$ exhibit the strongest negative correlations, indicating the dominant influence of geometric scale on the buckling capacity. Due to multicollinearity among the design parameters, these coefficients reflect marginal linear relationships and should not be interpreted as causal measures.

![图表, 条形图 AI 生成的内容可能不正确。](media/image15.tiff){width="3.15in" height="3.15in"}

Fig. 4 Pearson correlation between characteristics of parabolic arches and critical buckling loads

## Feature-Attention Enhanced LSTM Architecture

The nonlinear buckling response of a parabolic deep arch exhibits strong history dependence, where the load--displacement state $\mathbf{D}_{t}$​ at each step is governed by the preceding states. This sequential evolution, particularly during the transition from prebuckling to instability, involves progressive stiffness degradation and complex path-dependent behavior. Long short-term memory (LSTM) networks \[41,42\] are well suited for modeling such temporal dependencies due to their gated recurrent structure, which enables effective information propagation over long sequences.

Instead of reiterating the standard LSTM formulation, the present study focuses on enhancing its capability to incorporate structural design information. In addition to temporal dependencies, the buckling response is strongly influenced by static design parameters $\mathbf{B}$. To explicitly model the interactions between static features and time-varying states, a multi-head feature self-attention mechanism is integrated ahead of the LSTM.

The overall architecture of the proposed feature-attention-enhanced LSTM (FA-LSTM) model is illustrated in Fig. 5. The model consists of five main components: an input layer, an encoding layer (FeatureTokenizer), a multi-head feature self-attention module with attention pooling, a stacked LSTM with prior-informed initialization, and a dual-head decoding layer. The design parameter vector $\mathbf{B}$ is treated as static input, while the load--displacement state $\mathbf{D}_{t}$ serves as dynamic input that evolves with the load process.

Instead of directly predicting the absolute state, the model is trained to predict the state increment $\Delta\mathbf{D}_{t} = \mathbf{D}_{t} - \mathbf{D}_{t - 1}$, and the absolute response is recovered through cumulative summation. This incremental formulation improves the model's sensitivity to local stiffness variations and enhances its ability to capture nonlinear buckling features such as stiffness degradation and displacement jumps.

![图示 AI 生成的内容可能不正确。](media/image16.tiff){width="5.636805555555555in" height="4.094444444444444in"}

Fig. 5 Architecture of the proposed feature-attention-enhanced LSTM (FA-LSTM) network for nonlinear buckling prediction of parabolic deep arches.

In the encoding layer, all input features are standardized using z-score normalization. A FeatureTokenizer module maps each scalar physical feature into a 512-dimensional embedding space via a feature-wise linear projection followed by Layer Normalization. Two independent FeatureTokenizer modules are applied to the 16 static and 43 dynamic features, respectively, producing a sequence of 59 feature tokens.

To capture interactions between heterogeneous features, a multi-head feature self-attention module is employed to model high-order correlations among the combined static and dynamic tokens. The token sequence is projected into query, key, and value representations, and attention weights are computed to adaptively aggregate information across features.

Unlike conventional sequence attention that operates along temporal or spatial dimensions, the proposed mechanism performs attention across feature dimensions, enabling explicit modeling of the coupling between design parameters and evolving structural states.

The outputs of all attention heads are projected back to the embedding space, followed by a residual connection and LayerNorm. The number of heads and embedding dimension are determined through hyperparameter optimization to balance representational capacity and computational efficiency.

To compress the fused feature-token sequence into a compact representation suitable for LSTM input, an attention pooling module is introduced. This module adaptively aggregates the 59 feature tokens into a single 512-dimensional vector by learning their relative importance. Specifically, each token is assigned a learnable importance score, which is normalized across all tokens to produce attention weights. The pooled representation is then obtained as a weighted sum of the token embeddings.

This mechanism enables the model to dynamically focus on the most relevant features at each load step, allowing the contribution of static design parameters and evolving structural states to vary throughout the loading process. As a result, the model can better capture critical nonlinear behaviors such as stiffness degradation and displacement jumps.

To enhance the adaptability of the model to different arch configurations, the pooled feature representation is further modulated using Feature-wise Linear Modulation \[51\] (FiLM) conditioned on the design parameter vector $\mathbf{B}$. This conditioning mechanism enables static design parameters to directly influence the representation of dynamic features, allowing the model to explicitly capture the coupling between structural configuration and nonlinear response evolution. The FiLM module generates feature-wise scaling and shifting factors from $\mathbf{B}$, which modulate the representation as

The modulated feature sequence is then fed into a stacked LSTM network to model the temporal evolution of the load--displacement response. The recurrent structure enables the model to capture both smooth prebuckling behavior and abrupt transitions near instability.

To explicitly incorporate structural prior information into temporal modeling, the static design parameters are mapped to the initial hidden state and cell state of the LSTM.

In the decoding stage, a dual-head mechanism is adopted to account for the distinct characteristics of load and displacement responses, as illustrated in Fig. 6. The model predicts the state increment $\Delta\mathbf{D}_{t}$, which consists of both load and displacement components, providing a unified representation of the nonlinear response evolution

Additionally, a state residual connection is introduced to provide a shortcut from the current state to the predicted increment, facilitating stable training and improving the modeling of smooth prebuckling behavior.

![图示 AI 生成的内容可能不正确。](media/image19.tiff){width="2.65625in" height="1.9361111111111111in"}

Fig. 6. Structure of the dual-head decoder: displacement decoder (2-layer MLP) and load decoder (3-layer MLP).

## Training Strategy and Hyperparameter Optimization

As described above, the FA-LSTM model is trained using full dynamic sequence ${\{\mathbf{D}_{t}\}}_{t = 0}^{T}$, whereas during inference only the design parameter vector $\mathbf{B}$ and the initial state $\mathbf{D}_{0}$ are available, requiring autoregressive prediction.

To mitigate this train--inference discrepancy, a curriculum learning strategy is adopted, gradually transitioning the model from teacher-forced training to fully autoregressive prediction. The training process begins with ground-truth inputs at all steps, followed by a scheduled sampling strategy \[52\] phase in which the teacher forcing ratio is progressively reduced, and finally converges to fully autoregressive training where predictions are recursively fed back as inputs.

This progressive scheme stabilizes training while improving robustness to error accumulation during inference.

The key hyperparameters are optimized using the Optuna framework \[53\] with a Tree-structured Parzen Estimator (TPE) sampler over 30 trials, using the validation autoregressive loss as the objective. Early stopping is applied via a median pruner to eliminate unpromising trials, and the resulting optimal configurations are summarized in Table 2.

Table 2. Hyperparameter search space and optimal values identified by Optuna.

  Hyperparameter                                       Search space                                                  Optimal value
  ---------------------------------------------------- ------------------------------------------------------------- --------------------------
  Hidden dimension                                     $$\left\{ 128,256,384,512,640 \right\}$$                      512
  Number of LSTM layers                                $$\left\lbrack 1,\ 6 \right\rbrack$$                          4
  Number of attention heads                            $$\left\{ 2,\ 4,\ 8,\ 16 \right\}$$                           4
  Dropout rate                                         $$\left\lbrack 0,\ 0.35 \right\rbrack$$                       0.186
  Learning rate                                        $$\left\lbrack 10^{- 5},\ 5 \times 10^{- 4} \right\rbrack$$   $$3.05 \times 10^{- 4}$$
  Teacher-training epochs                              $$\left\lbrack 20,\ 120 \right\rbrack$$                       40
  Transition epochs                                    $$\left\lbrack 10,\ 80 \right\rbrack$$                        40
  Increment loss weight $\text{ω\!}_{\Delta}$          $$\left\lbrack 0.2,\ 0.8 \right\rbrack$$                      0.436
  Absolute-state loss weight $\omega_{\text{abs}}$​    $$1 - \text{ω\!}_{\Delta}$$                                   0.564
  Load task weight $\text{λ\!}_{\text{load}}$          $$\left\lbrack 0.2,\ 0.8 \right\rbrack$$                      0.209
  Displacement task weight ${\lambda}_{\text{disp}}$   $$1 - \text{λ\!}_{\text{load}}$$                              0.791

The optimized loss weights indicate that the absolute-state loss ($\omega_{\text{abs}} = 0.564$) slightly outweighs the increment loss ($\text{ω\!}_{\Delta} = 0.436$), highlighting the importance of maintaining global trajectory consistency. The displacement-related loss is assigned a significantly larger weight ($\text{λ\!}_{\text{disp}} = 0.791$) than the load component ($\text{λ\!}_{\text{load}} = 0.209$), reflecting the higher dimensionality and greater complexity of the displacement response. The model is trained using AdamW \[54\] with cosine annealing warm restarts and gradient clipping to ensure stable optimization.

## Loss Function

A joint multi-task loss function is formulated in the standardized feature space, comprising an increment loss $\mathcal{L}_{\Delta}$ and an absolute-state loss $\mathcal{L}_{\text{abs}}$.

The increment loss focuses on accurately capturing local load--displacement evolution, whereas the absolute-state loss enforces global trajectory consistency, thereby mitigating error accumulation during autoregressive prediction.

The increment loss is further decomposed into load and displacement components with task-specific weights:

The Huber loss is adopted as the base function to improve robustness, combining the advantages of MSE for small residuals and MAE for large deviations. The loss terms are defined as

To emphasize the early-step deformation behavior, an early-step weighting strategy is introduced:

where the weighting factor decays smoothly from $\omega_{\max}$ to unity, enhancing sensitivity to initial deformation while avoiding instability caused by abrupt weight transitions.

The absolute-state loss adopts the same task decomposition:

and is defined using the mean squared error to better capture smooth global responses:

## Indicators for model evaluation

Two categories of evaluation metrics are adopted to assess the performance of the FA-LSTM model: (i) full-sequence metrics that quantify prediction accuracy across all load steps and state-vector components, and (ii) buckling-load metrics that evaluate the accuracy of the predicted critical load.

Full-sequence metrics are computed over all state variables including the non-dimensional load and displacement components at each load step, and are averaged over all samples. The coefficient of determination is defined as

where ${\widehat{\mathbf{D}}}_{t}$ and $\mathbf{D}_{t}$ denote the predicted and ground-truth state vectors, respectively, and $\overline{\mathbf{D}}$ is the temporal mean of the ground-truth sequence. The mean absolute error (MAE) and mean squared error (MSE) are defined as

where ${\parallel \cdot \parallel}_{1}$ and ${\parallel \cdot \parallel}_{2}$ denote the $L1$ and $L2$ norms, respectively.

Buckling-load metrics are used to evaluate the accuracy of the predicted critical load ${\widehat{\widetilde{q}}}_{\text{cr}}$, defined as the load corresponding to the first vanishing tangent stiffness of the mid-span response, as defined in Section 3.2. The following metrics are adopted:

These buckling-load metrics are of particular engineering significance, as the critical load directly determines the safety margin and load-carrying capacity of the arch.

# Results

## Experimental setup

The dataset consisting of 2513 FE simulations is randomly divided into training, validation, and test sets with a ratio of 70%/15%/15% (1759/377/377 samples), as described in Section 3.1. The FA-LSTM model is implemented in PyTorch and trained on a single NVIDIA RTX 4090 GPU with 24 GB memory. The training hyperparameters are summarized in Table 2. During validation and testing, the model operates in a fully autoregressive manner and is initialized only with the design parameter vector $\mathbf{B}$ and the zero-displacement state $\mathbf{D}_{0}$.

## Training convergence

The training and validation loss curves over 150 epochs are presented in Fig. 7. The training loss converges within approximately 100 epochs. The validation loss, evaluated under fully autoregressive rollout, exhibits larger fluctuations during the early training stage but decreases consistently as the curriculum progresses into the mixed and autoregressive phases (epochs 41--150). The periodic spikes observed near the learning-rate reset epochs are a known characteristic of warm restarts \[55\], which help the optimizer escape local minima. The close agreement between the training and validation losses at the end of training indicates that the three-phase curriculum strategy effectively mitigates the gap between training and autoregressive inference.

![](media/image32.tiff){width="3.15in" height="3.15in"}

Fig. 7. Training and validation loss curves over 150 epochs

## Overall prediction accuracy

The trained FA-LSTM model is evaluated on the 377-sample test set under fully autoregressive conditions, and the overall evaluation metrics are summarized in Table 3. The $R^{2}$ values exceed 0.99 for both full-sequence and buckling-load predictions, while the buckling-load $R_{\text{cr}}^{2}$ reaches 0.996. These results indicate that the model not only reproduces the overall load--displacement trajectory with high accuracy but also effectively identifies the critical instability point.

Table 3. Overall evaluation metrics of the FA-LSTM model on the test set (377 samples).

  Metric                                   value
  ---------------------------------------- -----------
  Full-sequence MAE                        $$0.027$$
  Full-sequence MSE                        $$0.057$$
  Full-sequence $R^{2}$                    $$0.997$$
  Buckling-load $\text{MAE}_{\text{cr}}$   $$1.074$$
  Buckling-load $\text{MSE}_{\text{cr}}$   $$2.651$$
  Buckling-load $R_{\text{cr}}^{2}$        $$0.996$$

## Full-sequence prediction examples

Representative mid-span (crown) load--displacement curves are compared between the FA-LSTM predictions and the FE results for selected test samples covering different material types, rise-to-span ratios, and boundary conditions.

Fig. 8 presents the mid-span load--displacement curves for three arch samples with different rise-to-span ratios ($f/L\  \approx$ 0.10, 0.20, and 0.30). The FA-LSTM predictions closely follow the FE results throughout the prebuckling stage and accurately capture both the location and magnitude of the load peak (critical buckling load), as well as the postbuckling snap-through behavior. The agreement remains satisfactory even for the deepest arch ($f/L \approx$ 0.30), which exhibits the strongest geometric nonlinearity.

![](media/image33.tiff){width="5.905555555555556in" height="2.029166666666667in"}

Fig. 8. Comparison of FA-LSTM predictions and FE results for mid-span load--displacement curves of parabolic arches with three different rise-to-span ratios.(a)$\frac{f}{L} \approx 0.1$;(b) $\frac{f}{L} \approx 0.2$; (c) $\frac{f}{L} \approx 0.3$

![图表, 直方图 AI 生成的内容可能不正确。](media/image34.tiff){width="5.905555555555556in" height="2.029166666666667in"}

Fig. 9. Comparison of FA-LSTM predictions and FE results for mid-span load--displacement curves of laminated-composite parabolic arches with different axial--bending coupling stiffnesses: (a)$B_{11} = - 847.5$;(b) $B_{11} = 14100.4$; (c) $B_{11} = - 1752.1$

Fig. 9 presents the corresponding results for three laminated-composite arch samples. Despite the additional complexity introduced by the nonzero axial-bending coupling stiffness $\mathbf{B}_{11}$, the model reproduces the load--displacement paths with comparable accuracy, indicating that the unified design parameter representation effectively captures the influence of material heterogeneity. Minor deviations between the FA-LSTM predictions and FE results can be observed, particularly in the postbuckling stage for cases with strong coupling stiffness. Two factors contribute to these discrepancies. First, the training dataset consists of 2513 samples in total, of which approximately 1759 were used for training; this sample size, while sufficient for overall accuracy, may limit generalization to specific combinations of high coupling stiffness and geometric parameters at the boundaries of the parameter space. Second, despite the joint loss formulation and curriculum training strategy designed to suppress error accumulation, the autoregressive prediction over 200 load steps inevitably introduces cumulative errors, as each prediction step feeds its own output back as input for the next step. This effect is more pronounced in the postbuckling regime, where the load--displacement path undergoes abrupt changes.

## Buckling-load prediction accuracy

The accuracy of the predicted critical buckling loads is further evaluated by comparing the FA-LSTM predictions with the FE ground-truth values for all 377 test samples. Fig. 10 plots the predicted non-dimensional critical buckling load ${\widehat{\widetilde{q}}}_{\text{cr}}$ against the FE value ${\widetilde{q}}_{\text{cr}}$. The data points cluster closely around the 45° identity line ($y\  = \ x$), indicating high prediction accuracy across the full range of buckling loads in the dataset. The high $R_{\text{cr}}^{2}$ value of 0.996 and the narrow scatter band further demonstrate the strong generalization capability of the model for unseen arch configurations.

Fig. 11(a) presents the probability density distribution of the relative prediction error of the critical buckling load, defined as $e_{\text{cr}} = \frac{\left( {\widehat{\widetilde{q}}}_{\text{cr}} - {\widetilde{q}}_{\text{cr}} \right)}{{\widetilde{q}}_{\text{cr}}}$. The error distribution is approximately symmetric about zero, with a mean of 2.714%, a median of 2.149%, and a standard deviation of 2.321%. Fig. 11(b) shows the number of test samples within different relative error ranges. More than 87% of the test samples exhibit a relative error below 5%, only about 1% exceed 10% under extreme cases, and all samples remain within 15%.

![图表 AI 生成的内容可能不正确。](media/image35.jpeg){width="3.15in" height="3.15in"}

Fig. 10. Predicted versus FE non-dimensional critical buckling load for all 377 test samples.

![图表, 条形图, 直方图 AI 生成的内容可能不正确。](media/image36.tiff){width="5.905555555555556in" height="2.952777777777778in"}

Fig. 11. Distribution of the relative prediction error of the critical buckling load over the 377 test samples: (a) probability density distribution; (b) number of samples within different error ranges.

## Deformation prediction at buckling

Fig. 12 compares the predicted deformed shapes of four representative arches at the critical buckling point with the corresponding FE results. The selected samples cover different boundary conditions across the three material types. In all cases, the FA-LSTM-predicted deformed shapes agree closely with the FE results, indicating that the model captures not only the scalar critical buckling load but also the spatial distribution of the buckling deformation.

![图表, 折线图 AI 生成的内容可能不正确。](media/image37.tiff){width="5.905555555555556in" height="5.905555555555556in"}

Fig. 12. Comparison of predicted and FE buckling deformation shapes for four representative test samples

## Computational efficiency

Table 4 compares the computational time of the nonlinear FE method and the FA-LSTM model on the same hardware platform. For a single arch, the FA-LSTM model completes the 200-step prediction in approximately 2 s, corresponding to a speedup of about 300 times compared with nonlinear FE analysis. In batch mode (batch size = 16), the per-sample computational cost further decreases to approximately 0.6 s owing to GPU parallelism. This computational advantage makes the proposed framework well suited for large-scale applications such as parametric sensitivity analysis, reliability assessment, and design optimization.

Table 4. Computational cost comparison between nonlinear FE analysis and FA-LSTM inference.

+---------------------------------------+----------+----------------+----------------------+
| Method                                | Hardware | Time per case  | Speedup              |
+=======================================+==========+================+======================+
| Nonlinear FE (ANSYS, arc-length)      | CPU      | 10-15 min      | $1 \times$(baseline) |
+---------------------------------------+----------+----------------+----------------------+
| FA-LSTM (single-sample inference)     | GPU      | 2 s            | $$300 \times$$       |
+---------------------------------------+----------+----------------+----------------------+
| FA-LSTM (batch inference, 16 samples) | GPU      | 10 s           | $$960 \times$$       |
|                                       |          |                |                      |
|                                       |          | (0.6 s/sample) |                      |
+---------------------------------------+----------+----------------+----------------------+

# Ablation Studies

## Ablation Overview

To quantify the contributions of key architectural and training components in the FA-LSTM model, a series of ablation experiments are conducted, where each module is individually removed or replaced while all other settings are kept unchanged. All ablated models are retrained from scratch under the same data split and training schedule, and evaluated on the test set using fully autoregressive inference. The results are summarized in Table 5.

Table 5. Ablation study results on the test set (377 samples).

  Configuration        $$R^{2}$$   MAE     Change   MSE     Change   $$R_{\text{cr}}^{2}$$   MAE~cr~   Change   MSE~cr~   Change
  -------------------- ----------- ------- -------- ------- -------- ----------------------- --------- -------- --------- --------
  FA-LSTM              0.997       0.027   \-       0.057   \-       0.996                   1.074     \-       2.651     \-
  w/o FA               0.993       0.039   +44.4%   0.102   +78.9%   0.994                   1.140     +6.15%   3.654     +37.8%
  w/o AttnPool         0.995       0.034   +25.9%   0.096   +68.4%   0.993                   1.373     +27.8%   4.256     +60.5%
  w/o FiLM             0.995       0.038   +40.7%   0.108   +89.5%   0.993                   1.288     +19.9%   4.343     +63.8%
  w/o Prior            0.995       0.034   +25.9%   0.098   +71.9%   0.989                   1.681     +56.5%   6.744     +154%
  w/o State residual   0.993       0.047   +74.1%   0.143   +151%    0.995                   1.185     +10.3%   2.894     +9.17%
  Single-head output   0.993       0.043   +59.3%   0.157   +175%    0.995                   1.263     +17.6%   3.379     +27.5%

As shown in Table 5, all ablated variants exhibit consistent performance degradation compared with the full FA-LSTM model. Although the reduction in R² is relatively small (from 0.997 to 0.993--0.995), both MAE and MSE increase significantly, indicating a noticeable deterioration in prediction accuracy. This suggests that the removed components play a critical role in reducing large prediction errors, even when the overall trajectory shape is largely preserved.

In particular, the ablation results highlight that these modules primarily enhance model robustness under challenging conditions, such as deep arches near the buckling transition, rather than merely improving average-case performance.

## Feature-Attention

To evaluate the contribution of the multi-head feature self-attention module, the attention weights are fixed to uniform values while all other components remain trainable. The resulting performance degradation is substantial: the full-sequence MSE increases by 78.9%, and R² drops from 0.997 to 0.993 (Table 5).

Fig. 13 provides a representative comparison of mid-span load--displacement curves. Without the attention mechanism, the model fails to accurately capture the sharp load peak and subsequent snap-through behavior, producing a delayed and overly smoothed response that deviates from the FE reference. These results confirm that the feature self-attention module is essential for highlighting informative features and suppressing irrelevant ones, enabling more discriminative inputs to the LSTM at each load step. This improved feature weighting significantly enhances the model's sensitivity to buckling onset.

![图表, 直方图 AI 生成的内容可能不正确。](media/image38.tiff){width="3.15in" height="3.15in"}

Fig. 13. Effect of the feature self-attention module on the mid-span load--displacement prediction for a representative test sample

## Effect of prior-informed initialization

To assess the effect of the prior-informed LSTM initialization, the MLP that maps the design parameter vector **B** to the initial hidden and cell states is removed, and the states are initialized to zero. As shown in Table 5, removing this initialization leads to a significant degradation in performance: the full-sequence MSE increases by 71.9% and the buckling-load MAE~cr~ increases by 56.5%. Although the overall R² remains above 0.99, the notable increase in absolute error---particularly for the buckling load---indicates that prior-informed initialization provides a physically meaningful starting point for the temporal prediction. Notably, the impact on the buckling-load error (+154% in MSE~cr~) is substantially larger than that on the full-sequence MSE (+71.9%), highlighting that prior information is especially critical near the instability point, where accurate anticipation of the upcoming buckling behavior is essential.

Fig. 14 further illustrates this effect through a representative mid-span load--displacement response. Without prior-informed initialization, the model exhibits larger deviations during the early prebuckling stage, which accumulate and propagate during the autoregressive rollout. This results in a delayed and overestimated prediction of the buckling load, with a noticeable lag in capturing the onset of instability.

![图表 AI 生成的内容可能不正确。](media/image39.tiff){width="3.15in" height="3.15in"}

Fig. 14. Effect of prior-informed LSTM initialization on the mid-span load--displacement prediction for a representative test sample

## Effect of the dual-head decoder

To evaluate the effect of the dual-head decoding mechanism, the two separate decoder branches are replaced with a single shared 3-layer MLP that jointly predicts all 43 state-vector components. As shown in Table 5, this modification leads to a substantial performance degradation: the full-sequence MSE increases by 175%, and the buckling-load MAE~cr~ increases by 17.6%. These results indicate that the load and displacement outputs exhibit fundamentally different characteristics in terms of dimensionality, scale, and nonlinear behavior near the buckling point, and are therefore better handled by separate decoding paths. The shared decoder suffers from gradient competition between the two heterogeneous tasks, where the high-dimensional displacement output dominates parameter updates, leading to insufficient learning of the one-dimensional load signal. In contrast, the dual-head design alleviates this imbalance by assigning task-specific parameters, enabling more accurate simultaneous prediction of the load peak and displacement jump near the critical point.

## Attention weight analysis

To further examine the physical interpretability of the learned feature self-attention mechanism, the attention pooling weights are extracted from the trained FA-LSTM model and averaged over all test samples. Fig. 15(a) shows the mean attention weights across different feature categories. Deformation-related features receive the highest weights, indicating that the model predominantly relies on the deformation history of the arch to predict subsequent responses and buckling loads. Significant attention is also assigned to material stiffness parameters and the slenderness ratio, highlighting their critical roles in governing the buckling behavior. These results demonstrate that the model automatically identifies and prioritizes physically meaningful features, consistent with established structural mechanics principles.

![图表 AI 生成的内容可能不正确。](media/image40.tiff){width="5.905555555555556in" height="2.952777777777778in"}

Fig. 15. Attention weight analysis: (a) mean attention pooling weights for different feature categories averaged over all test samples and load steps; (b) evolution of attention weights with load step t for the 16 static features, load, and deformation.

Fig. 15(b) illustrates the evolution of the attention weights over the 16 static features, the load, and the deformation along the loading process. In the initial prebuckling stage, where both load and deformation remain small, the attention is predominantly assigned to material properties and geometric parameters (e.g., span $L$ and rise $f$), which govern the linear elastic response of the arch. As the load approaches the critical buckling point, the attention progressively shifts toward the load level and deformation states, indicating that the model increasingly relies on state-dependent signals to capture the imminent instability. Near $\text{t\ } \approx \ 120$, the attention weights on material and geometric features increase again, suggesting that the model begins to identify transition cues associated with the onset of postbuckling strengthening. Once the arch fully enters the postbuckling regime, the attention distribution shifts again, reflecting the changing dominance of different physical factors across deformation stages. This dynamic attention evolution aligns well with the underlying mechanics of structural response, providing an interpretable explanation of the model\'s internal decision-making process and enhancing confidence in its predictions.

# Conclusions

This paper proposed a feature-attention-enhanced LSTM (FA-LSTM) framework for data-driven prediction of the nonlinear in-plane buckling response of parabolic deep arches composed of homogeneous, laminated composite, and functionally graded materials. The main findings are summarized as follows:

\(1\) A unified design parameter vector, incorporating equivalent sectional stiffnesses, geometric parameters, and non-dimensional elastic support coefficients, is formulated to represent multiple material systems within a single feature space. This representation provides a scalable and consistent foundation for data-driven buckling analysis of arch structures.

\(2\) The proposed FA-LSTM architecture, which integrates multi-head feature self-attention, prior-informed LSTM initialization, FiLM conditioning, and a dual-head decoder, is systematically optimized via 30 Optuna TPE trials. Ablation studies demonstrate that each component contributes significantly to performance. Specifically, feature self-attention reduces the full-sequence MSE by 78.9%, prior-informed initialization reduces the buckling-load MAE~cr~ by 56.5%, and the dual-head decoder reduces the full-sequence MSE by 175%.

\(3\) A joint increment and absolute loss formulation, combined with a curriculum-based training strategy including teacher, mixed, and autoregressive phases (40, 40, and 70 epochs, respectively), effectively mitigates error accumulation during long-horizon autoregressive prediction over 200 load steps.

\(4\) Evaluated on 377 unseen configurations, the model achieves a full-sequence R² above 0.99 and a buckling-load R² of 0.996, while providing approximately 300 times computational speedup compared with nonlinear finite element analysis. These results demonstrate its strong potential for large-scale parametric studies and rapid design assessment of parabolic deep arches.

Several limitations remain. The current framework is restricted to two-dimensional in-plane buckling, and extension to out-of-plane instability would require both enriched datasets and modified model architectures. The use of fixed discretization, including control points and load steps, limits flexibility for variable-resolution analysis. In addition, the extrapolation capability beyond the training parameter space (Table 1) has not been systematically investigated. Future work will focus on incorporating physics-informed constraints and adaptive discretization strategies to further improve robustness and generalizability.

**Acknowledgments**

This work is financially supported by National Natural Science Foundation of China (No. 52279127), Technology Planning Project of Guangzhou City (No. 20212200004), 111 Project (No. D21021), Guangdong Provincial Key Laboratory of Earthquake Engineering and Applied Technology, Key Laboratory of Earthquake Resistance, Earthquake Mitigation and Structural Safety, Ministry of Education.

**References**

\[1\] Pi Y-L, Bradford MA, Uy B. In-plane stability of arches. Int J Solids Struct 2002;39:105--25. https://doi.org/10.1016/S0020-7683(01)00209-8.\[2\] Cai J, Feng J, Chen Y, Huang L. In-plane elastic stability of fixed parabolic shallow arches. Sci China Ser E: Technol Sci 2009;52:596--602. https://doi.org/10.1007/s11431-009-0057-9.\[3\] Moon J, Yoon K-Y, Lee T-H, Lee H-E. In-plane elastic buckling of pin-ended shallow parabolic arches. Eng Struct 2007;29:2611--7. https://doi.org/10.1016/j.engstruct.2007.01.004.\[4\] Fan L, Zhang Y, Zhuk Y, Goroshko I, Sareh P. Nonlinear in-plane buckling of shallow parabolic arches with tension cables under step loads. Arch Appl Mech 2022;92:335--49. https://doi.org/10.1007/s00419-021-02060-7.\[5\] Cai J, Xu Y, Feng J, Zhang J. In-plane elastic buckling of shallow parabolic arches under an external load and temperature changes. J Struct Eng 2012;138:1300--9. https://doi.org/10.1061/(ASCE)ST.1943-541X.0000570.\[6\] Moon J, Yoon K-Y, Lee T-H, Lee H-E. In-plane strength and design of parabolic arches. Eng Struct 2009;31:444--54. https://doi.org/10.1016/j.engstruct.2008.09.009.\[7\] Bradford MA, Wang T, Pi Y-L, Gilbert RI. In-plane stability of parabolic arches with horizontal spring supports. I: Theory. J Struct Eng 2007;133:1130--7. https://doi.org/10.1061/(ASCE)0733-9445(2007)133:8(1130).\[8\] Wang T, Bradford MA, Gilbert RI, Pi Y-L. In-plane stability of parabolic arches with horizontal spring supports. II: Experiments. J Struct Eng 2007;133:1138--45. https://doi.org/10.1061/(ASCE)0733-9445(2007)133:8(1138).\[9\] Sabale A, Gopal KN. Nonlinear in-plane stability of deep parabolic arches using geometrically exact beam theory. Int J Struct Stab Dyn 2018;18:1850006. https://doi.org/10.1142/S0219455418500062.\[10\] Bradford MA, Pi Y-L, Yang G, Fan X-C. Effects of approximations on non-linear in-plane elastic buckling and postbuckling analyses of shallow parabolic arches. Eng Struct 2015;101:58--67. https://doi.org/10.1016/j.engstruct.2015.07.008.\[11\] Hurdoganoglu D, Safaei B, Sahmani S, Onyibo EC, Qin Z. State-of-the-art review of computational static and dynamic behaviors of small-scaled functionally graded multilayer shallow arch structures from design to analysis. Arch Comput Methods Eng 2024;31:389--453. https://doi.org/10.1007/s11831-023-09983-0.\[12\] Han Q, Cheng Y, Lu Y, Li T, Lu P. Nonlinear buckling analysis of shallow arches with elastic horizontal supports. Thin Walled Struct 2016;109:88--102. https://doi.org/10.1016/j.tws.2016.09.016.\[13\] Hu C-F, Pi Y-L, Gao W, Li L. In-plane non-linear elastic stability of parabolic arches with different rise-to-span ratios. Thin Walled Struct 2018;129:74--84. https://doi.org/10.1016/j.tws.2018.03.019.\[14\] Han X, Wei C, Hu Q, Liu C, Wang Y. In-plane nonlinear buckling analysis and design method of concrete-filled steel tubular catenary arches. J Constr Steel Res 2024;214:108485. https://doi.org/10.1016/j.jcsr.2024.108485.\[15\] Adam C, Paulmichl I, Furtmüller T. Buckling and post-buckling analysis of three-layer shallow arches with geometric imperfections and interlayer slip. Thin Walled Struct 2023;193:111220. https://doi.org/10.1016/j.tws.2023.111220.\[16\] Zhang Z, Liu A, Yang J, Huang Y. Nonlinear in-plane elastic buckling of a laminated circular shallow arch subjected to a central concentrated load. International Journal of Mechanical Sciences 2019;161:105023.\[17\] Zhang Z, Liu A, Yang J, Pi Y, Huang Y, Fu J. Nonlinear in-plane buckling of shallow laminated arches incorporating shear deformation under a uniform radial loading. Compos Struct 2020;252:112732. https://doi.org/10.1016/j.compstruct.2020.112732.\[18\] Atacan AT, Yükseler RF. Nonlinear buckling and post-buckling analyses of functionally graded circular shallow arches. Proc Inst Mech Eng C: J Mech Eng Sci 2024;238:4768--89. https://doi.org/10.1177/09544062231209146.\[19\] Bateni M, Eslami M. Non-linear in-plane stability analysis of FG circular shallow arches under uniform radial pressure. Thin Walled Struct 2015;94:302--13. https://doi.org/10.1016/j.tws.2015.04.019.\[20\] Yang Z, Yang J, Liu A, Fu J. Nonlinear in-plane instability of functionally graded multilayer graphene reinforced composite shallow arches. Compos Struct 2018;204:301--12. https://doi.org/10.1016/j.compstruct.2018.07.072.\[21\] Yang Z, Huang Y, Liu A, Fu J, Wu D. Nonlinear in-plane buckling of fixed shallow functionally graded graphene reinforced composite arches subjected to mechanical and thermal loading. Appl Math Modell 2019;70:315--27. https://doi.org/10.1016/j.apm.2019.01.024.\[22\] Yang Z, Liu A, Yang J, Lai S-K, Lv J, Fu J. Analytical prediction for nonlinear buckling of elastically supported fg-gplrc arches under a central point load. Materials 2021;14:2026. https://doi.org/10.3390/ma14082026.\[23\] Li C, Zhu C, Lim C, Li S. Nonlinear in-plane thermal buckling of rotationally restrained functionally graded carbon nanotube reinforced composite shallow arches under uniform radial loading. Appl Math Mech 2022;43:1821--40. https://doi.org/10.1007/s10483-022-2917-7.\[24\] Zhao S, Yang Z, Kitipornchai S, Yang J. Dynamic instability of functionally graded porous arches reinforced by graphene platelets. Thin Walled Struct 2020;147:106491. https://doi.org/10.1016/j.tws.2019.106491.\[25\] Lu Y, Cheng Y, Han Q. Experimental investigation into the in-plane buckling and ultimate resistance of circular steel arches with elastic horizontal and rotational end restraints. Thin Walled Struct 2017;118:164--80. https://doi.org/10.1016/j.tws.2017.05.010.\[26\] Tang Z, Zhang W, Yu J, Pospı́šil S. Prediction of the elastoplastic in-plane buckling of parabolic steel arch bridges. J Constr Steel Res 2020;168:105988. https://doi.org/10.1016/j.jcsr.2020.105988.\[27\] Temel B, Noori AR. Transient analysis of laminated composite parabolic arches of uniform thickness. Mech Based Des Struct Mach 2019. https://doi.org/10.1080/15397734.2019.1572518.\[28\] Hu C-F, Li Z, Liu Z-W, Chen S-S. In-plane non-linear elastic stability of arches subjected to multi-pattern distributed load. Thin Walled Struct 2020;154:106810. https://doi.org/10.1016/j.tws.2020.106810.\[29\] Mandal P, Adil MT, Naz F, others. Application of artificial neural network to predict buckling load of thin cylindrical shells under axial compression. Eng Struct 2021;248:113221.\[30\] Li W, Bazant MZ, Zhu J. A physics-guided neural network framework for elastic plates: Comparison of governing equations-based and energy-based approaches. Comput Methods Appl Mech Eng 2021;383:113933. https://doi.org/10.1016/j.cma.2021.113933.\[31\] Haghighat E, Raissi M, Moure A, Gomez H, Juanes R. A physics-informed deep learning framework for inversion and surrogate modeling in solid mechanics. Comput Methods Appl Mech Eng 2021;379:113741. https://doi.org/10.1016/j.cma.2021.113741.\[32\] Kaveh A, Eskandari A, Movasat M. Buckling resistance prediction of high-strength steel columns using metaheuristic-trained artificial neural networks. Structures, vol. 56, Elsevier; 2023, p. 104853. https://doi.org/10.1016/j.istruc.2023.07.043.\[33\] Thai H-T. Machine learning for structural engineering: a state-of-the-art review. Structures, vol. 38, Elsevier; 2022, p. 448--91.\[34\] Ly H-B, Le LM, Duong HT, Nguyen TC, Pham TA, Le T-T, et al. Hybrid artificial intelligence approaches for predicting critical buckling load of structural members under compression considering the influence of initial geometric imperfections. Appl Sci 2019;9:2258. https://doi.org/10.3390/app9112258.\[35\] Asteris PG, Lemonis ME, Le T-T, Tsavdaridis KD. Evaluation of the ultimate eccentric load of rectangular CFSTs using advanced neural network modeling. Eng Struct 2021;248:113297. https://doi.org/10.1016/j.engstruct.2021.113297.\[36\] Mallela UK, Upadhyay A. Buckling load prediction of laminated composite stiffened panels subjected to in-plane shear using artificial neural networks. Thin Walled Struct 2016;102:158--64. https://doi.org/10.1016/j.tws.2016.01.025.\[37\] Abambres M, Rajana K, Tsavdaridis KD, Ribeiro TP. Neural network-based formula for the buckling load prediction of I-section cellular steel beams. Computers 2019;8:2. https://doi.org/10.3390/computers8010002.\[38\] Degtyarev VV, Tsavdaridis KD. Buckling and ultimate load prediction models for perforated steel beams using machine learning algorithms. J Build Eng 2022;51:104316. https://doi.org/10.1016/j.jobe.2022.104316.\[39\] Ahmed OS, Ali JSM, Aabid A, Hrairi M, Yatim NM. Parametric analysis of critical buckling in composite laminate structures under mechanical and thermal loads: a finite element and machine learning approach. Materials 2024;17:4367. https://doi.org/10.3390/ma17174367.\[40\] Zeinali M, Rahimi G, Hosseini S. Estimation of buckling load in sandwich beams with novel honeycomb core: a non-destructive approach enhanced by machine learning and genetic algorithm optimization. Int J Struct Stab Dyn 2025:2650248. https://doi.org/10.1142/S0219455426502482.\[41\] Hochreiter S, Schmidhuber J. Long short-term memory. Neural Comput 1997;9:1735--80. https://doi.org/10.1162/neco.1997.9.8.1735.\[42\] Gers FA, Schmidhuber J, Cummins F. Learning to forget: continual prediction with LSTM. Neural Comput 2000;12:2451--71. https://doi.org/10.1162/089976600300015015.\[43\] Vaswani A, Shazeer N, Parmar N, Uszkoreit J, Jones L, Gomez AN, et al. Attention is all you need. Adv Neural Inf Process Syst 2017;30.\[44\] Lipton ZC, Berkowitz J, Elkan C. A critical review of recurrent neural networks for sequence learning. arXiv Prepr arXiv:1506,00019 2015.\[45\] Wen X, Li W. Time series prediction based on LSTM-attention-LSTM model. IEEE Access : Pract Innov Open Solut 2023;11:48322--31.\[46\] Zhang Z, Tian Z, Wu Z, Xu B. Development and analysis of a BP-LSTM-kriging temperature field prediction model for the arch ring section of the reinforced concrete arch bridge. Structures, vol. 64, Elsevier; 2024, p. 106564. https://doi.org/10.1016/j.istruc.2024.106564.\[47\] Yao M, Chen Z, Li J, Guan S, Tang Y. Ultrasonic identification of CFST debonding via A novel Bayesian Optimized-LSTM network. Mech Syst Signal Process 2025;238:113175. https://doi.org/10.1016/j.ymssp.2025.113175.\[48\] Grandı́o J, Barros B, Cabaleiro M, Riveiro B. Point transformer network-based surrogate model for spatial prediction in bridges. Infrastructures 2025;10:70. https://doi.org/10.3390/infrastructures10040070.\[49\] Yu Z, Li B. Reinforced concrete beam full response prediction with hybrid feature-orientation transformer-LSTM model. Eng Struct 2025;332:120040. https://doi.org/10.1016/j.engstruct.2025.120040.\[50\] Yang Z, Zhao S, Yang J, Liu A, Fu J. Thermomechanical in-plane dynamic instability of asymmetric restrained functionally graded graphene reinforced composite arches via machine learning-based models. Compos Struct 2023;308:116709. https://doi.org/10.1016/j.compstruct.2023.116709.\[51\] Perez E, Strub F, De Vries H, Dumoulin V, Courville A. Film: Visual reasoning with a general conditioning layer. Proceedings of the AAAI conference on artificial intelligence, vol. 32, 2018. https://doi.org/10.1609/aaai.v32i1.11671.\[52\] Bengio S, Vinyals O, Jaitly N, Shazeer N. Scheduled sampling for sequence prediction with recurrent neural networks. Adv Neural Inf Process Syst 2015;28.\[53\] Akiba T, Sano S, Yanase T, Ohta T, Koyama M. Optuna: a next-generation hyperparameter optimization framework. Proceedings of the 25th ACM SIGKDD international conference on knowledge discovery & data mining, 2019, p. 2623--31. https://doi.org/10.1145/3292500.3330701.\[54\] Loshchilov I, Hutter F. Decoupled weight decay regularization. arXiv Prepr arXiv:1711,05101 2017.\[55\] Loshchilov I, Hutter F. SGDR: stochastic gradient descent with restarts. CoRR 2016;abs/1608.3983.
