# CausFlow

## Causal Disentanglement and Conditional Flow Matching for Single-Cell Drug Perturbation Prediction

---

## Introduction

Single-cell perturbation prediction aims to infer transcriptomic responses under unseen drug, dose, and cellular conditions. Existing approaches often struggle under distribution shift because drug effects, cellular identity, dose, and technical variation are entangled within observed gene expression.

**CausFlow** addresses this problem through a unified causal generative framework combining:

* **Structural Causal Modeling (SCM)** for disentangled, intervention-stable latent representation learning.
* **Conditional Flow Matching (CFM)** for continuous counterfactual transcriptome generation.
* **Causal Perturbation Magnitude (CPM)** for phenotype-oriented drug response prioritization.

The model supports both **in-distribution (ID)** and **out-of-distribution (OOD)** perturbation prediction across multiple single-cell benchmarks and downstream colorectal cancer drug-response validation.

---

# Graph and Strategy

## Graph Structure

<div align="center">

![CausFlow Architecture](img/Overall_architecture_of_CausFlow.png)

**Figure 1. Overall architecture of CausFlow.**

</div>

CausFlow is a unified framework that combines **causal disentanglement** and **conditional flow matching** for counterfactual single-cell transcriptome generation. Given log-normalized scRNA-seq profiles and biological condition annotations, the model first encodes observations into latent representations using a variational causal encoder. These latent variables are then propagated through a directed acyclic graph (DAG), which explicitly models causal relationships among biological factors such as cell state, perturbation condition, dose intensity, and transcriptional response. By separating intervention-sensitive and intervention-invariant factors, the model learns a structured latent space that supports biologically meaningful counterfactual reasoning.

The causal latent representation serves as the conditioning signal for the flow-matching generator. Instead of directly predicting expression values, CausFlow learns a continuous velocity field that transports cells from a prior state toward a target perturbation state. Cross-attention modules are employed to integrate information from multiple causal factors while preserving context-specific biological dependencies. This design enables the model to capture nonlinear transcriptomic transitions and generate realistic expression trajectories under both observed and unseen perturbation conditions.

The complete computational graph integrates causal reasoning and generative modeling into a single end-to-end architecture. The DAG provides structural constraints that improve interpretability and factor disentanglement, while the flow-matching module ensures smooth and high-fidelity transcriptomic generation. Additional regularization components, including adversarial supervision, sparsity constraints, and latent consistency objectives, further enhance robustness and generalization. Together, these components allow CausFlow to perform interpolation, extrapolation, and zero-shot perturbation prediction within a unified framework.

---

## Strategy

### 1. Dataset Preparation

All datasets were processed using a unified single-cell analysis pipeline. Raw count matrices underwent quality control, normalization, highly variable gene selection, and dimensionality reduction using Scanpy. Biological annotations were converted into concept labels representing different sources of variation, including cell identity, tissue region, developmental stage, perturbation condition, treatment response, and technical batch effects.

The benchmark datasets used in this study include:

| Dataset              | Task                              |
| -------------------- | --------------------------------- |
| Immune Atlas         | Concept disentanglement           |
| MERFISH-Brain        | Spatial transcriptomics           |
| Spatiotemporal Liver | Spatiotemporal modeling           |
| Limb Development     | Developmental trajectory modeling |
| ICI Response         | Clinical response prediction      |
| A549 Perturbation    | Drug perturbation pretraining     |
| CRC Validation       | Zero-shot drug screening          |

---

### 2. Concept-Based Data Splitting

To evaluate both interpolation and extrapolation capabilities, CausFlow adopts two complementary evaluation protocols:

#### In-Distribution (ID) Evaluation

For ID evaluation, all biological concept combinations are observed during training. Training and testing sets share identical concept structures, while individual cells are separated into different partitions. This setting evaluates whether the model can accurately interpolate among known biological conditions.

```text
Observed Combination
      │
 ┌────┴────┐
Training  Testing
 Different cells
 Same concept space
```

Characteristics:

* All concept combinations appear during training.
* Test cells are held out at the sample level.
* Evaluates interpolation performance.
* Measures reconstruction fidelity and latent consistency.

---

#### Out-of-Distribution (OOD) Evaluation

For OOD evaluation, selected concept combinations are completely removed from training and reserved exclusively for testing. The model must infer transcriptomic responses under previously unseen biological conditions.

```text
Observed Combinations ──► Training

Held-out Combinations ──► Testing
```

Characteristics:

* Entire concept combinations are hidden during training.
* Tests causal extrapolation ability.
* Evaluates robustness under distribution shift.
* Assesses disentanglement quality and transferability.

---

### 3. Training Strategy

CausFlow is trained end-to-end using a multi-objective optimization framework.

#### Optimization Settings

| Hyperparameter        | Value   |
| --------------------- | ------- |
| Optimizer             | Adam    |
| Learning Rate         | 5e-6    |
| Batch Size            | 64      |
| Training Steps        | 100,000 |
| Gradient Accumulation | 2       |
| EMA Decay             | 0.995   |
| Random Seed           | 888     |
| Latent Dimension      | 32      |
| ODE Integration Steps | 100     |

#### Model Architecture

| Component              | Configuration             |
| ---------------------- | ------------------------- |
| Backbone               | Conditional Flow Matching |
| Encoder                | Disentanglement Encoder   |
| Velocity Network       | MLP + Cross-Attention     |
| Cross-Attention Layers | 4                         |
| Attention Heads        | 4                         |
| Head Dimension         | 64                        |
| Activation             | Mish / GELU               |
| Normalization          | LayerNorm                 |
| Causal Prior           | Fixed DAG Matrix          |

---

### 4. Loss Functions

The training objective combines multiple complementary losses:

| Loss Component          | Weight |
| ----------------------- | ------ |
| Flow Matching Loss      | 1.0    |
| Concept Prediction Loss | 20.0   |
| Discriminator Loss      | 1.0    |
| KL Prior Loss           | 0.5    |
| Reconstruction Loss     | 1.0    |
| Sparsity Regularization | 1e-4   |

The flow-matching objective learns continuous transcriptomic transport, while auxiliary losses encourage factor disentanglement, causal consistency, and biological interpretability.

---

### 5. Causal Graph Construction

The causal graph is represented as a directed acyclic graph (DAG) whose nodes correspond to biological concepts and latent factors.

Typical nodes include:

* Cell Intrinsic State
* Drug Perturbation
* Dose Intensity
* Treatment Response
* Biological Context
* Transcriptomic Output

Edges represent hypothesized causal influences derived from biological prior knowledge.The sample data is as follows.

```text
Cell State
     │
     ▼
Drug Perturbation ──► Dose Effect
     │                    │
     ▼                    ▼
Response Program ──► Transcriptome
```

To prevent information leakage, latent factors connected by no causal edge are explicitly regularized toward independence. During counterfactual inference, interventions are performed by modifying perturbation-related latent variables while preserving background cellular states.

---

### 6. Evaluation Strategy

Model performance is evaluated under both ID and OOD settings using five complementary metrics.

| Metric       | Description                    |
| ------------ | ------------------------------ |
| PCC          | Expression trend preservation  |
| MSE          | Reconstruction error           |
| ARI          | Clustering consistency         |
| NMI          | Cluster structure preservation |
| Marker Score | Biological signal recovery     |

---

### 7. Stability and Statistical Analysis

To assess reproducibility, all experiments are repeated using five independent random seeds.

For each metric, we report:

* Mean ± Standard Deviation (SD)
* 95% Confidence Interval (CI)
* Seed-wise performance distribution

The confidence interval is calculated as:

CI95 = mean ± 1.96 × SD / √n

where n = 5 independent runs.

Statistical significance between CausFlow and competing methods is evaluated using paired Wilcoxon signed-rank tests followed by Benjamini–Hochberg false discovery rate correction.

---

### 8. Zero-Shot Drug Prediction Strategy

For zero-shot drug screening, CausFlow is first pretrained on A549 perturbation data containing known drug-response profiles. Unseen compounds are subsequently mapped into the learned perturbation space through mechanism-of-action (MoA) proxy assignments.

The workflow consists of three stages:

```text
A549 Perturbation Pretraining
            │
            ▼
MoA-Based Drug Mapping
            │
            ▼
Counterfactual Transcriptome Generation
            │
            ▼
CPM Score Calculation
            │
            ▼
Drug Ranking
```

The resulting latent perturbation displacement is summarized using the **Causal Perturbation Magnitude (CPM)** score, which provides a quantitative estimate of intervention strength and supports downstream drug prioritization across different biological backgrounds.


## Data Availability

All datasets used in this study are publicly available in https://drive.google.com/drive/folders/1kBnC7z5DrFzdgGEGcC4HVMjZIxH72LgU?usp=drive_link.
