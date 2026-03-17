# CryoDECO

**Deconstructing Compositional and Conformational Heterogeneity in Cryo-EM with Foundation Model Priors**

[![License](https://img.shields.io/badge/license-see%20LICENSE-green.svg)](LICENSE.txt)
[![Preprint](https://img.shields.io/badge/preprint-LangTaoSha-orange.svg)](https://doi.org/10.65215/LTSpreprints.2025.12.30.000075)

CryoDECO is an *ab initio* heterogeneous reconstruction algorithm designed to resolve complex structural mixtures in cryo-EM data. By leveraging structural priors from the pre-trained [Cryo-IEF](https://github.com/westlake-repl/Cryo-IEF) foundation model, CryoDECO bypasses the random initialization bottleneck common in traditional deep learning approaches, enabling robust classification of both compositional and conformational heterogeneity.

---

## Table of Contents

1. [Installation](#installation)
2. [Configuration & Parameters](#configuration--parameters)
3. [Usage](#usage)
4. [Analysis & Output](#analysis--output)
5. [Acknowledgments](#acknowledgments)
6. [Citation](#citation)

---

## Installation

We recommend installing CryoDECO in a clean Conda environment to avoid dependency conflicts.

```bash
# 1. Create and activate a virtual environment
conda create --name cryoDECO python=3.9
conda activate cryoDECO

# 2. Clone the repository
git clone https://github.com/yanyang1998/CryoDECO.git
cd CryoDECO/

# 3. Install dependencies
pip install -r requirements.txt
```

> **[CryoData](https://github.com/yanyang1998/cryoief-data)** is used internally for cryo-EM data preprocessing (normalization, LMDB conversion, and PyTorch integration). It is included in `requirements.txt` and installed automatically.

---

## Configuration & Parameters

### Required Parameters

| Parameter | Description |
|---|---|
| `particles` | Path to the CryoSPARC job directory (supports `Downsample`, `Extracted Particles`, `Restack Particles`, and `Particle Sets` jobs) |
| `outdir` | Directory where all output results will be saved |
| `pretrained_model_path` | Path to Cryo-IEF model weights ([download from HuggingFace](https://huggingface.co/westlake-repl/Cryo-IEF/tree/main/cryo_ief_checkpoint/cryo_ief_v1.5_vit_s)) |

> **Recommendation:** For efficient training and inference, we strongly recommend **downsampling input particles to a box size of 128 pixels**. ([Guide: How to downsample in CryoSPARC](readmes/downsample.md))

```yaml
particles='/path/to/cryosparc_job/'           # Path to input particle job directory
outdir='/path/to/save/your/results/'          # Directory for output results
pretrained_model_path='/path/to/checkpoint/'  # Path to Cryo-IEF weights
```

---

### Optional Hyperparameters

#### Latent Dimension (`feature_dim`)

The latent dimension `z` is a hyperparameter that can be changed to achieve best classification performance:

| Heterogeneity Type | Recommended `feature_dim` | Rationale |
|---|---|---|
| Compositional (discrete states) | `128` | High dimensionality ensures orthogonality between disjoint structures |
| Conformational — simple motion | `4` | Low dimensionality enforces smoothness on a simple manifold |
| Conformational — complex motion | `64` | Intermediate dimension balances expressiveness and regularity |

```yaml
feature_dim=128   # Default: 128
```

#### Clustering

The pipeline clusters learned latent features to generate initial volumes.

| Parameter | Default | Description |
|---|---|---|
| `k_num` | `8` | Number of clusters. For compositional heterogeneity, set to the expected number of distinct species. For conformational, controls the number of maps sampled from latent space. |
| `clustering_type` | `'gmm'` | Clustering algorithm. `'gmm'` (Gaussian Mixture Model) is more accurate; `'k-means++'` is much faster. |

```yaml
k_num=8               # Default: 8
clustering_type='gmm' # Default: 'gmm'. Option: 'k-means++'
```

#### Using Known Poses (CryoDECO-pose)

By default, CryoDECO estimates particle poses during training. If high-quality poses are available from a prior CryoSPARC refinement or *ab initio* job, they can be used to improve reconstruction quality and accelerate convergence.

**Requirement:** The input `particles` job must contain pose information. Generate this by running a `Restack Particles` or `Downsample Particles` job in CryoSPARC connected to your prior refinement result.

```yaml
use_gt_poses=True  # Default: False. Set to True to use CryoSPARC poses.
use_gt_trans=True  # Default: False. Set to True to use CryoSPARC translations.
```

#### Optimization

Batch sizes are tuned for an **NVIDIA A40 (40 GB)**. Adjust based on your available GPU memory.

```yaml
epochs_sgd=100        # Default: 100. Decrease for very large datasets (>1M particles).
batch_size_hps=22     # Batch size for Hierarchical Pose Search (per GPU)
batch_size_sgd=192    # Batch size for SGD Refinement (per GPU)
```

---

## Usage

```bash
accelerate launch --mixed_precision=bf16 CryoDECO_run.py \
    --particles $particles \
    --outdir $outdir \
    --pretrained_model_path $pretrained_model_path \
    --feature_dim 128 \
    --k_num 8 \
    --clustering_type 'gmm' \
    --epochs_sgd 100
```

> By default, `accelerate` uses all available GPUs. To restrict to specific GPUs, use `--gpu_ids` (e.g., `--gpu_ids 0,1`) or `--num_processes`.

---

## Analysis & Output

Results are saved to `outdir/out/analysis_(epoch_number)/`.

### A. Compositional Heterogeneity (Discrete States)

Focus on the **Clustering** results:

| Output | Description |
|---|---|
| `clustering(knum)/*.mrc` | Reconstructed density maps for each cluster center |
| `clustering(knum)/umap_clusters.png` | UMAP visualization of latent features, colored by cluster |
| `clustering(knum)/clustering_cs_star/` | `.star` files for each cluster, importable into CryoSPARC |

**Workflow:** Import `.star` files into CryoSPARC to create particle subsets for high-resolution refinement. If you used downsampled particles, map subsets back to the original full-resolution particles first. ([Guide: How to map back particles](readmes/mapback.md))

### B. Conformational Heterogeneity (Continuous Motion)

Focus on the **PCA Traversal** results:

| Output | Description |
|---|---|
| `pc1_10/` & `pc2_10/` | Maps reconstructed by traversing the latent space along PC1 and PC2 |
| `pca_traversal.png` | Visualization of the latent manifold with sampled reconstruction locations |

### Re-analyzing Without Retraining

If training is complete and you want to test different clustering parameters (e.g., a new `k_num`), use `--skip_train True`. Keep all other parameters consistent with the original run.

```bash
accelerate launch --mixed_precision=bf16 CryoDECO_run.py \
    --particles $particles \
    --outdir $outdir \
    --pretrained_model_path $pretrained_model_path \
    --feature_dim 128 \
    --k_num 8 \
    --clustering_type 'gmm' \
    --skip_train True
```

---

## Acknowledgments

CryoDECO adapts code from [DrgnAI](https://github.com/ml-struct-bio/drgnai). We thank the authors for their contributions to the open-source community.

---

## Citation

If you use CryoDECO in your research, please cite:

```bibtex
@article{yan_cryodeco_2026,
    title   = {{CryoDECO}: {Deconstructing} {Extreme} {Compositional} and {Conformational}
               {Heterogeneity} in {Cryo}-{EM} via {Foundation} {Model} {Priors}},
    author  = {Yan, Yang and Xi, Yanwanyu and Fan, Shiqi and Wang, Yifei and
               Tang, Ziyun and Yuan, Fajie and Shen, Huaizong},
    journal = {LangTaoSha Preprint Server},
    year    = {2026},
    doi     = {10.65215/LTSpreprints.2025.12.30.000075},
    url     = {https://doi.org/10.65215/LTSpreprints.2025.12.30.000075},
}
```
