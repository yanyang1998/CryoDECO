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
| `pretrained_model_path` | Path to DECO-IEF ViT-Small weights ([download from HuggingFace](https://huggingface.co/westlake-repl/Cryo-IEF/tree/main/cryo_ief_checkpoint/DECO_ief_vit_s)) |

> **Recommendation:** For efficient training and inference, we strongly recommend **downsampling input particles to a box size of 128 pixels**. ([Guide: How to downsample in CryoSPARC](readmes/downsample.md))

```yaml
particles='/path/to/cryosparc_job/'           # Path to input particle job directory
outdir='/path/to/save/your/results/'          # Directory for output results
pretrained_model_path='/path/to/checkpoint/'  # Path to Cryo-IEF weights
```

---

### Optional Hyperparameters

#### Adaptive Structural Latent Capacity

CryoDECO starts with a maximum structural latent capacity (`feature_dim`, or
`D_cap`) of 64 and uses DimAda to learn the decoder-visible capacity `D_act` for
the input dataset. The structural-z gate and AdaLN-conditioned decoder are
enabled by default and normally do not need tuning.

| Parameter | Default | Description |
|---|---:|---|
| `feature_dim` | `64` | Maximum structural latent capacity, `D_cap`. |
| `structural_z_gate_enabled` | `True` | Learn `D_act`. Disable for fixed-z ablations or old-model compatibility. |
| `decoder_adaln_enabled` | `True` | Condition residual decoder blocks on the gated structural latent. Disable for decoder ablations. |

The learned capacity and locked mask are written to `out/structural_z.json` and
stored in every model checkpoint.

#### Clustering

The pipeline clusters learned latent features to generate initial volumes.
The automatic partition-count estimator described in the manuscript is not yet
included in this release, so discrete compositional analysis still requires
`k_num` to be supplied by the user.

| Parameter | Default | Description |
|---|---|---|
| `k_num` | `4` | Number of clusters. Set this to the expected number of discrete groups; `4` is an example setting, not a universal recommendation. |
| `clustering_type` | `'gmm'` | Clustering algorithm. `'gmm'` (Gaussian Mixture Model) is more accurate; `'k-means++'` is much faster. |

```yaml
k_num=4               # Example/default; adjust for the dataset
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
batch_size_sgd=64     # Batch size for SGD Refinement (per GPU)
```

---

## Usage

```bash
accelerate launch --mixed_precision=bf16 CryoDECO_run.py \
    --particles $particles \
    --outdir $outdir \
    --pretrained_model_path $pretrained_model_path \
    --feature_dim 64 \
    --k_num 4 \
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
    --feature_dim 64 \
    --k_num 4 \
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
