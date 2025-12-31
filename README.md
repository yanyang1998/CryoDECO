# CryoDECO: Deconstructing Compositional and Conformational Heterogeneity in Cryo-EM with Foundation Model Priors #

CryoDECO is an _ab initio_ heterogeneous reconstruction algorithm designed to resolve complex structural mixtures in cryo-EM data. By leveraging structural priors from the pre-trained [Cryo-IEF](https://github.com/westlake-repl/Cryo-IEF) foundation model, CryoDECO bypasses the random initialization bottleneck common in traditional deep learning approaches, 
enabling robust classification of both compositional and conformational heterogeneity.
## 📑 Installation ##

We recommend installing CryoDECO in a clean Conda environment to avoid dependency conflicts.

```bash
    # 1. Create and activate a virtual environment
    conda create --name cryoDECO python=3.9
    conda activate cryoDECO
    
    # 2. Clone the repository
    git clone https://github.com/yanyang1998/CryoSolver.git
    cd cryoDECO/
    
    # 3. Install dependencies
    pip install -r requirements.txt
```

[//]: # (## 💻 Requirements ##)

[//]: # ()
[//]: # (- python==3.9)

[//]: # ()
[//]: # (- torch==2.8.0)

[//]: # (  )
[//]: # (- torchvision==0.23.0)

[//]: # ()
[//]: # (- cryodrgn==3.4.4)

[//]: # ()
[//]: # (- lmdb==1.7.3)

[//]: # ()
[//]: # (- numpy==1.24.4)



## ⚙️ Configuration & Parameters ##
### 1. Required Parameters
These paths must be defined by the user. 
*   **Model Weights:** The pre-trained Cryo-IEF model weights are available via [Huggingface](https://huggingface.co/westlake-repl/Cryo-IEF/tree/main/cryo_ief_checkpoint).
*   **Input Data:** CryoDECO accepts CryoSPARC job outputs (`Downsample`, `Extracted Particles`, `Restack Particles`, and `Particle Sets`).
*   **Recommendation:** For efficient training and inference, we strongly recommend **downsampling input particles to a box size of 128 pixels**. ([Guide: How to downsample in CryoSPARC](readmes/downsample.md))

[//]: # (Alternatively, they can be downloaded from the [Cryo-IEF google drive]&#40;https://drive.google.com/drive/folders/1C9jIdC5B58ohAwrfRalTngRtLtgIWfM8?usp=sharing&#41.)
[//]: # (This model is intended for **academic research only**. Commercial use is prohibited without explicit permission.)
```yaml
particles='/path/to/cryosparc_job/'           # Path to input particle job directory
outdir='/path/to/save/your/results/'          # Directory for output results
pretrained_model_path='/path/to/checkpoint/'  # Path to Cryo-IEF weights
```

### 2. Hyperparameters (Optional)
The following parameters control the latent space topology and clustering behavior. Default values are provided but should be adjusted based on the biological heterogeneity type.

**Latent Dimension (`feature_dim`):**

[//]: # (*   **Compositional Heterogeneity:** Set `feature_dim=128`. High dimensionality is required to ensure orthogonality between discrete structural states &#40;e.g., different complexes in a mixture&#41;.)
[//]: # (*   **Conformational Heterogeneity:** Set `feature_dim=4`. A tight information bottleneck forces the model to map continuous dynamics onto a low-dimensional manifold.)
Latent dimension (z) is a variable that can be changed to achieve best classification performance. 
Generally, it is recommended that discrete compositional heterogeneity demands a large dimension size (e.g., z=128) to ensure orthogonality between disjoint structures. Conversely, simple conformational dynamics require a small dimension size (e.g., z=4) to enforce smoothness, while complex conformational heterogeneity necessitates an intermediate dimension (e.g., z=64).



```yaml
feature_dim=128   # Default: 128 (Compositional). Use 4 (Simple Motion) or 64 (Complex Motion) for Conformational.
```

**Clustering (`k_num` & `clustering_type`):**

The pipeline performs clustering on the learned latent features to generate initial volumes.
*   **Compositional:** Set `k_num` based on the expected number of distinct species (if known).
*   **Conformational:** `k_num` determines the number of maps sampled by clustering in the latent space. 

```yaml
k_num=8               # Default: 8
clustering_type='gmm' # Default: 'gmm' (Gaussian Mixture Model). Option: 'k-means++' (much faster)
```

**Using the Known Poses (`use_gt_poses` & `use_gt_trans`):**

By default, CryoDECO estimates particle poses during training. 
However, if high-quality pose estimates are available (e.g., from a prior CryoSPARC refinement or ab initio job), they can be utilized to enhance reconstruction quality and accelerate convergence.
*   Requirement: The input `particles` job must contain pose information. You can generate this by running a `Restack Particles` or `Downsample Particles` job in CryoSPARC connected to your prior refinement or ab-initio job.
   
```yaml
use_gt_poses=True  # Default: False. Set to True to use CryoSPARC poses.
use_gt_trans=True  # Default: False. Set to True to use CryoSPARC translations.
```

**Optimization:**

Adjust batch sizes based on available GPU memory (Defaults tuned for NVIDIA A40 40GB).
```yaml
epochs_sgd=100        # Default: 100. Decrease for very large datasets (>1M particles).
batch_size_hps=22     # Batch size for Hierarchical Pose Search (per GPU)
batch_size_sgd=192    # Batch size for SGD Refinement (per GPU)
```

## 🚀 Usage

Execute the following command to launch CryoDECO. Arguments with default values can be omitted.

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

*Note: By default, `accelerate` utilizes all available GPUs. To specify GPUs, use `--gpu_ids` (e.g., `--gpu_ids 0,1`) or `--num_processes`.*


[//]: # (## 📈 Monitoring & Analysis)

[//]: # (### Training Monitoring)

[//]: # (Real-time training metrics &#40;loss curves, pose error, etc.&#41; are logged to TensorBoard.)

[//]: # ()
[//]: # (```bash)

[//]: # (conda activate CryoDECO)

[//]: # (tensorboard --logdir output/summaries)

[//]: # (```)

[//]: # (<p align="center">)

[//]: # (  <img src="tensorboard_detail.png" width="90%" alt="Tensorboard Example">)

[//]: # (</p>)

## 📈 Analysis

### Output Interpretation
Results are saved in `outdir/out/analysis_(epoch_number)/`.



#### A. Compositional Heterogeneity (Discrete States)
Focus on the **Clustering** results.
*   **`clustering(knum)/`**: Contains reconstructed density maps (`.mrc`) for each cluster center.
*   **`clustering(knum)/umap_clusters.png`**: UMAP visualization of latent features, colored by cluster assignment.
*   **`clustering(knum)/clustering_cs_star/`**: Contains `.star` files for each cluster.
    *   **Workflow:** Import these `.star` files into CryoSPARC to create particle subsets.
    *   **High-Res Refinement:** If you used downsampled particles, map these subsets back to the original full-resolution particles before final refinement. ([Guide: How to map back particles](readmes/mapback.md)).

#### B. Conformational Heterogeneity (Continuous Motion)
Focus on the **PCA Traversal** results.
*   **`pc1_10/` & `pc2_10/`**: Contains maps reconstructed by traversing the latent space along the 1st and 2nd Principal Components (PCs).
*   **`pca_traversal.png`**: Visualization of the latent manifold with markers indicating the sampled locations for reconstruction.

#### Re-analyzing (Skipping Training)

[//]: # (Once the training is complete and you want to try different clustering parameters, you can skip the training phase and directly run the analysis by adding `--skip_train True` to the command you used for training:)
If training is complete and you wish to test different clustering parameters (e.g., changing `k_num` ), use the `--skip_train True` flag. Ensure other parameters match the original run.
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

[//]: # (Make sure to keep the other parameters consistent with your previous training run.)

##  Acknowledgments

CryoDECO adapts code from [DrgnAI](https://github.com/ml-struct-bio/drgnai). We acknowledge the authors for their contributions to the open-source community.

[//]: # (## 📖 Reference ##)
If you use CryoDECO in your research, please cite:
```bibtex
@article{,
  title={CryoDECO: Deconstructing Compositional and Conformational Heterogeneity in Cryo-EM with Foundation Model Priors},
  url = {https://langtaosha.org.cn/index.php/lts/preprint/view/75},
  doi = {10.65215/LTSpreprints.2025.12.30.000075},
  journal = {浪淘沙预印本平台},
  author = {Yan, Yang and Xi, Yanwanyu and Fan, Shiqi and Tang, Ziyun and Yuan, Fajie and Shen, Huaizong},
  year = {2025},
```
}

