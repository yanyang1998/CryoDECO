import os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_split_comparison(z_2d, labels_before, labels_after,
                          macro_k, final_k, outdir, run_tag):
    z_2d = np.asarray(z_2d)
    labels_before = np.asarray(labels_before).astype(int)
    labels_after = np.asarray(labels_after).astype(int)

    cmap = plt.get_cmap('tab20' if final_k <= 20 else 'gist_rainbow')

    def _scatter(ax, labels, k_val, title):
        unique_labels = sorted(int(x) for x in np.unique(labels))
        for i, label in enumerate(unique_labels):
            mask = labels == label
            color = cmap(i % cmap.N)
            ax.scatter(
                z_2d[mask, 0], z_2d[mask, 1],
                c=[color], s=1.0, alpha=0.65,
                label=f'C{label} (n={int(mask.sum())})',
            )
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('UMAP-1', fontsize=12)
        ax.set_ylabel('UMAP-2', fontsize=12)
        ax.legend(loc='best', fontsize=8, markerscale=5, framealpha=0.85, ncol=2)
        ax.grid(True, alpha=0.3)

    # ---- Side-by-side comparison ----
    fig, axes = plt.subplots(1, 2, figsize=(22, 9))
    _scatter(axes[0], labels_before, macro_k, f'Before split: K = {macro_k}')
    _scatter(axes[1], labels_after, final_k, f'After split:  K = {final_k}')
    plt.suptitle(f'Split comparison ({run_tag})', fontsize=18)
    plt.tight_layout()

    side_path = os.path.join(outdir, 'umap_split_comparison.png')
    plt.savefig(side_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[Plot] saved side-by-side comparison: {side_path}", flush=True)

    # ---- Individual plots ----
    individual_specs = [
        ('before', labels_before, macro_k, 'umap_before_split.png'),
        ('after',  labels_after,  final_k, 'umap_after_split.png')]
    
    for tag, labels, k_val, fname in individual_specs:
        fig, ax = plt.subplots(figsize=(10, 8))
        _scatter(ax, labels, k_val, f'UMAP: {tag} split, K = {k_val}')
        plt.tight_layout()
        path = os.path.join(outdir, fname)
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"[Plot] saved: {path}", flush=True)
