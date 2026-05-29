"""
Visualisation helpers for training curves, sample images,
and cross-model metric comparisons.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import torch


# ── Training curves ────────────────────────────────────────────────────────

def plot_training_curves(history: dict, save_path: str,
                          model_name: str = 'Model'):
    """
    Save a single-figure training curve using dual y-axes:
      - Left  axis (red)  : train loss / val loss
      - Right axis (blue) : train accuracy / val accuracy

    No subplots or panel grids — one plt.figure() per call.

    Args:
        history    : dict with keys 'train_loss', 'val_loss',
                     'train_acc', 'val_acc'  (lists of per-epoch values)
        save_path  : output file path (.png)
        model_name : used in figure title
    """
    epochs = range(1, len(history['train_loss']) + 1)

    fig, ax_loss = plt.subplots(figsize=(8, 5))
    ax_loss.set_xlabel('Epoch', fontsize=12)
    ax_loss.set_ylabel('Loss', color='#E53935', fontsize=12)
    ax_loss.tick_params(axis='y', labelcolor='#E53935')

    l1, = ax_loss.plot(epochs, history['train_loss'],
                        color='#E53935', lw=2, ls='--', label='Train Loss')
    l2, = ax_loss.plot(epochs, history['val_loss'],
                        color='#E53935', lw=2, ls='-',  label='Val Loss')
    ax_loss.grid(alpha=0.3)

    ax_acc = ax_loss.twinx()
    ax_acc.set_ylabel('Accuracy (%)', color='#1E88E5', fontsize=12)
    ax_acc.tick_params(axis='y', labelcolor='#1E88E5')

    l3, = ax_acc.plot(epochs, history['train_acc'],
                       color='#1E88E5', lw=2, ls='--', label='Train Acc')
    l4, = ax_acc.plot(epochs, history['val_acc'],
                       color='#1E88E5', lw=2, ls='-',  label='Val Acc')

    lines  = [l1, l2, l3, l4]
    labels = [ln.get_label() for ln in lines]
    ax_loss.legend(lines, labels, loc='center right', fontsize=9)
    ax_loss.set_title(f'Training Curves — {model_name}',
                      fontsize=13, fontweight='bold')

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Plot] Training curves → {save_path}")


def plot_all_training_curves(histories: dict, save_path: str):
    """
    Overlay validation loss and accuracy for RGB, FFT, and Hybrid.

    Args:
        histories : {'RGB': history_dict, 'FFT': ..., 'Hybrid': ...}
        save_path : output file path
    """
    colors = {'RGB': '#2196F3', 'FFT': '#F44336', 'Hybrid': '#4CAF50'}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

    for name, hist in histories.items():
        c  = colors.get(name, 'black')
        ep = range(1, len(hist['val_loss']) + 1)
        ax1.plot(ep, hist['val_loss'], color=c, lw=2, label=name)
        ax2.plot(ep, hist['val_acc'],  color=c, lw=2, label=name)

    ax1.set(xlabel='Epoch', ylabel='Validation Loss',
            title='Validation Loss — All Models')
    ax1.legend(); ax1.grid(alpha=0.3)

    ax2.set(xlabel='Epoch', ylabel='Validation Accuracy (%)',
            title='Validation Accuracy — All Models')
    ax2.legend(); ax2.grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Plot] Combined training curves → {save_path}")


# ── Sample images ──────────────────────────────────────────────────────────

def plot_sample_images(loader, save_path: str,
                        num_per_class: int = 4,
                        title: str = 'Sample Images'):
    """
    Save a 2-row grid: top row = Real, bottom row = Fake.

    Works with 'rgb' and 'hybrid' DataLoaders.
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    real_imgs, fake_imgs = [], []

    for batch in loader:
        # Support both rgb (2-tuple) and hybrid (3-tuple) loaders
        imgs = batch[0]
        lbls = batch[-1]

        for i in range(len(lbls)):
            img_vis = (imgs[i] * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
            if lbls[i] == 0 and len(real_imgs) < num_per_class:
                real_imgs.append(img_vis)
            elif lbls[i] == 1 and len(fake_imgs) < num_per_class:
                fake_imgs.append(img_vis)

        if len(real_imgs) >= num_per_class and len(fake_imgs) >= num_per_class:
            break

    n   = min(num_per_class, len(real_imgs), len(fake_imgs))
    fig, axes = plt.subplots(2, n, figsize=(n * 3, 6))
    fig.suptitle(title, fontsize=13, fontweight='bold')

    for j in range(n):
        axes[0][j].imshow(real_imgs[j])
        axes[0][j].set_title('Real', color='#2E7D32', fontsize=10)
        axes[0][j].axis('off')

        axes[1][j].imshow(fake_imgs[j])
        axes[1][j].set_title('Fake', color='#C62828', fontsize=10)
        axes[1][j].axis('off')

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Plot] Sample images → {save_path}")


# ── Metric comparison bar chart ────────────────────────────────────────────

def plot_metrics_comparison(metrics_dict: dict, save_path: str):
    """
    Grouped bar chart comparing Accuracy, Precision, Recall, F1, ROC-AUC
    for all three models.

    Args:
        metrics_dict : {'RGB': {metric: value, ...}, 'FFT': ..., 'Hybrid': ...}
        save_path    : output file path
    """
    metric_keys  = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']
    model_names  = list(metrics_dict.keys())
    colors       = ['#2196F3', '#F44336', '#4CAF50']

    x     = np.arange(len(metric_keys))
    width = 0.22
    fig, ax = plt.subplots(figsize=(12, 5.5))

    for i, (name, metrics) in enumerate(metrics_dict.items()):
        vals = [metrics[k] for k in metric_keys]
        bars = ax.bar(x + i * width, vals, width, label=name,
                      color=colors[i % len(colors)], alpha=0.85,
                      edgecolor='black', linewidth=0.7)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f'{v:.3f}', ha='center', va='bottom', fontsize=7.5)

    ax.set(xticks=x + width,
           xticklabels=metric_labels,
           ylabel='Score',
           title='Performance Comparison: RGB vs FFT vs Hybrid',
           ylim=[0, 1.12])
    ax.tick_params(axis='x', labelsize=11)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Plot] Metrics comparison → {save_path}")


# ── Failure case gallery ────────────────────────────────────────────────────

def save_failure_cases(incorrect_samples: list, save_path: str,
                        model_name: str = 'Model'):
    """
    Save a plain image grid (no heatmap) of failure cases for quick review.

    Args:
        incorrect_samples : list of sample dicts (from generate_gradcam_visuals)
        save_path         : output path
        model_name        : used in title
    """
    if not incorrect_samples:
        print("[Plot] No failure cases to display.")
        return

    n    = len(incorrect_samples)
    names = ['Real', 'Fake']
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    fig, axes = plt.subplots(1, n, figsize=(n * 3, 3.5))
    if n == 1:
        axes = [axes]
    fig.suptitle(f'Failure Cases — {model_name}', fontsize=13,
                 fontweight='bold')

    for ax, s in zip(axes, incorrect_samples):
        img = (s['rgb'] * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
        ax.imshow(img)
        ax.set_title(f"True: {names[s['label']]}\n"
                     f"Pred: {names[s['pred']]}", fontsize=9, color='red')
        ax.axis('off')

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Plot] Failure cases → {save_path}")
