"""
Evaluation utilities: predictions, metrics, confusion matrix, ROC curves,
and a combined comparison plot across all three experiments.
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score, roc_curve,
)


# ── Prediction collection ──────────────────────────────────────────────────

@torch.no_grad()
def collect_predictions(model, loader, device,
                         mode: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run model over a DataLoader and collect predictions.

    Returns:
        preds  : (N,) int array of predicted class indices
        labels : (N,) int array of ground-truth class indices
        probs  : (N,) float array of P(fake) used for ROC / AUC
    """
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    for batch in loader:
        if mode == 'hybrid':
            rgb, fft, labels = batch
            rgb, fft = rgb.to(device), fft.to(device)
            logits = model(rgb, fft)
        else:
            images, labels = batch
            logits = model(images.to(device))

        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs[:, 1].cpu().numpy())   # P(fake)

    return (np.array(all_preds, dtype=int),
            np.array(all_labels, dtype=int),
            np.array(all_probs,  dtype=float))


# ── Metrics ────────────────────────────────────────────────────────────────

def compute_metrics(labels: np.ndarray, preds: np.ndarray,
                    probs: np.ndarray) -> dict:
    """Compute accuracy, precision, recall, F1, and ROC-AUC."""
    return {
        'accuracy' : float(accuracy_score(labels, preds)),
        'precision': float(precision_score(labels, preds, zero_division=0)),
        'recall'   : float(recall_score(labels, preds, zero_division=0)),
        'f1'       : float(f1_score(labels, preds, zero_division=0)),
        'roc_auc'  : float(roc_auc_score(labels, probs)),
    }


# ── Confusion matrix ───────────────────────────────────────────────────────

def plot_confusion_matrix(labels: np.ndarray, preds: np.ndarray,
                          save_path: str, title: str = 'Confusion Matrix'):
    """Plot a labelled confusion matrix and save it."""
    cm         = confusion_matrix(labels, preds)
    class_names = ['Real', 'Fake']

    fig, ax = plt.subplots(figsize=(5, 4))
    im      = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.colorbar(im, ax=ax)

    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(class_names, fontsize=12)
    ax.set_yticklabels(class_names, fontsize=12)
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True',      fontsize=12)
    ax.set_title(title,        fontsize=13)

    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]),
                    ha='center', va='center', fontsize=14,
                    color='white' if cm[i, j] > thresh else 'black')

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Eval] Confusion matrix → {save_path}")


# ── ROC curves ─────────────────────────────────────────────────────────────

def plot_roc_curve(labels: np.ndarray, probs: np.ndarray,
                   save_path: str, model_name: str = 'Model'):
    """Plot and save a single ROC curve."""
    fpr, tpr, _ = roc_curve(labels, probs)
    auc         = roc_auc_score(labels, probs)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, lw=2, color='darkorange',
            label=f'{model_name}  (AUC = {auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random baseline')
    ax.set(xlim=[0, 1], ylim=[0, 1.05],
           xlabel='False Positive Rate', ylabel='True Positive Rate',
           title=f'ROC Curve — {model_name}')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Eval] ROC curve → {save_path}")


def plot_combined_roc(roc_data: dict, save_path: str):
    """
    Overlay ROC curves for RGB, FFT, and Hybrid on a single axes.

    Args:
        roc_data : {'RGB': (labels, probs), 'FFT': (...), 'Hybrid': (...)}
        save_path: Output path
    """
    colors = {'RGB': '#2196F3', 'FFT': '#F44336', 'Hybrid': '#4CAF50'}

    fig, ax = plt.subplots(figsize=(6, 5))
    for name, (labels, probs) in roc_data.items():
        fpr, tpr, _ = roc_curve(labels, probs)
        auc         = roc_auc_score(labels, probs)
        ax.plot(fpr, tpr, lw=2, color=colors.get(name, 'gray'),
                label=f'{name}  (AUC = {auc:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
    ax.set(xlim=[0, 1], ylim=[0, 1.05],
           xlabel='False Positive Rate', ylabel='True Positive Rate',
           title='ROC Curve Comparison: RGB vs FFT vs Hybrid')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Eval] Combined ROC curve → {save_path}")


# ── Full evaluation pipeline ───────────────────────────────────────────────

def run_evaluation(model, loader, device, mode: str,
                   save_dir: str, name: str) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    """
    Collect predictions, compute all metrics, plot confusion matrix and
    ROC curve, save metrics JSON, and print a summary.

    Args:
        model    : trained nn.Module
        loader   : test DataLoader
        device   : torch.device
        mode     : 'rgb', 'fft', or 'hybrid'
        save_dir : directory for plots & JSON
        name     : model name used in titles and file names

    Returns:
        (metrics_dict, preds, labels, probs)
    """
    os.makedirs(save_dir, exist_ok=True)
    print(f"\n{'─'*50}")
    print(f"  Evaluating: {name}")
    print(f"{'─'*50}")

    preds, labels, probs = collect_predictions(model, loader, device, mode)
    metrics              = compute_metrics(labels, preds, probs)

    print(f"  Accuracy  : {metrics['accuracy']:.4f}  "
          f"({metrics['accuracy']*100:.2f}%)")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  F1-Score  : {metrics['f1']:.4f}")
    print(f"  ROC-AUC   : {metrics['roc_auc']:.4f}")

    tag = name.lower().replace(' ', '_')

    # Save JSON
    m_path = os.path.join(save_dir, f'{tag}_metrics.json')
    with open(m_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics JSON → {m_path}")

    # Confusion matrix
    plot_confusion_matrix(
        labels, preds,
        os.path.join(save_dir, f'{tag}_confusion_matrix.png'),
        title=f'Confusion Matrix — {name}',
    )

    # ROC curve
    plot_roc_curve(
        labels, probs,
        os.path.join(save_dir, f'{tag}_roc_curve.png'),
        model_name=name,
    )

    return metrics, preds, labels, probs
