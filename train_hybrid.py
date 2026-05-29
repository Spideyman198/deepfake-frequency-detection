"""
Experiment 3 — Hybrid RGB + FFT Model
======================================
Train a dual-branch model that fuses spatial (RGB) and frequency (FFT) features.

Architecture recap:
  RGB  → ResNet18 (pretrained)  →  512-d
  FFT  → ResNet18 (random)      →  512-d
  Concat 1024-d  →  Dropout → FC(256) → ReLU → Dropout → FC(2)

Quick start
-----------
    python train_hybrid.py
    python train_hybrid.py --max_samples 0   # use the entire dataset

Outputs
-------
    results/models/hybrid_model.pth
    results/models/hybrid_model_history.json
    results/figures/hybrid_training_curves.png
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.dataset import get_dataloaders
from src.models import build_model
from src.train import train_model
from src.visualize import plot_training_curves


def parse_args():
    p = argparse.ArgumentParser(
        description='Train Hybrid (RGB + FFT) deepfake detector',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--data_dir',    default='data')
    p.add_argument('--batch_size',  type=int,   default=32)
    p.add_argument('--epochs',      type=int,   default=10)
    p.add_argument('--lr',          type=float, default=5e-5,
                   help='Lower LR because two pretrained branches are present')
    p.add_argument('--patience',    type=int,   default=3)
    p.add_argument('--max_samples', type=int,   default=10000,
                   help='Max training images (balanced). 0 = use all.')
    p.add_argument('--backbone',    default='resnet18',
                   choices=['resnet18', 'efficientnet_b0'])
    p.add_argument('--num_workers', type=int,   default=0)
    return p.parse_args()


def main():
    args = parse_args()
    max_s = args.max_samples if args.max_samples > 0 else None

    print("\n" + "=" * 60)
    print("  EXPERIMENT 3 — Hybrid RGB + FFT Model")
    print("=" * 60)
    print(f"  Backbone    : {args.backbone}")
    print(f"  Batch size  : {args.batch_size}")
    print(f"  Max samples : {max_s or 'all'}")
    print(f"  Epochs      : {args.epochs}  (patience={args.patience})")
    print("=" * 60)

    # ── Data ──────────────────────────────────────────────────────────────
    print("\n[1/3] Loading dataset (hybrid mode) …")
    loaders = get_dataloaders(
        data_dir=args.data_dir,
        mode='hybrid',
        batch_size=args.batch_size,
        max_samples=max_s,
        num_workers=args.num_workers,
    )

    # ── Build model ────────────────────────────────────────────────────────
    print("\n[2/3] Building Hybrid model …")
    model = build_model('hybrid', backbone=args.backbone, pretrained=True)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {n_params:,}")

    # ── Train ──────────────────────────────────────────────────────────────
    print("\n[3/3] Training …")
    history = train_model(
        model=model,
        dataloaders=loaders,
        save_path='results/models/hybrid_model.pth',
        mode='hybrid',
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
    )

    os.makedirs('results/figures', exist_ok=True)
    plot_training_curves(history,
                         'results/figures/hybrid_training_curves.png',
                         model_name='Hybrid Model')

    print("\n✓  Experiment 3 complete.\n")


if __name__ == '__main__':
    main()
