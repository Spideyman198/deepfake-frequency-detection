"""
Experiment 2 — FFT Frequency-Domain Model
==========================================
Train a ResNet18 (randomly initialised) on FFT log-magnitude images.
The model learns to detect periodic upsampling artifacts introduced by GANs.

Quick start
-----------
    python train_fft.py
    python train_fft.py --max_samples 0   # use the entire dataset

Outputs
-------
    results/models/fft_model.pth
    results/models/fft_model_history.json
    results/figures/fft_training_curves.png
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
        description='Train FFT frequency-domain deepfake detector',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--data_dir',    default='data')
    p.add_argument('--batch_size',  type=int,   default=32)
    p.add_argument('--epochs',      type=int,   default=10)
    p.add_argument('--lr',          type=float, default=1e-3,
                   help='Higher LR than RGB because weights are random init')
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
    print("  EXPERIMENT 2 — FFT Frequency-Domain Model")
    print("=" * 60)
    print(f"  Backbone    : {args.backbone}  (random init, no pretrain)")
    print(f"  Batch size  : {args.batch_size}")
    print(f"  Max samples : {max_s or 'all'}")
    print(f"  Epochs      : {args.epochs}  (patience={args.patience})")
    print("=" * 60)

    # ── Data ──────────────────────────────────────────────────────────────
    print("\n[1/3] Loading dataset (FFT mode) …")
    loaders = get_dataloaders(
        data_dir=args.data_dir,
        mode='fft',
        batch_size=args.batch_size,
        max_samples=max_s,
        num_workers=args.num_workers,
    )

    # ── Build model ────────────────────────────────────────────────────────
    print("\n[2/3] Building FFT model …")
    model = build_model('fft', backbone=args.backbone)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {n_params:,}")

    # ── Train ──────────────────────────────────────────────────────────────
    print("\n[3/3] Training …")
    history = train_model(
        model=model,
        dataloaders=loaders,
        save_path='results/models/fft_model.pth',
        mode='fft',
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
    )

    os.makedirs('results/figures', exist_ok=True)
    plot_training_curves(history,
                         'results/figures/fft_training_curves.png',
                         model_name='FFT Model')

    print("\n✓  Experiment 2 complete.\n")


if __name__ == '__main__':
    main()
