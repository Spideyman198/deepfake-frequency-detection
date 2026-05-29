"""
Experiment 1 — RGB Baseline Model
===================================
Train a ResNet18 (or EfficientNet-B0) on raw RGB face images.
This is the spatial-domain baseline.

Quick start
-----------
    python train_rgb.py                          # use 10 000 training images
    python train_rgb.py --max_samples 0          # use the entire dataset
    python train_rgb.py --backbone efficientnet_b0

Outputs
-------
    results/models/rgb_model.pth
    results/models/rgb_model_history.json
    results/figures/rgb_training_curves.png
    results/figures/sample_images.png
    results/figures/rgb_vs_fft_comparison.png
"""

import argparse
import os
import sys

# Make sure src/ is importable when running from project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.dataset import get_dataloaders
from src.fft_utils import save_fft_comparison
from src.models import build_model
from src.train import train_model
from src.visualize import plot_sample_images, plot_training_curves


def parse_args():
    p = argparse.ArgumentParser(
        description='Train RGB baseline deepfake detector',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--data_dir',     default='data',
                   help='Root directory containing train/ val/ test/')
    p.add_argument('--batch_size',   type=int,   default=32)
    p.add_argument('--epochs',       type=int,   default=10)
    p.add_argument('--lr',           type=float, default=1e-4)
    p.add_argument('--patience',     type=int,   default=3,
                   help='Early-stopping patience (epochs without improvement)')
    p.add_argument('--max_samples',  type=int,   default=10000,
                   help='Max training images (balanced). 0 = use all.')
    p.add_argument('--backbone',     default='resnet18',
                   choices=['resnet18', 'efficientnet_b0'])
    p.add_argument('--num_workers',  type=int,   default=0,
                   help='DataLoader worker processes (0 = safe on Windows)')
    return p.parse_args()


def main():
    args = parse_args()
    max_s = args.max_samples if args.max_samples > 0 else None

    print("\n" + "=" * 60)
    print("  EXPERIMENT 1 — RGB Baseline Model")
    print("=" * 60)
    print(f"  Backbone    : {args.backbone}")
    print(f"  Batch size  : {args.batch_size}")
    print(f"  Max samples : {max_s or 'all'}")
    print(f"  Epochs      : {args.epochs}  (patience={args.patience})")
    print("=" * 60)

    # ── Data ──────────────────────────────────────────────────────────────
    print("\n[1/4] Loading dataset …")
    loaders = get_dataloaders(
        data_dir=args.data_dir,
        mode='rgb',
        batch_size=args.batch_size,
        max_samples=max_s,
        num_workers=args.num_workers,
    )

    # ── Visualise raw data ─────────────────────────────────────────────────
    os.makedirs('results/figures', exist_ok=True)
    plot_sample_images(loaders['train'],
                       'results/figures/sample_images.png',
                       title='Sample Real vs Fake Faces')
    save_fft_comparison(loaders['train'],
                        'results/figures',
                        num_samples=6)

    # ── Build model ────────────────────────────────────────────────────────
    print("\n[2/4] Building model …")
    model = build_model('rgb', backbone=args.backbone, pretrained=True)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {n_params:,}")

    # ── Train ──────────────────────────────────────────────────────────────
    print("\n[3/4] Training …")
    history = train_model(
        model=model,
        dataloaders=loaders,
        save_path='results/models/rgb_model.pth',
        mode='rgb',
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
    )

    # ── Plot ───────────────────────────────────────────────────────────────
    print("\n[4/4] Saving training curves …")
    plot_training_curves(history,
                         'results/figures/rgb_training_curves.png',
                         model_name='RGB Model')

    print("\n✓  Experiment 1 complete.")
    print("   Run evaluate_all.py to see test metrics and Grad-CAM.\n")


if __name__ == '__main__':
    main()
