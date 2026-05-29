"""
Unified Evaluation Script
==========================
Loads all three trained models, runs evaluation on the test split, and
produces:
  • Per-model metrics (accuracy, precision, recall, F1, AUC)
  • Confusion matrices
  • ROC curves (individual + combined overlay)
  • Grad-CAM visualisations (correct & failure cases)
  • Side-by-side performance bar chart
  • Combined training-curve overlay

Run this script after training all three models:
    python evaluate_all.py

Optional: evaluate only specific models:
    python evaluate_all.py --models rgb fft
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.dataset import get_dataloaders
from src.evaluate import (plot_combined_roc, plot_confusion_matrix,
                          plot_roc_curve, run_evaluation)
from src.gradcam import generate_gradcam_visuals
from src.models import build_model
from src.visualize import (plot_metrics_comparison, plot_training_curves,
                            save_failure_cases)


# ── Config ─────────────────────────────────────────────────────────────────

MODEL_CFG = {
    'rgb': {
        'mode'      : 'rgb',
        'ckpt'      : 'results/models/rgb_model.pth',
        'history'   : 'results/models/rgb_model_history.json',
        'display'   : 'RGB',
    },
    'fft': {
        'mode'      : 'fft',
        'ckpt'      : 'results/models/fft_model.pth',
        'history'   : 'results/models/fft_model_history.json',
        'display'   : 'FFT',
    },
    'hybrid': {
        'mode'      : 'hybrid',
        'ckpt'      : 'results/models/hybrid_model.pth',
        'history'   : 'results/models/hybrid_model_history.json',
        'display'   : 'Hybrid',
    },
}


def parse_args():
    p = argparse.ArgumentParser(
        description='Evaluate all trained deepfake detection models',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--data_dir',    default='data')
    p.add_argument('--batch_size',  type=int,   default=32)
    p.add_argument('--max_samples', type=int,   default=2000,
                   help='Max test images. 0 = use all test samples.')
    p.add_argument('--models',      nargs='+',  default=['rgb', 'fft', 'hybrid'],
                   choices=['rgb', 'fft', 'hybrid'],
                   help='Which models to evaluate')
    p.add_argument('--backbone',    default='resnet18',
                   choices=['resnet18', 'efficientnet_b0'])
    p.add_argument('--num_workers', type=int,   default=0)
    p.add_argument('--gradcam_n',   type=int,   default=6,
                   help='Samples per category in Grad-CAM grids')
    return p.parse_args()


def load_model(experiment: str, ckpt_path: str, backbone: str,
               device: torch.device) -> torch.nn.Module:
    """Load a trained model from a checkpoint file."""
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            f"Train first with: python train_{experiment}.py"
        )
    model = build_model(experiment, backbone=backbone, pretrained=False)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    print(f"  Loaded {experiment.upper()} model  ←  {ckpt_path}")
    return model


def main():
    args   = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    max_s  = args.max_samples if args.max_samples > 0 else None

    print("\n" + "=" * 60)
    print("  EVALUATION — All Experiments")
    print("=" * 60)
    print(f"  Device      : {device}")
    print(f"  Models      : {args.models}")
    print(f"  Max test    : {max_s or 'all'}")
    print("=" * 60)

    os.makedirs('results/figures', exist_ok=True)
    os.makedirs('results/metrics', exist_ok=True)

    all_metrics  = {}     # { 'RGB': metrics_dict, ... }
    roc_data     = {}     # { 'RGB': (labels, probs), ... }
    histories    = {}     # { 'RGB': history_dict, ... }

    # ── Per-model evaluation ───────────────────────────────────────────────
    for exp in args.models:
        cfg      = MODEL_CFG[exp]
        disp     = cfg['display']
        mode     = cfg['mode']

        print(f"\n{'─'*60}")
        print(f"  Model: {disp}")
        print(f"{'─'*60}")

        # Load data (shared loader per mode)
        loader_dict = get_dataloaders(
            data_dir=args.data_dir,
            mode=mode,
            batch_size=args.batch_size,
            max_samples=max_s,
            num_workers=args.num_workers,
        )
        test_loader = loader_dict['test']

        # Load model weights
        model = load_model(exp, cfg['ckpt'], args.backbone, device)

        # Evaluate (saves metrics JSON + copies to results/metrics/)
        metrics, preds, labels, probs = run_evaluation(
            model=model,
            loader=test_loader,
            device=device,
            mode=mode,
            save_dir='results/metrics',
            name=disp,
        )
        all_metrics[disp] = metrics
        roc_data[disp]    = (labels, probs)

        # ── Per-model figures → results/figures/ ──────────────────────────
        plot_confusion_matrix(
            labels, preds,
            save_path=f'results/figures/confusion_matrix_{exp}.png',
            title=f'Confusion Matrix — {disp}',
        )
        plot_roc_curve(
            labels, probs,
            save_path=f'results/figures/roc_{exp}.png',
            model_name=disp,
        )

        # Load training history and save individual training curve
        if os.path.exists(cfg['history']):
            with open(cfg['history']) as f:
                hist = json.load(f)
            histories[disp] = hist
            plot_training_curves(
                hist,
                save_path=f'results/figures/training_{exp}.png',
                model_name=disp,
            )

        # ── Grad-CAM (RGB and Hybrid only) ────────────────────────────────
        if exp in ('rgb', 'hybrid'):
            print(f"\n  Generating Grad-CAM visualisations …")
            try:
                correct, incorrect = generate_gradcam_visuals(
                    model=model,
                    loader=test_loader,
                    device=device,
                    mode=mode,
                    save_dir='results/figures',
                    max_per_category=args.gradcam_n,
                    correct_fname=f'gradcam_{exp}_correct.png',
                    wrong_fname=f'gradcam_{exp}_wrong.png',
                )
                # Plain failure-case gallery (no heatmap overlay)
                save_failure_cases(
                    incorrect,
                    save_path=f'results/figures/failure_cases_{exp}.png',
                    model_name=disp,
                )
            except Exception as e:
                print(f"  [Warning] Grad-CAM failed for {disp}: {e}")

    # ── Cross-model plots ──────────────────────────────────────────────────
    if len(roc_data) > 1:
        plot_combined_roc(roc_data, 'results/figures/roc_combined.png')

    if len(all_metrics) > 1:
        plot_metrics_comparison(all_metrics, 'results/figures/metrics_comparison.png')

    # ── Summary table ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  FINAL RESULTS SUMMARY")
    print("=" * 60)
    header = f"{'Model':<10}  {'Acc':>7}  {'Prec':>7}  {'Rec':>7}  {'F1':>7}  {'AUC':>7}"
    print(header)
    print("-" * len(header))
    for name, m in all_metrics.items():
        print(f"{name:<10}  "
              f"{m['accuracy']:>7.4f}  "
              f"{m['precision']:>7.4f}  "
              f"{m['recall']:>7.4f}  "
              f"{m['f1']:>7.4f}  "
              f"{m['roc_auc']:>7.4f}")

    # Save combined summary JSON
    summary_path = 'results/metrics/all_metrics_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n  Summary saved → {summary_path}")
    print("\n✓  Evaluation complete.  All outputs in results/\n")


if __name__ == '__main__':
    main()
