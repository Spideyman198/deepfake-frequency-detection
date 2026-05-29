"""
Grad-CAM (Gradient-weighted Class Activation Mapping) implementation.

Grad-CAM produces a coarse localisation map showing *which spatial regions*
influenced the model's decision.  We use it to:
  (a) Visualise what correct predictions attend to
  (b) Analyse failure cases — where did the model look when it was wrong?

Reference:
  Selvaraju et al. (2017) "Grad-CAM: Visual Explanations from Deep Networks
  via Gradient-based Localization" — ICCV 2017
"""

import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Core GradCAM class ─────────────────────────────────────────────────────

class GradCAM:
    """
    Computes Grad-CAM heatmaps for a given convolutional layer.

    Usage
    -----
        cam = GradCAM(model, target_layer=model.layer4[-1].conv2)
        heatmap, pred_class = cam.generate(img_tensor)
        cam.remove_hooks()
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model        = model
        self.target_layer = target_layer
        self._activations = None
        self._gradients   = None

        self._fwd_hook = target_layer.register_forward_hook(self._hook_fwd)
        self._bwd_hook = target_layer.register_full_backward_hook(self._hook_bwd)

    # ── Hooks ──────────────────────────────────────────────────────────────

    def _hook_fwd(self, module, input, output):
        self._activations = output.detach()

    def _hook_bwd(self, module, grad_in, grad_out):
        self._gradients = grad_out[0].detach()

    # ── Main method ────────────────────────────────────────────────────────

    def generate(self, input_tensor: torch.Tensor,
                 target_class: int = None) -> tuple[np.ndarray, int]:
        """
        Generate a Grad-CAM heatmap.

        Args:
            input_tensor : (1, 3, H, W) image tensor on the correct device.
            target_class : Class index to explain.
                           None  →  use the predicted (argmax) class.

        Returns:
            cam   : (H, W) float32 array, values in [0, 1]
            pred  : predicted class index (int)
        """
        self.model.eval()

        # Forward pass
        logits = self.model(input_tensor)
        pred   = int(logits.argmax(dim=1).item())

        if target_class is None:
            target_class = pred

        # Backward pass w.r.t. the target class score
        self.model.zero_grad()
        one_hot = torch.zeros_like(logits)
        one_hot[0, target_class] = 1.0
        logits.backward(gradient=one_hot)

        # Global-average-pool the gradients over spatial dims  →  channel weights
        weights = self._gradients.mean(dim=[2, 3], keepdim=True)   # (1, C, 1, 1)

        # Weighted sum of activations + ReLU
        cam = (weights * self._activations).sum(dim=1, keepdim=True)  # (1,1,h,w)
        cam = F.relu(cam)

        # Upsample to input resolution
        cam = F.interpolate(cam, size=input_tensor.shape[2:],
                            mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        # Normalise to [0, 1]
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())

        return cam.astype(np.float32), pred

    def remove_hooks(self):
        """Must be called when done to prevent memory leaks."""
        self._fwd_hook.remove()
        self._bwd_hook.remove()


# ── Helpers ────────────────────────────────────────────────────────────────

def overlay_heatmap(image_tensor: torch.Tensor,
                    cam: np.ndarray,
                    alpha: float = 0.45) -> np.ndarray:
    """
    Blend a Grad-CAM heatmap onto the original image.

    Args:
        image_tensor : (3, H, W) normalised tensor
        cam          : (H, W) heatmap in [0, 1]
        alpha        : heatmap blending weight

    Returns:
        (H, W, 3) uint8 RGB overlay
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img  = (image_tensor * std + mean).clamp(0, 1)
    img_u8 = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

    cam_u8  = (cam * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(cam_u8, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    blended = ((1 - alpha) * img_u8 + alpha * heatmap).astype(np.uint8)
    return blended


def _get_target_layer(model, mode: str):
    """Return the last conv layer of the backbone for Grad-CAM hooks."""
    if mode == 'hybrid':
        # Use the RGB branch of the hybrid model
        return model.rgb_branch.layer4[-1].conv2
    else:
        # RGBModel or FFTModel — both have model.features (the backbone)
        return model.features.layer4[-1].conv2


# ── Main visualisation function ────────────────────────────────────────────

def generate_gradcam_visuals(model: nn.Module,
                              loader,
                              device: torch.device,
                              mode: str,
                              save_dir: str,
                              max_per_category: int = 6,
                              correct_fname: str = 'gradcam_correct.png',
                              wrong_fname: str = 'gradcam_incorrect.png'):
    """
    Generate Grad-CAM grids for correct and incorrect predictions.

    Saves two files:
      - gradcam_correct.png   (✓ model got these right)
      - gradcam_incorrect.png (✗ failure cases)

    Args:
        model            : trained nn.Module
        loader           : DataLoader (test split preferred)
        device           : torch.device
        mode             : 'rgb', 'fft', or 'hybrid'
        save_dir         : output directory
        max_per_category : max samples per grid (correct / incorrect)
    """
    os.makedirs(save_dir, exist_ok=True)

    target_layer = _get_target_layer(model, mode)

    # GradCAM must run with gradients enabled
    correct_samples   = []
    incorrect_samples = []

    model.eval()

    for batch in loader:
        if len(correct_samples) >= max_per_category and \
           len(incorrect_samples) >= max_per_category:
            break

        if mode == 'hybrid':
            rgb_b, fft_b, labels_b = batch
        else:
            rgb_b, labels_b = batch
            fft_b = None

        for i in range(len(labels_b)):
            if len(correct_samples) >= max_per_category and \
               len(incorrect_samples) >= max_per_category:
                break

            label   = int(labels_b[i].item())
            rgb_img = rgb_b[i:i+1].to(device)          # (1,3,H,W)
            fft_img = fft_b[i:i+1].to(device) if fft_b is not None else None

            # For hybrid, we hook onto model.rgb_branch directly
            if mode == 'hybrid':
                hook_model = model
                cam_obj    = GradCAM(hook_model, model.rgb_branch.layer4[-1].conv2)

                # Override generate to use hybrid forward
                orig_gen = cam_obj.generate
                def _gen_hybrid(inp, target_class=None, _fft=fft_img):
                    cam_obj.model.eval()
                    logits = cam_obj.model(inp, _fft)
                    pred   = int(logits.argmax(dim=1).item())
                    if target_class is None:
                        target_class = pred
                    cam_obj.model.zero_grad()
                    oh = torch.zeros_like(logits)
                    oh[0, target_class] = 1.0
                    logits.backward(gradient=oh)
                    weights = cam_obj._gradients.mean(dim=[2,3], keepdim=True)
                    hmap    = (weights * cam_obj._activations).sum(dim=1, keepdim=True)
                    hmap    = F.relu(hmap)
                    hmap    = F.interpolate(hmap, size=inp.shape[2:],
                                            mode='bilinear', align_corners=False)
                    hmap    = hmap.squeeze().cpu().numpy()
                    if hmap.max() > hmap.min():
                        hmap = (hmap - hmap.min()) / (hmap.max() - hmap.min())
                    return hmap.astype(np.float32), pred

                heatmap, pred = _gen_hybrid(rgb_img)
                cam_obj.remove_hooks()

            else:
                cam_obj    = GradCAM(model, model.features.layer4[-1].conv2)
                heatmap, pred = cam_obj.generate(rgb_img)
                cam_obj.remove_hooks()

            overlay = overlay_heatmap(rgb_b[i].cpu(), heatmap)

            entry = {
                'rgb'    : rgb_b[i].cpu(),
                'cam'    : heatmap,
                'overlay': overlay,
                'label'  : label,
                'pred'   : pred,
            }

            if pred == label and len(correct_samples) < max_per_category:
                correct_samples.append(entry)
            elif pred != label and len(incorrect_samples) < max_per_category:
                incorrect_samples.append(entry)

    # ── Save grids ─────────────────────────────────────────────────────
    _save_sample_grid(
        correct_samples,
        os.path.join(save_dir, correct_fname),
        title='Grad-CAM: Correct Predictions',
    )
    _save_sample_grid(
        incorrect_samples,
        os.path.join(save_dir, wrong_fname),
        title='Grad-CAM: Failure Cases (Incorrect Predictions)',
    )

    return correct_samples, incorrect_samples


def _save_sample_grid(samples: list, save_path: str, title: str):
    """Save a grid of [original | heatmap | overlay] per sample."""
    if not samples:
        print(f"[GradCAM] No samples available for: {title}")
        return

    n           = len(samples)
    class_names = ['Real', 'Fake']
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    fig, axes = plt.subplots(n, 3, figsize=(11, 3.8 * n))
    fig.suptitle(title, fontsize=13, fontweight='bold')
    if n == 1:
        axes = [axes]

    for i, s in enumerate(samples):
        img_vis = (s['rgb'] * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
        status  = '✓' if s['pred'] == s['label'] else '✗'

        axes[i][0].imshow(img_vis)
        axes[i][0].set_title(f"Input\n True: {class_names[s['label']]}", fontsize=9)
        axes[i][0].axis('off')

        axes[i][1].imshow(s['cam'], cmap='jet')
        axes[i][1].set_title(f"Grad-CAM\n {status} Pred: {class_names[s['pred']]}",
                             fontsize=9)
        axes[i][1].axis('off')

        axes[i][2].imshow(s['overlay'])
        axes[i][2].set_title("Overlay", fontsize=9)
        axes[i][2].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[GradCAM] Saved → {save_path}")
