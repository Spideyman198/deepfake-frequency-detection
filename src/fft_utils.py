"""
FFT (Fast Fourier Transform) utilities for frequency-domain analysis.

Key idea:
  GAN-generated images often contain periodic upsampling artifacts that are
  invisible to the human eye but appear as distinct patterns in the frequency
  domain.  By converting images to their FFT magnitude spectrum we give the
  model a direct view of these artifacts.

References:
  - Frank et al. (2020) "Leveraging Frequency Analysis for Deep Fake Image
    Forgery Detection and Localization"
  - Durall et al. (2019) "Unmasking DeepFakes with Simple Features"
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms


# ImageNet stats used to re-normalise the FFT tensor so it can feed a
# pretrained ResNet (the per-channel mean/std values are reused here purely
# as a numerical scale — the network will learn to re-weight them).
_IMAGENET_NORM = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std =[0.229, 0.224, 0.225],
)


def rgb_to_fft(image_tensor: torch.Tensor) -> torch.Tensor:
    """
    Convert a (3, H, W) normalised RGB tensor to a (3, H, W) FFT
    log-magnitude tensor, also normalised with ImageNet statistics.

    Pipeline per channel
    --------------------
    1.  2-D FFT  →  shift zero-frequency to centre
    2.  log-magnitude  =  log( |FFT| + ε )
    3.  min-max normalise to [0, 1]
    4.  Stack channels, then apply ImageNet normalisation

    Args:
        image_tensor : Tensor of shape (3, H, W), already normalised.

    Returns:
        Tensor of shape (3, H, W) representing the FFT magnitude spectrum.
    """
    # Undo ImageNet normalisation so FFT is computed on [0,1] pixel values
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
    img_01 = (image_tensor * std + mean).clamp(0.0, 1.0)

    img_np = img_01.numpy()                # (3, H, W)  float32 in [0,1]
    fft_channels = []

    for c in range(3):
        channel  = img_np[c]                                  # (H, W)
        fft      = np.fft.fft2(channel)                       # complex
        shifted  = np.fft.fftshift(fft)                       # centre DC
        log_mag  = np.log(np.abs(shifted) + 1e-8)             # log-magnitude
        # min-max normalise to [0, 1]
        lo, hi   = log_mag.min(), log_mag.max()
        norm     = (log_mag - lo) / (hi - lo + 1e-8)
        fft_channels.append(norm.astype(np.float32))

    fft_tensor = torch.from_numpy(np.stack(fft_channels, axis=0))  # (3,H,W)
    # Normalise with ImageNet stats so pretrained heads get a familiar scale
    fft_tensor = _IMAGENET_NORM(fft_tensor)
    return fft_tensor


def compute_fft_spectrum_gray(image_tensor: torch.Tensor) -> np.ndarray:
    """
    Compute a grayscale (single-channel) log-magnitude FFT spectrum for
    *visualisation only* (no normalisation applied).

    Args:
        image_tensor : (3, H, W) tensor (may be normalised with ImageNet stats)

    Returns:
        (H, W) float array of log-magnitude values.
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img  = (image_tensor * std + mean).clamp(0, 1)

    gray    = img.mean(dim=0).numpy()               # average over channels
    fft     = np.fft.fft2(gray)
    shifted = np.fft.fftshift(fft)
    return np.log(np.abs(shifted) + 1.0)            # log-magnitude


def save_fft_comparison(dataloader, save_dir: str, num_samples: int = 8):
    """
    Save a side-by-side grid of RGB images and their FFT spectra.

    Works with 'rgb' or 'hybrid' dataloaders.

    Args:
        dataloader  : DataLoader that yields (rgb_batch, [fft_batch,] labels)
        save_dir    : Directory where the figure will be written
        num_samples : Number of image pairs to include in the grid
    """
    os.makedirs(save_dir, exist_ok=True)

    # Grab one batch
    batch = next(iter(dataloader))
    if len(batch) == 3:          # hybrid
        rgb_batch, _, labels = batch
    else:                        # rgb
        rgb_batch, labels = batch

    n = min(num_samples, len(rgb_batch))

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    fig, axes = plt.subplots(n, 2, figsize=(7, n * 3.2))
    if n == 1:
        axes = [axes]

    for i in range(n):
        rgb_vis  = (rgb_batch[i] * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
        fft_spec = compute_fft_spectrum_gray(rgb_batch[i])
        label    = "Real" if labels[i].item() == 0 else "Fake"

        axes[i][0].imshow(rgb_vis)
        axes[i][0].set_title(f"RGB — {label}", fontsize=10)
        axes[i][0].axis('off')

        axes[i][1].imshow(fft_spec, cmap='inferno')
        axes[i][1].set_title(f"FFT Spectrum — {label}", fontsize=10)
        axes[i][1].axis('off')

    plt.suptitle("RGB vs Frequency-Domain Comparison", fontsize=13,
                 fontweight='bold', y=1.01)
    plt.tight_layout()
    out = os.path.join(save_dir, 'rgb_vs_fft_comparison.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[FFT] Comparison grid saved → {out}")
