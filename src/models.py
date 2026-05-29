"""
Model architectures for the three experiments:

  1. RGBModel    — pretrained ResNet18 / EfficientNet-B0 on RGB images
  2. FFTModel    — same architecture, trained from scratch on FFT images
  3. HybridModel — dual-branch fusion of RGB and FFT features
"""

import torch
import torch.nn as nn

try:
    # torchvision >= 0.13 uses the Weights API
    from torchvision.models import (
        resnet18, ResNet18_Weights,
        efficientnet_b0, EfficientNet_B0_Weights,
    )
    def _resnet18(pretrained: bool):
        w = ResNet18_Weights.DEFAULT if pretrained else None
        return resnet18(weights=w)
    def _efficientnet_b0(pretrained: bool):
        w = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        return efficientnet_b0(weights=w)
except ImportError:
    # older torchvision
    from torchvision import models
    def _resnet18(pretrained: bool):
        return models.resnet18(pretrained=pretrained)
    def _efficientnet_b0(pretrained: bool):
        return models.efficientnet_b0(pretrained=pretrained)


# ── Helpers ────────────────────────────────────────────────────────────────

def _build_backbone(backbone: str, pretrained: bool):
    """Return (feature_extractor, feature_dim) for a given backbone."""
    if backbone == 'resnet18':
        m    = _resnet18(pretrained)
        dim  = m.fc.in_features           # 512
        m.fc = nn.Identity()              # strip final classifier
        return m, dim

    elif backbone == 'efficientnet_b0':
        m   = _efficientnet_b0(pretrained)
        dim = m.classifier[1].in_features  # 1280
        m.classifier = nn.Identity()
        return m, dim

    else:
        raise ValueError(f"Unknown backbone '{backbone}'. "
                         "Choose 'resnet18' or 'efficientnet_b0'.")


# ── Experiment 1: RGB Model ────────────────────────────────────────────────

class RGBModel(nn.Module):
    """
    Spatial baseline: a pretrained CNN fine-tuned to classify real vs fake
    images purely from pixel (RGB) information.

    Architecture:
        ResNet18 (ImageNet pretrained)  →  Dropout  →  Linear(2)
    """

    def __init__(self, backbone: str = 'resnet18', pretrained: bool = True,
                 num_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        self.backbone_name = backbone
        self.features, feat_dim = _build_backbone(backbone, pretrained)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats  = self.features(x)   # (B, feat_dim)
        logits = self.classifier(feats)
        return logits


# ── Experiment 2: FFT Model ────────────────────────────────────────────────

class FFTModel(nn.Module):
    """
    Frequency-domain model: same architecture as RGBModel but trained on
    FFT log-magnitude images.

    We intentionally *skip* ImageNet weights because FFT spectra have very
    different statistics from natural images.

    Architecture:
        ResNet18 (random init)  →  Dropout  →  Linear(2)
    """

    def __init__(self, backbone: str = 'resnet18', num_classes: int = 2,
                 dropout: float = 0.3):
        super().__init__()
        self.backbone_name = backbone
        # pretrained=False: FFT images ≠ natural images
        self.features, feat_dim = _build_backbone(backbone, pretrained=False)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats  = self.features(x)
        logits = self.classifier(feats)
        return logits


# ── Experiment 3: Hybrid Model ─────────────────────────────────────────────

class HybridModel(nn.Module):
    """
    Dual-branch fusion model that jointly processes RGB and FFT images.

    Architecture
    ────────────
    RGB  input  →  ResNet18 (pretrained)   →  512-d feature vector  ─┐
                                                                       ├→ concat(1024-d)
    FFT  input  →  ResNet18 (random init)  →  512-d feature vector  ─┘
                                                                       ↓
                                              Dropout → FC(256) → ReLU → Dropout → FC(2)

    The two branches share the same *architecture* but have different
    initialisations (and will learn different weights):
      - RGB branch  : ImageNet pretrained  (leverages natural-image features)
      - FFT branch  : Random init          (learns frequency-domain patterns)
    """

    def __init__(self, backbone: str = 'resnet18', pretrained_rgb: bool = True,
                 num_classes: int = 2, dropout: float = 0.4):
        super().__init__()

        self.rgb_branch, rgb_dim = _build_backbone(backbone, pretrained=pretrained_rgb)
        self.fft_branch, fft_dim = _build_backbone(backbone, pretrained=False)

        fused_dim = rgb_dim + fft_dim          # 1024 for ResNet18

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fused_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.75),
            nn.Linear(256, num_classes),
        )

    def forward(self, rgb: torch.Tensor, fft: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rgb : (B, 3, H, W) normalised RGB tensor
            fft : (B, 3, H, W) normalised FFT tensor
        Returns:
            logits : (B, num_classes)
        """
        rgb_feats = self.rgb_branch(rgb)               # (B, 512)
        fft_feats = self.fft_branch(fft)               # (B, 512)
        fused     = torch.cat([rgb_feats, fft_feats], dim=1)  # (B, 1024)
        return self.classifier(fused)


# ── Factory function ────────────────────────────────────────────────────────

def build_model(experiment: str, backbone: str = 'resnet18',
                pretrained: bool = True) -> nn.Module:
    """
    Convenience factory.

    Args:
        experiment : 'rgb', 'fft', or 'hybrid'
        backbone   : 'resnet18' or 'efficientnet_b0'
        pretrained : Use ImageNet pretrained weights for the RGB branch

    Returns:
        model : nn.Module ready for training
    """
    if experiment == 'rgb':
        return RGBModel(backbone=backbone, pretrained=pretrained)
    elif experiment == 'fft':
        return FFTModel(backbone=backbone)
    elif experiment == 'hybrid':
        return HybridModel(backbone=backbone, pretrained_rgb=pretrained)
    else:
        raise ValueError(f"Unknown experiment '{experiment}'. "
                         "Choose 'rgb', 'fft', or 'hybrid'.")
