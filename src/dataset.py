"""
Dataset loading utilities for Deepfake Detection.

Supports three modes:
  - 'rgb'    : Load raw RGB images (Experiment 1)
  - 'fft'    : Load FFT frequency-domain images (Experiment 2)
  - 'hybrid' : Load both RGB and FFT together (Experiment 3)
"""

import os
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from src.fft_utils import rgb_to_fft


class DeepfakeDataset(Dataset):
    """
    Loads real/fake face images from a directory with the structure:
        root_dir/
            real/  (label = 0)
            fake/  (label = 1)

    Args:
        root_dir   : Path to split directory (e.g. data/train)
        mode       : 'rgb', 'fft', or 'hybrid'
        transform  : torchvision transforms applied to the PIL image before
                     converting to tensor.  Must include ToTensor().
        max_samples: Cap on total images (balanced per class).  None = all.
        seed       : Random seed used when sub-sampling.
    """

    def __init__(self, root_dir, mode='rgb', transform=None,
                 max_samples=None, seed=42):
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform

        self.samples: list[str] = []
        self.labels:  list[int] = []

        # ── Collect paths ────────────────────────────────────────────────
        for label, class_name in enumerate(['real', 'fake']):
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                print(f"[Warning] Folder not found: {class_dir}")
                continue
            valid_ext = {'.jpg', '.jpeg', '.png', '.webp'}
            files = [
                os.path.join(class_dir, f)
                for f in os.listdir(class_dir)
                if os.path.splitext(f)[1].lower() in valid_ext
            ]
            self.samples.extend(files)
            self.labels.extend([label] * len(files))

        # ── Optional sub-sampling (balanced) ────────────────────────────
        if max_samples is not None:
            rng = random.Random(seed)
            real_idx = [i for i, l in enumerate(self.labels) if l == 0]
            fake_idx = [i for i, l in enumerate(self.labels) if l == 1]
            half = max_samples // 2
            rng.shuffle(real_idx)
            rng.shuffle(fake_idx)
            selected = sorted(real_idx[:half] + fake_idx[:half])
            self.samples = [self.samples[i] for i in selected]
            self.labels  = [self.labels[i]  for i in selected]

        n_real = self.labels.count(0)
        n_fake = self.labels.count(1)
        print(f"[Dataset] {root_dir}  |  real={n_real}  fake={n_fake}  "
              f"total={len(self.samples)}  mode={mode}")

    # ── Dunder methods ────────────────────────────────────────────────────

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path  = self.samples[idx]
        label = self.labels[idx]
        image = Image.open(path).convert('RGB')

        if self.transform:
            rgb_tensor = self.transform(image)
        else:
            rgb_tensor = transforms.ToTensor()(image)

        if self.mode == 'rgb':
            return rgb_tensor, label

        elif self.mode == 'fft':
            fft_tensor = rgb_to_fft(rgb_tensor)
            return fft_tensor, label

        elif self.mode == 'hybrid':
            fft_tensor = rgb_to_fft(rgb_tensor)
            return rgb_tensor, fft_tensor, label

        else:
            raise ValueError(f"Unknown mode '{self.mode}'. "
                             "Choose 'rgb', 'fft', or 'hybrid'.")


# ── Transform factories ────────────────────────────────────────────────────

def get_transforms(train: bool = True, image_size: int = 224) -> transforms.Compose:
    """
    Return image transforms.  Training adds light augmentation;
    validation/test use only resize + normalize.
    """
    # ImageNet statistics work well for pretrained ResNet / EfficientNet
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    if train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])


# ── DataLoader factory ─────────────────────────────────────────────────────

def _hybrid_collate(batch):
    """Custom collate for 'hybrid' mode (3-element tuples)."""
    rgbs   = torch.stack([b[0] for b in batch])
    ffts   = torch.stack([b[1] for b in batch])
    labels = torch.tensor([b[2] for b in batch], dtype=torch.long)
    return rgbs, ffts, labels


def get_dataloaders(data_dir: str,
                    mode: str        = 'rgb',
                    batch_size: int  = 32,
                    max_samples: int = None,
                    image_size: int  = 224,
                    num_workers: int = 0) -> dict:
    """
    Build train / val / test DataLoaders.

    Args:
        data_dir    : Root directory that contains train/, val/, test/
        mode        : 'rgb', 'fft', or 'hybrid'
        batch_size  : Images per batch
        max_samples : Max samples for the training split (balanced).
                      val and test use max_samples // 5 each.
        image_size  : Spatial size to resize images to
        num_workers : Worker processes for DataLoader (0 = main thread;
                      safe default on Windows)

    Returns:
        Dict  {'train': DataLoader, 'val': DataLoader, 'test': DataLoader}
    """
    train_tf = get_transforms(train=True,  image_size=image_size)
    val_tf   = get_transforms(train=False, image_size=image_size)

    # Scale down val/test subsets proportionally
    val_max  = (max_samples // 5) if max_samples else None
    test_max = (max_samples // 5) if max_samples else None

    datasets = {
        'train': DeepfakeDataset(
            os.path.join(data_dir, 'train'),
            mode=mode, transform=train_tf, max_samples=max_samples),
        'val':   DeepfakeDataset(
            os.path.join(data_dir, 'val'),
            mode=mode, transform=val_tf,   max_samples=val_max),
        'test':  DeepfakeDataset(
            os.path.join(data_dir, 'test'),
            mode=mode, transform=val_tf,   max_samples=test_max),
    }

    collate = _hybrid_collate if mode == 'hybrid' else None

    loaders = {}
    for split, ds in datasets.items():
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=collate,
        )

    return loaders
