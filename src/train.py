"""
Training utilities: single-epoch loops, full training pipeline with
early stopping, LR scheduling, and checkpoint saving.
"""

import json
import os
import time

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau


# ── Single-epoch helpers ───────────────────────────────────────────────────

def _unpack_batch(batch, mode: str, device: torch.device):
    """Return (inputs, labels) in a mode-agnostic way."""
    if mode == 'hybrid':
        rgb, fft, labels = batch
        return (rgb.to(device), fft.to(device)), labels.to(device)
    else:
        images, labels = batch
        return (images.to(device),), labels.to(device)


def _forward(model, inputs, mode: str) -> torch.Tensor:
    """Run forward pass; hybrid model receives two tensors."""
    if mode == 'hybrid':
        return model(*inputs)
    return model(inputs[0])


def train_one_epoch(model, loader, criterion, optimizer,
                    device, mode: str) -> tuple[float, float]:
    """
    One full pass over the training DataLoader.

    Returns:
        avg_loss : mean cross-entropy loss
        accuracy : percentage correct  (0–100)
    """
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for step, batch in enumerate(loader):
        inputs, labels = _unpack_batch(batch, mode, device)

        optimizer.zero_grad()
        outputs = _forward(model, inputs, mode)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds       = outputs.argmax(dim=1)
        correct    += preds.eq(labels).sum().item()
        total      += labels.size(0)

        if (step + 1) % 50 == 0:
            print(f"    step {step+1:>4}/{len(loader)}  "
                  f"loss={loss.item():.4f}  "
                  f"acc={100.*correct/total:.1f}%")

    return total_loss / len(loader), 100.0 * correct / total


@torch.no_grad()
def evaluate_one_epoch(model, loader, criterion,
                       device, mode: str) -> tuple[float, float]:
    """
    One full pass over a validation / test DataLoader (no gradient updates).

    Returns:
        avg_loss : mean cross-entropy loss
        accuracy : percentage correct  (0–100)
    """
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for batch in loader:
        inputs, labels = _unpack_batch(batch, mode, device)
        outputs        = _forward(model, inputs, mode)
        loss           = criterion(outputs, labels)

        total_loss += loss.item()
        preds       = outputs.argmax(dim=1)
        correct    += preds.eq(labels).sum().item()
        total      += labels.size(0)

    return total_loss / len(loader), 100.0 * correct / total


# ── Full training pipeline ─────────────────────────────────────────────────

def train_model(model: nn.Module,
                dataloaders: dict,
                save_path: str,
                mode: str        = 'rgb',
                epochs: int      = 10,
                lr: float        = 1e-4,
                weight_decay: float = 1e-4,
                patience: int    = 3,
                device           = None) -> dict:
    """
    Train a model with:
      • Adam optimiser + ReduceLROnPlateau scheduler
      • Best-model checkpointing (lowest validation loss)
      • Early stopping when validation loss does not improve for `patience` epochs

    Args:
        model       : nn.Module to train
        dataloaders : dict with keys 'train' and 'val'
        save_path   : .pth file path for the best model
        mode        : 'rgb', 'fft', or 'hybrid'
        epochs      : Maximum epochs
        lr          : Initial learning rate
        weight_decay: L2 regularisation
        patience    : Early-stop epochs without improvement
        device      : torch.device (auto-detected if None)

    Returns:
        history : dict with keys 'train_loss', 'train_acc',
                  'val_loss', 'val_acc'  (lists, one value per epoch)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*55}")
    print(f"  Device  : {device}")
    print(f"  Mode    : {mode}")
    print(f"  Epochs  : {epochs}  (early stop patience={patience})")
    print(f"  LR      : {lr}  |  weight_decay={weight_decay}")
    print(f"{'='*55}\n")

    model     = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                  patience=2, verbose=True)

    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)

    history = {'train_loss': [], 'train_acc': [],
               'val_loss':   [], 'val_acc':   []}

    best_val_loss   = float('inf')
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        print(f"Epoch [{epoch}/{epochs}]")
        print("-" * 40)

        train_loss, train_acc = train_one_epoch(
            model, dataloaders['train'], criterion, optimizer, device, mode)

        val_loss, val_acc = evaluate_one_epoch(
            model, dataloaders['val'], criterion, device, mode)

        scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        elapsed = time.time() - t0
        print(f"\n  Train  loss={train_loss:.4f}  acc={train_acc:.2f}%")
        print(f"  Val    loss={val_loss:.4f}  acc={val_acc:.2f}%")
        print(f"  Time   {elapsed:.1f}s\n")

        # ── Checkpoint ─────────────────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"  ✓ Saved best model  →  {save_path}\n")
        else:
            patience_counter += 1
            print(f"  No improvement.  Patience counter: "
                  f"{patience_counter}/{patience}\n")
            if patience_counter >= patience:
                print(f"  Early stopping triggered after epoch {epoch}.\n")
                break

    # Persist training curves alongside the model
    hist_path = save_path.replace('.pth', '_history.json')
    with open(hist_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Training history  →  {hist_path}")

    return history
