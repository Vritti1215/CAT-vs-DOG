"""
Training loop: trains the model, logs per-epoch loss/accuracy for
train and val, saves the best checkpoint (by val accuracy).

Usage:
    python src/train.py --epochs 10 --batch-size 32 --lr 0.001
"""

import argparse
import logging
import os
import time

import torch
import torch.nn as nn
from tqdm import tqdm

from config import CHECKPOINT_DIR, BEST_MODEL_PATH, TrainConfig
from dataset import get_dataloaders
from model import build_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for images, labels in tqdm(loader, leave=False):
            images, labels = images.to(device), labels.to(device)

            if train:
                optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            if train:
                loss.backward()
                optimizer.step()

            preds = outputs.argmax(dim=1)
            total_loss += loss.item() * images.size(0)
            total_correct += (preds == labels).sum().item()
            total_samples += images.size(0)

    return total_loss / total_samples, total_correct / total_samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--no-unfreeze", action="store_true",
                         help="Keep entire backbone frozen (head-only training)")
    args = parser.parse_args()

    cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        unfreeze_last_block=not args.no_unfreeze,
    )

    torch.manual_seed(cfg.seed)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    train_loader, val_loader, _, classes = get_dataloaders(
        batch_size=cfg.batch_size, num_workers=cfg.num_workers
    )
    logger.info(f"Classes: {classes}")

    model = build_model(unfreeze_last_block=cfg.unfreeze_last_block).to(device)

    criterion = nn.CrossEntropyLoss()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    best_val_acc = 0.0

    for epoch in range(1, cfg.epochs + 1):
        start = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        scheduler.step(val_acc)
        elapsed = time.time() - start

        logger.info(
            f"Epoch {epoch}/{cfg.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
            f"{elapsed:.1f}s"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            logger.info(f"  -> New best val_acc={val_acc:.4f}, checkpoint saved to {BEST_MODEL_PATH}")

    logger.info(f"Training complete. Best val_acc={best_val_acc:.4f}")


if __name__ == "__main__":
    main()
