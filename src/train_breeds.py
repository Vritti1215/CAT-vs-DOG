"""
Trains a MobileNetV2 breed classifier on the Oxford-IIIT Pet dataset (37 breeds).
Run prepare_breeds.py first to download and organise the data.

Usage:
    python src/train_breeds.py --epochs 15 --batch-size 32

Output: checkpoints/breed_model.pt  (best val accuracy checkpoint)
"""

import argparse
import logging
import os
import time

import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR    = os.path.join(PROJECT_ROOT, "data", "breeds", "train")
VAL_DIR      = os.path.join(PROJECT_ROOT, "data", "breeds", "val")
CHECKPOINT   = os.path.join(PROJECT_ROOT, "checkpoints", "breed_model.pt")
CLASSES_FILE = os.path.join(PROJECT_ROOT, "checkpoints", "breed_classes.txt")

IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

train_tf = transforms.Compose([
    transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

val_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def build_breed_model(num_classes: int) -> nn.Module:
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    # Freeze base, unfreeze last 4 layers for fine-tuning
    for p in model.features.parameters():
        p.requires_grad = False
    for p in model.features[-4:].parameters():
        p.requires_grad = True

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes),
    )
    return model


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss, total_correct, total = 0.0, 0, 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
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
            total_loss    += loss.item() * images.size(0)
            total_correct += (preds == labels).sum().item()
            total         += images.size(0)

    return total_loss / total, total_correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int,   default=15)
    parser.add_argument("--batch-size", type=int,   default=32)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--workers",    type=int,   default=2)
    args = parser.parse_args()

    if not os.path.isdir(TRAIN_DIR):
        print("ERROR: Train dir not found. Run: python src/prepare_breeds.py first.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    train_ds = datasets.ImageFolder(TRAIN_DIR, transform=train_tf)
    val_ds   = datasets.ImageFolder(VAL_DIR,   transform=val_tf)
    classes  = train_ds.classes
    logger.info(f"Breeds: {len(classes)}  Train: {len(train_ds)}  Val: {len(val_ds)}")

    # Save class list so inference knows the label mapping
    os.makedirs(os.path.dirname(CLASSES_FILE), exist_ok=True)
    with open(CLASSES_FILE, "w") as f:
        f.write("\n".join(classes))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers, pin_memory=True)

    model     = build_breed_model(len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        vl_loss, vl_acc = run_epoch(model, val_loader,   criterion, optimizer, device, train=False)
        scheduler.step()

        logger.info(
            f"Epoch {epoch:>2}/{args.epochs} | "
            f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} | "
            f"val_loss={vl_loss:.4f} val_acc={vl_acc:.4f} | "
            f"{time.time()-t0:.1f}s"
        )

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save({"state_dict": model.state_dict(), "classes": classes}, CHECKPOINT)
            logger.info(f"  → New best val_acc={vl_acc:.4f}, saved to {CHECKPOINT}")

    logger.info(f"Training complete. Best val_acc={best_val_acc:.4f}")
    logger.info(f"Checkpoint: {CHECKPOINT}")


if __name__ == "__main__":
    main()