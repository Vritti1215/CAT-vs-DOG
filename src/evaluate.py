"""
Evaluates a trained checkpoint on the test set:
- accuracy, precision, recall, F1
- confusion matrix (printed + saved as PNG)
- saves misclassified images for manual error analysis

Usage:
    python src/evaluate.py --checkpoint checkpoints/best_model.pt
"""

import argparse
import os
import shutil

import torch
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix, ConfusionMatrixDisplay
)

from config import CHECKPOINT_DIR, CLASS_NAMES
from dataset import get_dataloaders
from model import load_model_for_inference


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model_for_inference(args.checkpoint, device=device)

    _, _, test_loader, classes = get_dataloaders(batch_size=args.batch_size)

    all_preds, all_labels, all_paths = [], [], []
    samples = test_loader.dataset.samples  # list of (path, label)

    idx = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu()

            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

            batch_size = images.size(0)
            all_paths.extend([samples[i][0] for i in range(idx, idx + batch_size)])
            idx += batch_size

    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="binary"
    )

    print("\n=== Test Set Metrics ===")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    cm = confusion_matrix(all_labels, all_preds)
    print("\nConfusion Matrix:")
    print(cm)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap="Blues")
    cm_path = os.path.join(CHECKPOINT_DIR, "confusion_matrix.png")
    plt.savefig(cm_path)
    print(f"\nConfusion matrix saved to {cm_path}")

    # Save misclassified examples for manual error analysis
    mis_dir = os.path.join(CHECKPOINT_DIR, "misclassified")
    if os.path.exists(mis_dir):
        shutil.rmtree(mis_dir)
    os.makedirs(mis_dir, exist_ok=True)

    mis_count = 0
    for path, true_label, pred_label in zip(all_paths, all_labels, all_preds):
        if true_label != pred_label:
            fname = os.path.basename(path)
            true_name = CLASS_NAMES[true_label]
            pred_name = CLASS_NAMES[pred_label]
            dest = os.path.join(mis_dir, f"true_{true_name}_pred_{pred_name}_{fname}")
            shutil.copy(path, dest)
            mis_count += 1

    print(f"Saved {mis_count} misclassified images to {mis_dir} for error analysis.")


if __name__ == "__main__":
    main()
