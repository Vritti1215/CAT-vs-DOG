"""
Run inference on a single image.

Usage:
    python src/infer.py --image path/to/image.jpg --checkpoint checkpoints/best_model.pt
"""

import argparse

import torch
from PIL import Image

from config import CLASS_NAMES
from dataset import eval_transforms
from model import load_model_for_inference


def predict(image_path: str, checkpoint_path: str, device: str = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model_for_inference(checkpoint_path, device=device)

    image = Image.open(image_path).convert("RGB")
    tensor = eval_transforms(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        pred_idx = probs.argmax().item()

    return {
        "class": CLASS_NAMES[pred_idx],
        "confidence": round(probs[pred_idx].item(), 4),
        "probabilities": {CLASS_NAMES[i]: round(p.item(), 4) for i, p in enumerate(probs)},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()

    result = predict(args.image, args.checkpoint)
    print(result)
