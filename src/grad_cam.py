"""
Grad-CAM: generates a heatmap showing which parts of the image the model
focused on to make its prediction. Works by hooking into the last
convolutional layer of MobileNetV2, capturing activations and gradients
during a forward + backward pass, and combining them into a heatmap.

Usage (standalone):
    python src/grad_cam.py --image path/to/image.jpg --checkpoint checkpoints/best_model.pt

This also exposes `generate_gradcam_overlay()` for use in the API.
"""

import argparse
import io

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from config import CLASS_NAMES
from dataset import eval_transforms, IMAGENET_MEAN, IMAGENET_STD
from model import load_model_for_inference


class GradCAM:
    """Hooks into a target layer to capture activations + gradients."""

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.activations = None
        self.gradients = None

        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor, class_idx: int = None):
        self.model.zero_grad()
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        score = output[0, class_idx]
        score.backward()

        # Global-average-pool the gradients -> channel importance weights
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = cam.squeeze().cpu().numpy()
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam, class_idx, F.softmax(output, dim=1)[0]


def overlay_heatmap_on_image(pil_image: Image.Image, cam: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """Resizes the CAM to image size and blends it as a red-hot heatmap overlay."""
    cam_resized = Image.fromarray((cam * 255).astype(np.uint8)).resize(pil_image.size, Image.BILINEAR)
    cam_arr = np.array(cam_resized).astype(np.float32) / 255.0

    # simple red-yellow heatmap colormap (no matplotlib dependency needed here)
    heatmap = np.zeros((*cam_arr.shape, 3), dtype=np.uint8)
    heatmap[..., 0] = (cam_arr * 255).astype(np.uint8)               # Red channel
    heatmap[..., 1] = (np.clip(cam_arr * 2 - 1, 0, 1) * 255).astype(np.uint8)  # Green ramps up at high activation

    base = np.array(pil_image.convert("RGB")).astype(np.float32)
    heatmap = heatmap.astype(np.float32)

    blended = base * (1 - alpha) + heatmap * alpha
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    return Image.fromarray(blended)


def generate_gradcam_overlay(model: torch.nn.Module, pil_image: Image.Image, device: str = "cpu"):
    """
    High-level helper: takes a loaded model + PIL image, returns
    (overlay_image, predicted_class_name, confidence).
    """
    target_layer = model.features[-1]  # last conv block of MobileNetV2
    cam_engine = GradCAM(model, target_layer)

    input_tensor = eval_transforms(pil_image).unsqueeze(0).to(device)
    input_tensor.requires_grad_(True)

    cam, class_idx, probs = cam_engine.generate(input_tensor)
    overlay = overlay_heatmap_on_image(pil_image, cam)

    return overlay, CLASS_NAMES[class_idx], round(probs[class_idx].item(), 4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default="gradcam_output.png")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model_for_inference(args.checkpoint, device=device)

    image = Image.open(args.image).convert("RGB")
    overlay, pred_class, confidence = generate_gradcam_overlay(model, image, device=device)

    overlay.save(args.output)
    print(f"Prediction: {pred_class} ({confidence * 100:.1f}% confidence)")
    print(f"Grad-CAM overlay saved to {args.output}")


if __name__ == "__main__":
    main()