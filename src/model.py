"""
Model: MobileNetV2 pretrained on ImageNet, with a new classifier head
for binary cat/dog classification. Last conv block can optionally be
unfrozen for fine-tuning.
"""

import torch
import torch.nn as nn
from torchvision import models


def build_model(unfreeze_last_block: bool = True) -> nn.Module:
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

    # Freeze all base layers initially
    for param in model.features.parameters():
        param.requires_grad = False

    if unfreeze_last_block:
        # Unfreeze the last few layers of the feature extractor for fine-tuning
        for param in model.features[-3:].parameters():
            param.requires_grad = True

    # Replace classifier head: binary output (2 classes -> cat/dog)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 2),
    )

    return model


def load_model_for_inference(checkpoint_path: str, device: str = "cpu") -> nn.Module:
    model = build_model(unfreeze_last_block=False)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model
