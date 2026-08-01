"""
Basic unit tests. Run with:
    pytest tests/

Note: these assume data/train and data/val exist with at least one
image per class (run prepare_data.py first). The transform shape test
doesn't need real data.
"""

import os
import sys

import torch
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from dataset import eval_transforms  # noqa: E402
from model import build_model  # noqa: E402


def test_eval_transform_output_shape():
    img = Image.new("RGB", (300, 400), color=(255, 0, 0))
    tensor = eval_transforms(img)
    assert tensor.shape == (3, 224, 224)


def test_model_forward_pass():
    model = build_model(unfreeze_last_block=False)
    model.eval()
    dummy_input = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        output = model(dummy_input)
    assert output.shape == (2, 2)  # batch_size=2, 2 classes


def test_model_output_is_finite():
    model = build_model(unfreeze_last_block=False)
    model.eval()
    dummy_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        output = model(dummy_input)
    assert torch.isfinite(output).all()
