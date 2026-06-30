"""
Central configuration for the project.
Keeping paths/hyperparams here (instead of hardcoding in every script)
is what makes the codebase maintainable and "production-style".
"""

import os
from dataclasses import dataclass

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
TEST_DIR = os.path.join(DATA_DIR, "test")

CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pt")

CLASS_NAMES = ["cat", "dog"]  # index 0 = cat, index 1 = dog
IMAGE_SIZE = 224


@dataclass
class TrainConfig:
    epochs: int = 10
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 2
    unfreeze_last_block: bool = True
    seed: int = 42
