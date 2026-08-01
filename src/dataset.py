"""
Dataset/DataLoader setup using torchvision's ImageFolder (expects
data/train/cats, data/train/dogs style structure) plus augmentation
transforms for training and plain resize/normalize for val/test.
"""

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from config import TRAIN_DIR, VAL_DIR, TEST_DIR, IMAGE_SIZE

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

eval_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def get_dataloaders(batch_size: int = 32, num_workers: int = 2):
    train_ds = datasets.ImageFolder(TRAIN_DIR, transform=train_transforms)
    val_ds = datasets.ImageFolder(VAL_DIR, transform=eval_transforms)
    test_ds = datasets.ImageFolder(TEST_DIR, transform=eval_transforms)

    # Sanity check: class_to_idx should be {'cats': 0, 'dogs': 1}
    assert train_ds.classes == val_ds.classes == test_ds.classes, \
        "Class mismatch between splits — check folder names."

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, train_ds.classes
