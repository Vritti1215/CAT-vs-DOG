"""
Prepares the dataset:
1. (optional) Reorganizes a flat Kaggle-style folder into data/raw/cats, data/raw/dogs
2. Validates images (skips corrupted files)
3. Splits into train/val/test (70/15/15) preserving class balance
4. Prints class distribution stats

Usage:
    python src/prepare_data.py                 # split data/raw into train/val/test
    python src/prepare_data.py --reorganize     # first sort a flat folder into cats/dogs
"""

import argparse
import os
import random
import shutil
from PIL import Image

from config import RAW_DIR, TRAIN_DIR, VAL_DIR, TEST_DIR

random.seed(42)

SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}


def reorganize_flat_folder(flat_dir: str):
    """
    If your Kaggle download is a single folder with files like
    'cat.0.jpg', 'dog.0.jpg', this sorts them into data/raw/cats and data/raw/dogs.
    """
    cats_dir = os.path.join(RAW_DIR, "cats")
    dogs_dir = os.path.join(RAW_DIR, "dogs")
    os.makedirs(cats_dir, exist_ok=True)
    os.makedirs(dogs_dir, exist_ok=True)

    moved = 0
    for fname in os.listdir(flat_dir):
        lower = fname.lower()
        src = os.path.join(flat_dir, fname)
        if lower.startswith("cat"):
            shutil.copy(src, os.path.join(cats_dir, fname))
            moved += 1
        elif lower.startswith("dog"):
            shutil.copy(src, os.path.join(dogs_dir, fname))
            moved += 1
    print(f"Reorganized {moved} files into {RAW_DIR}/cats and {RAW_DIR}/dogs")


def is_valid_image(path: str) -> bool:
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def split_and_copy(class_name: str, src_dir: str):
    files = [f for f in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, f))]
    random.shuffle(files)

    valid_files = []
    corrupt_count = 0
    for f in files:
        full_path = os.path.join(src_dir, f)
        if is_valid_image(full_path):
            valid_files.append(f)
        else:
            corrupt_count += 1

    n = len(valid_files)
    n_train = int(n * SPLIT_RATIOS["train"])
    n_val = int(n * SPLIT_RATIOS["val"])

    splits = {
        "train": valid_files[:n_train],
        "val": valid_files[n_train:n_train + n_val],
        "test": valid_files[n_train + n_val:],
    }

    dest_roots = {"train": TRAIN_DIR, "val": VAL_DIR, "test": TEST_DIR}

    for split_name, file_list in splits.items():
        dest_dir = os.path.join(dest_roots[split_name], class_name)
        os.makedirs(dest_dir, exist_ok=True)
        for f in file_list:
            shutil.copy(os.path.join(src_dir, f), os.path.join(dest_dir, f))

    print(f"[{class_name}] total={n} corrupt_skipped={corrupt_count} "
          f"train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reorganize", type=str, default=None,
                         help="Path to a flat folder of cat.*/dog.* images to sort first")
    args = parser.parse_args()

    if args.reorganize:
        reorganize_flat_folder(args.reorganize)

    for class_name in ["cats", "dogs"]:
        src_dir = os.path.join(RAW_DIR, class_name)
        if not os.path.isdir(src_dir):
            print(f"WARNING: {src_dir} not found, skipping. "
                  f"Make sure data/raw/cats and data/raw/dogs exist.")
            continue
        split_and_copy(class_name, src_dir)

    print("\nData preparation complete.")


if __name__ == "__main__":
    main()
