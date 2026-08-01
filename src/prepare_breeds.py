"""
Downloads and prepares the Oxford-IIIT Pet dataset (37 breeds).
Run this ONCE before training the breed model.

Usage:
    python src/prepare_breeds.py

What it does:
1. Downloads the Oxford-IIIT Pet dataset (~800MB) from official source
2. Organises images into data/breeds/train and data/breeds/val (80/20 split)
3. Prints class distribution

After running this, train with:
    python src/train_breeds.py
"""

import os
import random
import shutil
import tarfile
import urllib.request

random.seed(42)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BREEDS_DIR   = os.path.join(PROJECT_ROOT, "data", "breeds")
TRAIN_DIR    = os.path.join(BREEDS_DIR, "train")
VAL_DIR      = os.path.join(BREEDS_DIR, "val")
RAW_DIR      = os.path.join(BREEDS_DIR, "raw")

IMAGES_URL  = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"
IMAGES_TAR  = os.path.join(RAW_DIR, "images.tar.gz")

VAL_SPLIT = 0.20


def download_file(url: str, dest: str):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest):
        print(f"  Already downloaded: {dest}")
        return
    print(f"  Downloading {url} …")
    def _progress(count, block, total):
        pct = min(count * block / total * 100, 100)
        print(f"\r  {pct:.1f}%", end="", flush=True)
    urllib.request.urlretrieve(url, dest, reporthook=_progress)
    print()


def extract(tar_path: str, dest: str):
    print(f"  Extracting {os.path.basename(tar_path)} …")
    with tarfile.open(tar_path, "r:gz") as t:
        t.extractall(dest)


def breed_name_from_filename(fname: str) -> str:
    """
    Oxford-IIIT filenames are like:
      Abyssinian_100.jpg  →  Abyssinian
      english_setter_10.jpg  →  English Setter
    The breed name is everything before the last _<number>.
    """
    stem = os.path.splitext(fname)[0]
    parts = stem.rsplit("_", 1)
    raw = parts[0]
    return raw.replace("_", " ").title()


def main():
    os.makedirs(RAW_DIR, exist_ok=True)

    print("=== Step 1: Download ===")
    download_file(IMAGES_URL, IMAGES_TAR)

    print("=== Step 2: Extract ===")
    images_dir = os.path.join(RAW_DIR, "images")
    if not os.path.exists(images_dir):
        extract(IMAGES_TAR, RAW_DIR)

    print("=== Step 3: Organise into train/val ===")
    # Collect all valid images grouped by breed
    breed_files: dict[str, list] = {}
    for fname in sorted(os.listdir(images_dir)):
        if not fname.lower().endswith(".jpg"):
            continue
        # Skip files that start with a dot or are clearly not pet images
        if fname.startswith("."):
            continue
        breed = breed_name_from_filename(fname)
        breed_files.setdefault(breed, []).append(fname)

    print(f"  Found {len(breed_files)} breeds, "
          f"{sum(len(v) for v in breed_files.values())} total images")

    # Clear existing split dirs
    for d in [TRAIN_DIR, VAL_DIR]:
        if os.path.exists(d):
            shutil.rmtree(d)

    for breed, files in sorted(breed_files.items()):
        random.shuffle(files)
        n_val   = max(1, int(len(files) * VAL_SPLIT))
        val_f   = files[:n_val]
        train_f = files[n_val:]

        for split, file_list in [("train", train_f), ("val", val_f)]:
            dest_dir = os.path.join(BREEDS_DIR, split, breed)
            os.makedirs(dest_dir, exist_ok=True)
            for f in file_list:
                shutil.copy(os.path.join(images_dir, f), os.path.join(dest_dir, f))

        print(f"  {breed:<35} train={len(train_f):>3}  val={len(val_f):>3}")

    print("\n=== Done ===")
    print(f"Train dir: {TRAIN_DIR}")
    print(f"Val dir:   {VAL_DIR}")
    print("Now run: python src/train_breeds.py")


if __name__ == "__main__":
    main()