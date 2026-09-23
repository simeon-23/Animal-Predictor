"""Validate and split an image-folder dataset."""

import argparse
import random
import shutil
from pathlib import Path

from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_animal_root(raw_data):
    root = Path(raw_data)
    folders = [p for p in root.iterdir() if p.is_dir()]
    if len(folders) == 1 and not any(p.suffix.lower() in IMAGE_EXTENSIONS for p in root.iterdir()):
        nested = [p for p in folders[0].iterdir() if p.is_dir()]
        if nested:
            return folders[0]
    return root


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_data", required=True)
    parser.add_argument("--train_out", required=True)
    parser.add_argument("--test_out", required=True)
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--min_images", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = find_animal_root(args.raw_data)
    classes = sorted(p for p in root.iterdir() if p.is_dir())
    if len(classes) < 2:
        raise ValueError("Dataset must contain at least two animal folders")
    if not 0 < args.test_ratio < 1:
        raise ValueError("test_ratio must be between 0 and 1")

    rows = []
    for class_dir in classes:
        files = sorted(p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
        for image_file in files:
            with Image.open(image_file) as image:
                image.verify()
        if len(files) < args.min_images:
            raise ValueError(f"{class_dir.name} has {len(files)} images; need at least {args.min_images}")
        shuffled = files[:]
        random.Random(args.seed).shuffle(shuffled)
        test_count = max(1, int(round(len(shuffled) * args.test_ratio)))
        test_files = shuffled[:test_count]
        train_files = shuffled[test_count:]
        for output, selected in ((Path(args.train_out), train_files), (Path(args.test_out), test_files)):
            destination = output / class_dir.name
            destination.mkdir(parents=True, exist_ok=True)
            for source in selected:
                shutil.copy2(source, destination / source.name)
        rows.append((class_dir.name, len(train_files), len(test_files)))

    print("Animal        Train  Test")
    print("-------------------------")
    for name, train_count, test_count in rows:
        print(f"{name:<13}{train_count:<7}{test_count}")
    print(f"Found {len(classes)} animals")


if __name__ == "__main__":
    main()