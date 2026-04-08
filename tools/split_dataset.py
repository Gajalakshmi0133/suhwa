#!/usr/bin/env python3
"""
Split ASL dataset into train and test sets.

Copies images from dataset/American_Sign_Language_Letters_Multiclass/
into dataset/train/ and dataset/test/ with 80/20 split per class.
"""
import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split

def split_dataset(source_dir: Path, train_dir: Path, test_dir: Path, test_size=0.2, random_state=42):
    """Split dataset into train and test."""
    if not source_dir.exists():
        print(f"Source directory {source_dir} does not exist.")
        return

    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    classes = [d for d in source_dir.iterdir() if d.is_dir()]
    print(f"Found {len(classes)} classes: {[c.name for c in classes]}")

    for cls in classes:
        images = list(cls.glob('*.jpg'))
        if not images:
            print(f"No images in {cls}")
            continue

        train_imgs, test_imgs = train_test_split(images, test_size=test_size, random_state=random_state)

        # Create subdirs
        (train_dir / cls.name).mkdir(exist_ok=True)
        (test_dir / cls.name).mkdir(exist_ok=True)

        # Copy train
        for img in train_imgs:
            shutil.copy(img, train_dir / cls.name / img.name)

        # Copy test
        for img in test_imgs:
            shutil.copy(img, test_dir / cls.name / img.name)

        print(f"Class {cls.name}: {len(train_imgs)} train, {len(test_imgs)} test")

if __name__ == '__main__':
    base_dir = Path('dataset')
    source = base_dir / 'American_Sign_Language_Letters_Multiclass'
    train_dest = base_dir / 'train'
    test_dest = base_dir / 'test'

    split_dataset(source, train_dest, test_dest)
    print("Dataset split complete.")
