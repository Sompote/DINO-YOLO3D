#!/usr/bin/env python3
"""
Create validation split for KITTI dataset
Splits training images into train/val directories
"""

import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split

def create_val_split(kitti_path, val_percent=10, seed=42):
    """
    Create validation split by copying images to val directory

    Args:
        kitti_path: Path to KITTI dataset
        val_percent: Percentage of data for validation
        seed: Random seed for reproducibility
    """
    kitti_path = Path(kitti_path)
    img_dir = kitti_path / 'training' / 'image_2'
    label_dir = kitti_path / 'training' / 'label_2'
    calib_dir = kitti_path / 'training' / 'calib'

    # Create val directories
    val_img_dir = kitti_path / 'val' / 'image_2'
    val_label_dir = kitti_path / 'val' / 'label_2'
    val_calib_dir = kitti_path / 'val' / 'calib'

    val_img_dir.mkdir(parents=True, exist_ok=True)
    val_label_dir.mkdir(parents=True, exist_ok=True)
    val_calib_dir.mkdir(parents=True, exist_ok=True)

    # Get all image filenames
    img_files = sorted([f.name for f in img_dir.glob('*.png')])
    total_images = len(img_files)

    # Calculate validation size
    val_size = int(total_images * val_percent / 100)

    print(f"Dataset: {total_images} images")
    print(f"Creating {val_percent}% validation split: {val_size} images")
    print(f"Training will use: {total_images - val_size} images")

    # Split into train/val
    train_imgs, val_imgs = train_test_split(
        img_files,
        test_size=val_size,
        random_state=seed,
        shuffle=True
    )

    print(f"\nValidation images: {len(val_imgs)}")

    # Copy validation files
    print("\nCopying validation files...")
    for img_file in val_imgs:
        # Copy image
        shutil.copy2(img_dir / img_file, val_img_dir / img_file)

        # Copy label (same filename)
        label_file = img_file.replace('.png', '.txt')
        if (label_dir / label_file).exists():
            shutil.copy2(label_dir / label_file, val_label_dir / label_file)

        # Copy calibration
        if (calib_dir / img_file).exists():
            shutil.copy2(calib_dir / img_file, val_calib_dir / img_file)

    print(f"✅ Created validation split: {val_percent}% ({val_size} images)")
    print(f"   Val images: {val_img_dir}")
    print(f"   Val labels: {val_label_dir}")
    print(f"   Val calib: {val_calib_dir}")

    # Update kitti-3d.yaml
    yaml_file = kitti_path.parent.parent / 'kitti-3d.yaml'
    if yaml_file.exists():
        print(f"\n📝 Updating {yaml_file}...")
        with open(yaml_file, 'r') as f:
            content = f.read()

        # Update paths
        content = content.replace(
            "val: training/image_2",
            "val: val/image_2"
        )

        with open(yaml_file, 'w') as f:
            f.write(content)

        print("✅ Updated kitti-3d.yaml with new validation path")

    return len(val_imgs)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Create KITTI validation split')
    parser.add_argument(
        '--kitti',
        type=str,
        default='/Users/sompoteyouwai/Downloads/datakitti/datasets/kitti',
        help='Path to KITTI dataset'
    )
    parser.add_argument(
        '--val-percent',
        type=int,
        default=10,
        help='Validation percentage (default: 10)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )

    args = parser.parse_args()

    # Create validation split
    create_val_split(
        kitti_path=args.kitti,
        val_percent=args.val_percent,
        seed=args.seed
    )

    print("\n🎉 Done! You can now run training with:")
    print(f"   python yolo3d.py train --data kitti-3d.yaml --valpercent {args.val_percent}")
