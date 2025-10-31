"""
KITTI 3D Object Detection Dataset Download Script

Developed by AI Research Group
Department of Civil Engineering
King Mongkut's University of Technology Thonburi (KMUTT)
Bangkok, Thailand

This script downloads and prepares the KITTI 3D Object Detection dataset
for training with YOLOv12-3D.

IMPORTANT:
- KITTI dataset requires manual download due to license agreement
- You must register at http://www.cvlibs.net/datasets/kitti/
- This script helps organize the downloaded files

Usage:
    python scripts/download_kitti.py --data_dir ./datasets/kitti
"""

import argparse
import os
import shutil
import zipfile
from pathlib import Path
import requests
from tqdm import tqdm


def print_header():
    """Print script header."""
    print("=" * 80)
    print("KITTI 3D Object Detection Dataset Setup")
    print("Developed by AI Research Group, Department of Civil Engineering, KMUTT")
    print("=" * 80)
    print()


def print_instructions():
    """Print download instructions."""
    print("\n" + "=" * 80)
    print("MANUAL DOWNLOAD REQUIRED")
    print("=" * 80)
    print("\nThe KITTI dataset requires accepting a license agreement.")
    print("Please follow these steps:\n")
    print("1. Visit: http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d")
    print("2. Create an account or log in")
    print("3. Download the following files:\n")
    print("   Required files:")
    print("   ├─ Left color images of object data set (12 GB)")
    print("   │  File: data_object_image_2.zip")
    print("   ├─ Training labels of object data set (5 MB)")
    print("   │  File: data_object_label_2.zip")
    print("   └─ Camera calibration matrices of object data set (16 MB)")
    print("      File: data_object_calib.zip")
    print("\n4. Place the downloaded .zip files in the download directory")
    print("5. Run this script again to extract and organize the files")
    print("=" * 80 + "\n")


def check_downloaded_files(download_dir):
    """Check if required files are downloaded."""
    download_dir = Path(download_dir)
    required_files = {
        "images": "data_object_image_2.zip",
        "labels": "data_object_label_2.zip",
        "calib": "data_object_calib.zip",
    }

    found_files = {}
    missing_files = []

    for key, filename in required_files.items():
        filepath = download_dir / filename
        if filepath.exists():
            found_files[key] = filepath
            print(f"✓ Found: {filename} ({filepath.stat().st_size / (1024**3):.2f} GB)")
        else:
            missing_files.append(filename)
            print(f"✗ Missing: {filename}")

    return found_files, missing_files


def extract_zip(zip_path, extract_to, desc="Extracting"):
    """Extract zip file with progress bar."""
    print(f"\n{desc}: {zip_path.name}")

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        members = zip_ref.namelist()
        with tqdm(total=len(members), desc=desc) as pbar:
            for member in members:
                zip_ref.extract(member, extract_to)
                pbar.update(1)


def organize_kitti_structure(data_dir):
    """Organize KITTI files into proper structure."""
    data_dir = Path(data_dir)

    print("\nOrganizing KITTI dataset structure...")

    # Expected structure after extraction
    training_dir = data_dir / "training"
    testing_dir = data_dir / "testing"

    # Check if structure already exists
    if training_dir.exists() and (training_dir / "image_2").exists():
        print("✓ Dataset structure looks good!")

        # Count files
        n_train_images = len(list((training_dir / "image_2").glob("*.png")))
        n_train_labels = len(list((training_dir / "label_2").glob("*.txt")))
        n_train_calib = len(list((training_dir / "calib").glob("*.txt")))

        print(f"\nTraining set:")
        print(f"  - Images: {n_train_images}")
        print(f"  - Labels: {n_train_labels}")
        print(f"  - Calibration files: {n_train_calib}")

        if testing_dir.exists():
            n_test_images = len(list((testing_dir / "image_2").glob("*.png")))
            n_test_calib = len(list((testing_dir / "calib").glob("*.txt")))
            print(f"\nTesting set:")
            print(f"  - Images: {n_test_images}")
            print(f"  - Calibration files: {n_test_calib}")

        return True
    else:
        print("⚠ Dataset structure needs organization")
        return False


def create_train_val_split(data_dir, val_split=0.2, seed=42):
    """Create train/val split files."""
    import random

    data_dir = Path(data_dir)
    training_dir = data_dir / "training"
    image_dir = training_dir / "image_2"

    if not image_dir.exists():
        print("Error: Training images not found!")
        return

    # Get all image files
    image_files = sorted(list(image_dir.glob("*.png")))
    n_total = len(image_files)
    n_val = int(n_total * val_split)
    n_train = n_total - n_val

    # Set seed for reproducibility
    random.seed(seed)

    # Shuffle and split
    indices = list(range(n_total))
    random.shuffle(indices)

    train_indices = sorted(indices[:n_train])
    val_indices = sorted(indices[n_train:])

    # Create split files
    splits_dir = data_dir / "ImageSets"
    splits_dir.mkdir(exist_ok=True)

    # Write train.txt
    train_file = splits_dir / "train.txt"
    with open(train_file, "w") as f:
        for idx in train_indices:
            img_name = image_files[idx].stem
            f.write(f"{img_name}\n")

    # Write val.txt
    val_file = splits_dir / "val.txt"
    with open(val_file, "w") as f:
        for idx in val_indices:
            img_name = image_files[idx].stem
            f.write(f"{img_name}\n")

    # Write trainval.txt (all training data)
    trainval_file = splits_dir / "trainval.txt"
    with open(trainval_file, "w") as f:
        for img in image_files:
            f.write(f"{img.stem}\n")

    print(f"\n✓ Created train/val split:")
    print(f"  - Training samples: {n_train} ({(1 - val_split) * 100:.0f}%)")
    print(f"  - Validation samples: {n_val} ({val_split * 100:.0f}%)")
    print(f"  - Split files saved to: {splits_dir}")


def verify_dataset(data_dir):
    """Verify dataset integrity."""
    data_dir = Path(data_dir)
    training_dir = data_dir / "training"

    print("\nVerifying dataset integrity...")

    # Check directories exist
    required_dirs = [training_dir / "image_2", training_dir / "label_2", training_dir / "calib"]

    all_good = True
    for dir_path in required_dirs:
        if dir_path.exists():
            print(f"✓ {dir_path.relative_to(data_dir)}")
        else:
            print(f"✗ {dir_path.relative_to(data_dir)} - MISSING!")
            all_good = False

    if not all_good:
        return False

    # Check file counts match
    n_images = len(list((training_dir / "image_2").glob("*.png")))
    n_labels = len(list((training_dir / "label_2").glob("*.txt")))
    n_calib = len(list((training_dir / "calib").glob("*.txt")))

    print(f"\nFile counts:")
    print(f"  - Images: {n_images}")
    print(f"  - Labels: {n_labels}")
    print(f"  - Calibration: {n_calib}")

    if n_images != n_labels or n_images != n_calib:
        print("\n⚠ Warning: File counts don't match!")
        print("  This might cause issues during training.")
        return False

    print("\n✓ Dataset verification passed!")
    return True


def create_dataset_yaml(data_dir, output_path=None):
    """Create dataset YAML configuration file."""
    data_dir = Path(data_dir).resolve()

    if output_path is None:
        output_path = data_dir / "kitti-3d.yaml"
    else:
        output_path = Path(output_path)

    yaml_content = f"""# KITTI 3D Object Detection Dataset Configuration
# Auto-generated by download_kitti.py

# Path configuration
path: {data_dir}  # dataset root dir
train: training/image_2  # train images (relative to 'path')
val: training/image_2    # val images (relative to 'path')
test: testing/image_2    # test images (optional)

# Classes (KITTI 3D Object Detection)
names:
  0: Car
  1: Truck
  2: Pedestrian
  3: Cyclist
  4: Misc
  5: Van
  6: Tram
  7: Person_sitting

# Number of classes
nc: 8

# Task type
task: detect3d

# Train/val split files (if created)
train_split: ImageSets/train.txt  # optional
val_split: ImageSets/val.txt      # optional

# Camera calibration parameters
camera:
  fx: 721.5377  # focal length x
  fy: 721.5377  # focal length y
  cx: 609.5593  # principal point x
  cy: 172.854   # principal point y

# Data augmentation (conservative for 3D detection)
augment:
  hsv_h: 0.015      # HSV-Hue augmentation
  hsv_s: 0.7        # HSV-Saturation augmentation
  hsv_v: 0.4        # HSV-Value augmentation
  degrees: 0.0      # rotation (disabled for 3D)
  translate: 0.1    # translation
  scale: 0.5        # scale
  shear: 0.0        # shear (disabled for 3D)
  perspective: 0.0  # perspective (disabled for 3D)
  flipud: 0.0       # flip up-down (disabled)
  fliplr: 0.5       # flip left-right
  mosaic: 1.0       # mosaic augmentation
  mixup: 0.0        # mixup (disabled for 3D)
"""

    with open(output_path, "w") as f:
        f.write(yaml_content)

    print(f"\n✓ Created dataset configuration: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Download and setup KITTI 3D Object Detection dataset")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./datasets/kitti",
        help="Directory to store KITTI dataset (default: ./datasets/kitti)",
    )
    parser.add_argument(
        "--download_dir",
        type=str,
        default="./downloads",
        help="Directory containing downloaded .zip files (default: ./downloads)",
    )
    parser.add_argument("--val_split", type=float, default=0.2, help="Validation split ratio (default: 0.2)")
    parser.add_argument("--skip_extract", action="store_true", help="Skip extraction if already done")
    parser.add_argument("--create_yaml", action="store_true", help="Create dataset YAML configuration file")

    args = parser.parse_args()

    print_header()

    # Create directories
    data_dir = Path(args.data_dir)
    download_dir = Path(args.download_dir)

    data_dir.mkdir(parents=True, exist_ok=True)
    download_dir.mkdir(parents=True, exist_ok=True)

    print(f"Data directory: {data_dir.resolve()}")
    print(f"Download directory: {download_dir.resolve()}")

    # Check for downloaded files
    print("\nChecking for downloaded files...")
    found_files, missing_files = check_downloaded_files(download_dir)

    if missing_files:
        print(f"\n⚠ Missing {len(missing_files)} required file(s)")
        print_instructions()
        return

    print("\n✓ All required files found!")

    # Extract files if needed
    if not args.skip_extract:
        print("\nExtracting files...")

        # Extract images
        if "images" in found_files:
            extract_zip(found_files["images"], data_dir, "Extracting images")

        # Extract labels
        if "labels" in found_files:
            extract_zip(found_files["labels"], data_dir, "Extracting labels")

        # Extract calibration
        if "calib" in found_files:
            extract_zip(found_files["calib"], data_dir, "Extracting calibration")

        print("\n✓ Extraction complete!")
    else:
        print("\nSkipping extraction (--skip_extract flag set)")

    # Organize structure
    organize_kitti_structure(data_dir)

    # Create train/val split
    print("\nCreating train/validation split...")
    create_train_val_split(data_dir, val_split=args.val_split)

    # Verify dataset
    if verify_dataset(data_dir):
        print("\n" + "=" * 80)
        print("SUCCESS! Dataset is ready for training")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("WARNING: Dataset verification failed")
        print("Please check the errors above")
        print("=" * 80)

    # Create YAML configuration
    if args.create_yaml:
        create_dataset_yaml(data_dir)

    # Print next steps
    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print("\n1. Verify the dataset structure:")
    print(f"   ls -la {data_dir}/training/")
    print("\n2. Update the dataset path in your config:")
    print(f"   ultralytics/cfg/datasets/kitti-3d.yaml")
    print("\n3. Start training:")
    print("   python examples/train_kitti_3d.py")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
