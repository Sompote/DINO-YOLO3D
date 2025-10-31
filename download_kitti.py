#!/usr/bin/env python3
"""
KITTI 3D Object Detection Dataset Download Script

This script downloads the KITTI 3D object detection dataset files.
The dataset is used for training and evaluating 3D object detection models.

Dataset source: http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d

Required files:
- data_object_image_2.zip (12 GB) - Left color images
- data_object_label_2.zip (5 MB) - Training labels
- data_object_calib.zip (16 MB) - Camera calibration matrices

Usage:
    python download_kitti.py [--output-dir DIR]

Note: KITTI dataset requires manual download due to terms of use.
This script provides instructions and can verify/extract downloaded files.
"""

import os
import sys
import argparse
import zipfile
from pathlib import Path
from typing import List, Tuple

# KITTI dataset file information
KITTI_FILES = [
    {
        "name": "data_object_image_2.zip",
        "size": "12 GB",
        "description": "Left color images of object dataset",
        "url": "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip",
    },
    {
        "name": "data_object_label_2.zip",
        "size": "5 MB",
        "description": "Training labels of object dataset",
        "url": "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip",
    },
    {
        "name": "data_object_calib.zip",
        "size": "16 MB",
        "description": "Camera calibration matrices of object dataset",
        "url": "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_calib.zip",
    },
]


def print_header():
    """Print script header."""
    print("=" * 80)
    print("KITTI 3D Object Detection Dataset Downloader")
    print("=" * 80)
    print()


def print_instructions():
    """Print download instructions."""
    print("KITTI Dataset Download Instructions:")
    print("-" * 80)
    print()
    print("Option 1: Manual Download (Recommended)")
    print("  1. Visit: http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d")
    print("  2. Accept the terms of use")
    print("  3. Download the following files to ./downloads/ directory:")
    print()
    for file_info in KITTI_FILES:
        print(f"     - {file_info['name']} ({file_info['size']}) - {file_info['description']}")
    print()
    print("Option 2: Direct Download (using wget/curl)")
    print("  Run the following commands:")
    print()
    print("  mkdir -p downloads")
    for file_info in KITTI_FILES:
        print(f"  wget {file_info['url']} -P downloads/")
    print()
    print("-" * 80)
    print()


def check_file_exists(filepath: Path) -> Tuple[bool, str]:
    """
    Check if a file exists and return status message.

    Args:
        filepath: Path to check

    Returns:
        Tuple of (exists, status_message)
    """
    if filepath.exists():
        size_mb = filepath.stat().st_size / (1024 * 1024)
        if size_mb > 1024:
            size_str = f"{size_mb / 1024:.2f} GB"
        else:
            size_str = f"{size_mb:.2f} MB"
        return True, f"✓ Found: {filepath.name} ({size_str})"
    else:
        return False, f"✗ Missing: {filepath.name}"


def verify_downloads(download_dir: Path) -> List[Path]:
    """
    Verify that all required files have been downloaded.

    Args:
        download_dir: Directory containing downloaded files

    Returns:
        List of paths to downloaded files
    """
    print("Verifying downloaded files...")
    print("-" * 80)

    downloaded_files = []
    all_present = True

    for file_info in KITTI_FILES:
        filepath = download_dir / file_info["name"]
        exists, message = check_file_exists(filepath)
        print(message)

        if exists:
            downloaded_files.append(filepath)
        else:
            all_present = False

    print("-" * 80)
    print()

    if all_present:
        print("✓ All required files are present!")
    else:
        print("✗ Some files are missing. Please download them first.")
        print()
        print_instructions()

    return downloaded_files if all_present else []


def extract_zip(zip_path: Path, extract_dir: Path) -> bool:
    """
    Extract a zip file.

    Args:
        zip_path: Path to zip file
        extract_dir: Directory to extract to

    Returns:
        True if successful, False otherwise
    """
    try:
        print(f"Extracting {zip_path.name}...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)
        print(f"✓ Extracted {zip_path.name}")
        return True
    except Exception as e:
        print(f"✗ Error extracting {zip_path.name}: {e}")
        return False


def extract_all(downloaded_files: List[Path], output_dir: Path) -> bool:
    """
    Extract all downloaded zip files.

    Args:
        downloaded_files: List of downloaded zip files
        output_dir: Directory to extract files to

    Returns:
        True if all extractions successful, False otherwise
    """
    print("Extracting files...")
    print("-" * 80)

    output_dir.mkdir(parents=True, exist_ok=True)

    success = True
    for zip_path in downloaded_files:
        if not extract_zip(zip_path, output_dir):
            success = False

    print("-" * 80)
    print()

    if success:
        print(f"✓ All files extracted to: {output_dir}")
        print()
        print("Dataset structure:")
        print(f"  {output_dir}/")
        print("    ├── training/")
        print("    │   ├── image_2/    (left color camera images)")
        print("    │   ├── label_2/    (training labels)")
        print("    │   └── calib/      (calibration files)")
        print("    └── testing/")
        print("        ├── image_2/")
        print("        └── calib/")
    else:
        print("✗ Some files failed to extract.")

    return success


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Download and setup KITTI 3D object detection dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--download-dir",
        type=str,
        default="./downloads",
        help="Directory containing downloaded zip files (default: ./downloads)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./datasets/kitti",
        help="Directory to extract dataset to (default: ./datasets/kitti)",
    )
    parser.add_argument("--extract", action="store_true", help="Extract downloaded files")
    parser.add_argument("--verify-only", action="store_true", help="Only verify downloads without extracting")

    args = parser.parse_args()

    print_header()

    download_dir = Path(args.download_dir)
    output_dir = Path(args.output_dir)

    # Create download directory if it doesn't exist
    download_dir.mkdir(parents=True, exist_ok=True)

    # Print instructions
    print_instructions()

    # Verify downloads
    downloaded_files = verify_downloads(download_dir)

    if not downloaded_files:
        print("Please download the required files and run this script again.")
        return 1

    # Extract if requested
    if args.verify_only:
        print("Verification complete. Use --extract to extract the files.")
        return 0

    if args.extract or input("\nExtract files now? [y/N]: ").lower().strip() == "y":
        if extract_all(downloaded_files, output_dir):
            print()
            print("✓ KITTI dataset setup complete!")
            print()
            print("Next steps:")
            print("  1. Update the dataset path in ultralytics/cfg/datasets/kitti-3d.yaml")
            print(f"     Set 'path: {output_dir.absolute()}'")
            print("  2. Start training: python train.py --data kitti-3d.yaml --model yolov12-3d.yaml")
            return 0
        else:
            return 1
    else:
        print("Extraction skipped. Run with --extract to extract files.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
