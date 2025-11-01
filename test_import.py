#!/usr/bin/env python3
"""
Test script to verify YOLOv12-3D installation and imports.

This script checks if all required packages and modules can be imported correctly.
Run this before training to diagnose any import issues.

Usage:
    python test_import.py
"""

import sys
from pathlib import Path

# Add current directory to path (same as yolo3d.py does)
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


def test_import(module_name, from_import=None):
    """Test importing a module."""
    try:
        if from_import:
            exec(f"from {module_name} import {from_import}")
            print(f"✓ Successfully imported: from {module_name} import {from_import}")
        else:
            exec(f"import {module_name}")
            print(f"✓ Successfully imported: {module_name}")
        return True
    except ImportError as e:
        print(f"✗ Failed to import {module_name}: {e}")
        return False
    except Exception as e:
        print(f"✗ Error with {module_name}: {e}")
        return False


def main():
    print("=" * 80)
    print("YOLOv12-3D Import Test")
    print("=" * 80)
    print()

    print("Testing basic dependencies...")
    print("-" * 80)

    results = []

    # Test basic packages
    results.append(test_import("torch"))
    results.append(test_import("torchvision"))
    results.append(test_import("numpy"))
    results.append(test_import("cv2"))
    results.append(test_import("yaml"))
    results.append(test_import("PIL"))

    print()
    print("Testing ultralytics package...")
    print("-" * 80)

    # Test ultralytics
    results.append(test_import("ultralytics"))
    results.append(test_import("ultralytics", "YOLO"))
    results.append(test_import("ultralytics.nn.modules.head", "Detect3D"))
    results.append(test_import("ultralytics.nn.tasks", "Detection3DModel"))
    results.append(test_import("ultralytics.models.yolo.detect3d"))
    results.append(test_import("ultralytics.data.dataset", "KITTIDataset"))

    print()
    print("=" * 80)

    if all(results):
        print("✓ All imports successful! Your environment is ready.")
        print()
        print("Next steps:")
        print("  1. Download KITTI dataset: ./download_kitti.sh")
        print("  2. Setup dataset: python scripts/kitti_setup.py all --create-yaml")
        print("  3. Start training: python yolo3d.py train --data kitti-3d.yaml --epochs 100 -y")
        return 0
    else:
        print("✗ Some imports failed. Please fix the issues above.")
        print()
        print("Common solutions:")
        print("  - Install missing packages: pip install -r requirements.txt")
        print("  - Ensure you're in the project root directory")
        print("  - Check Python version (requires Python 3.8+)")
        return 1


if __name__ == "__main__":
    sys.exit(main())
