#!/usr/bin/env python3
"""
Visualize what the model receives as input
"""
import cv2
import numpy as np
from pathlib import Path

source = "/Users/sompoteyouwai/Downloads/KITTI/valid/images/000235_png.rf.fa1b7e1bebbab60e0582fdcdb3b6a59f.jpg"

print("="*80)
print("Image Analysis")
print("="*80)

# Load image
img = cv2.imread(source)

if img is None:
    print(f"ERROR: Could not load image from {source}")
    print("The file might be corrupted or the path is wrong")
else:
    print(f"\n✓ Image loaded successfully")
    print(f"  Path: {source}")
    print(f"  Shape: {img.shape} (H×W×C)")
    print(f"  Dtype: {img.dtype}")
    print(f"  Value range: [{img.min()}, {img.max()}]")
    print(f"  File size: {Path(source).stat().st_size / 1024:.1f} KB")

    # Check if image is mostly empty
    mean_val = img.mean()
    print(f"  Mean pixel value: {mean_val:.1f}")

    if mean_val < 10:
        print("  ⚠ WARNING: Image is very dark (mean < 10)")
    elif mean_val > 245:
        print("  ⚠ WARNING: Image is very bright (mean > 245)")

    # Show image
    print("\nDisplaying image (press any key to close)...")

    # Resize for display if too large
    display_img = img.copy()
    max_display = 1200
    if max(display_img.shape[:2]) > max_display:
        scale = max_display / max(display_img.shape[:2])
        new_size = (int(display_img.shape[1] * scale), int(display_img.shape[0] * scale))
        display_img = cv2.resize(display_img, new_size)
        print(f"  Resized to {new_size} for display")

    cv2.imshow("Input Image", display_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("\nImage looks okay? If you can see cars/objects, the model should detect them.")
    print("If the image is blank or corrupted, that's the problem.")

print("="*80)
