#!/usr/bin/env python3
"""
Check depth value ranges in KITTI dataset to understand normalization needs
"""
import numpy as np
from pathlib import Path

# Path to KITTI labels
label_dir = Path("/Users/sompoteyouwai/Downloads/datakitti/datasets/kitti/training/label_2")

depth_values = []
dim_values = []

# Read a sample of labels
for label_file in list(label_dir.glob("*.txt"))[:100]:  # Check first 100 files
    with open(label_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 15:
                continue

            # Get depth (z coordinate from location_3d)
            z = float(parts[13])  # location_3d z coordinate
            depth_values.append(z)

            # Get dimensions
            h, w, l = float(parts[8]), float(parts[9]), float(parts[10])
            dim_values.extend([h, w, l])

if depth_values:
    print("Depth (z coordinate) statistics:")
    print(f"  Min: {np.min(depth_values):.2f}m")
    print(f"  Max: {np.max(depth_values):.2f}m")
    print(f"  Mean: {np.mean(depth_values):.2f}m")
    print(f"  Median: {np.median(depth_values):.2f}m")
    print(f"  Std: {np.std(depth_values):.2f}m")

    print("\nDimension statistics:")
    print(f"  Min: {np.min(dim_values):.2f}m")
    print(f"  Max: {np.max(dim_values):.2f}m")
    print(f"  Mean: {np.mean(dim_values):.2f}m")
    print(f"  Median: {np.median(dim_values):.2f}m")
    print(f"  Std: {np.std(dim_values):.2f}m")
else:
    print("No data found!")
