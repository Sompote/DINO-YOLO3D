#!/usr/bin/env python3
"""
Debug script to check model predictions
"""
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from ultralytics import YOLO
import torch

# Load model
weights = "/Users/sompoteyouwai/Downloads/best-5.pt"
source = "/Users/sompoteyouwai/Downloads/KITTI/valid/images/000235_png.rf.fa1b7e1bebbab60e0582fdcdb3b6a59f.jpg"

print("Loading model...")
model = YOLO(weights)

print(f"Model task: {model.task}")
print(f"Model classes: {model.names}")

# Run with very low confidence
print("\n" + "="*80)
print("Testing with conf=0.001 (very low)")
print("="*80)

results = model.predict(
    source=source,
    conf=0.001,  # Very low confidence
    iou=0.45,
    verbose=True,
    save=False,
)

for result in results:
    boxes = result.boxes
    if boxes is None:
        print("No boxes returned!")
    else:
        print(f"\nTotal detections: {len(boxes)}")
        if len(boxes) > 0:
            print(f"Box data shape: {boxes.data.shape}")
            print(f"Is 3D: {boxes.is_3d if hasattr(boxes, 'is_3d') else 'N/A'}")
            print(f"\nTop 5 detections:")
            for i in range(min(5, len(boxes))):
                box = boxes.data[i]
                print(f"  [{i}] conf={box[4]:.4f}, cls={int(box[5])}, "
                      f"bbox=[{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}]")
                if len(box) >= 11:
                    print(f"       depth={box[6]:.2f}m, dims=[{box[7]:.2f}, {box[8]:.2f}, {box[9]:.2f}]m, "
                          f"rot={box[10]:.2f}rad")
        else:
            print("No detections even with conf=0.001!")
            print("\nThis suggests:")
            print("1. The model might not be trained properly")
            print("2. The image preprocessing might have issues")
            print("3. The model architecture might not match the weights")

print("\n" + "="*80)
print("Testing with conf=0.5 (high)")
print("="*80)

results = model.predict(
    source=source,
    conf=0.5,
    iou=0.45,
    verbose=True,
    save=False,
)

for result in results:
    boxes = result.boxes
    if boxes and len(boxes) > 0:
        print(f"\nDetections at conf=0.5: {len(boxes)}")
    else:
        print("No detections at conf=0.5")

print("\n" + "="*80)
print("Debug complete!")
print("="*80)
