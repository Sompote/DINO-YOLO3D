#!/usr/bin/env python3
"""
Test script to understand KITTI 3D bounding box visualization.
Check how the model outputs match the ground truth annotations.
"""
import sys
from pathlib import Path
import numpy as np
import cv2

# Add current directory to path
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from ultralytics import YOLO

# Load model
weights = "/Users/sompoteyouwai/Downloads/best-5.pt"
model = YOLO(weights)

# Run inference
source = "/Users/sompoteyouwai/Downloads/000005.png"
results = model.predict(source=source, conf=0.25, verbose=False)

print("=" * 80)
print("Model Output Analysis")
print("=" * 80)

for result in results:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        continue

    boxes_data = boxes.data.cpu().numpy()
    print(f"\nTotal detections: {len(boxes_data)}")
    print(f"Boxes shape: {boxes_data.shape}")
    print(f"\nColumns: [x1, y1, x2, y2, conf, cls, depth, h_3d, w_3d, l_3d, rotation_y]")

    for i, box in enumerate(boxes_data):
        x1, y1, x2, y2 = box[:4]
        conf = box[4]
        cls_id = int(box[5])
        depth = box[6]
        h, w, l = box[7], box[8], box[9]
        rotation_y = box[10]

        print(f"\n[Detection {i}] {model.names[cls_id]} (conf={conf:.2f})")
        print(f"  2D bbox: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")
        print(f"  2D box size: {x2-x1:.1f} x {y2-y1:.1f} pixels")
        print(f"  Depth (z): {depth:.2f}m")
        print(f"  Dimensions [h,w,l]: [{h:.2f}, {w:.2f}, {l:.2f}]m")
        print(f"  Rotation_y: {rotation_y:.2f} rad ({np.degrees(rotation_y):.1f}°)")

        # Note: Model only predicts depth (z), not full 3D location (x,y,z)
        # The x,y coordinates need to be computed from 2D bbox and depth using camera calibration
        print(f"\n  Note: The model predicts depth only, not full 3D location.")
        print(f"  To get 3D location (x,y,z), we need to back-project from 2D bbox center:")

        # Example back-projection (simplified, assumes pinhole camera)
        img_h, img_w = result.orig_img.shape[:2]

        # Estimate focal length (common heuristic: f ≈ image width)
        fx = fy = img_w
        cx = img_w / 2
        cy = img_h / 2

        # 2D box center
        cx_2d = (x1 + x2) / 2
        cy_2d = (y1 + y2) / 2  # or use y2 (bottom) for better ground plane alignment

        # Back-project to 3D (simple pinhole model)
        x_3d = (cx_2d - cx) * depth / fx
        y_3d = (cy_2d - cy) * depth / fy
        z_3d = depth

        print(f"  Computed 3D location (approx): x={x_3d:.2f}, y={y_3d:.2f}, z={z_3d:.2f}m")

print("\n" + "=" * 80)
print("Key Insight:")
print("  - The model outputs: depth, dimensions (h,w,l), rotation_y")
print("  - The model does NOT output full 3D location (x,y,z)")
print("  - For proper 3D visualization, we need camera calibration to back-project")
print("  - Without calibration, we can only show approximate 3D boxes")
print("=" * 80 + "\n")
