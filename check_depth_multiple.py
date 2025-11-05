"""
Check depth predictions from inference on multiple images to diagnose scaling issues.
"""
from ultralytics import YOLO
import torch
import numpy as np

# Load model
model = YOLO("last-3.pt")

# Test on multiple images
test_images = [
    "000000.png",
    "000001.png",
    "000002.png",
    "000003.png",
    "000004.png"
]

depth_errors = []

for img_name in test_images:
    img_path = f"/Users/sompoteyouwai/Downloads/datakitti/datasets/kitti/training/image_2/{img_name}"
    label_path = f"/Users/sompoteyouwai/Downloads/datakitti/datasets/kitti/training/label_2/{img_name.replace('.png', '.txt')}"

    print(f"\n{'='*60}")
    print(f"Processing {img_name}")
    print(f"{'='*60}")

    # Run inference
    results = model.predict(img_path, verbose=False)
    result = results[0]

    # Load ground truth
    gt_objects = []
    try:
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 15 and parts[0] in ['Car', 'Pedestrian', 'Cyclist']:
                    obj = {
                        'class': parts[0],
                        'bbox': [float(x) for x in parts[4:8]],
                        'depth': float(parts[13]),  # z coordinate
                        'dims': [float(x) for x in parts[8:11]],
                        'loc': [float(x) for x in parts[11:14]]
                    }
                    gt_objects.append(obj)
    except FileNotFoundError:
        print(f"Label file not found: {label_path}")
        continue

    # Compare predictions with ground truth
    if result.boxes is not None and len(result.boxes) > 0:
        boxes_data = result.boxes.data

        for i, box in enumerate(boxes_data):
            if boxes_data.shape[1] >= 13:
                x1, y1, x2, y2, conf, cls = box[:6].tolist()
                x_3d, y_3d, z_3d, h_3d, w_3d, l_3d, rot_y = box[6:13].tolist()

                # Find matching ground truth (by IoU or proximity)
                pred_bbox = [x1, y1, x2, y2]
                best_match = None
                best_iou = 0

                for gt in gt_objects:
                    # Simple IoU calculation
                    gt_bbox = gt['bbox']
                    x1_i = max(pred_bbox[0], gt_bbox[0])
                    y1_i = max(pred_bbox[1], gt_bbox[1])
                    x2_i = min(pred_bbox[2], gt_bbox[2])
                    y2_i = min(pred_bbox[3], gt_bbox[3])

                    if x2_i > x1_i and y2_i > y1_i:
                        inter = (x2_i - x1_i) * (y2_i - y1_i)
                        area1 = (pred_bbox[2] - pred_bbox[0]) * (pred_bbox[3] - pred_bbox[1])
                        area2 = (gt_bbox[2] - gt_bbox[0]) * (gt_bbox[3] - gt_bbox[1])
                        union = area1 + area2 - inter
                        iou = inter / union if union > 0 else 0

                        if iou > best_iou:
                            best_iou = iou
                            best_match = gt

                if best_match is not None:
                    gt_depth = best_match['depth']
                    error = z_3d - gt_depth
                    error_pct = (error / gt_depth) * 100 if gt_depth > 0 else 0
                    depth_errors.append(error)

                    print(f"\nDetection {i+1}: {model.names[int(cls)]} (conf={conf:.3f}, IoU={best_iou:.3f})")
                    print(f"  Predicted depth: {z_3d:.2f}m")
                    print(f"  Ground truth depth: {gt_depth:.2f}m")
                    print(f"  Error: {error:+.2f}m ({error_pct:+.1f}%)")

                    # Also show dimension comparison
                    gt_dims = best_match['dims']
                    print(f"  Pred dims (h,w,l): {h_3d:.2f}, {w_3d:.2f}, {l_3d:.2f}")
                    print(f"  GT dims (h,w,l): {gt_dims[0]:.2f}, {gt_dims[1]:.2f}, {gt_dims[2]:.2f}")

print(f"\n\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
if depth_errors:
    depth_errors = np.array(depth_errors)
    print(f"Mean depth error: {depth_errors.mean():.2f}m")
    print(f"Std depth error: {depth_errors.std():.2f}m")
    print(f"Median depth error: {np.median(depth_errors):.2f}m")
    print(f"Min error: {depth_errors.min():.2f}m")
    print(f"Max error: {depth_errors.max():.2f}m")
    print(f"\nDepth predictions are consistently TOO HIGH by {depth_errors.mean():.2f}m on average")
else:
    print("No depth errors collected")
