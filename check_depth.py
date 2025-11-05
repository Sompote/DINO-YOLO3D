"""
Check depth predictions from inference to diagnose scaling issues.
"""
from ultralytics import YOLO
import torch

# Load model
model = YOLO("last-3.pt")

# Run inference on a single image
results = model.predict("/Users/sompoteyouwai/Downloads/datakitti/datasets/kitti/training/image_2/000000.png", verbose=False)

# Get the first result
result = results[0]

# Check if we have 3D detections
if result.boxes is not None and len(result.boxes) > 0:
    boxes_data = result.boxes.data
    print(f"Number of detections: {len(boxes_data)}")
    print(f"Boxes data shape: {boxes_data.shape}")

    # Print format info
    print("\nExpected format: [x1, y1, x2, y2, conf, cls, x_3d, y_3d, z_3d, h_3d, w_3d, l_3d, rot_y]")
    print(f"Actual shape: {boxes_data.shape}")

    # Check if we have 3D params
    if boxes_data.shape[1] >= 13:
        print("\n3D Detection data available!")
        for i, box in enumerate(boxes_data):
            x1, y1, x2, y2, conf, cls = box[:6].tolist()
            if boxes_data.shape[1] == 13:
                x_3d, y_3d, z_3d, h_3d, w_3d, l_3d, rot_y = box[6:13].tolist()
            else:
                # Handle case where there might be 11 columns
                depth = box[6].item()
                h_3d, w_3d, l_3d = box[7:10].tolist()
                rot_y = box[10].item()
                x_3d, y_3d, z_3d = 0, 0, depth

            print(f"\nDetection {i+1}:")
            print(f"  Class: {int(cls)} ({model.names[int(cls)]}), Confidence: {conf:.3f}")
            print(f"  2D Box: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")
            if boxes_data.shape[1] == 13:
                print(f"  3D Location: x={x_3d:.2f}m, y={y_3d:.2f}m, z={z_3d:.2f}m (depth)")
            else:
                print(f"  Depth: z={z_3d:.2f}m")
            print(f"  3D Dimensions: h={h_3d:.2f}m, w={w_3d:.2f}m, l={l_3d:.2f}m")
            print(f"  Rotation: {rot_y:.3f} rad ({rot_y*57.2958:.1f}°)")
    else:
        print(f"\nWarning: Expected at least 13 columns for 3D detection, got {boxes_data.shape[1]}")
else:
    print("No detections found")

# Also check ground truth for comparison
print("\n" + "="*60)
print("GROUND TRUTH from label file:")
print("="*60)
label_file = "/Users/sompoteyouwai/Downloads/datakitti/datasets/kitti/training/label_2/000000.txt"
try:
    with open(label_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 15:
                cls_name = parts[0]
                bbox = [float(x) for x in parts[4:8]]
                dims = [float(x) for x in parts[8:11]]  # h, w, l
                loc = [float(x) for x in parts[11:14]]  # x, y, z
                rot = float(parts[14])
                print(f"\n{cls_name}:")
                print(f"  2D Box: {bbox}")
                print(f"  3D Location: x={loc[0]:.2f}m, y={loc[1]:.2f}m, z={loc[2]:.2f}m (depth)")
                print(f"  3D Dimensions: h={dims[0]:.2f}m, w={dims[1]:.2f}m, l={dims[2]:.2f}m")
                print(f"  Rotation: {rot:.3f} rad ({rot*57.2958:.1f}°)")
except FileNotFoundError:
    print("Ground truth file not found")
