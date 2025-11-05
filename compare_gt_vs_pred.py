#!/usr/bin/env python3
"""
Compare Ground Truth vs Model Predictions for 3D Object Detection
Visualizes both GT and predicted 3D boxes on the same image
"""

import cv2
import numpy as np
import torch
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from ultralytics import YOLO


class KITTIGroundTruthLoader:
    """Load KITTI ground truth labels."""

    CLASS_NAMES = {
        'Car': 0, 'Truck': 1, 'Pedestrian': 2, 'Cyclist': 3,
        'Misc': 4, 'Van': 5, 'Tram': 6, 'Person_sitting': 7,
        'DontCare': -1
    }

    def load_label(self, label_path):
        """Load ground truth from KITTI label file."""
        if not Path(label_path).exists():
            return []

        objects = []
        with open(label_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) < 15:
                    continue

                obj_type = parts[0]
                if obj_type == 'DontCare':
                    continue

                cls_id = self.CLASS_NAMES.get(obj_type, -1)
                if cls_id == -1:
                    continue

                # 2D bbox
                x1, y1, x2, y2 = [float(x) for x in parts[4:8]]

                # 3D dimensions: h, w, l
                h, w, l = [float(x) for x in parts[8:11]]

                # 3D location: x, y, z
                x_3d, y_3d, z_3d = [float(x) for x in parts[11:14]]

                # Rotation
                rot_y = float(parts[14])

                objects.append({
                    'type': obj_type,
                    'cls': cls_id,
                    'bbox_2d': [x1, y1, x2, y2],
                    'dimensions_3d': [h, w, l],
                    'location_3d': [x_3d, y_3d, z_3d],
                    'rotation_y': rot_y
                })

        return objects


class GTvsPredVisualizer:
    """Visualize ground truth vs predictions."""

    def __init__(self):
        self.class_names = {
            0: 'Car', 1: 'Truck', 2: 'Pedestrian', 3: 'Cyclist',
            4: 'Misc', 5: 'Van', 6: 'Tram', 7: 'Person_sitting'
        }

        # Colors: GT = Blue, Pred = Green
        self.gt_color = (255, 0, 0)    # Blue for ground truth
        self.pred_color = (0, 255, 0)  # Green for predictions

    def project_3d_to_2d(self, location, dimensions, rotation_y, P2):
        """Project 3D box to 2D image coordinates."""
        h, w, l = dimensions
        x, y, z = location

        # Create 8 corners - location is at BOTTOM CENTER
        corners_3d = np.array([
            [w/2, 0, l/2],       # back bottom right
            [w/2, -h, l/2],      # back top right
            [-w/2, -h, l/2],     # back top left
            [-w/2, 0, l/2],      # back bottom left
            [w/2, 0, -l/2],      # front bottom right
            [w/2, -h, -l/2],     # front top right
            [-w/2, -h, -l/2],    # front top left
            [-w/2, 0, -l/2],     # front bottom left
        ])

        # Rotation matrix around Y axis
        R = np.array([
            [np.cos(rotation_y), 0, np.sin(rotation_y)],
            [0, 1, 0],
            [-np.sin(rotation_y), 0, np.cos(rotation_y)]
        ])

        # Rotate and translate
        corners_3d = corners_3d @ R.T
        corners_3d = corners_3d + location

        # Project to 2D
        corners_3d_homo = np.hstack([corners_3d, np.ones((8, 1))])
        corners_2d_homo = corners_3d_homo @ P2.T
        corners_2d = corners_2d_homo[:, :2] / corners_2d_homo[:, 2:3]

        return corners_2d

    def draw_3d_box(self, img, corners_2d, color, thickness=2, label=""):
        """Draw 3D bounding box on image."""
        corners_2d = corners_2d.astype(int)

        # Draw back face (0,1,2,3)
        for i in range(4):
            pt1 = tuple(corners_2d[i])
            pt2 = tuple(corners_2d[(i + 1) % 4])
            cv2.line(img, pt1, pt2, color, thickness)

        # Draw front face (4,5,6,7)
        for i in range(4, 8):
            pt1 = tuple(corners_2d[i])
            pt2 = tuple(corners_2d[4 + (i + 1) % 4])
            cv2.line(img, pt1, pt2, color, thickness)

        # Connect back and front
        for i in range(4):
            pt1 = tuple(corners_2d[i])
            pt2 = tuple(corners_2d[i + 4])
            cv2.line(img, pt1, pt2, color, thickness)

        # Add label if provided
        if label:
            label_pos = tuple(corners_2d[1])  # Top back right corner
            cv2.putText(img, label, label_pos, cv2.FONT_HERSHEY_SIMPLEX,
                       0.5, color, 1, cv2.LINE_AA)

    def compare_objects(self, img, gt_objects, pred_objects, P2, save_path=None):
        """Create comparison visualization with GT and predictions."""
        img_vis = img.copy()

        # Draw ground truth in BLUE
        print(f"\nGround Truth: {len(gt_objects)} objects")
        for i, obj in enumerate(gt_objects):
            try:
                corners_2d = self.project_3d_to_2d(
                    obj['location_3d'],
                    obj['dimensions_3d'],
                    obj['rotation_y'],
                    P2
                )
                label = f"GT:{obj['type']}"
                self.draw_3d_box(img_vis, corners_2d, self.gt_color,
                               thickness=2, label=label)

                # Draw 2D bbox
                x1, y1, x2, y2 = [int(x) for x in obj['bbox_2d']]
                cv2.rectangle(img_vis, (x1, y1), (x2, y2), self.gt_color, 1)

                print(f"  GT {i+1}: {obj['type']} at z={obj['location_3d'][2]:.1f}m")
            except Exception as e:
                print(f"  Error drawing GT {i+1}: {e}")

        # Draw predictions in GREEN
        print(f"\nPredictions: {len(pred_objects)} objects")
        for i, det in enumerate(pred_objects):
            try:
                # Compute 3D location from 2D box center
                x1, y1, x2, y2 = det['bbox_2d']
                center_2d_x = (x1 + x2) / 2.0
                center_2d_y = (y1 + y2) / 2.0

                fx = P2[0, 0]
                fy = P2[1, 1]
                cx = P2[0, 2]
                cy = P2[1, 2]

                z = det['location_3d'][2]
                x = (center_2d_x - cx) * z / fx
                y = (center_2d_y - cy) * z / fy

                location = [x, y, z]

                corners_2d = self.project_3d_to_2d(
                    location,
                    det['dimensions_3d'],
                    det['rotation_y'],
                    P2
                )

                cls_name = self.class_names.get(det['cls'], f"Class{det['cls']}")
                label = f"Pred:{cls_name}({det['conf']:.2f})"
                self.draw_3d_box(img_vis, corners_2d, self.pred_color,
                               thickness=2, label=label)

                # Draw 2D bbox
                x1, y1, x2, y2 = [int(x) for x in det['bbox_2d']]
                cv2.rectangle(img_vis, (x1, y1), (x2, y2), self.pred_color, 1)

                print(f"  Pred {i+1}: {cls_name} (conf={det['conf']:.2f}) at z={z:.1f}m")
            except Exception as e:
                print(f"  Error drawing Pred {i+1}: {e}")

        # Add legend
        legend_y = 30
        cv2.putText(img_vis, "Ground Truth", (10, legend_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.gt_color, 2)
        cv2.putText(img_vis, "Prediction", (10, legend_y + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.pred_color, 2)

        if save_path:
            cv2.imwrite(str(save_path), img_vis)
            print(f"\nSaved comparison to: {save_path}")

        return img_vis


def load_calib(calib_file):
    """Load KITTI calibration file."""
    with open(calib_file, 'r') as f:
        lines = f.readlines()

    for line in lines:
        if line.startswith('P2:'):
            P2 = np.array([float(x) for x in line.split()[1:]], dtype=np.float32)
            P2 = P2.reshape(3, 4)
            return P2

    # Default P2 if not found
    P2 = np.array([
        [721.5377, 0, 609.5593, 44.85728],
        [0, 721.5377, 172.854, 0.2163791],
        [0, 0, 1, 0.002745884]
    ])
    return P2


def run_comparison(model_path, image_ids, data_root, output_dir='comparison_results'):
    """Run comparison between GT and predictions on specified images."""

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Load model
    print(f"Loading model: {model_path}")
    model = YOLO(model_path, task='detect3d')

    # Initialize loaders
    gt_loader = KITTIGroundTruthLoader()
    visualizer = GTvsPredVisualizer()

    data_root = Path(data_root)

    # Process each image
    for img_id in image_ids:
        print("\n" + "="*60)
        print(f"Processing image: {img_id}")
        print("="*60)

        # Paths
        img_path = data_root / "training" / "image_2" / f"{img_id}.png"
        label_path = data_root / "training" / "label_2" / f"{img_id}.txt"
        calib_path = data_root / "training" / "calib" / f"{img_id}.txt"

        if not img_path.exists():
            print(f"Error: Image not found: {img_path}")
            continue

        # Load image
        img = cv2.imread(str(img_path))
        print(f"Image size: {img.shape[1]}x{img.shape[0]}")

        # Load calibration
        P2 = load_calib(calib_path)
        print(f"Loaded calibration")

        # Load ground truth
        gt_objects = gt_loader.load_label(label_path)
        print(f"Loaded {len(gt_objects)} ground truth objects")

        # Run inference
        print("Running inference...")
        results = model.predict(img_path, save=False, conf=0.25, verbose=False)

        # Parse predictions
        pred_objects = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            if hasattr(boxes, 'data') and boxes.data.shape[1] >= 13:
                for det in boxes.data.cpu().numpy():
                    pred_objects.append({
                        'bbox_2d': det[:4].tolist(),
                        'conf': float(det[4]),
                        'cls': int(det[5]),
                        'location_3d': det[6:9].tolist(),
                        'dimensions_3d': det[9:12].tolist(),
                        'rotation_y': float(det[12])
                    })

        print(f"Got {len(pred_objects)} predictions")

        # Create comparison visualization
        output_path = output_dir / f"{img_id}_comparison.jpg"
        img_vis = visualizer.compare_objects(img, gt_objects, pred_objects, P2, output_path)

        # Display using matplotlib
        plt.figure(figsize=(20, 10))
        plt.imshow(cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title(f'GT (Blue) vs Prediction (Green): {img_id}', fontsize=16)
        plt.tight_layout()

        plot_path = output_dir / f"{img_id}_comparison_plot.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to: {plot_path}")
        plt.show()
        plt.close()

    print("\n" + "="*60)
    print("Comparison complete!")
    print(f"Results saved to: {output_dir}/")
    print("="*60)


if __name__ == "__main__":
    # Configuration
    MODEL_PATH = "last-2.pt"
    DATA_ROOT = "/Users/sompoteyouwai/Downloads/datakitti/datasets/kitti"

    # Specify 3 image IDs to compare
    IMAGE_IDS = [
        "000000",  # Image 1
        "000010",  # Image 2
        "000050",  # Image 3
    ]

    print("="*60)
    print("3D Object Detection: Ground Truth vs Prediction Comparison")
    print("="*60)
    print(f"\nModel: {MODEL_PATH}")
    print(f"Data root: {DATA_ROOT}")
    print(f"Images to compare: {IMAGE_IDS}")
    print()

    # Run comparison
    run_comparison(MODEL_PATH, IMAGE_IDS, DATA_ROOT)
