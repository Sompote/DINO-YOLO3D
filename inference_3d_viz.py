#!/usr/bin/env python3
"""
3D Object Detection Inference and Visualization Script
Uses trained YOLOv12-3D model to detect objects and visualize 3D bounding boxes
"""

import cv2
import numpy as np
import torch
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import math

from ultralytics import YOLO


class KITTI3DVisualizer:
    """Visualize 3D detections on KITTI images."""

    def __init__(self):
        # KITTI class names
        self.class_names = {
            0: 'Car', 1: 'Truck', 2: 'Pedestrian', 3: 'Cyclist',
            4: 'Misc', 5: 'Van', 6: 'Tram', 7: 'Person_sitting'
        }

        # Colors for each class (BGR format)
        self.colors = {
            0: (0, 255, 0),    # Car - Green
            1: (255, 0, 0),    # Truck - Blue
            2: (0, 0, 255),    # Pedestrian - Red
            3: (255, 255, 0),  # Cyclist - Cyan
            4: (128, 128, 128),# Misc - Gray
            5: (255, 0, 255),  # Van - Magenta
            6: (0, 255, 255),  # Tram - Yellow
            7: (128, 0, 128),  # Person_sitting - Purple
        }

    def draw_3d_box(self, img, corners_2d, color=(0, 255, 0), thickness=2):
        """
        Draw 3D bounding box on image.

        Args:
            img: Image to draw on
            corners_2d: 8x2 array of 2D projected corners
            color: Box color (BGR)
            thickness: Line thickness
        """
        # Draw back face
        for i in range(4):
            pt1 = tuple(corners_2d[i].astype(int))
            pt2 = tuple(corners_2d[(i + 1) % 4].astype(int))
            cv2.line(img, pt1, pt2, color, thickness)

        # Draw front face
        for i in range(4, 8):
            pt1 = tuple(corners_2d[i].astype(int))
            pt2 = tuple(corners_2d[4 + (i + 1) % 4].astype(int))
            cv2.line(img, pt1, pt2, color, thickness)

        # Connect back and front faces
        for i in range(4):
            pt1 = tuple(corners_2d[i].astype(int))
            pt2 = tuple(corners_2d[i + 4].astype(int))
            cv2.line(img, pt1, pt2, color, thickness)

    def project_3d_to_2d(self, location, dimensions, rotation_y, P2):
        """
        Project 3D bounding box to 2D image coordinates.

        Args:
            location: (x, y, z) 3D center in camera coordinates
            dimensions: (h, w, l) object dimensions
            rotation_y: rotation around Y axis
            P2: 3x4 projection matrix

        Returns:
            corners_2d: 8x2 array of projected 2D corners
        """
        h, w, l = dimensions
        x, y, z = location

        # 3D bounding box corners (in object coordinate system)
        # Order: back face (clockwise from top-left), front face (clockwise)
        corners_3d = np.array([
            [l/2, -h/2, w/2],   # back top right
            [l/2, h/2, w/2],    # back bottom right
            [-l/2, h/2, w/2],   # back bottom left
            [-l/2, -h/2, w/2],  # back top left
            [l/2, -h/2, -w/2],  # front top right
            [l/2, h/2, -w/2],   # front bottom right
            [-l/2, h/2, -w/2],  # front bottom left
            [-l/2, -h/2, -w/2], # front top left
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
        corners_3d_homo = np.hstack([corners_3d, np.ones((8, 1))])  # 8x4
        corners_2d_homo = corners_3d_homo @ P2.T  # 8x3
        corners_2d = corners_2d_homo[:, :2] / corners_2d_homo[:, 2:3]  # 8x2

        return corners_2d

    def visualize_detections(self, img, detections, P2, save_path=None):
        """
        Visualize all detections on image.

        Args:
            img: Input image (BGR)
            detections: Detection results with 3D parameters
            P2: Projection matrix
            save_path: Path to save visualization
        """
        img_vis = img.copy()

        if detections is None or len(detections) == 0:
            print("No detections found")
            return img_vis

        print(f"\nFound {len(detections)} detections:")

        for i, det in enumerate(detections):
            # Parse detection
            # Format: [x1, y1, x2, y2, conf, class, x_3d, y_3d, z_3d, h_3d, w_3d, l_3d, rot_y]
            if len(det) < 13:
                print(f"  Detection {i}: Invalid format (only {len(det)} values)")
                continue

            x1, y1, x2, y2 = det[:4]
            conf = det[4]
            cls = int(det[5])
            x_3d, y_3d, z_3d = det[6:9]
            h_3d, w_3d, l_3d = det[9:12]
            rot_y = det[12]

            # Get class info
            class_name = self.class_names.get(cls, f'Class_{cls}')
            color = self.colors.get(cls, (255, 255, 255))

            print(f"  {i+1}. {class_name} (conf={conf:.3f})")
            print(f"     2D bbox: [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")
            print(f"     3D location: x={x_3d:.2f}m, y={y_3d:.2f}m, z={z_3d:.2f}m (depth)")
            print(f"     3D dimensions: h={h_3d:.2f}m, w={w_3d:.2f}m, l={l_3d:.2f}m")
            print(f"     Rotation: {rot_y:.3f} rad ({math.degrees(rot_y):.1f}°)")

            # Draw 2D bbox
            cv2.rectangle(img_vis,
                         (int(x1), int(y1)),
                         (int(x2), int(y2)),
                         color, 2)

            # Draw label
            label = f'{class_name} {conf:.2f}'
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img_vis,
                         (int(x1), int(y1) - label_size[1] - 4),
                         (int(x1) + label_size[0], int(y1)),
                         color, -1)
            cv2.putText(img_vis, label,
                       (int(x1), int(y1) - 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Project and draw 3D box
            try:
                location = np.array([x_3d, y_3d, z_3d])
                dimensions = np.array([h_3d, w_3d, l_3d])
                corners_2d = self.project_3d_to_2d(location, dimensions, rot_y, P2)
                self.draw_3d_box(img_vis, corners_2d, color, thickness=2)
            except Exception as e:
                print(f"     Warning: Could not draw 3D box: {e}")

        if save_path:
            cv2.imwrite(str(save_path), img_vis)
            print(f"\nSaved visualization to: {save_path}")

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

    # Default P2 matrix if not found
    print("Warning: P2 not found in calib file, using default")
    P2 = np.array([
        [721.5377, 0, 609.5593, 44.85728],
        [0, 721.5377, 172.854, 0.2163791],
        [0, 0, 1, 0.002745884]
    ])
    return P2


def run_inference(model_path, image_paths, data_yaml, output_dir='inference_results'):
    """Run inference on images and visualize results."""

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Load model
    print(f"Loading model from: {model_path}")
    model = YOLO(model_path, task='detect3d')

    # Create visualizer
    visualizer = KITTI3DVisualizer()

    # Process each image
    for img_path in image_paths:
        img_path = Path(img_path)
        print(f"\n{'='*60}")
        print(f"Processing: {img_path.name}")
        print('='*60)

        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Error: Could not load image {img_path}")
            continue

        print(f"Image size: {img.shape[1]}x{img.shape[0]}")

        # Load calibration
        calib_path = str(img_path).replace('image_2', 'calib').replace('.png', '.txt')
        if Path(calib_path).exists():
            P2 = load_calib(calib_path)
            print(f"Loaded calibration from: {calib_path}")
        else:
            print(f"Warning: Calib file not found: {calib_path}")
            P2 = np.array([
                [721.5377, 0, 609.5593, 44.85728],
                [0, 721.5377, 172.854, 0.2163791],
                [0, 0, 1, 0.002745884]
            ])

        # Run inference
        print("Running inference...")
        results = model.predict(img_path, save=False, conf=0.25, verbose=False)

        # Get detections
        if len(results) > 0 and results[0].boxes is not None:
            # Extract detections with 3D parameters
            boxes = results[0].boxes
            detections = []

            # Check if we have 3D parameters in the results
            if hasattr(boxes, 'data') and boxes.data.shape[1] >= 13:
                detections = boxes.data.cpu().numpy()
            else:
                print("Warning: No 3D parameters found in results")
                detections = None
        else:
            detections = None

        # Visualize
        output_path = output_dir / f"{img_path.stem}_result.jpg"
        img_vis = visualizer.visualize_detections(img, detections, P2, output_path)

        # Display using matplotlib
        plt.figure(figsize=(16, 9))
        plt.imshow(cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title(f'3D Object Detection: {img_path.name}')
        plt.tight_layout()

        # Save matplotlib figure
        plt_path = output_dir / f"{img_path.stem}_plot.png"
        plt.savefig(plt_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to: {plt_path}")

        # Show plot
        plt.show()
        plt.close()


if __name__ == "__main__":
    # Configuration
    MODEL_PATH = "last-2.pt"
    DATA_YAML = "kitti-3d.yaml"
    IMAGE_DIR = "/Users/sompoteyouwai/Downloads/datakitti/datasets/kitti/training/image_2"

    # Select a few test images
    image_dir = Path(IMAGE_DIR)
    test_images = [
        image_dir / "000000.png",
        image_dir / "000001.png",
        image_dir / "000010.png",
        image_dir / "000050.png",
        image_dir / "000100.png",
    ]

    # Filter existing images
    test_images = [img for img in test_images if img.exists()]

    if not test_images:
        print(f"Error: No test images found in {IMAGE_DIR}")
        print("Please check the path in kitti-3d.yaml")
        exit(1)

    print(f"Found {len(test_images)} test images")

    # Run inference
    run_inference(MODEL_PATH, test_images, DATA_YAML)

    print("\n" + "="*60)
    print("Inference completed!")
    print(f"Results saved to: inference_results/")
    print("="*60)
