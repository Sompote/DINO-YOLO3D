#!/usr/bin/env python3
"""
YOLOv12-3D Inference Script

Performs inference on images and videos with trained YOLOv12-3D models,
visualizing 2D bounding boxes with 3D information (depth, dimensions, rotation).

Developed by AI Research Group
Department of Civil Engineering
King Mongkut's University of Technology Thonburi (KMUTT)

Usage:
    # Image inference
    python infer.py --weights runs/detect3d/train/weights/best.pt --source image.jpg

    # Video inference
    python infer.py --weights runs/detect3d/train/weights/best.pt --source video.mp4

    # Directory inference
    python infer.py --weights runs/detect3d/train/weights/best.pt --source images/

    # Webcam inference
    python infer.py --weights runs/detect3d/train/weights/best.pt --source 0
"""

import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np

# Add current directory to path
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from ultralytics import YOLO
from ultralytics.utils import LOGGER
from ultralytics.utils.plotting import Annotator, Colors


def compute_3d_box_corners(location, dimensions, rotation_y):
    """
    Compute 3D bounding box corners in camera coordinates (KITTI format).

    Args:
        location: [x, y, z] - center bottom of 3D box in camera coords
        dimensions: [h, w, l] - height, width, length
        rotation_y: rotation around Y-axis in radians

    Returns:
        corners_3d: (8, 3) array of corner coordinates
    """
    h, w, l = dimensions

    # 3D box corners in object coordinate system (before rotation)
    # Bottom face center is at origin, extends downward (y-down in camera)
    x_corners = np.array([l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2])
    y_corners = np.array([0, 0, 0, 0, -h, -h, -h, -h])
    z_corners = np.array([w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2])

    # Rotation matrix around Y-axis
    R = np.array([
        [np.cos(rotation_y), 0, np.sin(rotation_y)],
        [0, 1, 0],
        [-np.sin(rotation_y), 0, np.cos(rotation_y)]
    ])

    # Rotate and translate
    corners_3d = np.vstack([x_corners, y_corners, z_corners])  # (3, 8)
    corners_3d = R @ corners_3d
    corners_3d = corners_3d.T + location  # (8, 3)

    return corners_3d


def project_3d_to_2d(corners_3d, P):
    """
    Project 3D corners to 2D image plane using camera projection matrix.

    Args:
        corners_3d: (8, 3) array of 3D corners
        P: (3, 4) camera projection matrix

    Returns:
        corners_2d: (8, 2) array of 2D pixel coordinates
    """
    # Convert to homogeneous coordinates
    corners_3d_homo = np.hstack([corners_3d, np.ones((8, 1))])  # (8, 4)

    # Project: corners_2d_homo = P @ corners_3d_homo^T
    corners_2d_homo = (P @ corners_3d_homo.T).T  # (8, 3)

    # Normalize by depth
    corners_2d = corners_2d_homo[:, :2] / corners_2d_homo[:, 2:3]

    return corners_2d.astype(np.int32)


def draw_3d_box(img, box_2d, depth, dims, rotation, color, thickness=2, P=None):
    """
    Draw proper KITTI-style 3D bounding box using camera projection.

    Based on: https://github.com/ehsanrs2/Kitti-3D-bounding-box

    Args:
        img: Image array
        box_2d: 2D bounding box (x1, y1, x2, y2)
        depth: Object depth in meters (z-coordinate)
        dims: Object dimensions [h, w, l] in meters
        rotation: Rotation angle in radians (rotation_y)
        color: Box color tuple (B, G, R)
        thickness: Line thickness
        P: Camera projection matrix (3, 4)
    """
    # Validate inputs
    if depth <= 0 or depth > 100:
        return img

    # Use KITTI P2 matrix if not provided
    # These are the average parameters from KITTI dataset
    if P is None:
        img_h, img_w = img.shape[:2]
        # KITTI P2 average values, scaled to match image size
        # Original KITTI image size is ~1242x375, we scale focal length proportionally
        kitti_w = 1242.0
        scale = img_w / kitti_w
        fx = fy = 721.5377 * scale
        cx = img_w / 2.0
        cy = img_h / 2.0
        P = np.array([
            [fx, 0.0, cx, 0.0],
            [0.0, fy, cy, 0.0],
            [0.0, 0.0, 1.0, 0.0]
        ])

    # Back-project 2D bottom center to 3D location
    # In KITTI, location is the bottom center of the 3D box
    x1, y1, x2, y2 = box_2d
    cx_2d = (x1 + x2) / 2
    # Use bottom of 2D box for y (ground plane alignment)
    cy_2d = y2

    fx, fy = P[0, 0], P[1, 1]
    cx_cam, cy_cam = P[0, 2], P[1, 2]

    # Compute 3D location from 2D projection and depth
    x_3d = (cx_2d - cx_cam) * depth / fx
    y_3d = (cy_2d - cy_cam) * depth / fy
    z_3d = depth

    location = np.array([x_3d, y_3d, z_3d])

    # Compute 3D box corners and project to 2D
    try:
        # Scale dimensions for better visualization (2.0x larger)
        # This helps the 3D box better match the visual size of the 2D box
        dims_scaled = dims * 2.0
        corners_3d = compute_3d_box_corners(location, dims_scaled, rotation)
        corners_2d = project_3d_to_2d(corners_3d, P)

        # Check if corners are within reasonable bounds
        if np.any(corners_2d < -1000) or np.any(corners_2d > 10000):
            return img

        # Draw all 12 edges of the 3D box
        # Bottom face: 0-1-2-3, Top face: 4-5-6-7, Vertical: 0-4, 1-5, 2-6, 3-7
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # bottom
            (4, 5), (5, 6), (6, 7), (7, 4),  # top
            (0, 4), (1, 5), (2, 6), (3, 7),  # vertical
        ]

        for start, end in edges:
            pt1 = tuple(corners_2d[start])
            pt2 = tuple(corners_2d[end])
            cv2.line(img, pt1, pt2, color, thickness)

        # Highlight front face (facing camera)
        front_edges = [(0, 1), (1, 5), (5, 4), (4, 0)]
        for start, end in front_edges:
            pt1 = tuple(corners_2d[start])
            pt2 = tuple(corners_2d[end])
            cv2.line(img, pt1, pt2, color, thickness + 1)

    except Exception as e:
        pass

    return img


def save_labels(save_path, boxes_data, img_shape, save_conf=False):
    """
    Save detection labels to text file.

    Args:
        save_path: Path to save txt file
        boxes_data: Boxes tensor with shape [N, 11] containing
                   [x1, y1, x2, y2, conf, cls, depth, h_3d, w_3d, l_3d, rotation_y]
        img_shape: Image shape (height, width)
        save_conf: Whether to include confidence scores
    """
    save_path.parent.mkdir(exist_ok=True, parents=True)

    with open(save_path, "w") as f:
        for box in boxes_data:
            cls_id = int(box[5])
            conf = float(box[4])

            # Convert to normalized xywh
            x1, y1, x2, y2 = box[:4]
            x_center = ((x1 + x2) / 2) / img_shape[1]
            y_center = ((y1 + y2) / 2) / img_shape[0]
            width = (x2 - x1) / img_shape[1]
            height = (y2 - y1) / img_shape[0]

            line = [cls_id, x_center, y_center, width, height]

            # Add 3D params if available
            if len(box) >= 11:
                depth = float(box[6])
                dim_h = float(box[7])
                dim_w = float(box[8])
                dim_l = float(box[9])
                rotation = float(box[10])
                line.extend([depth, dim_h, dim_w, dim_l, rotation])

            if save_conf:
                line.insert(1, conf)

            f.write(("%g " * len(line)).rstrip() % tuple(line) + "\n")


def run_inference(
    weights,
    source,
    imgsz=640,
    conf=0.25,
    iou=0.45,
    device="",
    save_dir="runs/infer3d",
    save_txt=False,
    save_conf=False,
    classes=None,
    visualize_3d=True,
    view_img=False,
    line_width=2,
):
    """
    Run YOLOv12-3D inference on images or videos.

    Args:
        weights: Path to model weights
        source: Input source (image/video/directory/webcam)
        imgsz: Inference image size
        conf: Confidence threshold
        iou: NMS IoU threshold
        device: Device to use (cuda:0 or cpu)
        save_dir: Directory to save results
        save_txt: Save labels to txt files
        save_conf: Include confidence in labels
        classes: Filter by class IDs
        visualize_3d: Draw 3D bounding boxes
        view_img: Display results
        line_width: Bounding box line width
    """
    # Load model
    model = YOLO(weights)
    names = model.names
    colors_palette = Colors()

    LOGGER.info(f"Loaded YOLOv12-3D model from {weights}")
    LOGGER.info(f"Model classes: {names}")

    # Create save directory
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Run inference using model's built-in predict
    results = model.predict(
        source=source,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        device=device,
        classes=classes,
        stream=True,  # Use streaming for videos
        verbose=False,
    )

    # Process results
    for idx, result in enumerate(results):
        # Get image (ensure contiguous for cv2 operations)
        img = np.ascontiguousarray(result.orig_img)
        img_path = Path(result.path) if result.path else Path(f"image_{idx}")

        # Get boxes
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            LOGGER.info(f"No detections in {img_path.name}")
            continue

        # boxes.data contains: [x1, y1, x2, y2, conf, cls, depth, h_3d, w_3d, l_3d, rotation_y]
        boxes_data = boxes.data.cpu().numpy()

        # Check if 3D params are available
        has_3d = boxes_data.shape[1] >= 11

        # Create annotator (force cv2 mode for consistency)
        annotator = Annotator(img, line_width=line_width, pil=False)

        # Draw each detection
        for box in boxes_data:
            xyxy = box[:4]
            conf_val = float(box[4])
            cls_id = int(box[5])

            # Get color
            color = colors_palette(cls_id, True)

            # Build simple label (class name only)
            label = f"{names[cls_id]} {conf_val:.2f}"

            # Draw 2D box with simple label
            annotator.box_label(xyxy, label, color=color)

        # Get result image with 2D annotations
        img_result = annotator.result()

        # Draw 3D boxes on top of 2D annotations
        if visualize_3d and has_3d:
            for box in boxes_data:
                xyxy = box[:4]
                cls_id = int(box[5])
                color = colors_palette(cls_id, True)
                depth = float(box[6])
                dims = np.array([box[7], box[8], box[9]])
                rot = float(box[10])
                draw_3d_box(img_result, xyxy, depth, dims, rot, color, line_width)

        # Save image/video
        if result.path:
            stem = Path(result.path).stem
            suffix = Path(result.path).suffix
        else:
            stem = f"result_{idx}"
            suffix = ".jpg"

        save_path = save_dir / f"{stem}_result{suffix}"

        if suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
            # Save image
            cv2.imwrite(str(save_path), img_result)
            LOGGER.info(f"Saved result to {save_path}")
        else:
            # For videos, save frame (video saving handled by model.predict with save=True)
            pass

        # Save labels
        if save_txt:
            txt_path = save_dir / "labels" / f"{stem}_{idx}.txt"
            save_labels(txt_path, boxes_data, img.shape[:2], save_conf)

        # Display
        if view_img:
            cv2.imshow(str(img_path), img_result)
            if cv2.waitKey(1) == ord("q"):
                break

    if view_img:
        cv2.destroyAllWindows()

    LOGGER.info(f"Inference complete. Results saved to {save_dir}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="YOLOv12-3D Inference")
    parser.add_argument("--weights", type=str, required=True, help="Path to trained weights")
    parser.add_argument("--source", type=str, required=True, help="Image/video path, directory, or webcam (0)")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--device", type=str, default="", help="Device (cuda:0 or cpu)")
    parser.add_argument("--save-dir", type=str, default="runs/infer3d", help="Save directory")
    parser.add_argument("--save-txt", action="store_true", help="Save results to *.txt")
    parser.add_argument("--save-conf", action="store_true", help="Save confidences in labels")
    parser.add_argument("--classes", nargs="+", type=int, help="Filter by class")
    parser.add_argument("--visualize-3d", dest="visualize_3d", action="store_true", help="Draw 3D boxes")
    parser.add_argument("--no-visualize-3d", dest="visualize_3d", action="store_false", help="Don't draw 3D boxes")
    parser.set_defaults(visualize_3d=True)
    parser.add_argument("--view-img", action="store_true", help="Show results")
    parser.add_argument("--line-width", type=int, default=2, help="Bounding box line width")

    return parser.parse_args()


def main():
    """Main inference function."""
    args = parse_args()

    # Print header
    print("\n" + "=" * 80)
    print("YOLOv12-3D Inference Tool")
    print("AI Research Group, Civil Engineering, KMUTT")
    print("=" * 80 + "\n")

    # Check weights exist
    if not Path(args.weights).exists():
        LOGGER.error(f"Weights file not found: {args.weights}")
        sys.exit(1)

    # Check source exists (except webcam)
    if not args.source.isnumeric() and not Path(args.source).exists():
        LOGGER.error(f"Source not found: {args.source}")
        sys.exit(1)

    # Run inference
    run_inference(
        weights=args.weights,
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        save_dir=args.save_dir,
        save_txt=args.save_txt,
        save_conf=args.save_conf,
        classes=args.classes,
        visualize_3d=args.visualize_3d,
        view_img=args.view_img,
        line_width=args.line_width,
    )

    print("\n" + "=" * 80)
    print("Inference completed successfully!")
    print(f"Results saved to: {args.save_dir}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
