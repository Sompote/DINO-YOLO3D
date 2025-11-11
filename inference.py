#!/usr/bin/env python3
"""
Unified 3D Object Detection Inference Script

Supports both image and video inference with CLI arguments.

Usage:
    # Single image
    python inference.py --input image.png --model last-5.pt
    
    # Multiple images
    python inference.py --input image1.png image2.png image3.png --model last-5.pt
    
    # Video
    python inference.py --input video.mov --model last-5.pt --output result.mp4
    
    # With custom confidence threshold
    python inference.py --input image.png --model last-5.pt --conf 0.3
    
    # Video with frame skip
    python inference.py --input video.mov --model last-5.pt --skip 2 --max-frames 100
"""

import argparse
import math
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from ultralytics import YOLO


class KITTI3DVisualizer:
    """Visualize 3D detections on KITTI images."""

    def __init__(self):
        """Initialize KITTI 3D visualizer."""
        # KITTI class names
        self.class_names = {
            0: "Car",
            1: "Truck",
            2: "Pedestrian",
            3: "Cyclist",
            4: "Misc",
            5: "Van",
            6: "Tram",
            7: "Person_sitting",
        }

        # Colors for each class (BGR format)
        self.colors = {
            0: (0, 255, 0),      # Car - Green
            1: (255, 0, 0),      # Truck - Blue
            2: (0, 0, 255),      # Pedestrian - Red
            3: (255, 255, 0),    # Cyclist - Cyan
            4: (128, 128, 128),  # Misc - Gray
            5: (255, 0, 255),    # Van - Magenta
            6: (0, 255, 255),    # Tram - Yellow
            7: (128, 0, 128),    # Person_sitting - Purple
        }

    def draw_3d_box(self, img, corners_2d, color=(0, 255, 0), thickness=2):
        """Draw 3D bounding box on image."""
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
            location: (x, y, z) 3D center in camera coordinates - BOTTOM CENTER
            dimensions: (h, w, l) object dimensions
            rotation_y: rotation around Y axis
            P2: 3x4 projection matrix

        Returns:
            corners_2d: 8x2 array of projected 2D corners
        """
        h, w, l = dimensions
        x, y, z = location

        # CRITICAL: In KITTI, location (x,y,z) is at the BOTTOM CENTER of the box!
        corners_3d = np.array(
            [
                [w / 2, 0, l / 2],      # back bottom right
                [w / 2, -h, l / 2],     # back top right
                [-w / 2, -h, l / 2],    # back top left
                [-w / 2, 0, l / 2],     # back bottom left
                [w / 2, 0, -l / 2],     # front bottom right
                [w / 2, -h, -l / 2],    # front top right
                [-w / 2, -h, -l / 2],   # front top left
                [-w / 2, 0, -l / 2],    # front bottom left
            ]
        )

        # Rotation matrix around Y axis
        R = np.array(
            [[np.cos(rotation_y), 0, np.sin(rotation_y)], 
             [0, 1, 0], 
             [-np.sin(rotation_y), 0, np.cos(rotation_y)]]
        )

        # Rotate and translate
        corners_3d = corners_3d @ R.T
        corners_3d = corners_3d + location

        # Project to 2D
        corners_3d_homo = np.hstack([corners_3d, np.ones((8, 1))])
        corners_2d_homo = corners_3d_homo @ P2.T
        corners_2d = corners_2d_homo[:, :2] / corners_2d_homo[:, 2:3]

        return corners_2d

    def visualize_detections(self, img, detections, P2, verbose=True):
        """
        Visualize all detections on image.

        Args:
            img: Input image (BGR)
            detections: Detection results with 3D parameters
            P2: Projection matrix
            verbose: Print detection info

        Returns:
            img_vis: Visualized image
        """
        img_vis = img.copy()

        if detections is None or len(detections) == 0:
            if verbose:
                print("No detections found")
            return img_vis

        if verbose:
            print(f"Found {len(detections)} detections:")

        # Extract camera intrinsics from P2
        fx = P2[0, 0]
        fy = P2[1, 1]
        cx = P2[0, 2]
        cy = P2[1, 2]

        for i, det in enumerate(detections):
            if len(det) < 13:
                continue

            x1, y1, x2, y2 = det[:4]
            conf = det[4]
            cls = int(det[5])
            x_3d_pred, y_3d_pred, z_3d = det[6:9]
            h_3d, w_3d, l_3d = det[9:12]
            rot_y = det[12]

            # Compute 3D location from 2D box center and depth
            center_2d_x = (x1 + x2) / 2.0
            center_2d_y = (y1 + y2) / 2.0

            # Project 2D center to 3D using depth (z) and camera intrinsics
            x_3d = (center_2d_x - cx) * z_3d / fx
            y_3d = (center_2d_y - cy) * z_3d / fy + h_3d / 2.0

            # Get class info
            class_name = self.class_names.get(cls, f"Class_{cls}")
            color = self.colors.get(cls, (255, 255, 255))

            if verbose:
                print(f"  {i + 1}. {class_name} (conf={conf:.3f})")
                print(f"     3D location: x={x_3d:.2f}m, y={y_3d:.2f}m, z={z_3d:.2f}m")

            # Project and draw 3D box
            try:
                location = np.array([x_3d, y_3d, z_3d])
                dimensions = np.array([h_3d, w_3d, l_3d])
                corners_2d = self.project_3d_to_2d(location, dimensions, rot_y, P2)
                self.draw_3d_box(img_vis, corners_2d, color, thickness=2)

                # Draw label
                label = f"{class_name} {conf:.2f}"
                top_corner = corners_2d[1]
                text_pos = (int(top_corner[0]), int(top_corner[1]) - 5)
                cv2.putText(
                    img_vis,
                    label,
                    text_pos,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                    lineType=cv2.LINE_AA,
                )
            except Exception as e:
                if verbose:
                    print(f"     Warning: Could not draw 3D box: {e}")

        return img_vis


def load_calib(calib_file):
    """Load KITTI calibration file."""
    if not Path(calib_file).exists():
        return None
        
    with open(calib_file, "r") as f:
        lines = f.readlines()

    for line in lines:
        if line.startswith("P2:"):
            P2 = np.array([float(x) for x in line.split()[1:]], dtype=np.float32)
            P2 = P2.reshape(3, 4)
            return P2

    return None


def get_default_P2():
    """Get default KITTI P2 calibration matrix."""
    return np.array(
        [[721.5377, 0, 609.5593, 44.85728], 
         [0, 721.5377, 172.854, 0.2163791], 
         [0, 0, 1, 0.002745884]],
        dtype=np.float32
    )


def process_image(model, img_path, visualizer, args):
    """Process a single image."""
    img_path = Path(img_path)
    print(f"\n{'='*80}")
    print(f"Processing: {img_path.name}")
    print(f"{'='*80}")

    # Load image
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Error: Could not load image {img_path}")
        return

    print(f"Image size: {img.shape[1]}x{img.shape[0]}")

    # Load calibration
    calib_path = str(img_path).replace("/image_2/", "/label_2/").replace(".png", ".txt").replace(".jpg", ".txt")
    calib_path = str(img_path).replace("/image_2/", "/calib/").replace(".png", ".txt").replace(".jpg", ".txt")
    
    P2 = load_calib(calib_path)
    if P2 is None:
        print(f"Using default calibration")
        P2 = get_default_P2()
    else:
        print(f"Loaded calibration from: {calib_path}")

    # Run inference
    print("Running inference...")
    results = model.predict(img_path, save=False, conf=args.conf, verbose=False)

    # Get detections
    detections = None
    if len(results) > 0 and results[0].boxes is not None:
        boxes = results[0].boxes
        if hasattr(boxes, "data") and boxes.data.shape[1] >= 13:
            detections = boxes.data.cpu().numpy()

    # Visualize
    img_vis = visualizer.visualize_detections(img, detections, P2, verbose=not args.quiet)

    # Save result
    if args.output:
        output_path = Path(args.output) / f"{img_path.stem}_result.jpg"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), img_vis)
        print(f"Saved: {output_path}")

    # Display
    if args.show:
        plt.figure(figsize=(16, 9))
        plt.imshow(cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title(f"3D Detection: {img_path.name}")
        plt.tight_layout()
        plt.show()


def process_video(model, video_path, visualizer, args):
    """Process a video file."""
    print(f"\n{'='*80}")
    print(f"Processing Video: {video_path}")
    print(f"{'='*80}")

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Resolution: {width}x{height}")
    print(f"FPS: {fps}")
    print(f"Total frames: {total_frames}")
    
    # Setup output
    if args.output:
        output_path = Path(args.output)
        if output_path.is_dir():
            output_path = output_path / f"{Path(video_path).stem}_3d.mp4"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        print(f"Output: {output_path}")
    else:
        out = None
        print("Warning: No output specified, video will not be saved")

    # Get calibration
    P2 = get_default_P2()

    # Process frames
    frame_num = 0
    processed_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_num += 1

        # Check limits
        if args.max_frames > 0 and processed_count >= args.max_frames:
            break

        # Skip frames
        if frame_num % args.skip != 0:
            continue

        processed_count += 1

        # Run inference
        results = model.predict(frame, save=False, conf=args.conf, verbose=False)

        # Get detections
        detections = None
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            if hasattr(boxes, "data") and boxes.data.shape[1] >= 13:
                detections = boxes.data.cpu().numpy()

        # Visualize
        frame_vis = visualizer.visualize_detections(frame, detections, P2, verbose=False)

        # Write output
        if out is not None:
            out.write(frame_vis)

        # Progress
        if processed_count % 30 == 0:
            print(f"Processed: {processed_count}/{total_frames if args.max_frames == 0 else args.max_frames} frames")

    cap.release()
    if out is not None:
        out.release()

    print(f"\nDone! Processed {processed_count} frames")
    if args.output:
        print(f"Saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="3D Object Detection Inference - Images and Videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single image
  python inference.py --input image.png --model last-5.pt
  
  # Multiple images
  python inference.py --input img1.png img2.png --model last-5.pt --output results/
  
  # Video
  python inference.py --input video.mov --model last-5.pt --output result.mp4
  
  # Video with options
  python inference.py --input video.mov --model last-5.pt --skip 2 --max-frames 100
        """
    )
    
    parser.add_argument(
        "--input", 
        nargs="+",
        required=True,
        help="Input image(s) or video file path"
    )
    parser.add_argument(
        "--model",
        default="last-5.pt",
        help="Model weights path (default: last-5.pt)"
    )
    parser.add_argument(
        "--output",
        default="inference_results",
        help="Output directory for images or output video path (default: inference_results)"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold (default: 0.25)"
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display results (images only)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress detection details"
    )
    
    # Video-specific options
    parser.add_argument(
        "--skip",
        type=int,
        default=1,
        help="Process every Nth frame (default: 1, video only)"
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Max frames to process, 0=all (default: 0, video only)"
    )

    args = parser.parse_args()

    # Load model
    print(f"Loading model: {args.model}")
    model = YOLO(args.model, task="detect3d")
    print("Model loaded!")

    # Create visualizer
    visualizer = KITTI3DVisualizer()

    # Process inputs
    for input_path in args.input:
        input_path = Path(input_path)
        
        if not input_path.exists():
            print(f"Error: {input_path} not found")
            continue

        # Check if video or image
        video_exts = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'}
        if input_path.suffix.lower() in video_exts:
            process_video(model, input_path, visualizer, args)
        else:
            process_image(model, input_path, visualizer, args)

    print(f"\n{'='*80}")
    print("All processing complete!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
