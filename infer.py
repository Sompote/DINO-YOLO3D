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
import torch

# Add current directory to path
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from ultralytics import YOLO
from ultralytics.utils import LOGGER, ops
from ultralytics.utils.plotting import Annotator, Colors


class YOLO3DInference:
    """YOLOv12-3D inference class for images and videos."""

    def __init__(self, weights, device="", conf=0.25, iou=0.45, imgsz=640, classes=None):
        """
        Initialize YOLOv12-3D inference.

        Args:
            weights: Path to trained weights file
            device: Device to run inference on (cuda device, i.e. 0 or cpu)
            conf: Confidence threshold
            iou: NMS IoU threshold
            imgsz: Inference image size
            classes: Filter by class, e.g. [0, 2, 3]
        """
        self.model = YOLO(weights)
        self.device = device
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.classes = classes
        self.names = self.model.names
        self.colors = Colors()

        LOGGER.info(f"Loaded YOLOv12-3D model from {weights}")
        LOGGER.info(f"Model classes: {self.names}")

    def _convert_pred_params(self, params):
        """Convert raw network outputs into physical depth/dimension/rotation values."""
        if params is None or params.shape[1] < 5:
            return None, None, None
        depth = torch.sigmoid(params[:, 0]) * 100.0  # 0-100m
        dims = torch.sigmoid(params[:, 1:4]) * 10.0  # 0-10m (h, w, l)
        rot = (torch.sigmoid(params[:, 4]) - 0.5) * 2 * math.pi  # -pi to pi
        return depth, dims, rot

    def _draw_3d_box(self, img, box_2d, depth, dims, rotation, color):
        """
        Draw a 3D bounding box projection on the image.

        Args:
            img: Image array
            box_2d: 2D bounding box (x1, y1, x2, y2)
            depth: Object depth in meters
            dims: Object dimensions (h, w, l) in meters
            rotation: Rotation angle in radians
            color: Box color
        """
        # Get 2D box center
        x1, y1, x2, y2 = box_2d
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        # Simplified 3D box corners projection
        # This is a simplified visualization - proper projection needs camera calibration
        h, w, l = dims

        # Scale factor based on depth (perspective)
        scale = max(0.3, 1.0 - depth / 100.0)

        # Define 3D box corners in object coordinate system
        # Front face and back face
        corners_3d = np.array([
            [-l/2, -h/2, -w/2], [l/2, -h/2, -w/2], [l/2, h/2, -w/2], [-l/2, h/2, -w/2],  # Front
            [-l/2, -h/2, w/2], [l/2, -h/2, w/2], [l/2, h/2, w/2], [-l/2, h/2, w/2]      # Back
        ])

        # Apply rotation around Y axis
        cos_r = np.cos(rotation)
        sin_r = np.sin(rotation)
        R = np.array([
            [cos_r, 0, sin_r],
            [0, 1, 0],
            [-sin_r, 0, cos_r]
        ])
        corners_3d = corners_3d @ R.T

        # Project to image plane (simplified - uses 2D center)
        # Scale based on box size and depth
        img_scale = (x2 - x1) / max(l, 0.1) * scale
        corners_2d = corners_3d[:, [0, 1]] * img_scale
        corners_2d[:, 0] += cx
        corners_2d[:, 1] += cy
        corners_2d = corners_2d.astype(np.int32)

        # Draw back face
        for i in range(4):
            j = (i + 1) % 4
            cv2.line(img, tuple(corners_2d[4 + i]), tuple(corners_2d[4 + j]), color, 2)

        # Draw front face
        for i in range(4):
            j = (i + 1) % 4
            cv2.line(img, tuple(corners_2d[i]), tuple(corners_2d[j]), color, 3)

        # Draw connecting lines
        for i in range(4):
            cv2.line(img, tuple(corners_2d[i]), tuple(corners_2d[4 + i]), color, 2)

        return img

    def predict(self, source, save_dir="runs/infer3d", save_txt=False, save_conf=False,
                visualize_3d=True, view_img=False, line_width=2):
        """
        Run inference on source.

        Args:
            source: Image/video path, directory, or webcam index
            save_dir: Directory to save results
            save_txt: Save results to *.txt
            save_conf: Save confidences in labels
            visualize_3d: Draw 3D bounding boxes
            view_img: Show results in window
            line_width: Bounding box line width
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Get source type
        source = str(source)
        is_file = Path(source).suffix[1:] in ("jpg", "jpeg", "png", "bmp", "gif", "tiff")
        is_video = Path(source).suffix[1:] in ("mp4", "avi", "mov", "mkv", "webm")
        is_webcam = source.isnumeric()
        is_dir = Path(source).is_dir()

        # Setup video writer if needed
        vid_writer = None

        # Get source files
        if is_file:
            files = [Path(source)]
        elif is_dir:
            files = list(Path(source).glob("*.*"))
            files = [f for f in files if f.suffix[1:] in ("jpg", "jpeg", "png", "bmp", "gif", "tiff")]
        elif is_video or is_webcam:
            files = [source]
        else:
            raise ValueError(f"Invalid source: {source}")

        LOGGER.info(f"Processing {len(files) if not (is_video or is_webcam) else 1} source(s)")

        # Process each source
        for path in files:
            # Setup capture
            if is_video or is_webcam:
                cap = cv2.VideoCapture(int(path) if is_webcam else str(path))
                if not cap.isOpened():
                    LOGGER.error(f"Failed to open video: {path}")
                    continue

                # Get video properties
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                # Setup video writer
                save_path = save_dir / f"{Path(path).stem}_result.mp4"
                vid_writer = cv2.VideoWriter(
                    str(save_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (w, h)
                )

                frame_idx = 0
                while True:
                    ret, im0 = cap.read()
                    if not ret:
                        break

                    # Run inference
                    results = self.model.predict(
                        im0,
                        imgsz=self.imgsz,
                        conf=self.conf,
                        iou=self.iou,
                        classes=self.classes,
                        verbose=False
                    )

                    # Process results
                    im_result = self._process_results(
                        im0.copy(), results[0], visualize_3d, line_width, save_txt,
                        save_dir, Path(path).stem, frame_idx, save_conf
                    )

                    # Write frame
                    if vid_writer is not None:
                        vid_writer.write(im_result)

                    # Display
                    if view_img:
                        cv2.imshow(str(path), im_result)
                        if cv2.waitKey(1) == ord('q'):
                            break

                    frame_idx += 1

                    if frame_idx % 30 == 0:
                        LOGGER.info(f"Processed {frame_idx} frames from {Path(path).name}")

                cap.release()
                if vid_writer is not None:
                    vid_writer.release()
                LOGGER.info(f"Video saved to {save_path}")

            else:
                # Image inference
                im0 = cv2.imread(str(path))
                if im0 is None:
                    LOGGER.error(f"Failed to load image: {path}")
                    continue

                # Run inference
                results = self.model.predict(
                    im0,
                    imgsz=self.imgsz,
                    conf=self.conf,
                    iou=self.iou,
                    classes=self.classes,
                    verbose=False
                )

                # Process results
                im_result = self._process_results(
                    im0.copy(), results[0], visualize_3d, line_width, save_txt,
                    save_dir, path.stem, 0, save_conf
                )

                # Save image
                save_path = save_dir / f"{path.stem}_result{path.suffix}"
                cv2.imwrite(str(save_path), im_result)
                LOGGER.info(f"Saved result to {save_path}")

                # Display
                if view_img:
                    cv2.imshow(str(path), im_result)
                    cv2.waitKey(0)

        if view_img:
            cv2.destroyAllWindows()

    def _process_results(self, im0, result, visualize_3d, line_width, save_txt,
                        save_dir, stem, frame_idx, save_conf):
        """Process detection results and draw visualizations."""
        annotator = Annotator(im0, line_width=line_width)

        # Get detections
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return im0

        # Get predictions
        pred = boxes.data  # xyxy, conf, cls, [3d_params]

        # Extract 3D parameters if available
        depth_vals = None
        dims_vals = None
        rot_vals = None

        if pred.shape[1] > 6:
            extra_params = pred[:, 6:]
            if extra_params.shape[1] >= 5:
                depth_vals, dims_vals, rot_vals = self._convert_pred_params(extra_params)

        # Save labels
        if save_txt:
            txt_file = save_dir / "labels" / f"{stem}_{frame_idx}.txt"
            txt_file.parent.mkdir(exist_ok=True, parents=True)
            with open(txt_file, "w") as f:
                for i, box in enumerate(pred):
                    cls_id = int(box[5])
                    conf = float(box[4])
                    xyxy = box[:4]

                    # Normalized coordinates
                    xywh = ops.xyxy2xywh(xyxy.unsqueeze(0)).squeeze()
                    xywh[0::2] /= im0.shape[1]  # width
                    xywh[1::2] /= im0.shape[0]  # height

                    line = [cls_id, *xywh.tolist()]

                    # Add 3D params
                    if depth_vals is not None:
                        line.extend([
                            float(depth_vals[i]),
                            float(dims_vals[i, 0]),
                            float(dims_vals[i, 1]),
                            float(dims_vals[i, 2]),
                            float(rot_vals[i])
                        ])

                    if save_conf:
                        line.insert(1, conf)

                    f.write(("%g " * len(line)).rstrip() % tuple(line) + "\n")

        # Draw detections
        for i, box in enumerate(pred):
            xyxy = box[:4].cpu().numpy()
            conf = float(box[4])
            cls_id = int(box[5])

            # Get color
            color = self.colors(cls_id, True)

            # Build label
            label = f"{self.names[cls_id]} {conf:.2f}"

            # Add 3D info to label
            if depth_vals is not None:
                depth = float(depth_vals[i])
                h, w, l = float(dims_vals[i, 0]), float(dims_vals[i, 1]), float(dims_vals[i, 2])
                rot = float(rot_vals[i])
                label += f"\nD:{depth:.1f}m H:{h:.1f}m W:{w:.1f}m L:{l:.1f}m"
                label += f"\nR:{math.degrees(rot):.1f}°"

            # Draw 2D box
            annotator.box_label(xyxy, label, color=color)

            # Draw 3D box
            if visualize_3d and depth_vals is not None:
                depth = float(depth_vals[i])
                dims = dims_vals[i].cpu().numpy()
                rot = float(rot_vals[i])
                self._draw_3d_box(im0, xyxy, depth, dims, rot, color)

        return annotator.result()


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
    parser.add_argument("--visualize-3d", action="store_true", default=True, help="Draw 3D boxes")
    parser.add_argument("--no-visualize-3d", dest="visualize_3d", action="store_false", help="Don't draw 3D boxes")
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

    # Initialize inference
    inferencer = YOLO3DInference(
        weights=args.weights,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        classes=args.classes
    )

    # Run inference
    inferencer.predict(
        source=args.source,
        save_dir=args.save_dir,
        save_txt=args.save_txt,
        save_conf=args.save_conf,
        visualize_3d=args.visualize_3d,
        view_img=args.view_img,
        line_width=args.line_width
    )

    print("\n" + "=" * 80)
    print("Inference completed successfully!")
    print(f"Results saved to: {args.save_dir}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
