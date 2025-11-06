#!/usr/bin/env python3
"""
Clean 3D Object Detection on Video - Only 3D Bounding Boxes

Usage:
    python video_3d_clean.py
    python video_3d_clean.py --input /path/to/video.mov --output result.mp4 --max-frames 100
"""

import argparse
import cv2
import numpy as np
from ultralytics import YOLO

parser = argparse.ArgumentParser(description="3D Object Detection on Video")
parser.add_argument("--input", default="/Users/sompoteyouwai/Downloads/1106.mov", help="Input video path")
parser.add_argument("--output", default="output_3d_only.mp4", help="Output video path")
parser.add_argument("--model", default="/Users/sompoteyouwai/env/yolo3d/yolov12/last-4.pt", help="Model path")
parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
parser.add_argument("--max-frames", type=int, default=0, help="Max frames to process (0 = all)")
parser.add_argument("--skip", type=int, default=1, help="Process every Nth frame (1 = all frames)")

args = parser.parse_args()

print(f"Loading model: {args.model}")
model = YOLO(args.model, task="detect3d")
print("Model loaded!")

# Open video
cap = cv2.VideoCapture(args.input)

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Input: {args.input}")
print(f"Resolution: {width}x{height}")
print(f"FPS: {fps}")
print(f"Total frames: {total_frames}")
print(f"Max frames: {args.max_frames if args.max_frames > 0 else 'all'}")
print(f"Skip every: {args.skip} frames")
print(f"Output: {args.output}\n")

# Setup output
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

# Default KITTI calibration
P2 = np.array(
    [[721.5377, 0, 609.5593, 44.85728],
     [0, 721.5377, 172.854, 0.2163791],
     [0, 0, 1, 0.002745884]],
    dtype=np.float32
)

# Class config
colors = {0: (0, 255, 0), 1: (255, 0, 0), 2: (0, 0, 255), 3: (255, 255, 0)}

def project_3d_to_2d(location, dimensions, rotation_y, P2):
    h, w, l = dimensions
    x, y, z = location

    corners_3d = np.array([
        [w/2, 0, l/2], [w/2, -h, l/2], [-w/2, -h, l/2], [-w/2, 0, l/2],
        [w/2, 0, -l/2], [w/2, -h, -l/2], [-w/2, -h, -l/2], [-w/2, 0, -l/2]
    ])

    R = np.array([
        [np.cos(rotation_y), 0, np.sin(rotation_y)],
        [0, 1, 0],
        [-np.sin(rotation_y), 0, np.cos(rotation_y)]
    ])

    corners_3d = corners_3d @ R.T + location
    corners_3d_homo = np.hstack([corners_3d, np.ones((8, 1))])
    corners_2d_homo = corners_3d_homo @ P2.T
    corners_2d = corners_2d_homo[:, :2] / corners_2d_homo[:, 2:3]

    return corners_2d

def draw_3d_box(img, corners_2d, color, thickness=2):
    for i in range(4):
        cv2.line(img, tuple(corners_2d[i].astype(int)), tuple(corners_2d[(i+1)%4].astype(int)), color, thickness)
    for i in range(4, 8):
        cv2.line(img, tuple(corners_2d[i].astype(int)), tuple(corners_2d[4 + (i+1)%4].astype(int)), color, thickness)
    for i in range(4):
        cv2.line(img, tuple(corners_2d[i].astype(int)), tuple(corners_2d[i+4].astype(int)), color, thickness)

# Process frames
frame_num = 0
processed_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_num += 1

    # Check max frames limit
    if args.max_frames > 0 and processed_count >= args.max_frames:
        break

    # Skip frames
    if frame_num % args.skip != 0:
        continue

    processed_count += 1

    results = model.predict(frame, save=False, conf=args.conf, verbose=False)

    if len(results) > 0 and results[0].boxes is not None:
        boxes = results[0].boxes
        if hasattr(boxes, "data") and boxes.data.shape[1] >= 13:
            detections = boxes.data.cpu().numpy()

            fx, fy = P2[0,0], P2[1,1]
            cx, cy = P2[0,2], P2[1,2]

            for det in detections:
                if len(det) < 13:
                    continue

                x1, y1, x2, y2 = det[:4]
                conf = det[4]
                cls = int(det[5])
                z_3d = det[8]
                h_3d, w_3d, l_3d = det[9:12]
                rot_y = det[12]

                center_2d_x = (x1 + x2) / 2.0
                center_2d_y = (y1 + y2) / 2.0
                x_3d = (center_2d_x - cx) * z_3d / fx
                y_3d = (center_2d_y - cy) * z_3d / fy + h_3d / 2.0

                try:
                    location = np.array([x_3d, y_3d, z_3d])
                    dimensions = np.array([h_3d, w_3d, l_3d])
                    corners_2d = project_3d_to_2d(location, dimensions, rot_y, P2)
                    color = colors.get(cls, (255, 255, 255))
                    draw_3d_box(frame, corners_2d, color, thickness=2)
                except:
                    pass

    out.write(frame)

    # Simple progress indicator
    if processed_count % 30 == 0:
        print(f"Processed: {processed_count} frames")

cap.release()
out.release()

print(f"\nDone!")
print(f"Processed {processed_count} frames")
print(f"Output saved to: {args.output}")
