# YOLOv12-3D

[![KMUTT](https://img.shields.io/badge/Made%20by-KMUTT%20Civil%20Engineering-orange)](https://www.kmutt.ac.th/)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-AGPL--3.0-green.svg)](LICENSE)

Monocular YOLOv12 extended with a 3D detection head for the KITTI benchmark. The repository provides scripts for dataset preparation, training, validation, and inference with consistent CLI ergonomics.

![YOLOv12-3D detection result on KITTI street scene](assets/yolov12-3d-detection-sample.jpg)

## Features

- 3D-aware detections (depth, dimensions, yaw) on top of YOLOv12 accuracy and speed
- Turnkey CLI (`yolo3d.py`) for training, validation, inference, and export
- **Video 3D detection** with real-time 3D bounding box visualization
- Automated KITTI setup utilities with verification and split management
- Multiple model scales (n/s/m/l/x) to balance FPS and accuracy

## Quick Start

```bash
git clone https://github.com/Sompote/YOLOv12-3D.git
cd YOLOv12-3D
pip install -r requirements.txt
pip install ultralytics torch torchvision
```

### Prepare KITTI

```bash
# Recommended: end-to-end download and extraction (Linux)
./download_kitti.sh

# or for cross-platform setups
python download_kitti.py --extract
python scripts/kitti_setup.py all --create-yaml
```

### Train, Validate, Infer

```bash
# Train (default config)
python yolo3d.py train --data kitti-3d.yaml --epochs 100 -y

# Validate a checkpoint
python yolo3d.py val --model runs/detect/train/weights/best.pt --data kitti-3d.yaml

# Run inference on an image folder
python yolo3d.py predict --model runs/detect/train/weights/best.pt --source path/to/images --conf 0.25

# Video 3D Detection
# Process video with 3D bounding boxes (only 3D boxes, no 2D)
python video_3d_clean.py --input path/to/video.mov --output result.mp4

# Fast processing options
python video_3d_clean.py --max-frames 100           # First 100 frames only
python video_3d_clean.py --skip 5                   # Every 5th frame for speed
python video_3d_clean.py --max-frames 100 --skip 3  # Combined for fast preview
```

For exports: `python yolo3d.py export --model best.pt --format onnx`.

## Documentation

- `CLI_GUIDE.md` – full command reference
- `YOLO3D_QUICKREF.md` – parameter cheat sheet
- `KITTI_DOWNLOAD_GUIDE.md` – manual download walkthrough
- `scripts/README.md` – dataset automation details
- `IMPLEMENTATION_SUMMARY.md` – architecture and loss overview

## Repository Layout

```
YOLOv12-3D/
├── yolo3d.py                 # Main CLI entry point
├── video_3d_clean.py         # Video 3D detection script
├── scripts/kitti_setup.py    # Dataset automation
├── ultralytics/              # Model, loss, and trainer extensions
├── assets/                   # Project figures (includes detection sample)
├── result.mp4                # Example 3D detection video output
└── docs & guides             # *.md reference material
```

## Example Results

![3D Detection on Street Scene](assets/yolov12-3d-detection-sample.jpg)

**Video Output:** `result.mp4` - Demonstrates real-time 3D object detection with bounding boxes projected from depth estimation on street scene video.

## Contributing & License

Contributions are welcome—please open an issue or submit a PR with improvements, fixes, or new features. The project is released under the AGPL-3.0 license (see `LICENSE`).

For collaboration or academic inquiries, contact the AI Research Group, Department of Civil Engineering, KMUTT.

