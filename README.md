# cd YOLOv12-3D

[![KMUTT](https://img.shields.io/badge/Made%20by-KMUTT%20Civil%20Engineering-orange)](https://www.kmutt.ac.th/)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-AGPL--3.0-green.svg)](LICENSE)

**3D Object Detection for KITTI Dataset**

**Developed by AI Research Group, Department of Civil Engineering**
**King Mongkut's University of Technology Thonburi (KMUTT)**

> **⚠️ IMPORTANT RESEARCH DISCLAIMER ⚠️**
>
> This is an **UNDERDEVELOPED RESEARCH VERSION** of YOLOv12-3D currently being used within our research group. This implementation is:
> - 🔬 **Actively in development** - features may change or break
> - 🚧 **Not production-ready** - use with caution
> - 🧪 **Experimental** - expected accuracy and performance may vary
> - 📝 **Not fully documented** - some features may lack complete documentation
>
> **USE AT YOUR OWN RISK.** We strongly recommend:
> - Thoroughly testing before any critical use
> - Validating results independently
> - Using appropriate backup/safety measures
>
> **We welcome ALL feedback, bug reports, suggestions, and contributions** to help improve this project!
> Please open issues, submit PRs, or contact us with your experiences and ideas.
>
> Together, we can make this project better for everyone! 🚀

A professional implementation extending YOLOv12 with 3D object detection capabilities for autonomous driving applications using the KITTI dataset. Features complete CLI tools for dataset setup, training, validation, and inference.

---

## 🚀 Key Features

- **🎯 3D Bounding Box Detection**: Predicts 3D location, dimensions, and rotation from monocular images
- **⚡ YOLOv12 Architecture**: Built on latest YOLO backbone for speed and accuracy
- **🗂️ KITTI Dataset Support**: Native support for KITTI 3D object detection format
- **🖥️ Professional CLI Tools**: Industry-standard command-line interface for all operations
- **🔧 End-to-End Pipeline**: From dataset download to model export in simple commands
- **📊 Multiple Model Sizes**: n/s/m/l/x variants for different speed/accuracy trade-offs

---

## 📦 Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/Sompote/YOLOv12-3D.git
cd YOLOv12-3D

# Install dependencies
pip install -r requirements.txt
pip install ultralytics torch torchvision
```

### 2. Dataset Setup

#### Option A: Automated Download (Recommended for Linux)

```bash
# One-command setup - downloads, extracts, and configures everything
./download_kitti.sh

# Or using Python script
python download_kitti.py --extract
```

#### Option B: Manual Download + Automated Setup

```bash
# Step 1: Download KITTI dataset manually
# Visit: http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d
# Download and place files in ./downloads/ directory:
#   - data_object_image_2.zip (12 GB)
#   - data_object_label_2.zip (5 MB)
#   - data_object_calib.zip (16 MB)

# Step 2: Setup dataset automatically
python scripts/kitti_setup.py all --create-yaml

# This will:
#   ✓ Extract all files
#   ✓ Verify dataset structure
#   ✓ Create train/val splits (80/20)
#   ✓ Generate YAML configuration
```

📖 **Complete Download Guide**: [KITTI_DOWNLOAD_GUIDE.md](KITTI_DOWNLOAD_GUIDE.md)

```bash
# Train with default settings
python yolo3d.py train --data kitti-3d.yaml --epochs 100 -y

# Train with custom settings
python yolo3d.py train \
    --model s \
    --data kitti-3d.yaml \
    --epochs 100 \
    --batch 16 \
    --imgsz 640 \
    --device 0 \
    --name yolov12s-kitti \
    -y
```

### 4. Run Inference

```bash
# Predict on images
python yolo3d.py predict \
    --model runs/detect/train/weights/best.pt \
    --source path/to/images/ \
    --conf 0.25

# Predict on video
python yolo3d.py predict \
    --model best.pt \
    --source video.mp4 \
    --save
```

### 5. Validate Model

```bash
python yolo3d.py val \
    --model runs/detect/train/weights/best.pt \
    --data kitti-3d.yaml
```

### 6. Export Model

```bash
# Export to ONNX
python yolo3d.py export --model best.pt --format onnx

# Export to TensorRT
python yolo3d.py export --model best.pt --format engine --half
```

---

## 🖥️ CLI Tools

### Dataset Setup Tool: `kitti_setup.py`

Professional CLI for KITTI dataset preparation.

```bash
# Check downloads
python scripts/kitti_setup.py download

# Extract files
python scripts/kitti_setup.py extract

# Verify structure
python scripts/kitti_setup.py verify

# Create splits
python scripts/kitti_setup.py split --val-split 0.2

# Complete setup (all-in-one)
python scripts/kitti_setup.py all --create-yaml
```

**Features:**
- ✅ Color-coded output with progress indicators
- ✅ Automatic file verification
- ✅ Configurable train/val splits
- ✅ YAML config generation

📖 **Full Documentation**: [scripts/README.md](scripts/README.md)

### Training Tool: `yolo3d.py`

Unified CLI for all YOLOv12-3D operations.

**Commands:**
- `train` - Train 3D detection model
- `val` - Validate model performance
- `predict` - Run inference on images/videos
- `export` - Export to ONNX, TensorRT, TFLite, etc.

```bash
# Get help for any command
python yolo3d.py train --help
python yolo3d.py predict --help

# Example: Training with all options
python yolo3d.py train \
    --model n \
    --data kitti-3d.yaml \
    --epochs 100 \
    --batch 16 \
    --imgsz 640 \
    --device 0 \
    --workers 8 \
    --optimizer SGD \
    --lr0 0.01 \
    --name my-experiment \
    -y
```

**Features:**
- ✅ Professional banner and colored output
- ✅ Comprehensive parameter validation
- ✅ Progress tracking and metrics reporting
- ✅ Confirmation prompts (skippable with `-y`)

📖 **Full Documentation**: [CLI_GUIDE.md](CLI_GUIDE.md)
📋 **Quick Reference**: [YOLO3D_QUICKREF.md](YOLO3D_QUICKREF.md)

---

## 📊 Model Variants

| Model | Parameters | Speed | mAP50-95* | Use Case |
|-------|-----------|-------|-----------|----------|
| **cd YOLOv12n-3D** | 2.6M | ⚡⚡⚡⚡⚡ | ~30% | Edge devices, real-time |
| **cd YOLOv12s-3D** | 9.1M | ⚡⚡⚡⚡ | ~35% | Balanced performance |
| **cd YOLOv12m-3D** | 19.7M | ⚡⚡⚡ | ~40% | High accuracy |
| **cd YOLOv12l-3D** | 26.5M | ⚡⚡ | ~42% | Best accuracy |
| **cd YOLOv12x-3D** | 59.4M | ⚡ | ~44% | Maximum accuracy |

*Estimated performance on KITTI dataset

---

## 🏗️ Architecture Overview

### Model Components

![YOLOv12-3D Architecture](yolov12-3d-architecture.svg)

**Key Components:**

```
┌─────────────────────────────────────────────────────────────┐
│                     YOLOv12-3D Pipeline                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Input Image (640×640)                                      │
│       ↓                                                     │
│  Backbone (YOLOv12)  ──→  Feature Extraction                │
│       ↓                                                     │
│  Neck (FPN/PAN)      ──→  Multi-scale Features              │
│       ↓                                                     │
│  Detect3D Head       ──→  Predictions:                      │
│                           • 2D Box [x, y, w, h]             │
│                           • Class Probabilities             │
│                           • 3D Depth (z)                    │
│                           • 3D Dimensions [h, w, l]         │
│                           • Rotation (yaw angle)            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 3D Detection Head

The `Detect3D` class extends standard YOLO detection with 3D parameters:

```python
# Output per detection
{
    "2d_box": [x, y, w, h],              # 2D bounding box
    "class": class_id,                    # Object class
    "confidence": conf,                   # Detection confidence
    "depth": z,                           # Depth (meters)
    "dimensions_3d": [h, w, l],          # 3D size (meters)
    "rotation_y": theta                   # Yaw angle (radians)
}
```

### Loss Function

Multi-task loss combining 2D detection and 3D estimation:

```
Total Loss = box_loss + cls_loss + dfl_loss + depth_loss + dim_loss + rot_loss
```

- **box_loss**: 2D bounding box regression (IoU loss)
- **cls_loss**: Classification loss (BCE)
- **dfl_loss**: Distribution focal loss
- **depth_loss**: Depth estimation (L1)
- **dim_loss**: 3D dimensions (L1)
- **rot_loss**: Rotation angle (Smooth L1 with sin/cos encoding)

---

## 📁 Project Structure

```
cd YOLOv12-3D/
├── yolo3d.py                          # Main CLI tool
├── scripts/
│   ├── kitti_setup.py                 # Dataset setup CLI
│   └── README.md                      # Setup guide
├── ultralytics/
│   ├── nn/
│   │   ├── modules/head.py            # Detect3D head
│   │   └── tasks.py                   # Detection3DModel
│   ├── models/yolo/detect3d/          # 3D detection module
│   │   ├── train.py                   # Training logic
│   │   ├── val.py                     # Validation logic
│   │   └── predict.py                 # Inference logic
│   ├── data/dataset.py                # KITTIDataset loader
│   ├── utils/loss.py                  # v8Detection3DLoss
│   └── cfg/
│       ├── models/v12/yolov12-3d.yaml # Model config
│       └── datasets/kitti-3d.yaml     # Dataset config
├── CLI_GUIDE.md                       # Complete CLI documentation
├── YOLO3D_QUICKREF.md                # Quick reference
└── README.md                          # This file
```

---

## 📋 KITTI Dataset Format

Each label file contains one object per line:

```
Type Truncated Occluded Alpha Bbox_2D[4] Dimensions_3D[3] Location_3D[3] Rotation_y
```

**Example:**
```
Car 0.00 0 -1.58 587.01 173.33 614.12 200.12 1.65 1.67 3.64 -0.65 1.71 46.70 -1.59
```

**Fields:**
- `Type`: Object class (Car, Van, Truck, Pedestrian, Cyclist, etc.)
- `Bbox_2D`: 2D box [left, top, right, bottom] in pixels
- `Dimensions_3D`: 3D size [height, width, length] in meters
- `Location_3D`: 3D center [x, y, z] in camera coordinates (meters)
- `Rotation_y`: Rotation around Y-axis in radians [-π, π]

**KITTI Classes (8 total):**
```
0: Car              4: Misc
1: Truck            5: Van
2: Pedestrian       6: Tram
3: Cyclist          7: Person_sitting
```

---

## 🎓 Training Tips

### Recommended Settings

```bash
# For NVIDIA RTX 3090 (24GB)
python yolo3d.py train --data kitti-3d.yaml --batch 32 --imgsz 640 -y

# For NVIDIA RTX 4090 (24GB)
python yolo3d.py train --data kitti-3d.yaml --batch 48 --imgsz 640 -y

# For smaller GPUs (8GB-12GB)
python yolo3d.py train --data kitti-3d.yaml --batch 8 --imgsz 512 -y

# Multi-GPU training
python yolo3d.py train --data kitti-3d.yaml --device 0,1,2,3 --batch 64 -y
```

### Hyperparameter Tuning

```bash
# Conservative augmentation (preserves 3D geometry)
python yolo3d.py train \
    --data kitti-3d.yaml \
    --epochs 200 \
    --patience 50 \
    --lr0 0.01 \
    --lrf 0.01 \
    --momentum 0.937 \
    --weight-decay 0.0005 \
    -y

# Tune loss weights if needed
# Edit ultralytics/utils/loss.py:
#   self.depth_weight = 1.0
#   self.dim_weight = 1.0
#   self.rot_weight = 1.0
```

### Transfer Learning

```bash
# Start from pretrained 2D YOLOv12
python yolo3d.py train \
    --model yolov12n.pt \
    --data kitti-3d.yaml \
    --pretrained \
    --epochs 100 \
    -y
```

### Monitoring Training

Results are saved to `runs/detect/<name>/`:
- `weights/best.pt` - Best model checkpoint
- `weights/last.pt` - Latest checkpoint
- `results.png` - Training curves
- `val_batch*.jpg` - Validation visualizations
- `confusion_matrix.png` - Class confusion matrix

---

## 🔍 Advanced Usage

### Custom Dataset

```yaml
# Create custom-3d.yaml
path: /path/to/dataset
train: images/train
val: images/val

names:
  0: Class1
  1: Class2
  # ...

nc: 2
task: detect3d
```

### Modify Architecture

```yaml
# ultralytics/cfg/models/v12/yolov12-custom-3d.yaml
nc: 8
depth_multiple: 0.33  # Adjust model depth
width_multiple: 0.25  # Adjust model width

# Head configuration
head:
  - [[14, 17, 20], 1, Detect3D, [nc]]
```

### Python API

```python
from ultralytics import YOLO

# Load model
model = YOLO('yolov12n-3d.yaml')

# Train
results = model.train(
    data='kitti-3d.yaml',
    epochs=100,
    batch=16,
    imgsz=640,
    device=0,
    name='my_experiment',
    # Additional parameters...
)

# Validate
metrics = model.val()
print(f"mAP50-95: {metrics.box.map:.3f}")

# Predict
results = model.predict('image.jpg', save=True)

# Export
model.export(format='onnx')
```

---

## 🐛 Known Issues & Roadmap

### Current Limitations

- ⚠️ `Boxes3D` results class not yet implemented
- ⚠️ No 3D bounding box visualization
- ⚠️ Camera calibration not integrated in inference
- ⚠️ Depth map refinement not implemented

> **✅ FIXED**: KmAP (KITTI 3D mAP) now displays correctly! See [KMAP_FIX_SUMMARY.md](KMAP_FIX_SUMMARY.md) for details.

### Roadmap

**Short-term:**
- [x] ✅ Add 3D IoU calculation and KmAP evaluation
- [ ] Implement `Boxes3D` class in `engine/results.py`
- [ ] Add 3D box visualization (projected onto image)
- [ ] Integrate camera calibration

**Mid-term:**
- [ ] Bird's eye view visualization
- [ ] Depth map refinement module
- [ ] Orientation binning (like SMOKE)

**Long-term:**
- [ ] Multi-modal fusion (LiDAR + camera)
- [ ] Temporal consistency for video
- [ ] Support for nuScenes and Waymo datasets
- [ ] KITTI 3D benchmark submission

---

## 📚 References

### Key Papers

1. **SMOKE** - Single-Stage Monocular 3D Object Detection (2020)
2. **M3D-RPN** - Monocular 3D Region Proposal Network (2019)
3. **MonoDLE** - Delving into Localization Errors for Monocular 3D Detection (2021)
4. **GUPNet** - Geometry Uncertainty Projection Network (2021)
5. **YOLOv8** - Ultralytics YOLO architecture

### Resources

- **KITTI Dataset**: http://www.cvlibs.net/datasets/kitti/
- **Ultralytics**: https://github.com/ultralytics/ultralytics
- **YOLO-3D Reference**: https://github.com/niconielsen32/YOLO-3D
- **MMDetection3D**: https://github.com/open-mmlab/mmdetection3d

---

## 🤝 Contributing

We welcome contributions! Areas for improvement:

- 🎨 3D visualization utilities
- 📊 Additional 3D metrics (3D IoU, 3D mAP)
- 🗂️ Support for other datasets (nuScenes, Waymo)
- 🔬 Multi-modal fusion approaches
- 🎬 Temporal consistency for video sequences
- 📝 Documentation improvements

**How to contribute:**
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## 📝 License

This project extends [Ultralytics YOLOv12](https://github.com/ultralytics/ultralytics), licensed under **AGPL-3.0**.

See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

**Developed by:**

**AI Research Group**
**Department of Civil Engineering**
**King Mongkut's University of Technology Thonburi (KMUTT)**
Bangkok, Thailand

🌐 Website: [https://www.kmutt.ac.th/](https://www.kmutt.ac.th/)

**Special Thanks:**
- Ultralytics team for the excellent YOLO framework
- KITTI dataset creators for the comprehensive benchmark
- Open-source 3D detection research community
- KMUTT for supporting AI research and development

---

## 📧 Contact

**Academic Collaboration:**
- AI Research Group, Department of Civil Engineering, KMUTT
- Email: [Contact KMUTT Civil Engineering](https://www.kmutt.ac.th/en/contact/)

**Technical Issues:**
- Open an issue on GitHub
- Check [CLI_GUIDE.md](CLI_GUIDE.md) for troubleshooting

---

## 📖 Documentation

- **[CLI_GUIDE.md](CLI_GUIDE.md)** - Complete CLI tool documentation
- **[YOLO3D_QUICKREF.md](YOLO3D_QUICKREF.md)** - Quick reference guide
- **[scripts/README.md](scripts/README.md)** - Dataset setup guide
- **[IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)** - Technical details

---

**Happy 3D Detection! 🚗 📦 🎯**

*Powered by YOLOv12 | Built for KITTI | Made with ❤️ at KMUTT*
