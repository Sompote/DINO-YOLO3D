# YOLOv12-3D with DINO ViT-B Integration

## Overview

This project implements YOLOv12-3D with DINO (Self-supervised Vision Transformer) integration at P0 and P3 levels, providing enhanced 3D object detection capabilities for datasets like KITTI.

## Architecture

### Model Variants

1. **YOLOv12m-3D-DINO** (Medium)
   - Model size: Medium (m)
   - DINO Integration: P0 only OR P0+P3 (configurable)
   - DINO Model: ViT-B (dinov3_vitb16)
   - Total Parameters: ~100M (P0 only) or ~120M (P0+P3) (estimated)
   - Memory: ~3GB (P0 only) or ~4GB (P0+P3) GPU memory

2. **YOLOv12l-3D-DINO** (Large)
   - Model size: Large (l)
   - DINO Integration: P0 only OR P0+P3 (configurable)
   - DINO Model: ViT-B (dinov3_vitb16)
   - Total Parameters: ~160M (P0 only) or ~180M (P0+P3) (estimated)
   - Memory: ~5GB (P0 only) or ~6GB (P0+P3) GPU memory

### DINO Integration Strategy

The implementation offers **two DINO integration modes**:

#### 1. Single-Scale Integration (P0 only) - Lightweight
- **P0 Level Integration**: DINO features are integrated at the early stage (after first convolution)
- Provides high-level semantic understanding from the beginning
- Faster training and inference
- Lower memory requirements
- Best for: Limited computational resources, faster experimentation

#### 2. Dual-Scale Integration (P0+P3) - Maximum Performance
- **P0 Level Integration**: DINO features at early stage for high-level semantic understanding
- **P3 Level Integration**: DINO features at mid-level stage (1/8 resolution) to enhance object detection
- More comprehensive feature enhancement
- Better accuracy for complex scenes
- Best for: Maximum performance, complex datasets, production use

### Model Architecture

```
Input (3, H, W)
    ↓
Conv (64, 3x3) → P0
    ↓
DINO3Backbone (P0) → Enhanced P0 Features
    ↓
Conv (128, 3x3) → P2
    ↓
C3k2 → P3
    ↓
DINO3Backbone (P3) → Enhanced P3 Features
    ↓
C3k2 → P4
    ↓
A2C2f → P4 (4x)
    ↓
Conv (1024, 3x3) → P5
    ↓
A2C2f → P5 (4x)
    ↓
Head with Detect3D
```

## Installation

1. **Install Dependencies**
   ```bash
   pip install ultralytics
   pip install torch torchvision
   pip install transformers
   ```

2. **Clone/Setup Environment**
   - Ensure you're in the YOLOv12-3D directory
   - The DINO3Backbone is already integrated in `ultralytics/nn/modules/block.py`

## Usage

### Training

#### Train YOLOv12m-3D-DINO with Dual Integration (P0+P3) - Maximum Performance
```bash
python yolodio3d.py train \
    --data ultralytics/cfg/datasets/kitti-3d.yaml \
    --yolo-size m \
    --dino-integration dual \
    --epochs 100 \
    --batch-size 16 \
    --imgsz 640
```

#### Train YOLOv12l-3D-DINO with Single Integration (P0 only) - Lightweight
```bash
python yolodio3d.py train \
    --data ultralytics/cfg/datasets/kitti-3d.yaml \
    --yolo-size l \
    --dino-integration single \
    --epochs 100 \
    --batch-size 8 \
    --imgsz 640
```

#### Train YOLOv12m-3D-DINO with Dual Integration (P0+P3)
```bash
python yolodio3d.py train \
    --data ultralytics/cfg/datasets/kitti-3d.yaml \
    --yolo-size m \
    --dino-integration dual \
    --epochs 100 \
    --batch-size 16 \
    --imgsz 640
```

#### Train without DINO (Base YOLOv12-3D)
```bash
python yolodio3d.py train \
    --data ultralytics/cfg/datasets/kitti-3d.yaml \
    --yolo-size m \
    --no-dino \
    --epochs 100
```

#### Available Options:
- `--dino-integration single`: Use DINO at P0 level only (lightweight, faster)
- `--dino-integration dual`: Use DINO at both P0 and P3 levels (maximum performance)
- `--no-dino`: Disable DINO integration (use base YOLOv12-3D)
- `--freeze-dino`: Freeze DINO weights for faster training (enabled by default)

### Validation

```bash
# Validate with dual DINO integration
python yolodio3d.py val \
    --data ultralytics/cfg/datasets/kitti-3d.yaml \
    --yolo-size m \
    --dino-integration dual \
    --model runs/train/exp/weights/best.pt

# Validate with single DINO integration
python yolodio3d.py val \
    --data ultralytics/cfg/datasets/kitti-3d.yaml \
    --yolo-size m \
    --dino-integration single \
    --model runs/train/exp/weights/best.pt
```

### Prediction

```bash
python yolodio3d.py predict \
    --model runs/train/exp/weights/best.pt \
    --source path/to/images/ \
    --conf 0.25 \
    --save
```

### Export

```bash
# Export dual DINO model
python yolodio3d.py export \
    --format onnx \
    --yolo-size m \
    --dino-integration dual \
    --simplify

# Export single DINO model
python yolodio3d.py export \
    --format onnx \
    --yolo-size m \
    --dino-integration single \
    --simplify
```

## Model Configurations

### Model Config Files

#### Single DINO Integration (P0 only) - Lightweight
- `ultralytics/cfg/models/v12/yolov12m-3d-dino-p0.yaml` - Medium model with DINO at P0 only
- `ultralytics/cfg/models/v12/yolov12l-3d-dino-p0.yaml` - Large model with DINO at P0 only

#### Dual DINO Integration (P0+P3) - Maximum Performance
- `ultralytics/cfg/models/v12/yolov12m-3d-dino-p0p3.yaml` - Medium model with DINO at P0 and P3
- `ultralytics/cfg/models/v12/yolov12l-3d-dino-p0p3.yaml` - Large model with DINO at P0 and P3

#### Base Model (no DINO)
- `ultralytics/cfg/models/v12/yolov12-3d.yaml` - Base YOLOv12-3D model

### DINO3Backbone Parameters

The `DINO3Backbone` class accepts the following parameters:

- `model_name`: DINO model variant (default: 'dinov3_vitb16')
- `freeze_backbone`: Whether to freeze DINO weights (default: True)
- `output_channels`: Output channel dimension (default: 512)
- `input_channels`: Input channel dimension (inferred dynamically)

### Supported DINO Variants

- **ViT Models**: dinov3_vits16, dinov3_vitb16, dinov3_vitl16, dinov3_vith16_plus, dinov3_vit7b16
- **ConvNeXt Models**: dinov3_convnext_tiny, dinov3_convnext_small, dinov3_convnext_base, dinov3_convnext_large
- **Satellite Variants**: dinov3_vits16_sat, dinov3_vitb16_sat, dinov3_vitl16_sat, etc.

## DINO Loading Strategy

The DINO3Backbone attempts to load DINO models in the following order:

1. **PyTorch Hub** - Official DINOv3 from facebookresearch/dinov3
2. **Hugging Face** - DINOv3 models from facebook organization
3. **DINOv2 Fallback** - Compatible DINOv2 models based on embedding dimensions
4. **Random Initialization** - Fallback with matching specifications

## Performance Considerations

### Training Time
- Base YOLOv12-3D: 1x (baseline)
- YOLOv12-3D-DINO (P0 only): ~1.5-2x longer training time
- YOLOv12-3D-DINO (P0+P3): ~2-3x longer training time

### Memory Usage
- YOLOv12m-3D-DINO (P0 only): ~3-4GB GPU memory
- YOLOv12m-3D-DINO (P0+P3): ~4-6GB GPU memory
- YOLOv12l-3D-DINO (P0 only): ~5-6GB GPU memory
- YOLOv12l-3D-DINO (P0+P3): ~6-8GB GPU memory

### Recommended Batch Sizes
- YOLOv12m-3D-DINO (P0 only): 16-32 (depending on GPU memory)
- YOLOv12m-3D-DINO (P0+P3): 8-16 (depending on GPU memory)
- YOLOv12l-3D-DINO (P0 only): 8-16 (depending on GPU memory)
- YOLOv12l-3D-DINO (P0+P3): 4-8 (depending on GPU memory)

### Performance vs Efficiency Trade-off

| Configuration | Speed | Memory | Accuracy | Best For |
|---------------|-------|--------|----------|----------|
| Base YOLOv12-3D | Fastest | Lowest | Baseline | Quick prototyping, low-resource environments |
| YOLOv12-3D-DINO (P0 only) | Fast | Low | Good | Balanced performance with limited resources |
| YOLOv12-3D-DINO (P0+P3) | Slower | Higher | Best | Maximum accuracy, complex scenes, production use |

## Key Features

✅ **Flexible DINO Integration**: Choose between single P0 or dual P0+P3 enhancement
✅ **ViT-Backbone Integration**: Using dinov3_vitb16 for optimal performance
✅ **3D Detection**: Support for KITTI and other 3D object detection datasets
✅ **Multiple Model Variants**: Support for M and L model sizes
✅ **Frozen Backbone**: DINO weights can be frozen for faster training
✅ **Multiple Export Formats**: ONNX, TorchScript, OpenVINO, CoreML
✅ **Configurable Performance**: Trade-off between speed and accuracy

## File Structure

```
YOLOv12-3D/
├── yolodio3d.py                    # Main CLI tool
├── ultralytics/
│   ├── nn/
│   │   └── modules/
│   │       ├── block.py            # Contains DINO3Backbone class
│   │       └── __init__.py         # Updated to export DINO3Backbone
│   └── cfg/models/v12/
│       ├── yolov12m-3d-dino-p0.yaml     # Medium - Single P0 integration
│       ├── yolov12m-3d-dino-p0p3.yaml   # Medium - Dual P0+P3 integration
│       ├── yolov12l-3d-dino-p0.yaml     # Large - Single P0 integration
│       ├── yolov12l-3d-dino-p0p3.yaml   # Large - Dual P0+P3 integration
│       └── yolov12-3d.yaml             # Base YOLOv12-3D (no DINO)
└── YOLODIO3D_README.md            # This file
```

## References

- [DINOv3 GitHub Repository](https://github.com/facebookresearch/dinov3)
- [DINOv2 Paper](https://arxiv.org/abs/2204.07110)
- [YOLOv12 Ultralytics](https://github.com/ultralytics/ultralytics)
- [KITTI Dataset](http://www.cvlibs.net/datasets/kitti/)

## License

This project follows the AGPL-3.0 license from Ultralytics.

## Development Notes

### Implementation Details

1. **DINO3Backbone Class**: Integrated from the dino3_yolo_6sep_official project with modifications for YOLOv12-3D
2. **Model Loading**: Supports multiple loading strategies with fallback options
3. **Feature Fusion**: DINO features are fused with CNN features using projection layers
4. **Spatial Adaptation**: Maintains spatial dimensions through patch-to-grid conversion

### Future Improvements

- [ ] Add support for P4 and P5 level DINO integration
- [ ] Implement dynamic DINO model selection
- [ ] Add support for additional 3D datasets
- [ ] Optimize memory usage for larger models
- [ ] Add mixed-precision training support

## Author

**AI Research Group**
Department of Civil Engineering
King Mongkut's University of Technology Thonburi (KMUTT)
