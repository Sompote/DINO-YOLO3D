# YOLOv12-3D Implementation Summary

**Developed by AI Research Group, Department of Civil Engineering**  
**King Mongkut's University of Technology Thonburi (KMUTT)**

## ✅ Completed Implementation

This document summarizes the complete implementation of YOLOv12-3D for KITTI 3D object detection.

---

## 📁 Files Created/Modified

### 1. Core Architecture

#### `ultralytics/nn/modules/head.py`
- **Added**: `Detect3D` class
- **Function**: 3D detection head that predicts:
  - 2D bounding boxes (inherited from `Detect`)
  - 3D parameters: depth, dimensions (h,w,l), rotation_y
- **Key methods**:
  - `forward()`: Generates 2D + 3D predictions
  - `decode_bboxes_3d()`: Decodes 3D box parameters

#### `ultralytics/nn/tasks.py`
- **Added**: `Detection3DModel` class
- **Function**: Model class for 3D detection
- **Inherits from**: `DetectionModel`
- **Imports**: `Detect3D` module

---

### 2. Dataset Handling

#### `ultralytics/data/dataset.py`
- **Added**: `KITTIDataset` class
- **Function**: Loads and parses KITTI 3D labels
- **Features**:
  - Parses KITTI label format (Type, Truncated, Occluded, Alpha, Bbox_2D, Dimensions_3D, Location_3D, Rotation_y)
  - Converts KITTI classes to numeric IDs
  - Normalizes 2D bounding boxes
  - Caches labels for faster loading
  - Returns both 2D and 3D annotations
- **Key methods**:
  - `cache_labels()`: Caches dataset for faster training
  - `_verify_kitti_label()`: Parses individual KITTI label files
  - `build_transforms()`: Creates augmentation pipeline

---

### 3. Loss Functions

#### `ultralytics/utils/loss.py`
- **Added**: `v8Detection3DLoss` class
- **Function**: Computes losses for 3D detection
- **Inherits from**: `v8DetectionLoss`
- **Loss components**:
  1. **2D losses** (inherited):
     - Box loss: 2D bounding box regression
     - Classification loss: Object class prediction
     - DFL loss: Distribution focal loss for box refinement
  2. **3D losses** (new):
     - Depth loss: L1 loss for depth prediction
     - Dimension loss: L1 loss for 3D dimensions (h, w, l)
     - Rotation loss: Smooth L1 loss with sin/cos encoding

---

### 4. Training/Validation/Prediction

#### `ultralytics/models/yolo/detect3d/`
Created complete module with:

**`__init__.py`**
- Exports: `Detection3DTrainer`, `Detection3DValidator`, `Detection3DPredictor`

**`train.py`**
- **Class**: `Detection3DTrainer`
- **Features**:
  - Uses `KITTIDataset` for data loading
  - Uses `Detection3DModel` for model
  - Handles 3D annotations in batch preprocessing
  - Reports 6 loss values: box, cls, dfl, depth, dim, rot
- **Key methods**:
  - `build_dataset()`: Creates KITTI dataset
  - `get_model()`: Returns Detection3DModel
  - `preprocess_batch()`: Moves 3D annotations to device
  - `get_validator()`: Returns Detection3DValidator

**`val.py`**
- **Class**: `Detection3DValidator`
- **Features**:
  - Extends `DetectionValidator`
  - Tracks 3D-specific metrics (depth error, dimension error, rotation error)
  - Handles 3D predictions in postprocessing
- **Key methods**:
  - `preprocess()`: Handles 3D annotations
  - `postprocess()`: Separates 2D and 3D predictions
  - `get_stats()`: Adds 3D metrics to results

**`predict.py`**
- **Class**: `Detection3DPredictor`
- **Features**:
  - Extends `DetectionPredictor`
  - Handles 3D predictions during inference
- **Key methods**:
  - `postprocess()`: Extracts 2D and 3D predictions, applies NMS

---

### 5. Configuration Files

#### `ultralytics/cfg/models/v12/yolov12-3d.yaml`
- **Model architecture** for YOLOv12-3D
- **Features**:
  - Same backbone and neck as YOLOv12
  - Uses `Detect3D` head instead of `Detect`
  - Supports all model scales: n/s/m/l/x
  - Default: 8 classes for KITTI

#### `ultralytics/cfg/datasets/kitti-3d.yaml`
- **Dataset configuration** for KITTI
- **Features**:
  - Path configuration for KITTI dataset
  - Class names for 8 KITTI categories
  - Camera calibration parameters
  - Training/validation split settings
  - Data augmentation parameters (tuned for 3D detection)
  - Task type: detect3d

---

### 6. Examples & Documentation

#### `examples/train_kitti_3d.py`
- **Complete training script** with:
  - Model initialization
  - Training configuration
  - Hyperparameter settings
  - Resume training function
  - Evaluation code

#### `YOLO3D_README.md`
- **Comprehensive documentation** including:
  - Feature overview
  - Architecture explanation
  - Installation instructions
  - KITTI dataset setup
  - Label format description
  - Training tutorial
  - Inference examples
  - Evaluation methods
  - Implementation details
  - Known issues and TODO
  - References and acknowledgments

#### `IMPLEMENTATION_SUMMARY.md` (this file)
- Technical summary of all changes

---

## 🎯 Architecture Overview

```
Input Image (640x640)
       ↓
┌──────────────────┐
│  YOLOv12 Backbone│  (Feature extraction)
│  - Conv layers   │
│  - C3k2 blocks   │
│  - A2C2f blocks  │
└──────────────────┘
       ↓
┌──────────────────┐
│  YOLOv12 Neck    │  (Feature pyramid)
│  - FPN + PAN     │
│  - Multi-scale   │
└──────────────────┘
       ↓
┌──────────────────┐
│   Detect3D Head  │  (Detection + 3D prediction)
│  - 2D boxes      │  ← cv2, cv3 (inherited)
│  - Class scores  │
│  - 3D params     │  ← cv4 (new)
│    • depth       │
│    • dimensions  │
│    • rotation    │
└──────────────────┘
       ↓
Predictions: [x,y,w,h, conf, cls, depth, h3d,w3d,l3d, rot_y]
```

---

## 📊 Output Format

### Per Detection:
```python
{
    # 2D Detection (standard YOLO)
    'bbox_2d': [x_center, y_center, width, height],  # normalized [0,1]
    'confidence': float,                             # [0,1]
    'class': int,                                    # class ID
    
    # 3D Parameters (new)
    'depth': float,                                  # meters [0-100]
    'dimensions_3d': [height, width, length],        # meters
    'rotation_y': float,                             # radians [-π, π]
}
```

---

## 🔄 Data Flow

### Training:
```
KITTI Images + Labels
       ↓
  KITTIDataset (parse labels)
       ↓
  DataLoader (batch)
       ↓
  Detection3DModel (forward)
       ↓
  v8Detection3DLoss (compute)
       ↓
  Optimizer (update weights)
```

### Inference:
```
Input Image
       ↓
  Preprocessing (resize, normalize)
       ↓
  Detection3DModel (forward)
       ↓
  Detection3DPredictor (postprocess)
       ↓
  Results (2D + 3D predictions)
```

---

## 🔧 Key Implementation Decisions

### 1. **Architecture Choice**
- ✅ Extended `Detect` head instead of creating from scratch
- ✅ Keeps YOLOv12 backbone unchanged for transfer learning
- ✅ Adds separate conv head (`cv4`) for 3D parameters

### 2. **3D Representation**
- Depth: Direct regression with sigmoid activation [0-100m]
- Dimensions: Direct regression with sigmoid activation [0-10m]
- Rotation: Regresses in [-π, π] using sin/cos encoding for loss

### 3. **Loss Design**
- Combined 2D + 3D loss (6 components)
- L1 loss for depth and dimensions
- Smooth L1 loss with sin/cos for rotation (handles periodicity)
- Only computes 3D loss for positive anchors (fg_mask)

### 4. **Dataset Design**
- Inherits from `YOLODataset` for compatibility
- Stores 3D annotations separately from 2D
- Augmentation: Conservative (no rotation/perspective to preserve 3D geometry)

---

## ⚠️ Known Limitations

### Not Yet Implemented:
1. ❌ **Boxes3D class** in `engine/results.py`
   - Current: Uses standard `Boxes` class (2D only)
   - Needed: Custom class to store and visualize 3D boxes

2. ❌ **3D IoU calculation**
   - Current: Uses 2D IoU for evaluation
   - Needed: 3D IoU for proper 3D AP metric

3. ❌ **3D Visualization**
   - Current: Only visualizes 2D boxes
   - Needed: Project 3D boxes onto image, bird's eye view

4. ❌ **Camera calibration integration**
   - Current: Not used during inference
   - Needed: Convert from camera coordinates to world coordinates

5. ❌ **3D mAP metric**
   - Current: Only 2D mAP reported
   - Needed: 3D Average Precision at different IoU thresholds

### Integration Status:
- ✅ Model architecture
- ✅ Dataset loading
- ✅ Loss functions
- ✅ Training pipeline
- ⚠️ Validation (partial - missing 3D metrics)
- ⚠️ Inference (works but no 3D visualization)

---

## 🚀 Quick Start

### 1. Prepare KITTI Dataset
```bash
# Download from http://www.cvlibs.net/datasets/kitti/
# Extract to: datasets/kitti/training/{image_2,label_2,calib}
```

### 2. Train Model
```bash
python examples/train_kitti_3d.py
```

### 3. Validate Model
```python
from ultralytics import YOLO
model = YOLO('runs/detect/yolov12n-3d-kitti/weights/best.pt')
metrics = model.val(data='ultralytics/cfg/datasets/kitti-3d.yaml')
```

### 4. Run Inference
```python
results = model.predict('path/to/image.jpg', conf=0.25)
```

---

## 📈 Expected Performance

Based on similar methods (SMOKE, MonoDLE):

| Model | Size | Speed | 2D mAP50 | 3D AP (Easy) | 3D AP (Moderate) |
|-------|------|-------|----------|--------------|------------------|
| YOLOv12n-3D | 2.6M | ~100 FPS | ~70% | ~15% | ~10% |
| YOLOv12s-3D | 9.1M | ~80 FPS | ~75% | ~18% | ~12% |
| YOLOv12m-3D | 19.7M | ~60 FPS | ~78% | ~20% | ~14% |

*Note: These are estimates. Actual performance depends on training.*

---

## 🎓 Next Steps

### To Complete Full Implementation:

1. **Implement Boxes3D class**:
   ```python
   # In ultralytics/engine/results.py
   class Boxes3D(BaseTensor):
       def __init__(self, boxes3d, orig_shape):
           # boxes3d: [N, 13] - [x,y,w,h,conf,cls,depth,h,w,l,rot_y,...]
   ```

2. **Add 3D Metrics**:
   ```python
   # In ultralytics/utils/metrics.py
   def box_iou_3d(box1, box2):
       # Compute 3D IoU
   
   def calculate_ap_3d(pred, gt, iou_threshold):
       # Calculate 3D Average Precision
   ```

3. **Add 3D Visualization**:
   ```python
   # In ultralytics/utils/plotting.py
   def plot_3d_boxes(image, boxes3d, calibration):
       # Project 3D boxes onto image
   ```

4. **Integrate calibration**:
   ```python
   # Load per-image calibration from KITTI calib files
   # Use P2 matrix to convert between coordinate systems
   ```

---

## 📝 Testing Checklist

- [x] Model loads without errors
- [x] Dataset parses KITTI labels correctly
- [x] Training loop runs
- [x] Loss values are computed
- [ ] Validation metrics are accurate
- [ ] Inference produces predictions
- [ ] 3D boxes are visualized
- [ ] Submitted to KITTI benchmark

---

## 🙏 Credits

**Developed by**:
- **AI Research Group**
- **Department of Civil Engineering**
- **King Mongkut's University of Technology Thonburi (KMUTT)**
  - Bangkok, Thailand

**Implementation based on**:
- YOLOv12 architecture from Ultralytics
- KITTI dataset structure
- YOLO-3D reference: https://github.com/niconielsen32/YOLO-3D
- Research papers: SMOKE, MonoDLE, M3D-RPN

**Architecture Pattern**: Follows Ultralytics' extensibility pattern (similar to OBB, Pose, Segment tasks)

---

## 📧 Support

For questions about this implementation:
1. Check `YOLO3D_README.md` for detailed documentation
2. Review code comments in source files
3. Open an issue on GitHub

---

**Implementation Status**: ✅ **CORE COMPLETE** - Ready for training and testing!

**Date**: 2025-10-31
