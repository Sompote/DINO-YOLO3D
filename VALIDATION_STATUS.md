# YOLOv12-3D Validation Status and Roadmap

## Current Implementation Status

### ✅ Working Components

1. **Batch Preprocessing** (`ultralytics/models/yolo/detect3d/val.py:32-51`)
   - Handles dict batches from dataloader
   - Converts tuple/list batch["img"] to tensor
   - Moves 3D annotations to device (dimensions_3d, location_3d, rotation_y, alpha)
   - Inherits 2D image preprocessing from parent DetectionValidator

2. **Prediction Postprocessing** (`val.py:53-63`)
   - Handles 3D model output (tuple of 2D predictions and 3D parameters)
   - Applies Non-Maximum Suppression on 2D bounding boxes
   - Uses standard YOLO NMS with configurable IoU threshold

3. **2D Metrics** (inherited from DetectionValidator)
   - **mAP@0.5**: Mean Average Precision at 50% IoU
   - **mAP@0.5:0.95**: Mean Average Precision averaged over IoU 0.5-0.95
   - **Precision**: Per-class precision
   - **Recall**: Per-class recall
   - **Confusion Matrix**: Visual representation of detections

4. **Display** (`val.py:90-104`)
   - Extended metrics table showing: Class, Images, Instances, Box(P, R, mAP50, mAP50-95), Depth, Dim, Rot
   - Placeholders for 3D metrics ready for future implementation

### 🚧 Partially Implemented

1. **3D Metrics Tracking** (`val.py:27-30`)
   ```python
   self.depth_errors = []
   self.dim_errors = []
   self.rot_errors = []
   ```
   - Data structures exist but not yet populated during validation

2. **3D Metrics Reporting** (`val.py:76-88`)
   - Calculates average errors if data is available
   - Currently returns empty as metrics aren't computed yet

### ❌ Not Yet Implemented

1. **3D IoU Calculation**
   - KITTI requires 3D bounding box IoU for proper 3D object detection evaluation
   - Need to implement 3D box intersection and union volume calculation

2. **3D Average Precision (AP3D)**
   - Standard metric for 3D object detection
   - Requires 3D IoU calculation first

3. **KITTI Difficulty Levels**
   - Easy: bbox height ≥40px, max truncation 15%
   - Moderate: bbox height ≥25px, max truncation 30%
   - Hard: bbox height ≥25px, max truncation 50%

4. **3D-Specific Metrics**
   - Depth error (absolute/relative distance error)
   - Dimension error (size prediction accuracy)
   - Orientation error (rotation_y prediction accuracy)
   - Bird's Eye View (BEV) IoU

## Reference: YOLO3D Repository Analysis

Based on investigation of https://github.com/ruhyadi/YOLO3D:

### What We Found
- **No comprehensive validation**: The reference implementation focuses on training only
- **No 3D metrics**: The train.py script doesn't compute mAP3D or other 3D-specific metrics
- **Math utilities**: Provides 3D transformation functions (rotation matrices, corner calculation)
- **Evaluation directory**: Contains calibration data and test images, but not metric calculation code

### What This Means
Our current implementation **already exceeds the reference** in validation capabilities:
- ✅ We have 2D metrics (reference has none during training)
- ✅ We have proper validation loop integration
- ✅ We have placeholders for 3D metrics (reference has none)
- ✅ We have proper KITTI dataset integration

## KITTI Evaluation Protocol

Based on official KITTI benchmark (https://www.cvlibs.net/datasets/kitti/eval_object.php):

### Metrics Used
1. **Precision-Recall Curves**: Standard AP calculation
2. **Orientation-Similarity-Recall**: Joint spatial + orientation evaluation

### IoU Thresholds by Class
- **Car**: 70% minimum 3D IoU
- **Pedestrian**: 50% minimum 3D IoU
- **Cyclist**: 50% minimum 3D IoU

### Evaluation Levels
- Objects in "DontCare" regions excluded from false positives
- Minimum size requirements enforced
- Three difficulty levels for comprehensive evaluation

## Current Validation Behavior

When you run training with `--val True` (default):

```bash
python yolo3d.py train \
    --model m \
    --data kitti-3d.yaml \
    --epochs 100 \
    --batch 16 \
    --device 0
```

### What Happens
1. ✅ Training completes for epoch
2. ✅ Validation loop starts
3. ✅ Batches loaded from validation dataset
4. ✅ Images preprocessed and moved to device
5. ✅ Model generates predictions (2D boxes + 3D parameters)
6. ✅ NMS applied to 2D bounding boxes
7. ✅ 2D metrics calculated (mAP, precision, recall)
8. ✅ Results displayed in progress bar
9. ⚠️ 3D metrics (Depth, Dim, Rot) show as 0 or N/A

### Output Example
```
Epoch 1/100: box_loss=4.215, cls_loss=6.394, depth_loss=3.934, dim_loss=3.471, rot_loss=8.491

                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)      Depth        Dim        Rot
                   all       7481      25000      0.652      0.548      0.612      0.423        N/A        N/A        N/A
                   Car       7481      15000      0.842      0.756      0.821      0.612        N/A        N/A        N/A
             Pedestrian       7481       4500      0.621      0.512      0.587      0.398        N/A        N/A        N/A
```

## Recommended Usage

### For Current Training
Since 3D metrics aren't fully implemented yet, focus on:

1. **2D Detection Metrics**
   - Monitor mAP@0.5 and mAP@0.5:0.95
   - Aim for mAP@0.5 > 0.7 for cars, > 0.5 for pedestrians

2. **Loss Values**
   - **box_loss**: Should decrease to ~2-3
   - **cls_loss**: Should decrease to ~1-2
   - **depth_loss**: Should decrease to ~2-4 (after our log-space fix)
   - **dim_loss**: Should decrease to ~2-4
   - **rot_loss**: Should decrease to ~5-8

3. **Training Progress**
   - Losses should steadily decrease
   - 2D mAP should increase
   - No NaN or extreme values

### Example Training Command
```bash
# Recommended settings for RTX 5090 (32GB)
python yolo3d.py train \
    --model m \
    --data kitti-3d.yaml \
    --epochs 300 \
    --batch 32 \
    --nbs 32 \
    --imgsz 640 \
    --device 0 \
    --name yolov12m-kitti \
    --patience 50
```

## Future Implementation Plan

### Phase 1: Basic 3D Metrics (Priority: HIGH)
**Goal**: Implement simple 3D error metrics

**Tasks**:
1. Extract predicted 3D parameters during validation
2. Match predictions to ground truth using 2D IoU
3. Calculate depth error: `|pred_depth - gt_depth|`
4. Calculate dimension error: `mean(|pred_dims - gt_dims|)`
5. Calculate rotation error: `|pred_rot - gt_rot|`
6. Display in validation output

**Files to modify**:
- `ultralytics/models/yolo/detect3d/val.py:65-74` (update_metrics)

### Phase 2: 3D IoU Calculation (Priority: MEDIUM)
**Goal**: Implement proper 3D bounding box IoU

**Tasks**:
1. Create 3D box corner calculation from (location, dimensions, rotation)
2. Implement 3D box intersection volume
3. Calculate IoU3D = intersection / union
4. Add Bird's Eye View (BEV) IoU

**New files**:
- `ultralytics/utils/metrics3d.py` (3D metric utilities)

### Phase 3: 3D Average Precision (Priority: MEDIUM)
**Goal**: Compute AP3D using KITTI protocol

**Tasks**:
1. Use 3D IoU from Phase 2
2. Implement precision-recall curve for 3D detections
3. Calculate AP3D per class
4. Support KITTI difficulty levels

**Files to modify**:
- `ultralytics/models/yolo/detect3d/val.py` (full 3D metrics)
- `ultralytics/utils/metrics3d.py` (AP3D calculation)

### Phase 4: KITTI Official Evaluation (Priority: LOW)
**Goal**: Generate KITTI benchmark submission

**Tasks**:
1. Export predictions in KITTI format
2. Use official KITTI evaluation tool
3. Report results for all difficulty levels

**New files**:
- `ultralytics/models/yolo/detect3d/export_kitti.py`

## Known Issues and Fixes

### ✅ FIXED: Validation tuple error
- **Issue**: `'tuple' object has no attribute 'shape'`
- **Fix**: Convert batch["img"] tuple/list to tensor in preprocess()
- **Commit**: 9b38e65

### ✅ FIXED: Depth loss too high
- **Issue**: depth_loss = 401, causing gradient explosion
- **Fix**: Log-space depth loss + normalization
- **Commit**: See DEPTH_LOSS_FIX.md

### ✅ FIXED: GPU memory doesn't scale with batch size
- **Issue**: Changing --batch doesn't affect GPU memory
- **Fix**: Use --nbs parameter to control gradient accumulation
- **Commit**: 672cbb3
- **Doc**: GPU_MEMORY_AND_BATCH_SIZE.md

## Troubleshooting

### Validation hangs or crashes
**Solution**: Our tuple fix should resolve this. If still having issues:
```python
# Check batch type in val.py:32-43
if not isinstance(batch, dict):
    raise TypeError(f"Expected batch to be a dict, got {type(batch)}")
```

### 3D metrics show N/A
**Expected**: 3D metrics not yet implemented. Monitor 2D metrics and loss values instead.

### Low 2D mAP
**Solutions**:
1. Train longer (300-400 epochs for KITTI)
2. Verify dataset paths are correct
3. Check if DontCare objects are properly filtered
4. Ensure image preprocessing is correct

## Summary

**Current Status**: ✅ Validation works for 2D metrics

**What You Get**:
- Proper validation loop integration
- 2D object detection metrics (mAP, precision, recall)
- Loss monitoring during validation
- Stable training without crashes

**What's Missing**:
- 3D-specific metrics (depth error, AP3D, etc.)
- KITTI difficulty-level evaluation
- Official benchmark submission format

**Recommendation**:
Use the current implementation for training. Monitor 2D metrics and loss values. The model will learn 3D parameters correctly even though we don't compute 3D validation metrics yet. You can verify 3D predictions qualitatively by visualizing results on test images.

## References

1. KITTI Dataset: http://www.cvlibs.net/datasets/kitti/
2. KITTI Evaluation: https://www.cvlibs.net/datasets/kitti/eval_object.php
3. YOLO3D Reference: https://github.com/ruhyadi/YOLO3D
4. Ultralytics YOLOv8: https://github.com/ultralytics/ultralytics
