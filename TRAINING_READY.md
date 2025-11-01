# YOLOv12-3D Training Ready Guide

## ✅ Status: Ready for Training

All fixes have been applied and the system is ready to train on KITTI 3D Object Detection dataset.

## 🔧 What Was Fixed

### 1. Import and Module Issues
- ✅ Added `Detect3D` to `ultralytics.nn.modules` exports
- ✅ Fixed `Detect3D` handling in `parse_model` function
- ✅ Fixed forward pass handling in model initialization
- ✅ Fixed prediction unpacking in `v8Detection3DLoss`

### 2. Dataset Loading
- ✅ Created `_get_kitti_label_files()` to convert image_2 → label_2 paths
- ✅ Override `get_labels()` to use KITTI structure
- ✅ Updated YAML to point to `training/image_2`
- ✅ Dataset reads KITTI format natively

### 3. User Experience
- ✅ Removed training confirmation prompt
- ✅ Improved error messages
- ✅ Added comprehensive documentation

## 📋 Current Setup

### Model Architecture
- **Base**: YOLOv12 with Detect3D head
- **Outputs**: 
  - 2D bounding box [x, y, w, h]
  - Object class
  - Depth (z) [0-100m]
  - 3D dimensions [h, w, l] [0-10m each]
  - Rotation Y [-π, π]
- **Parameters**: 2.59M (nano model)
- **GFLOPs**: 6.2

### Dataset Format
- **Source**: KITTI 3D Object Detection
- **Images**: 7,481 training images
- **Classes**: 8 (Car, Truck, Pedestrian, Cyclist, etc.)
- **Annotations**: Native KITTI format
- **Location**: `training/image_2/` → `training/label_2/`

## 🚀 Training Instructions

### On Mac (CPU - Testing Only)

**Note**: Training on CPU is VERY slow. Use only for testing.

```bash
# Delete old caches first
find /Users/sompoteyouwai/Downloads/datakitti/datasets/kitti -name "*.cache" -delete

# Run training
python yolo3d.py train \
    --model s \
    --data ultralytics/cfg/datasets/kitti-3d.yaml \
    --epochs 100 \
    --batch 2 \
    --imgsz 640 \
    --device cpu \
    --name yolov12s-kitti
```

### On Cloud Server (GPU - Recommended)

```bash
cd /workspace/yolor2/YOLOv12-3D
git pull

# Update dataset path in YAML
# Edit ultralytics/cfg/datasets/kitti-3d.yaml
# Change 'path:' to your KITTI location

# Delete any old caches
find /path/to/kitti -name "*.cache" -delete

# Run training
python yolo3d.py train \
    --model s \
    --data kitti-3d.yaml \
    --epochs 100 \
    --batch 16 \
    --imgsz 640 \
    --device 0 \
    --name yolov12s-kitti
```

## ⚠️ Important Notes

### First-Time Cache Creation

The first time you run training, the dataset will:
1. Scan all 7,481 images
2. Parse all 7,481 KITTI label files
3. Create a cache file in `training/label_2/` directory

This can take **2-5 minutes** on the first run. Be patient!

**Expected output**:
```
train: Scanning /path/to/kitti/training/label_2... 
7481 images, 0 backgrounds, 0 corrupt: 100%|██████████| 7481/7481
```

**Good sign**: `7481 images` (not "0 images")
**Bad sign**: `0 images, 7481 backgrounds` = labels not found

### Cache Issues

If you see "0 images, 7481 backgrounds":
1. Delete all `.cache` files: `find /path/to/kitti -name "*.cache" -delete`
2. Verify label files exist: `ls /path/to/kitti/training/label_2/*.txt | wc -l` (should show 7481)
3. Check YAML path is correct
4. Run training again

### Training on CPU vs GPU

**CPU (Mac)**:
- Batch size: 1-2
- Very slow (hours per epoch)
- Only for testing setup

**GPU (Cloud)**:
- Batch size: 16-32 (depends on GPU memory)
- Fast (minutes per epoch)
- For actual training

## 📊 Expected Training Behavior

### Initialization
```
YOLOv12-3d summary: 522 layers, 2,593,095 parameters
Freezing layer 'model.21.dfl.conv.weight'
train: Scanning .../training/label_2... 7481 images, 0 backgrounds
```

### Training Loop
```
Epoch  GPU_mem  box_loss  cls_loss  dfl_loss  depth_loss  dim_loss  rot_loss  Instances  Size
  0/100    0.0G    1.234     0.567     0.890       0.234      0.123     0.456      28.5    640
```

**Loss Components**:
- `box_loss`: 2D bounding box regression
- `cls_loss`: Classification
- `dfl_loss`: Distribution focal loss
- `depth_loss`: Depth prediction (NEW for 3D)
- `dim_loss`: 3D dimensions (NEW for 3D)
- `rot_loss`: Rotation angle (NEW for 3D)

## 🎯 Training Tips

### Recommended Settings

**Quick Test** (verify setup works):
```bash
python yolo3d.py train \
    --model s \
    --data kitti-3d.yaml \
    --epochs 1 \
    --batch 4 \
    --imgsz 640 \
    --device 0 \
    --name test-run
```

**Full Training**:
```bash
python yolo3d.py train \
    --model s \
    --data kitti-3d.yaml \
    --epochs 100 \
    --batch 16 \
    --imgsz 640 \
    --device 0 \
    --patience 50 \
    --name yolov12s-kitti-full
```

### Model Sizes

| Model | Params | Speed | Use Case |
|-------|--------|-------|----------|
| n | 2.6M | ⚡⚡⚡⚡⚡ | Testing, edge devices |
| s | 9.1M | ⚡⚡⚡⚡ | Balanced (recommended) |
| m | 19.7M | ⚡⚡⚡ | Better accuracy |
| l | 26.5M | ⚡⚡ | High accuracy |
| x | 59.4M | ⚡ | Maximum accuracy |

### Hardware Requirements

**Minimum**:
- GPU: 6GB VRAM
- Batch size: 8
- Image size: 640

**Recommended**:
- GPU: 12GB+ VRAM
- Batch size: 16-32
- Image size: 640

**High Performance**:
- GPU: 24GB+ VRAM
- Batch size: 32-48
- Image size: 640

## 📁 Output Structure

After training, results will be in:
```
runs/detect/yolov12s-kitti/
├── weights/
│   ├── best.pt          # Best model checkpoint
│   └── last.pt          # Last epoch checkpoint
├── results.csv          # Training metrics
├── results.png          # Training curves
├── confusion_matrix.png # Confusion matrix
├── val_batch*.jpg       # Validation visualizations
└── args.yaml            # Training arguments
```

## 🔍 Monitoring Training

### TensorBoard
```bash
tensorboard --logdir runs/detect/yolov12s-kitti
# Open http://localhost:6006
```

### Training Metrics
- **box_loss, cls_loss, dfl_loss**: Should decrease steadily
- **depth_loss, dim_loss, rot_loss**: Should decrease (3D metrics)
- **mAP50, mAP50-95**: Should increase (validation)

### Good Training Signs
✅ Losses decreasing
✅ mAP increasing
✅ Validation not diverging from training
✅ No NaN values

### Bad Training Signs
❌ Losses stuck or increasing
❌ NaN values appear
❌ Validation much worse than training (overfitting)
❌ Out of memory errors

## 🐛 Troubleshooting

### "No labels found"
**Cause**: Cache created before fixes were applied
**Solution**: Delete cache and retry
```bash
find /path/to/kitti -name "*.cache" -delete
```

### "shape '[5, 72, -1]' is invalid"
**Cause**: Old cache with no labels
**Solution**: Same as above - delete cache

### "Out of memory"
**Cause**: Batch size too large for GPU
**Solution**: Reduce batch size
```bash
--batch 8  # instead of 16
--batch 4  # if still fails
```

### "CUDA initialization failed"
**Cause**: GPU not available or driver issue
**Solution**: Use CPU temporarily
```bash
--device cpu
```

## 📚 Additional Resources

- **3D Predictions**: See `3D_PREDICTIONS_VERIFICATION.md`
- **Download Guide**: See `KITTI_DOWNLOAD_GUIDE.md`
- **CLI Reference**: See `CLI_GUIDE.md`
- **Quick Start**: See `QUICK_START.md`

## ✅ Verification Checklist

Before starting full training, verify:

- [ ] Git repo is up to date (`git pull`)
- [ ] KITTI dataset downloaded and extracted
- [ ] Dataset path in YAML is correct
- [ ] Old cache files deleted
- [ ] GPU is available (`nvidia-smi`)
- [ ] Test run completes without errors

## 🎉 You're Ready!

Everything is set up and ready to train YOLOv12-3D on KITTI dataset!

**Next step**: Run training on your cloud server with GPU.

---

**Last Updated**: 2025-11-01  
**Status**: ✅ All systems ready  
**Tested On**: macOS (CPU), Linux (GPU pending)
