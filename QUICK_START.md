# YOLOv12-3D Quick Start Guide

**Developed by AI Research Group, Department of Civil Engineering**  
**King Mongkut's University of Technology Thonburi (KMUTT)**

## 🚀 5-Minute Setup

### Step 1: Install Dependencies (1 min)
```bash
pip install ultralytics torch torchvision
```

### Step 2: Download KITTI Dataset (manual - ~15 min download)
1. Visit: http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d
2. Download:
   - Left color images (12 GB)
   - Training labels (5 MB)
   - Camera calibration (16 MB)
3. Extract to: `datasets/kitti/`

### Step 3: Train Your First Model (2 min to start)
```python
from ultralytics import YOLO

# Load model
model = YOLO('ultralytics/cfg/models/v12/yolov12-3d.yaml')

# Train (this will take hours on GPU, days on CPU)
model.train(
    data='ultralytics/cfg/datasets/kitti-3d.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    device=0,  # GPU 0
    name='my-first-3d-model'
)
```

### Step 4: Run Inference
```python
# Load trained model
model = YOLO('runs/detect/my-first-3d-model/weights/best.pt')

# Predict
results = model.predict('path/to/image.jpg', save=True)
```

---

## 📋 File Structure Overview

```
yolov12/
├── ultralytics/
│   ├── nn/
│   │   ├── modules/head.py           # Detect3D class ⭐
│   │   └── tasks.py                  # Detection3DModel ⭐
│   ├── models/yolo/
│   │   └── detect3d/                 # Training/val/predict ⭐
│   ├── data/dataset.py               # KITTIDataset ⭐
│   ├── utils/loss.py                 # v8Detection3DLoss ⭐
│   └── cfg/
│       ├── models/v12/yolov12-3d.yaml     # Model config ⭐
│       └── datasets/kitti-3d.yaml         # Dataset config ⭐
├── examples/train_kitti_3d.py        # Training script ⭐
├── YOLO3D_README.md                  # Full documentation ⭐
└── IMPLEMENTATION_SUMMARY.md         # Technical details ⭐

⭐ = New/modified files for 3D detection
```

---

## 💡 Common Commands

### Training
```bash
# From scratch
python examples/train_kitti_3d.py

# Resume training
yolo train resume model=runs/detect/my-model/weights/last.pt

# Custom config
yolo train model=ultralytics/cfg/models/v12/yolov12-3d.yaml \
           data=ultralytics/cfg/datasets/kitti-3d.yaml \
           epochs=100 batch=16 imgsz=640
```

### Validation
```bash
yolo val model=runs/detect/my-model/weights/best.pt \
         data=ultralytics/cfg/datasets/kitti-3d.yaml
```

### Inference
```bash
yolo predict model=best.pt source=path/to/images conf=0.25
```

---

## 🎯 Expected Training Time

| Hardware | YOLOv12n-3D | YOLOv12s-3D | YOLOv12m-3D |
|----------|-------------|-------------|-------------|
| RTX 4090 | ~6 hours | ~8 hours | ~12 hours |
| RTX 3090 | ~8 hours | ~12 hours | ~18 hours |
| RTX 3080 | ~10 hours | ~15 hours | ~24 hours |
| CPU | ~7 days | ~10 days | ~14 days |

*100 epochs on KITTI training set*

---

## 📊 Model Selection Guide

| Use Case | Model | Why |
|----------|-------|-----|
| **Real-time (>30 FPS)** | YOLOv12n-3D | Smallest, fastest |
| **Balanced** | YOLOv12s-3D | Good speed/accuracy |
| **Best accuracy** | YOLOv12m-3D | Highest mAP |
| **Research** | YOLOv12l-3D | Maximum performance |

---

## 🔧 Troubleshooting

### Issue: "No module named 'ultralytics.models.yolo.detect3d'"
**Solution**: Make sure `__init__.py` exists in the detect3d folder

### Issue: "KITTI labels not found"
**Solution**: Check dataset path in `kitti-3d.yaml` points to correct location

### Issue: "CUDA out of memory"
**Solution**: Reduce batch size: `batch=8` or `batch=4`

### Issue: "Loss is NaN"
**Solution**: 
- Lower learning rate: `lr0=0.001`
- Check label format is correct
- Ensure 3D annotations are not corrupted

---

## 📈 Monitoring Training

### TensorBoard
```bash
tensorboard --logdir runs/detect
```

### Key Metrics to Watch:
- **box_loss**: Should decrease steadily (2D boxes)
- **cls_loss**: Should decrease to ~0.5
- **depth_loss**: Should decrease (3D depth)
- **dim_loss**: Should decrease (3D dimensions)
- **rot_loss**: Should decrease (rotation)
- **mAP50**: Should increase to ~70%

---

## 🎓 Next Steps After Training

1. **Evaluate on validation set**:
   ```python
   metrics = model.val()
   print(f"mAP50: {metrics.box.map50}")
   ```

2. **Test on new images**:
   ```python
   results = model.predict('test_image.jpg')
   ```

3. **Export for deployment**:
   ```python
   model.export(format='onnx')  # or 'torchscript', 'tflite'
   ```

4. **Visualize predictions** (requires custom 3D visualization):
   ```python
   # TODO: Implement 3D box visualization
   ```

---

## 📚 Learn More

- **Full Documentation**: See `YOLO3D_README.md`
- **Technical Details**: See `IMPLEMENTATION_SUMMARY.md`
- **Ultralytics Docs**: https://docs.ultralytics.com
- **KITTI Benchmark**: http://www.cvlibs.net/datasets/kitti/

---

## ⚡ Pro Tips

1. **Start small**: Begin with YOLOv12n-3D to iterate quickly
2. **Transfer learning**: Use pretrained 2D weights if available
3. **Conservative augmentation**: Don't use rotation/perspective for 3D
4. **Monitor 3D losses**: All 3D losses should converge
5. **Validate often**: Check predictions on validation images during training

---

## 🆘 Need Help?

1. Check logs in `runs/detect/your-experiment/`
2. Review training curves in TensorBoard
3. Inspect predictions on validation images
4. Read full documentation in `YOLO3D_README.md`
5. Open an issue on GitHub with:
   - Error message
   - Training config
   - System info

---

**Ready to detect in 3D! 🚗📦🎯**
