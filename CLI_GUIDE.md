# YOLOv12-3D CLI Guide

**AI Research Group, Department of Civil Engineering, KMUTT**

Complete command-line interface for training, validation, and inference with YOLOv12-3D.

---

## 🚀 Quick Start

```bash
# Train a model
python yolo3d.py train --data ultralytics/cfg/datasets/kitti-3d.yaml --epochs 100

# Validate model
python yolo3d.py val --model runs/detect/train/weights/best.pt --data kitti-3d.yaml

# Run inference
python yolo3d.py predict --model best.pt --source images/ --conf 0.25

# Export model
python yolo3d.py export --model best.pt --format onnx
```

---

## 📋 CLI Commands

### Overview

```
python yolo3d.py <command> [options]

Commands:
  train       Train YOLOv12-3D model
  val         Validate trained model
  predict     Run inference on images/videos
  export      Export model to different formats
```

---

## 🎓 Training

### Basic Training

```bash
python yolo3d.py train \
  --data ultralytics/cfg/datasets/kitti-3d.yaml \
  --epochs 100 \
  --batch 16 \
  --name my-model
```

### Training with Custom Model Size

```bash
# Use model size shortcuts
python yolo3d.py train --model n --data kitti-3d.yaml  # nano
python yolo3d.py train --model s --data kitti-3d.yaml  # small
python yolo3d.py train --model m --data kitti-3d.yaml  # medium
python yolo3d.py train --model l --data kitti-3d.yaml  # large
python yolo3d.py train --model x --data kitti-3d.yaml  # xlarge
```

### Training from Pretrained Weights

```bash
# Resume from checkpoint
python yolo3d.py train \
  --model runs/detect/my-model/weights/last.pt \
  --data kitti-3d.yaml \
  --epochs 100
```

### Advanced Training

```bash
python yolo3d.py train \
  --model m \
  --data kitti-3d.yaml \
  --epochs 200 \
  --batch 32 \
  --imgsz 640 \
  --device 0,1 \
  --workers 16 \
  --optimizer AdamW \
  --lr0 0.001 \
  --name kitti-medium \
  --patience 100 \
  --verbose \
  -y
```

### Training Options

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | `n` | Model size (n/s/m/l/x) or path to .yaml/.pt |
| `--data` | *required* | Dataset config (.yaml) |
| `--epochs` | `100` | Training epochs |
| `--batch` | `16` | Batch size |
| `--imgsz` | `640` | Input image size |
| `--device` | `0` | GPU device (0, 1, cpu, 0,1) |
| `--workers` | `8` | DataLoader workers |
| `--name` | `train` | Experiment name |
| `--optimizer` | `SGD` | Optimizer (SGD/Adam/AdamW) |
| `--lr0` | `0.01` | Initial learning rate |
| `--lrf` | `0.01` | Final LR factor |
| `--momentum` | `0.937` | SGD momentum |
| `--weight-decay` | `0.0005` | Weight decay |
| `--warmup-epochs` | `3.0` | Warmup epochs |
| `--box` | `7.5` | Box loss weight |
| `--cls` | `0.5` | Classification loss weight |
| `--dfl` | `1.5` | DFL loss weight |
| `--patience` | `50` | Early stopping patience |
| `--seed` | `0` | Random seed |
| `--pretrained` | `False` | Use pretrained weights |
| `--verbose` | `False` | Verbose output |
| `-y, --yes` | `False` | Skip confirmation |

---

## 🔍 Validation

### Basic Validation

```bash
python yolo3d.py val \
  --model runs/detect/train/weights/best.pt \
  --data ultralytics/cfg/datasets/kitti-3d.yaml
```

### Validation with Custom Settings

```bash
python yolo3d.py val \
  --model best.pt \
  --data kitti-3d.yaml \
  --batch 32 \
  --imgsz 640 \
  --device 0 \
  --plots \
  --save-json
```

### Validation Options

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | *required* | Model weights (.pt) |
| `--data` | *required* | Dataset config (.yaml) |
| `--batch` | `16` | Batch size |
| `--imgsz` | `640` | Image size |
| `--device` | `0` | Device |
| `--plots` | `False` | Save validation plots |
| `--save-json` | `False` | Save results to JSON |
| `--verbose` | `False` | Verbose output |

---

## 📸 Prediction (Inference)

### Basic Prediction

```bash
# Single image
python yolo3d.py predict --model best.pt --source image.jpg

# Folder of images
python yolo3d.py predict --model best.pt --source images/

# Video
python yolo3d.py predict --model best.pt --source video.mp4
```

### Prediction with Custom Settings

```bash
python yolo3d.py predict \
  --model runs/detect/train/weights/best.pt \
  --source datasets/kitti/testing/image_2/ \
  --conf 0.3 \
  --iou 0.5 \
  --imgsz 640 \
  --save \
  --save-txt \
  --save-conf \
  --name kitti-test
```

### Webcam Inference

```bash
python yolo3d.py predict --model best.pt --source 0 --show
```

### Prediction Options

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | *required* | Model weights (.pt) |
| `--source` | *required* | Source (image/folder/video/webcam) |
| `--conf` | `0.25` | Confidence threshold |
| `--iou` | `0.45` | NMS IoU threshold |
| `--imgsz` | `640` | Image size |
| `--device` | `0` | Device |
| `--save` | `True` | Save results |
| `--save-txt` | `False` | Save as txt labels |
| `--save-conf` | `False` | Save confidence scores |
| `--show` | `False` | Display results |
| `--project` | `runs/detect` | Save directory |
| `--name` | `predict` | Experiment name |
| `--verbose` | `False` | Verbose output |

---

## 📤 Export

### Export to ONNX

```bash
python yolo3d.py export --model best.pt --format onnx
```

### Export Options

| Format | Command | Use Case |
|--------|---------|----------|
| ONNX | `--format onnx` | General deployment |
| TorchScript | `--format torchscript` | PyTorch production |
| TensorFlow Lite | `--format tflite` | Mobile (Android/iOS) |
| Edge TPU | `--format edgetpu` | Google Coral |
| TensorFlow.js | `--format tfjs` | Web browsers |
| CoreML | `--format coreml` | Apple devices |
| TensorRT | `--format engine` | NVIDIA GPUs |

### Advanced Export

```bash
# ONNX with FP16 and simplification
python yolo3d.py export \
  --model best.pt \
  --format onnx \
  --half \
  --simplify \
  --dynamic

# TFLite with INT8 quantization
python yolo3d.py export \
  --model best.pt \
  --format tflite \
  --int8
```

### Export Options

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | *required* | Model weights (.pt) |
| `--format` | `onnx` | Export format |
| `--imgsz` | `640` | Image size |
| `--half` | `False` | FP16 quantization |
| `--int8` | `False` | INT8 quantization |
| `--dynamic` | `False` | Dynamic axes |
| `--simplify` | `False` | Simplify ONNX |
| `--optimize` | `False` | Optimize for mobile |

---

## 💡 Complete Workflow Example

### 1. Setup Dataset

```bash
# Download and prepare KITTI dataset
python scripts/kitti_setup.py all --create-yaml
```

### 2. Train Model

```bash
# Train YOLOv12n-3D for 100 epochs
python yolo3d.py train \
  --model n \
  --data datasets/kitti/kitti-3d.yaml \
  --epochs 100 \
  --batch 16 \
  --imgsz 640 \
  --name kitti-yolov12n-3d \
  --device 0 \
  --verbose
```

### 3. Validate Model

```bash
# Validate on test set
python yolo3d.py val \
  --model runs/detect/kitti-yolov12n-3d/weights/best.pt \
  --data datasets/kitti/kitti-3d.yaml \
  --plots \
  --save-json
```

### 4. Run Inference

```bash
# Test on images
python yolo3d.py predict \
  --model runs/detect/kitti-yolov12n-3d/weights/best.pt \
  --source datasets/kitti/testing/image_2/ \
  --conf 0.25 \
  --save \
  --save-txt
```

### 5. Export for Deployment

```bash
# Export to ONNX
python yolo3d.py export \
  --model runs/detect/kitti-yolov12n-3d/weights/best.pt \
  --format onnx \
  --simplify
```

---

## 🎨 CLI Output Examples

### Training Output

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                        YOLOv12-3D Command Line Tool                          ║
║              AI Research Group, Civil Engineering, KMUTT                     ║
╚══════════════════════════════════════════════════════════════════════════════╝

================================================================================
Training Configuration
================================================================================

  Model:           ultralytics/cfg/models/v12/yolov12n-3d.yaml
  Dataset:         ultralytics/cfg/datasets/kitti-3d.yaml
  Epochs:          100
  Batch Size:      16
  Image Size:      640
  Device:          0
  Workers:         8
  Experiment Name: train

Start training? [y/N]: y

================================================================================
Starting Training
================================================================================
... training logs ...

================================================================================
Training Complete!
================================================================================
✓ Model saved to: runs/detect/train/weights/best.pt

  mAP50-95: 0.453
  mAP50:    0.721
```

### Validation Output

```
================================================================================
Validation Results
================================================================================

  Metrics (2D Detection):
    mAP50-95: 0.4534
    mAP50:    0.7214
    mAP75:    0.4892
    
  Per Class:
    Precision: 0.7645
    Recall:    0.6823

✓ Validation complete!
```

### Prediction Output

```
================================================================================
Inference Results
================================================================================

  Images processed: 248
  Total detections: 1,543
  Average per image: 6.2

✓ Results saved to: runs/detect/predict
```

---

## 🔧 Tips & Best Practices

### Training Tips

1. **Start Small**: Begin with YOLOv12n-3D for faster iterations
2. **Use GPU**: Training on CPU is extremely slow
3. **Monitor Training**: Use `--verbose` to see detailed progress
4. **Save Checkpoints**: Keep `--save` enabled (default)
5. **Early Stopping**: Use `--patience 50` to prevent overfitting

### Multi-GPU Training

```bash
# Use multiple GPUs
python yolo3d.py train \
  --model m \
  --data kitti-3d.yaml \
  --device 0,1,2,3 \
  --batch 64
```

### Resume Training

```bash
# Resume from last checkpoint
python yolo3d.py train \
  --model runs/detect/train/weights/last.pt \
  --data kitti-3d.yaml
```

### Hyperparameter Tuning

```bash
# Conservative for 3D detection
python yolo3d.py train \
  --model m \
  --data kitti-3d.yaml \
  --lr0 0.001 \
  --lrf 0.01 \
  --box 10.0 \
  --cls 0.5 \
  --dfl 2.0
```

---

## 🐛 Troubleshooting

### "CUDA out of memory"

**Solution**: Reduce batch size
```bash
python yolo3d.py train --batch 8  # or 4, or 2
```

### "Dataset not found"

**Solution**: Check dataset path
```bash
# Ensure dataset exists
ls -la datasets/kitti/training/image_2/

# Update path in config
vim ultralytics/cfg/datasets/kitti-3d.yaml
```

### "Model not found"

**Solution**: Use correct model path
```bash
# List available models
ls runs/detect/*/weights/best.pt

# Use full path
python yolo3d.py val --model runs/detect/train/weights/best.pt
```

### Slow Training

**Solutions**:
1. Reduce image size: `--imgsz 512`
2. Use smaller model: `--model n`
3. Increase workers: `--workers 16`
4. Use GPU: `--device 0`

---

## 📚 Help Commands

```bash
# General help
python yolo3d.py --help

# Command-specific help
python yolo3d.py train --help
python yolo3d.py val --help
python yolo3d.py predict --help
python yolo3d.py export --help
```

---

## 🔗 Integration Examples

### With Python Scripts

```python
import subprocess

# Train from Python
subprocess.run([
    'python', 'yolo3d.py', 'train',
    '--data', 'kitti-3d.yaml',
    '--epochs', '100',
    '--batch', '16',
    '-y'
])
```

### In Shell Scripts

```bash
#!/bin/bash
# train_all_sizes.sh

for size in n s m l x; do
    python yolo3d.py train \
        --model $size \
        --data kitti-3d.yaml \
        --epochs 100 \
        --name kitti-yolov12${size}-3d \
        -y
done
```

### Batch Processing

```bash
# Process multiple test sets
for dir in test1 test2 test3; do
    python yolo3d.py predict \
        --model best.pt \
        --source data/$dir \
        --name results-$dir
done
```

---

## 📊 Expected Performance

### Training Time (KITTI, 100 epochs)

| Model | GPU | Batch | Time |
|-------|-----|-------|------|
| YOLOv12n-3D | RTX 4090 | 16 | ~6 hours |
| YOLOv12s-3D | RTX 4090 | 16 | ~8 hours |
| YOLOv12m-3D | RTX 4090 | 16 | ~12 hours |
| YOLOv12l-3D | RTX 3090 | 16 | ~18 hours |

### Inference Speed

| Model | GPU | FPS | mAP50 |
|-------|-----|-----|-------|
| YOLOv12n-3D | RTX 4090 | ~120 | ~70% |
| YOLOv12s-3D | RTX 4090 | ~100 | ~75% |
| YOLOv12m-3D | RTX 4090 | ~80 | ~78% |

---

## ✅ Next Steps

After training:

1. **Evaluate thoroughly**
   ```bash
   python yolo3d.py val --model best.pt --data kitti-3d.yaml --plots
   ```

2. **Test on real images**
   ```bash
   python yolo3d.py predict --model best.pt --source my_images/
   ```

3. **Export for deployment**
   ```bash
   python yolo3d.py export --model best.pt --format onnx
   ```

4. **Submit to KITTI benchmark** (if applicable)

---

**For more information:**
- Full Documentation: `YOLO3D_README.md`
- Quick Start: `QUICK_START.md`
- Dataset Setup: `scripts/README.md`

**Developed by AI Research Group**  
**Department of Civil Engineering, KMUTT**
