# YOLOv12-3D CLI - Quick Reference

**AI Research Group, Department of Civil Engineering, KMUTT**

---

## 🚀 One-Liners

```bash
# Train
python yolo3d.py train --data kitti-3d.yaml --epochs 100 -y

# Validate
python yolo3d.py val --model best.pt --data kitti-3d.yaml

# Predict
python yolo3d.py predict --model best.pt --source images/

# Export
python yolo3d.py export --model best.pt --format onnx
```

---

## 📋 Common Commands

### Training

```bash
# Quick train (nano model, default settings)
python yolo3d.py train --data kitti-3d.yaml -y

# Production train (medium model, optimized)
python yolo3d.py train --model m --data kitti-3d.yaml --epochs 200 --batch 32 -y

# Resume training
python yolo3d.py train --model runs/detect/train/weights/last.pt --data kitti-3d.yaml
```

### Validation

```bash
# Basic validation
python yolo3d.py val --model best.pt --data kitti-3d.yaml

# With plots and JSON
python yolo3d.py val --model best.pt --data kitti-3d.yaml --plots --save-json
```

### Prediction

```bash
# Single image
python yolo3d.py predict --model best.pt --source image.jpg

# Folder
python yolo3d.py predict --model best.pt --source images/

# Video
python yolo3d.py predict --model best.pt --source video.mp4

# Webcam
python yolo3d.py predict --model best.pt --source 0 --show
```

### Export

```bash
# ONNX (general)
python yolo3d.py export --model best.pt --format onnx

# TFLite (mobile)
python yolo3d.py export --model best.pt --format tflite --int8

# TensorRT (NVIDIA)
python yolo3d.py export --model best.pt --format engine
```

---

## 🎯 Model Sizes

| Size | Command | Speed | Accuracy | Use Case |
|------|---------|-------|----------|----------|
| Nano | `--model n` | Fastest | Good | Real-time, edge |
| Small | `--model s` | Fast | Better | Balanced |
| Medium | `--model m` | Medium | Best | Production |
| Large | `--model l` | Slow | Higher | Research |
| XLarge | `--model x` | Slowest | Highest | Maximum accuracy |

---

## ⚙️ Key Options

### Training
```bash
--model n/s/m/l/x    # Model size
--data CONFIG.yaml   # Dataset config
--epochs 100         # Training epochs
--batch 16          # Batch size
--device 0          # GPU device
--name EXP_NAME     # Experiment name
-y                  # Skip confirmation
```

### Prediction
```bash
--model best.pt     # Trained model
--source PATH       # Images/video/folder
--conf 0.25        # Confidence threshold
--iou 0.45         # NMS IoU
--save             # Save results
--show             # Display results
```

---

## 📊 Complete Workflow

```bash
# 1. Setup dataset
python scripts/kitti_setup.py all --create-yaml

# 2. Train model
python yolo3d.py train --model m --data datasets/kitti/kitti-3d.yaml --epochs 100 -y

# 3. Validate
python yolo3d.py val --model runs/detect/train/weights/best.pt --data kitti-3d.yaml

# 4. Inference
python yolo3d.py predict --model runs/detect/train/weights/best.pt --source test_images/

# 5. Export
python yolo3d.py export --model runs/detect/train/weights/best.pt --format onnx
```

---

## 🔧 Troubleshooting

| Problem | Solution |
|---------|----------|
| CUDA OOM | `--batch 8` or smaller |
| Slow training | `--workers 16`, use GPU |
| Low accuracy | More `--epochs`, larger `--model` |
| File not found | Check paths with `ls` |

---

## 📚 Get Help

```bash
python yolo3d.py --help              # General help
python yolo3d.py train --help        # Training help
python yolo3d.py val --help          # Validation help
python yolo3d.py predict --help      # Prediction help
python yolo3d.py export --help       # Export help
```

---

## 🎓 Examples

### Multi-GPU Training
```bash
python yolo3d.py train --model m --data kitti-3d.yaml --device 0,1,2,3 --batch 64
```

### High Confidence Detection
```bash
python yolo3d.py predict --model best.pt --source images/ --conf 0.5
```

### Batch Export
```bash
for fmt in onnx torchscript tflite; do
    python yolo3d.py export --model best.pt --format $fmt
done
```

---

**Full Documentation**: See `CLI_GUIDE.md`

**Developed by AI Research Group, Civil Engineering, KMUTT**
