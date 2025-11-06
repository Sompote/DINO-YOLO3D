# DINO Integration Options - Quick Reference

## Available DINO Integration Modes

### 1. Single P0 Integration (Lightweight) 🚀
- **DINO levels**: P0 only
- **Memory usage**: Lower (~3-5GB)
- **Training speed**: Faster (~1.5-2x)
- **Accuracy**: Good
- **Best for**: Limited GPU resources, faster experimentation

**Usage:**
```bash
# Medium model with single P0 DINO
python yolodio3d.py train --data kitti-3d.yaml --yolo-size m --dino-integration single

# Large model with single P0 DINO
python yolodio3d.py train --data kitti-3d.yaml --yolo-size l --dino-integration single
```

### 2. Dual P0+P3 Integration (Maximum Performance) 🎯
- **DINO levels**: P0 and P3
- **Memory usage**: Higher (~4-6GB)
- **Training speed**: Slower (~2-3x)
- **Accuracy**: Best
- **Best for**: Maximum accuracy, complex datasets, production

**Usage:**
```bash
# Medium model with dual P0+P3 DINO
python yolodio3d.py train --data kitti-3d.yaml --yolo-size m --dino-integration dual

# Large model with dual P0+P3 DINO
python yolodio3d.py train --data kitti-3d.yaml --yolo-size l --dino-integration dual
```

## Model Config Files

| File | Model | DINO Integration | Parameters | Memory |
|------|-------|------------------|------------|--------|
| `yolov12m-3d-dino-p0.yaml` | Medium | P0 only | ~100M | ~3-4GB |
| `yolov12m-3d-dino-p0p3.yaml` | Medium | P0+P3 | ~120M | ~4-6GB |
| `yolov12l-3d-dino-p0.yaml` | Large | P0 only | ~160M | ~5-6GB |
| `yolov12l-3d-dino-p0p3.yaml` | Large | P0+P3 | ~180M | ~6-8GB |

## When to Use Each Option

### Choose **Single P0** if you:
- ✅ Have limited GPU memory (4-6GB)
- ✅ Need faster training/inference
- ✅ Want to experiment quickly
- ✅ Dataset is relatively simple
- ✅ Resource-constrained environment

### Choose **Dual P0+P3** if you:
- ✅ Have sufficient GPU memory (8GB+)
- ✅ Need maximum accuracy
- ✅ Working with complex scenes
- ✅ Production deployment
- ✅ Accuracy is more important than speed

## Comparison Table

| Feature | Single P0 | Dual P0+P3 |
|---------|-----------|------------|
| Training Time | 1.5-2x | 2-3x |
| GPU Memory | 3-5GB | 4-6GB |
| Model Parameters | ~100-160M | ~120-180M |
| Inference Speed | Faster | Slower |
| Accuracy | Good | Best |
| Best For | Quick experiments | Production use |

## Quick Commands

### Training
```bash
# Fast training (single P0)
python yolodio3d.py train --data kitti-3d.yaml --yolo-size m --dino-integration single --batch-size 32

# Maximum performance (dual P0+P3)
python yolodio3d.py train --data kitti-3d.yaml --yolo-size m --dino-integration dual --batch-size 16

# Base model (no DINO)
python yolodio3d.py train --data kitti-3d.yaml --yolo-size m --no-dino --batch-size 64
```

### Validation
```bash
# Validate single P0 model
python yolodio3d.py val --data kitti-3d.yaml --yolo-size m --dino-integration single

# Validate dual P0+P3 model
python yolodio3d.py val --data kitti-3d.yaml --yolo-size m --dino-integration dual
```

### Export
```bash
# Export single P0 model
python yolodio3d.py export --format onnx --yolo-size m --dino-integration single

# Export dual P0+P3 model
python yolodio3d.py export --format onnx --yolo-size m --dino-integration dual
```

## Architecture Visualization

### Single P0 Integration
```
Input → Conv → DINO(P0) → Conv → C3k2 → P3 → ... → Head
               ↑
           (Early enhancement)
```

### Dual P0+P3 Integration
```
Input → Conv → DINO(P0) → Conv → C3k2 → DINO(P3) → ... → Head
               ↑                          ↑
        (Early enhancement)        (Mid-level enhancement)
```

## Recommendations

1. **Start with Single P0** for initial experiments
2. **Upgrade to Dual P0+P3** if you need more accuracy
3. **Use `--freeze-dino`** for faster training (enabled by default)
4. **Adjust batch size** based on your GPU memory
5. **Validate both** on your dataset to compare performance

---

For more details, see `YOLODIO3D_README.md`
