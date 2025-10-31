# KITTI Setup - Quick Reference Card

**AI Research Group, Department of Civil Engineering, KMUTT**

---

## One-Liner Setup

```bash
# Complete setup in one command
python scripts/kitti_setup.py all --create-yaml
```

---

## Common Commands

### Check Downloads
```bash
python scripts/kitti_setup.py download
```

### Extract Files
```bash
python scripts/kitti_setup.py extract
```

### Verify Dataset
```bash
python scripts/kitti_setup.py verify
```

### Create Splits
```bash
python scripts/kitti_setup.py split --val-split 0.2 --create-yaml
```

### Full Setup
```bash
python scripts/kitti_setup.py all --create-yaml
```

---

## Quick Options

| Option | Default | Description |
|--------|---------|-------------|
| `--data-dir` | `./datasets/kitti` | Dataset location |
| `--download-dir` | `./downloads` | Downloaded files location |
| `--val-split` | `0.2` | Validation split ratio |
| `--seed` | `42` | Random seed |
| `--create-yaml` | `False` | Create YAML config |

---

## Example Workflows

### First Time Setup
```bash
# 1. Download files from KITTI website
# 2. Place in downloads/ folder
# 3. Run setup
python scripts/kitti_setup.py all --create-yaml
```

### Re-split Dataset
```bash
# Change validation split to 30%
python scripts/kitti_setup.py split --val-split 0.3
```

### Verify Existing Dataset
```bash
# Check if dataset is OK
python scripts/kitti_setup.py verify
```

### Custom Paths
```bash
python scripts/kitti_setup.py all \
  --data-dir /data/kitti \
  --download-dir ~/Downloads \
  --create-yaml
```

---

## File Locations

After setup:
```
datasets/kitti/
├── training/image_2/       # 7,481 images
├── training/label_2/       # 7,481 labels
├── training/calib/         # 7,481 calibrations
├── ImageSets/train.txt     # Train split (~5,985)
├── ImageSets/val.txt       # Val split (~1,496)
└── kitti-3d.yaml          # Config (optional)
```

---

## Help Commands

```bash
# General help
python scripts/kitti_setup.py --help

# Command help
python scripts/kitti_setup.py download --help
python scripts/kitti_setup.py extract --help
python scripts/kitti_setup.py verify --help
python scripts/kitti_setup.py split --help
python scripts/kitti_setup.py all --help
```

---

## Next Steps After Setup

```bash
# 1. Start training
python examples/train_kitti_3d.py

# 2. Or use YOLO CLI
yolo train \
  model=ultralytics/cfg/models/v12/yolov12-3d.yaml \
  data=datasets/kitti/kitti-3d.yaml \
  epochs=100
```

---

**For full documentation, see: `scripts/README.md`**
