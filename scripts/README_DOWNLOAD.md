# KITTI Dataset Download Guide

**AI Research Group, Department of Civil Engineering, KMUTT**

This guide explains how to download and prepare the KITTI 3D Object Detection dataset for training.

---

## 📋 Quick Start

### Option 1: Automated Setup (Recommended)

If you already have the KITTI zip files:

```bash
# 1. Place downloaded .zip files in downloads folder
mkdir -p downloads

# 2. Run the automated setup script
bash scripts/download_kitti_auto.sh ./datasets/kitti
```

### Option 2: Manual Setup with Python Script

```bash
# 1. Download KITTI files manually (see instructions below)
# 2. Extract using Python script
python scripts/download_kitti.py --data_dir ./datasets/kitti --download_dir ./downloads
```

---

## 📥 Step-by-Step Download Instructions

### Step 1: Register on KITTI Website

1. Visit: **http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d**
2. Click "Register" and create an account
3. Verify your email address
4. Log in to your account

### Step 2: Accept License Agreement

1. Read the KITTI Vision Benchmark Suite license
2. Accept the terms and conditions
3. You'll get access to download links

### Step 3: Download Required Files

Download these 3 files (total ~12.3 GB):

| File | Size | Description | Download Link |
|------|------|-------------|---------------|
| `data_object_image_2.zip` | 12 GB | Left color camera images | [Download](http://www.cvlibs.net/download.php?file=data_object_image_2.zip) |
| `data_object_label_2.zip` | 5 MB | Training labels with 3D annotations | [Download](http://www.cvlibs.net/download.php?file=data_object_label_2.zip) |
| `data_object_calib.zip` | 16 MB | Camera calibration matrices | [Download](http://www.cvlibs.net/download.php?file=data_object_calib.zip) |

**Note:** Download links require authentication. Log in first!

### Step 4: Organize Downloaded Files

```bash
# Create download directory
mkdir -p downloads

# Move downloaded files to downloads folder
mv ~/Downloads/data_object_image_2.zip downloads/
mv ~/Downloads/data_object_label_2.zip downloads/
mv ~/Downloads/data_object_calib.zip downloads/
```

### Step 5: Extract and Setup

**Option A: Automated (bash script)**
```bash
bash scripts/download_kitti_auto.sh ./datasets/kitti
```

**Option B: Python script with options**
```bash
python scripts/download_kitti.py \
    --data_dir ./datasets/kitti \
    --download_dir ./downloads \
    --val_split 0.2 \
    --create_yaml
```

---

## 📁 Expected Directory Structure

After setup, your directory should look like this:

```
datasets/kitti/
├── training/
│   ├── image_2/          # 7,481 training images (.png)
│   │   ├── 000000.png
│   │   ├── 000001.png
│   │   └── ...
│   ├── label_2/          # 7,481 label files (.txt)
│   │   ├── 000000.txt
│   │   ├── 000001.txt
│   │   └── ...
│   └── calib/            # 7,481 calibration files (.txt)
│       ├── 000000.txt
│       ├── 000001.txt
│       └── ...
├── testing/
│   ├── image_2/          # 7,518 test images (.png)
│   └── calib/            # 7,518 calibration files (.txt)
├── ImageSets/            # Train/val splits (auto-generated)
│   ├── train.txt         # Training image IDs
│   ├── val.txt           # Validation image IDs
│   └── trainval.txt      # All training image IDs
└── kitti-3d.yaml         # Dataset config (optional)
```

---

## 🔍 Verification

### Verify File Counts

```bash
# Count training files
echo "Training images: $(ls datasets/kitti/training/image_2/*.png | wc -l)"
echo "Training labels: $(ls datasets/kitti/training/label_2/*.txt | wc -l)"
echo "Calibration files: $(ls datasets/kitti/training/calib/*.txt | wc -l)"
```

Expected output:
```
Training images: 7481
Training labels: 7481
Calibration files: 7481
```

### Verify Label Format

```bash
# Check a sample label file
head -n 5 datasets/kitti/training/label_2/000000.txt
```

Expected format:
```
Truck 0.00 0 -1.57 599.41 156.40 629.75 189.25 2.85 2.63 12.34 0.47 1.49 69.44 -1.56
Car 0.00 0 1.85 387.63 181.54 423.81 203.12 1.67 1.87 3.69 -16.53 2.39 58.49 1.57
Cyclist 0.00 3 -1.65 676.60 163.95 688.98 193.93 1.86 0.60 2.02 4.59 1.32 45.84 -1.55
```

### Test with Python

```python
from pathlib import Path

kitti_path = Path('datasets/kitti')

# Check structure
assert (kitti_path / 'training/image_2').exists(), "Images not found"
assert (kitti_path / 'training/label_2').exists(), "Labels not found"
assert (kitti_path / 'training/calib').exists(), "Calibration not found"

# Count files
n_images = len(list((kitti_path / 'training/image_2').glob('*.png')))
n_labels = len(list((kitti_path / 'training/label_2').glob('*.txt')))
n_calib = len(list((kitti_path / 'training/calib').glob('*.txt')))

print(f"✓ Images: {n_images}")
print(f"✓ Labels: {n_labels}")
print(f"✓ Calibration: {n_calib}")

assert n_images == n_labels == n_calib == 7481, "File count mismatch!"
print("\n✓ Dataset verification passed!")
```

---

## ⚙️ Script Options

### Python Script (`download_kitti.py`)

```bash
python scripts/download_kitti.py --help
```

**Options:**
- `--data_dir`: Directory to store KITTI dataset (default: `./datasets/kitti`)
- `--download_dir`: Directory with downloaded .zip files (default: `./downloads`)
- `--val_split`: Validation split ratio (default: `0.2` = 20%)
- `--skip_extract`: Skip extraction if already done
- `--create_yaml`: Create dataset YAML configuration file

**Examples:**

```bash
# Basic usage
python scripts/download_kitti.py

# Custom directories
python scripts/download_kitti.py --data_dir ./data/kitti --download_dir ~/Downloads

# Custom validation split (30%)
python scripts/download_kitti.py --val_split 0.3

# Skip extraction, only create splits
python scripts/download_kitti.py --skip_extract --create_yaml

# Full setup with YAML
python scripts/download_kitti.py --data_dir ./datasets/kitti --create_yaml
```

### Bash Script (`download_kitti_auto.sh`)

```bash
bash scripts/download_kitti_auto.sh [data_directory]
```

**Examples:**

```bash
# Default location (./datasets/kitti)
bash scripts/download_kitti_auto.sh

# Custom location
bash scripts/download_kitti_auto.sh /path/to/kitti

# With verbose output
bash -x scripts/download_kitti_auto.sh ./datasets/kitti
```

---

## 🎯 Train/Val Split

The scripts automatically create train/validation splits:

- **Default split**: 80% train / 20% validation
- **Random seed**: 42 (for reproducibility)
- **Files created**: `ImageSets/train.txt`, `ImageSets/val.txt`, `ImageSets/trainval.txt`

### Custom Split

To create a custom split:

```python
python scripts/download_kitti.py --val_split 0.3  # 70% train, 30% val
```

Or manually edit the split files in `ImageSets/`.

---

## 🐛 Troubleshooting

### Issue: "Permission denied"

```bash
chmod +x scripts/download_kitti_auto.sh
```

### Issue: "unzip: command not found"

```bash
# Ubuntu/Debian
sudo apt-get install unzip

# macOS
brew install unzip

# Or use Python script instead
python scripts/download_kitti.py
```

### Issue: "File count mismatch"

Some label files might be missing. This is normal for images without objects. Check:

```bash
# Find images without labels
cd datasets/kitti/training
comm -23 <(ls image_2/*.png | xargs -n1 basename | cut -d. -f1 | sort) \
         <(ls label_2/*.txt | xargs -n1 basename | cut -d. -f1 | sort)
```

### Issue: Download is slow

KITTI servers can be slow. Consider:
1. Using a download manager (e.g., `wget`, `aria2c`)
2. Downloading during off-peak hours
3. Check your internet connection

### Issue: Extraction takes too long

The image archive is 12GB. Extraction can take 5-15 minutes depending on your disk speed. Be patient!

---

## 📊 Dataset Statistics

| Split | Images | Labels | Total Size |
|-------|--------|--------|------------|
| Training | 7,481 | 7,481 | ~12 GB |
| Testing | 7,518 | - | ~12 GB |
| **Total** | **14,999** | **7,481** | **~24 GB** |

### Class Distribution (Training Set)

| Class | Count | Percentage |
|-------|-------|------------|
| Car | 28,742 | 52.3% |
| Pedestrian | 4,487 | 8.2% |
| Cyclist | 1,627 | 3.0% |
| Van | 2,914 | 5.3% |
| Truck | 1,094 | 2.0% |
| Person (sitting) | 222 | 0.4% |
| Tram | 511 | 0.9% |
| Misc | 973 | 1.8% |

---

## ✅ Next Steps

After successful download and setup:

1. **Update dataset config**:
   ```yaml
   # ultralytics/cfg/datasets/kitti-3d.yaml
   path: ./datasets/kitti  # Update this path
   ```

2. **Verify with quick test**:
   ```python
   from ultralytics.data.dataset import KITTIDataset
   
   dataset = KITTIDataset(
       img_path='datasets/kitti/training/image_2',
       imgsz=640,
       task='detect3d'
   )
   print(f"Dataset size: {len(dataset)}")
   ```

3. **Start training**:
   ```bash
   python examples/train_kitti_3d.py
   ```

---

## 📚 Additional Resources

- **KITTI Website**: http://www.cvlibs.net/datasets/kitti/
- **KITTI Devkit**: http://www.cvlibs.net/datasets/kitti/setup.php
- **KITTI Paper**: "Are we ready for Autonomous Driving? The KITTI Vision Benchmark Suite"
- **Evaluation Server**: http://www.cvlibs.net/datasets/kitti/eval_object.php

---

## 📧 Support

For issues with:
- **Dataset download**: Contact KITTI team or check their FAQ
- **Script errors**: Open an issue on GitHub
- **Training questions**: See `YOLO3D_README.md`

---

**Prepared by AI Research Group, Department of Civil Engineering, KMUTT**
