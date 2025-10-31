# KITTI Dataset Download Scripts - Usage Guide

**AI Research Group, Department of Civil Engineering, KMUTT**

---

## 📦 Available Scripts

### 1. `download_kitti.py` - Python Download Manager
Full-featured Python script for downloading and organizing KITTI dataset.

### 2. `download_kitti_auto.sh` - Bash Automation Script
Automated bash script for quick setup (requires manual download first).

---

## 🚀 Quick Start (Choose One Method)

### Method 1: Bash Script (Fastest)

```bash
# Step 1: Download KITTI files manually from website
# Place .zip files in 'downloads' folder

# Step 2: Run the bash script
bash scripts/download_kitti_auto.sh ./datasets/kitti

# Done! Dataset is ready for training
```

### Method 2: Python Script (More Control)

```bash
# Step 1: Download KITTI files manually from website

# Step 2: Run Python script with options
python scripts/download_kitti.py \
    --data_dir ./datasets/kitti \
    --download_dir ./downloads \
    --val_split 0.2 \
    --create_yaml

# Done! Dataset is ready for training
```

---

## 📥 Manual Download Steps

### Required Files (Download from KITTI Website)

1. **Visit**: http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d
2. **Register/Login** to your KITTI account
3. **Download these 3 files**:

| Filename | Size | Description |
|----------|------|-------------|
| `data_object_image_2.zip` | 12 GB | Training images (left camera) |
| `data_object_label_2.zip` | 5 MB | Training labels (3D annotations) |
| `data_object_calib.zip` | 16 MB | Camera calibration files |

4. **Place files in**: `./downloads/` folder

---

## 🛠️ Detailed Usage

### Python Script Options

```bash
python scripts/download_kitti.py --help
```

**Available Options:**

```
--data_dir PATH          Where to extract dataset (default: ./datasets/kitti)
--download_dir PATH      Where .zip files are located (default: ./downloads)
--val_split FLOAT        Validation split ratio (default: 0.2)
--skip_extract          Skip extraction if already done
--create_yaml           Create dataset YAML config file
```

**Common Examples:**

```bash
# Basic usage (default settings)
python scripts/download_kitti.py

# Custom data location
python scripts/download_kitti.py --data_dir /data/kitti

# 30% validation split
python scripts/download_kitti.py --val_split 0.3

# Only create train/val splits (skip extraction)
python scripts/download_kitti.py --skip_extract

# Full setup with YAML config
python scripts/download_kitti.py --create_yaml
```

### Bash Script Usage

```bash
bash scripts/download_kitti_auto.sh [data_directory]
```

**Examples:**

```bash
# Default location
bash scripts/download_kitti_auto.sh

# Custom location
bash scripts/download_kitti_auto.sh /mnt/data/kitti

# Verbose mode (see all commands)
bash -x scripts/download_kitti_auto.sh ./datasets/kitti
```

---

## 📁 What the Scripts Do

### Automatic Steps:

1. ✅ Create necessary directories
2. ✅ Check for downloaded .zip files
3. ✅ Extract all archives
4. ✅ Organize files into KITTI structure
5. ✅ Create train/validation split files
6. ✅ Verify dataset integrity
7. ✅ Generate dataset statistics
8. ✅ Create YAML config (optional)

### Final Structure:

```
datasets/kitti/
├── training/
│   ├── image_2/      # 7,481 images
│   ├── label_2/      # 7,481 labels
│   └── calib/        # 7,481 calibration files
├── testing/
│   ├── image_2/      # 7,518 images
│   └── calib/        # 7,518 calibration files
├── ImageSets/
│   ├── train.txt     # Training split
│   ├── val.txt       # Validation split
│   └── trainval.txt  # All training data
└── kitti-3d.yaml     # Dataset config (if --create_yaml)
```

---

## ✅ Verification Commands

### Check File Counts

```bash
# Should show 7481 for each
ls datasets/kitti/training/image_2/*.png | wc -l
ls datasets/kitti/training/label_2/*.txt | wc -l
ls datasets/kitti/training/calib/*.txt | wc -l
```

### Check Label Format

```bash
# View sample label
head datasets/kitti/training/label_2/000000.txt
```

Expected format:
```
Type Truncated Occluded Alpha Bbox[4] Dimensions[3] Location[3] Rotation_y
Car 0.00 0 -1.58 587.01 173.33 614.12 200.12 1.65 1.67 3.64 -0.65 1.71 46.70 -1.59
```

### Check Split Files

```bash
# Should show ~5985 (80% of 7481)
wc -l datasets/kitti/ImageSets/train.txt

# Should show ~1496 (20% of 7481)
wc -l datasets/kitti/ImageSets/val.txt
```

---

## 🔧 Troubleshooting

### Problem: "Files not found"

**Solution:**
```bash
# Check download directory
ls -lh downloads/

# Expected files:
# data_object_image_2.zip
# data_object_label_2.zip
# data_object_calib.zip
```

### Problem: "Permission denied"

**Solution:**
```bash
# Make bash script executable
chmod +x scripts/download_kitti_auto.sh

# Or use Python script instead
python scripts/download_kitti.py
```

### Problem: "Extraction failed"

**Solution:**
```bash
# Check disk space (need ~25GB free)
df -h .

# Check if unzip is installed
which unzip

# Install if needed (Ubuntu/Debian)
sudo apt-get install unzip

# Install if needed (macOS)
brew install unzip
```

### Problem: "File count mismatch"

**Cause:** Some images don't have annotations (valid in KITTI)

**Verify:**
```python
from pathlib import Path

kitti = Path('datasets/kitti/training')
images = set(f.stem for f in (kitti / 'image_2').glob('*.png'))
labels = set(f.stem for f in (kitti / 'label_2').glob('*.txt'))

missing_labels = images - labels
print(f"Images without labels: {len(missing_labels)}")
# This is normal! Empty images are valid.
```

---

## 📊 Expected Results

### Dataset Statistics

```
Training Set: 7,481 images
├─ With labels: 7,481 files
├─ Train split: ~5,985 images (80%)
└─ Val split: ~1,496 images (20%)

Testing Set: 7,518 images
└─ No labels (for competition submission)

Total Download: ~12.3 GB
Total Extracted: ~25 GB
```

### Class Distribution

```
Total Objects: 80,256

Cars: 28,742 (35.8%)
Pedestrians: 4,487 (5.6%)
Cyclists: 1,627 (2.0%)
Vans: 2,914 (3.6%)
Trucks: 1,094 (1.4%)
Trams: 511 (0.6%)
Persons (sitting): 222 (0.3%)
Misc: 973 (1.2%)
DontCare: 39,686 (49.5%)
```

---

## ⏱️ Time Estimates

| Step | Time | Notes |
|------|------|-------|
| Manual download | 10-30 min | Depends on internet speed |
| Extraction | 5-15 min | Depends on disk speed (SSD faster) |
| Organization | 1-2 min | Automatic |
| Verification | < 1 min | Automatic |
| **Total** | **15-45 min** | First-time setup |

---

## 🎯 After Setup

### 1. Update Dataset Config

Edit `ultralytics/cfg/datasets/kitti-3d.yaml`:

```yaml
path: ./datasets/kitti  # ← Update this to your actual path
train: training/image_2
val: training/image_2
```

### 2. Test Dataset Loading

```python
from ultralytics.data.dataset import KITTIDataset

# Create dataset
dataset = KITTIDataset(
    img_path='datasets/kitti/training/image_2',
    imgsz=640,
    task='detect3d',
    data={'nc': 8, 'names': ['Car', 'Truck', 'Pedestrian', 'Cyclist', 
                              'Misc', 'Van', 'Tram', 'Person_sitting']}
)

print(f"✓ Dataset loaded: {len(dataset)} samples")

# Test getting one sample
sample = dataset[0]
print(f"✓ Image shape: {sample['img'].shape}")
print(f"✓ Labels shape: {sample['cls'].shape}")
```

### 3. Start Training

```bash
# Using example script
python examples/train_kitti_3d.py

# Or directly with YOLO
from ultralytics import YOLO

model = YOLO('ultralytics/cfg/models/v12/yolov12-3d.yaml')
model.train(
    data='ultralytics/cfg/datasets/kitti-3d.yaml',
    epochs=100,
    imgsz=640,
    batch=16
)
```

---

## 📚 Additional Resources

- **KITTI Website**: http://www.cvlibs.net/datasets/kitti/
- **KITTI Devkit**: http://www.cvlibs.net/datasets/kitti/setup.php
- **Dataset Paper**: Geiger et al., "Are we ready for Autonomous Driving?"
- **Full Documentation**: See `scripts/README_DOWNLOAD.md`

---

## 💡 Tips

1. **Download during off-peak hours** - KITTI servers can be slow
2. **Use SSD for extraction** - Much faster than HDD
3. **Keep .zip files** - In case you need to re-extract
4. **Backup splits** - Save `ImageSets/` folder for reproducibility
5. **Monitor disk space** - Need ~25GB free for full dataset

---

## 📞 Support

**Script issues:**
- Check `scripts/README_DOWNLOAD.md` for detailed guide
- Open GitHub issue with error message

**KITTI download issues:**
- Check KITTI website status
- Contact KITTI support team
- Try downloading at different times

**Training issues:**
- See `YOLO3D_README.md`
- Check `QUICK_START.md`

---

**Ready to train! 🚀**

**Developed by AI Research Group**  
**Department of Civil Engineering, KMUTT**
