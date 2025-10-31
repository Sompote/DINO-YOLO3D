# KITTI Dataset Setup - CLI Tool

**AI Research Group, Department of Civil Engineering, KMUTT**

A command-line tool for downloading and preparing the KITTI 3D Object Detection dataset.

---

## 🚀 Quick Start

```bash
# 1. Download KITTI files from website (manual step)
# 2. Place .zip files in downloads/ folder
# 3. Run one command to setup everything:

python scripts/kitti_setup.py all --create-yaml
```

---

## 📋 CLI Commands

### Overview

```
python scripts/kitti_setup.py <command> [options]

Commands:
  download    Check downloads and show instructions
  extract     Extract downloaded .zip files
  verify      Verify dataset structure
  split       Create train/validation splits
  all         Run complete setup (extract + verify + split)
```

---

## 📥 Step-by-Step Usage

### Step 1: Check Downloads

Shows which files are downloaded and provides download instructions:

```bash
python scripts/kitti_setup.py download
```

**Output:**
```
╔══════════════════════════════════════════════════════════════════════════════╗
║                    KITTI 3D Object Detection Setup Tool                      ║
║              AI Research Group, Civil Engineering, KMUTT                     ║
╚══════════════════════════════════════════════════════════════════════════════╝

================================================================================
Checking Downloaded Files
================================================================================
✓ data_object_image_2.zip (12.00 GB)
✓ data_object_label_2.zip (0.01 GB)
✓ data_object_calib.zip (0.02 GB)
```

**If files are missing**, it shows download instructions.

---

### Step 2: Extract Files

Extract all downloaded .zip files:

```bash
python scripts/kitti_setup.py extract
```

**Features:**
- ✅ Progress bars for extraction (if tqdm installed)
- ✅ Automatic directory structure creation
- ✅ Error handling

**Output:**
```
================================================================================
Extracting Files
================================================================================
ℹ Extracting data_object_image_2.zip...
  data_object_image_2.zip |████████████████████| 12396/12396
✓ data_object_image_2.zip extracted successfully
```

---

### Step 3: Verify Dataset

Verify the extracted dataset structure:

```bash
python scripts/kitti_setup.py verify
```

**Checks:**
- Directory structure is correct
- File counts match (7,481 files)
- All required directories exist

**Output:**
```
================================================================================
Verifying Dataset Structure
================================================================================
✓ training/image_2
✓ training/label_2
✓ training/calib

================================================================================
Dataset Statistics
================================================================================
  Training Images:          7481
  Training Labels:          7481
  Calibration Files:        7481

✓ Dataset verification passed!
```

---

### Step 4: Create Train/Val Split

Create train and validation splits:

```bash
python scripts/kitti_setup.py split --val-split 0.2
```

**Options:**
- `--val-split`: Validation ratio (default: 0.2 = 20%)
- `--seed`: Random seed for reproducibility (default: 42)
- `--create-yaml`: Also create dataset YAML config

**Output:**
```
================================================================================
Creating Train/Val Split (80% / 20%)
================================================================================

  Training samples:         5985  (80%)
  Validation samples:       1496  (20%)
  Total samples:            7481
  
  Split files saved to: datasets/kitti/ImageSets

✓ Train/val split created successfully
```

---

### All-in-One Command

Run the complete setup with one command:

```bash
python scripts/kitti_setup.py all --create-yaml
```

**This command:**
1. ✅ Checks downloaded files
2. ✅ Extracts all archives
3. ✅ Verifies structure
4. ✅ Creates train/val splits
5. ✅ Generates YAML config

**Perfect for first-time setup!**

---

## ⚙️ Command Reference

### Global Options

Available for all commands:

```bash
--data-dir PATH        Dataset directory (default: ./datasets/kitti)
--download-dir PATH    Download directory (default: ./downloads)
```

**Example:**
```bash
python scripts/kitti_setup.py extract \
  --data-dir /data/kitti \
  --download-dir ~/Downloads
```

---

### Command: `download`

Check for downloaded files and show instructions.

```bash
python scripts/kitti_setup.py download [--data-dir DIR] [--download-dir DIR]
```

**Use case:** First step - check what needs to be downloaded

---

### Command: `extract`

Extract downloaded .zip files.

```bash
python scripts/kitti_setup.py extract [--data-dir DIR] [--download-dir DIR]
```

**Requirements:**
- Downloaded .zip files in download directory
- ~25GB free disk space

---

### Command: `verify`

Verify dataset structure and file counts.

```bash
python scripts/kitti_setup.py verify [--data-dir DIR]
```

**Checks:**
- All required directories exist
- File counts are correct
- Shows dataset statistics

---

### Command: `split`

Create train/validation splits.

```bash
python scripts/kitti_setup.py split [OPTIONS]

Options:
  --val-split RATIO    Validation split (default: 0.2)
  --seed SEED          Random seed (default: 42)
  --create-yaml        Create dataset YAML config
```

**Examples:**
```bash
# 80/20 split (default)
python scripts/kitti_setup.py split

# 70/30 split
python scripts/kitti_setup.py split --val-split 0.3

# Custom seed with YAML
python scripts/kitti_setup.py split --seed 123 --create-yaml
```

---

### Command: `all`

Run complete setup process.

```bash
python scripts/kitti_setup.py all [OPTIONS]

Options:
  --val-split RATIO    Validation split (default: 0.2)
  --seed SEED          Random seed (default: 42)
  --create-yaml        Create dataset YAML config
  --data-dir DIR       Dataset directory
  --download-dir DIR   Download directory
```

**Example:**
```bash
python scripts/kitti_setup.py all \
  --val-split 0.2 \
  --create-yaml \
  --data-dir ./datasets/kitti
```

---

## 📊 Output Files

After running the setup, you'll have:

```
datasets/kitti/
├── training/
│   ├── image_2/         # 7,481 training images
│   ├── label_2/         # 7,481 label files (3D annotations)
│   └── calib/           # 7,481 calibration files
├── testing/
│   ├── image_2/         # 7,518 test images
│   └── calib/           # 7,518 calibration files
├── ImageSets/
│   ├── train.txt        # Training image IDs (~5,985)
│   ├── val.txt          # Validation image IDs (~1,496)
│   └── trainval.txt     # All training image IDs (7,481)
└── kitti-3d.yaml        # Dataset config (if --create-yaml used)
```

---

## 🎨 Color-Coded Output

The CLI uses colors for better readability:

- 🟢 **Green (✓)**: Success messages
- 🔴 **Red (✗)**: Error messages
- 🟡 **Yellow (⚠)**: Warning messages
- 🔵 **Blue (ℹ)**: Info messages

---

## 💡 Usage Examples

### Example 1: First Time Setup

```bash
# Step by step
python scripts/kitti_setup.py download          # Check downloads
# ... download files manually ...
python scripts/kitti_setup.py extract           # Extract files
python scripts/kitti_setup.py verify            # Verify structure
python scripts/kitti_setup.py split --create-yaml  # Create splits + YAML

# Or use all-in-one command
python scripts/kitti_setup.py all --create-yaml
```

### Example 2: Re-create Splits Only

```bash
# Change validation split ratio
python scripts/kitti_setup.py split --val-split 0.3
```

### Example 3: Verify Existing Dataset

```bash
# Check if dataset is properly set up
python scripts/kitti_setup.py verify
```

### Example 4: Custom Directories

```bash
# Use custom paths
python scripts/kitti_setup.py all \
  --data-dir /mnt/datasets/kitti \
  --download-dir ~/Downloads \
  --create-yaml
```

---

## 🔧 Requirements

### Python Dependencies

```bash
pip install tqdm  # Optional but recommended for progress bars
```

### System Requirements

- **Disk Space**: ~25GB free
- **Python**: 3.7+
- **OS**: Linux, macOS, or Windows

### Downloaded Files

Place these in your download directory:

| File | Size | Required |
|------|------|----------|
| `data_object_image_2.zip` | 12 GB | ✅ Yes |
| `data_object_label_2.zip` | 5 MB | ✅ Yes |
| `data_object_calib.zip` | 16 MB | ✅ Yes |

---

## 🐛 Troubleshooting

### "Missing X file(s)"

**Problem:** Required .zip files not found

**Solution:**
```bash
# Check download directory
ls -lh downloads/

# Expected files:
# data_object_image_2.zip
# data_object_label_2.zip
# data_object_calib.zip

# Run download command for instructions
python scripts/kitti_setup.py download
```

---

### "Extraction failed"

**Problem:** Error during extraction

**Solutions:**
1. Check disk space: `df -h`
2. Verify .zip files aren't corrupted: `unzip -t downloads/*.zip`
3. Try extracting manually

---

### "File counts don't match"

**Problem:** Expected 7,481 files but got different count

**Explanation:** This can be normal - some images don't have annotations

**Check:**
```bash
python scripts/kitti_setup.py verify
```

---

### "Permission denied"

**Problem:** Cannot write to directory

**Solution:**
```bash
# Make script executable (Linux/Mac)
chmod +x scripts/kitti_setup.py

# Or run with python explicitly
python scripts/kitti_setup.py <command>
```

---

## 📚 Help & Documentation

### Get Help

```bash
# General help
python scripts/kitti_setup.py --help

# Command-specific help
python scripts/kitti_setup.py download --help
python scripts/kitti_setup.py extract --help
python scripts/kitti_setup.py split --help
```

### Verbose Output

For debugging, use Python's verbose mode:

```bash
python -v scripts/kitti_setup.py all
```

---

## ✅ Verification Checklist

After setup, verify everything is correct:

```bash
# 1. Check file counts
ls datasets/kitti/training/image_2/*.png | wc -l  # Should be 7481
ls datasets/kitti/training/label_2/*.txt | wc -l  # Should be 7481

# 2. Check split files exist
ls datasets/kitti/ImageSets/

# 3. Run verification command
python scripts/kitti_setup.py verify

# 4. Test loading with Python
python -c "from pathlib import Path; assert (Path('datasets/kitti/training/image_2').exists())"
```

---

## 🚀 Next Steps

After successful setup:

### 1. Update Dataset Config

If not using `--create-yaml`, update manually:

```yaml
# ultralytics/cfg/datasets/kitti-3d.yaml
path: ./datasets/kitti  # Update to your path
```

### 2. Verify Dataset Loading

```python
from ultralytics.data.dataset import KITTIDataset

dataset = KITTIDataset(
    img_path='datasets/kitti/training/image_2',
    imgsz=640,
    task='detect3d'
)
print(f"✓ Dataset loaded: {len(dataset)} samples")
```

### 3. Start Training

```bash
# Using example script
python examples/train_kitti_3d.py

# Or with YOLO CLI
yolo train \
  model=ultralytics/cfg/models/v12/yolov12-3d.yaml \
  data=ultralytics/cfg/datasets/kitti-3d.yaml \
  epochs=100 \
  imgsz=640
```

---

## 📖 Additional Resources

- **KITTI Website**: http://www.cvlibs.net/datasets/kitti/
- **Full Documentation**: See `YOLO3D_README.md`
- **Training Guide**: See `QUICK_START.md`

---

**Developed by AI Research Group**  
**Department of Civil Engineering, KMUTT**

For issues, open a GitHub issue or contact the team.
