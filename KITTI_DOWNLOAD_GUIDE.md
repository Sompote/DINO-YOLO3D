# KITTI 3D Object Detection Dataset Download Guide

This guide explains how to download and setup the KITTI 3D object detection dataset for training YOLOv12-3D.

## Dataset Information

**Source:** http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d

**Required Files:**
- `data_object_image_2.zip` (12 GB) - Left color camera images
- `data_object_label_2.zip` (5 MB) - Training labels with 3D bounding boxes
- `data_object_calib.zip` (16 MB) - Camera calibration matrices

**Total Size:** ~12 GB

## Download Methods

### Method 1: Automated Download (Bash Script - Recommended for Linux)

```bash
# Make script executable
chmod +x download_kitti.sh

# Run the script
./download_kitti.sh

# Or specify custom directories
./download_kitti.sh ./downloads ./datasets/kitti
```

**Features:**
- Automatically downloads all required files
- Resumes interrupted downloads
- Verifies downloaded files
- Extracts and organizes dataset
- Works with both `wget` and `curl`

### Method 2: Python Script

```bash
# Make script executable
chmod +x download_kitti.py

# Verify downloads only
python download_kitti.py --verify-only

# Download and extract
python download_kitti.py --extract

# Custom directories
python download_kitti.py --download-dir ./downloads --output-dir ./datasets/kitti --extract
```

**Options:**
- `--download-dir DIR` - Directory for zip files (default: ./downloads)
- `--output-dir DIR` - Directory to extract to (default: ./datasets/kitti)
- `--extract` - Automatically extract files
- `--verify-only` - Only verify without extracting

### Method 3: Manual Download

1. Visit: http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d
2. Accept the terms of use
3. Download the following files:
   - Left color images (data_object_image_2.zip)
   - Training labels (data_object_label_2.zip)
   - Camera calibration (data_object_calib.zip)
4. Place files in `./downloads/` directory
5. Run extraction:
   ```bash
   python download_kitti.py --extract
   ```

### Method 4: Direct Download with wget/curl

```bash
# Create download directory
mkdir -p downloads

# Download using wget
wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip -P downloads/
wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip -P downloads/
wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_calib.zip -P downloads/

# Or using curl
curl -o downloads/data_object_image_2.zip https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip
curl -o downloads/data_object_label_2.zip https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip
curl -o downloads/data_object_calib.zip https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_calib.zip

# Extract files
cd downloads
unzip data_object_image_2.zip -d ../datasets/kitti
unzip data_object_label_2.zip -d ../datasets/kitti
unzip data_object_calib.zip -d ../datasets/kitti
cd ..
```

## Dataset Structure

After extraction, the dataset will be organized as follows:

```
datasets/kitti/
├── training/
│   ├── image_2/      # 7,481 training images
│   ├── label_2/      # 7,481 label files with 3D annotations
│   └── calib/        # 7,481 calibration files
└── testing/
    ├── image_2/      # 7,518 testing images
    └── calib/        # 7,518 calibration files
```

## Configuration

After downloading, update the dataset path in your configuration file:

**File:** `ultralytics/cfg/datasets/kitti-3d.yaml`

```yaml
path: ./datasets/kitti  # Update this path
train: training/image_2
val: training/image_2    # KITTI doesn't have official val set
test: testing/image_2

# Class names
names:
  0: Car
  1: Pedestrian
  2: Cyclist
```

## Next Steps

1. **Verify Dataset:**
   ```bash
   # Check training images
   ls datasets/kitti/training/image_2/*.png | wc -l
   # Should show: 7481
   
   # Check labels
   ls datasets/kitti/training/label_2/*.txt | wc -l
   # Should show: 7481
   ```

2. **Start Training:**
   ```bash
   # Using the training script
   python train.py --data kitti-3d.yaml --model yolov12-3d.yaml --epochs 100
   
   # Or using yolo3d.py
   ./yolo3d.py train --data kitti-3d --model yolov12-3d --epochs 100
   ```

3. **Run Inference:**
   ```bash
   python predict.py --model runs/train/exp/weights/best.pt --source datasets/kitti/testing/image_2
   ```

## Troubleshooting

### Download Issues

**Problem:** Download is slow or times out
- **Solution:** Use the bash script which supports resume functionality
- **Alternative:** Download files manually from the KITTI website

**Problem:** Insufficient disk space
- **Solution:** Ensure you have at least 15 GB free space (12 GB data + 3 GB for extraction)

### Extraction Issues

**Problem:** `unzip` command not found
- **Solution:** Install unzip
  ```bash
  # Ubuntu/Debian
  sudo apt-get install unzip
  
  # macOS
  brew install unzip
  
  # CentOS/RHEL
  sudo yum install unzip
  ```

**Problem:** Corrupted zip files
- **Solution:** Re-download the corrupted file
  ```bash
  rm downloads/data_object_image_2.zip
  wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip -P downloads/
  ```

### Dataset Issues

**Problem:** Training fails with "Dataset not found"
- **Solution:** Check the path in `kitti-3d.yaml` is correct
- **Verify:** Run `ls datasets/kitti/training/image_2/` to ensure images exist

**Problem:** Label format errors
- **Solution:** KITTI labels are in a specific format. Ensure you downloaded `data_object_label_2.zip` (not other label formats)

## KITTI Label Format

Each label file contains lines with the following format:

```
Type Truncated Occluded Alpha Bbox_2D Dimensions_3D Location_3D Rotation_Y [Score]
```

Example:
```
Car 0.00 0 -1.58 587.01 173.33 614.12 200.12 1.65 1.67 3.64 -0.65 1.71 46.70 -1.59
```

Fields:
- **Type:** Object class (Car, Pedestrian, Cyclist, etc.)
- **Truncated:** 0-1 indicating truncation
- **Occluded:** 0-3 indicating occlusion level
- **Alpha:** Observation angle
- **Bbox_2D:** 2D bounding box (left, top, right, bottom)
- **Dimensions_3D:** Height, width, length in meters
- **Location_3D:** X, Y, Z in camera coordinates
- **Rotation_Y:** Rotation around Y-axis

## Dataset Statistics

- **Training images:** 7,481
- **Testing images:** 7,518
- **Image size:** 1242 × 375 pixels
- **Classes:** 8 (Car, Van, Truck, Pedestrian, Person_sitting, Cyclist, Tram, Misc)
- **Commonly used:** 3 classes (Car, Pedestrian, Cyclist)

## Citation

If you use the KITTI dataset, please cite:

```bibtex
@inproceedings{Geiger2012CVPR,
  author = {Andreas Geiger and Philip Lenz and Raquel Urtasun},
  title = {Are we ready for Autonomous Driving? The KITTI Vision Benchmark Suite},
  booktitle = {Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2012}
}
```

## Additional Resources

- **Official Website:** http://www.cvlibs.net/datasets/kitti/
- **3D Object Detection Benchmark:** http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d
- **Development Kit:** http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d
- **Paper:** http://www.cvlibs.net/publications/Geiger2012CVPR.pdf

## Support

For issues related to:
- **Dataset download:** Check KITTI website or use alternative download methods
- **YOLOv12-3D training:** See [README.md](README.md) and [QUICK_START.md](QUICK_START.md)
- **KITTI format:** See official KITTI documentation
