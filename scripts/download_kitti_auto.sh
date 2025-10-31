#!/bin/bash

################################################################################
# KITTI 3D Object Detection Dataset - Automated Download Script
#
# Developed by AI Research Group
# Department of Civil Engineering
# King Mongkut's University of Technology Thonburi (KMUTT)
# Bangkok, Thailand
#
# This script automatically downloads and prepares the KITTI dataset
#
# Usage:
#   bash scripts/download_kitti_auto.sh [data_directory]
#
# Example:
#   bash scripts/download_kitti_auto.sh ./datasets/kitti
################################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default data directory
DATA_DIR="${1:-./datasets/kitti}"
DOWNLOAD_DIR="$DATA_DIR/downloads"

# Print header
echo "================================================================================"
echo "KITTI 3D Object Detection Dataset - Automated Setup"
echo "Developed by AI Research Group, Department of Civil Engineering, KMUTT"
echo "================================================================================"
echo ""

# Create directories
echo -e "${BLUE}Creating directories...${NC}"
mkdir -p "$DATA_DIR"
mkdir -p "$DOWNLOAD_DIR"
echo -e "${GREEN}✓ Directories created${NC}"
echo ""

# KITTI download URLs (these are example URLs - actual KITTI requires registration)
# NOTE: KITTI requires manual download due to license agreement
# If you have download links from KITTI, replace these URLs

echo -e "${YELLOW}================================================================================${NC}"
echo -e "${YELLOW}IMPORTANT: KITTI Dataset License Agreement${NC}"
echo -e "${YELLOW}================================================================================${NC}"
echo ""
echo "The KITTI dataset requires accepting a license agreement."
echo "This script cannot automatically download without proper authentication."
echo ""
echo "Please follow these steps:"
echo ""
echo "1. Visit: http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d"
echo "2. Register/Login to your account"
echo "3. Accept the license agreement"
echo "4. Download the following files manually:"
echo ""
echo "   Required files (~12.3 GB total):"
echo "   ├─ data_object_image_2.zip     (12 GB) - Training images"
echo "   ├─ data_object_label_2.zip     (5 MB)  - Training labels"
echo "   └─ data_object_calib.zip       (16 MB) - Calibration files"
echo ""
echo "5. Place downloaded files in: $DOWNLOAD_DIR"
echo "6. Run the extraction script:"
echo "   python scripts/download_kitti.py --data_dir $DATA_DIR --download_dir $DOWNLOAD_DIR"
echo ""
echo -e "${YELLOW}================================================================================${NC}"
echo ""

# Check if files already exist
echo -e "${BLUE}Checking for downloaded files in $DOWNLOAD_DIR...${NC}"

FILES_FOUND=0

if [ -f "$DOWNLOAD_DIR/data_object_image_2.zip" ]; then
    echo -e "${GREEN}✓ Found: data_object_image_2.zip${NC}"
    FILES_FOUND=$((FILES_FOUND + 1))
else
    echo -e "${RED}✗ Missing: data_object_image_2.zip${NC}"
fi

if [ -f "$DOWNLOAD_DIR/data_object_label_2.zip" ]; then
    echo -e "${GREEN}✓ Found: data_object_label_2.zip${NC}"
    FILES_FOUND=$((FILES_FOUND + 1))
else
    echo -e "${RED}✗ Missing: data_object_label_2.zip${NC}"
fi

if [ -f "$DOWNLOAD_DIR/data_object_calib.zip" ]; then
    echo -e "${GREEN}✓ Found: data_object_calib.zip${NC}"
    FILES_FOUND=$((FILES_FOUND + 1))
else
    echo -e "${RED}✗ Missing: data_object_calib.zip${NC}"
fi

echo ""

# If all files are found, extract them
if [ $FILES_FOUND -eq 3 ]; then
    echo -e "${GREEN}✓ All required files found!${NC}"
    echo ""

    read -p "Do you want to extract the files now? (y/n) " -n 1 -r
    echo ""

    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${BLUE}Extracting files...${NC}"

        # Extract images
        echo "Extracting images..."
        unzip -q "$DOWNLOAD_DIR/data_object_image_2.zip" -d "$DATA_DIR"
        echo -e "${GREEN}✓ Images extracted${NC}"

        # Extract labels
        echo "Extracting labels..."
        unzip -q "$DOWNLOAD_DIR/data_object_label_2.zip" -d "$DATA_DIR"
        echo -e "${GREEN}✓ Labels extracted${NC}"

        # Extract calibration
        echo "Extracting calibration..."
        unzip -q "$DOWNLOAD_DIR/data_object_calib.zip" -d "$DATA_DIR"
        echo -e "${GREEN}✓ Calibration extracted${NC}"

        echo ""
        echo -e "${GREEN}✓ Extraction complete!${NC}"

        # Verify structure
        echo ""
        echo -e "${BLUE}Verifying dataset structure...${NC}"

        if [ -d "$DATA_DIR/training/image_2" ] && [ -d "$DATA_DIR/training/label_2" ] && [ -d "$DATA_DIR/training/calib" ]; then
            echo -e "${GREEN}✓ Dataset structure looks good!${NC}"

            # Count files
            IMG_COUNT=$(ls -1 "$DATA_DIR/training/image_2" | wc -l)
            LABEL_COUNT=$(ls -1 "$DATA_DIR/training/label_2" | wc -l)
            CALIB_COUNT=$(ls -1 "$DATA_DIR/training/calib" | wc -l)

            echo ""
            echo "Training set statistics:"
            echo "  - Images: $IMG_COUNT"
            echo "  - Labels: $LABEL_COUNT"
            echo "  - Calibration files: $CALIB_COUNT"

            # Create train/val split using Python
            echo ""
            echo -e "${BLUE}Creating train/validation split...${NC}"
            python3 - <<EOF
import random
from pathlib import Path

data_dir = Path("$DATA_DIR")
training_dir = data_dir / 'training'
image_dir = training_dir / 'image_2'

# Get all image files
image_files = sorted(list(image_dir.glob('*.png')))
n_total = len(image_files)
val_split = 0.2
n_val = int(n_total * val_split)
n_train = n_total - n_val

# Shuffle and split
random.seed(42)
indices = list(range(n_total))
random.shuffle(indices)

train_indices = sorted(indices[:n_train])
val_indices = sorted(indices[n_train:])

# Create split directory
splits_dir = data_dir / 'ImageSets'
splits_dir.mkdir(exist_ok=True)

# Write files
with open(splits_dir / 'train.txt', 'w') as f:
    for idx in train_indices:
        f.write(f"{image_files[idx].stem}\n")

with open(splits_dir / 'val.txt', 'w') as f:
    for idx in val_indices:
        f.write(f"{image_files[idx].stem}\n")

with open(splits_dir / 'trainval.txt', 'w') as f:
    for img in image_files:
        f.write(f"{img.stem}\n")

print(f"✓ Train samples: {n_train}")
print(f"✓ Val samples: {n_val}")
print(f"✓ Split files saved to: {splits_dir}")
EOF

            echo ""
            echo -e "${GREEN}================================================================================${NC}"
            echo -e "${GREEN}SUCCESS! KITTI dataset is ready for training${NC}"
            echo -e "${GREEN}================================================================================${NC}"
            echo ""
            echo "Next steps:"
            echo "1. Update dataset path in: ultralytics/cfg/datasets/kitti-3d.yaml"
            echo "   Change 'path' to: $DATA_DIR"
            echo ""
            echo "2. Start training:"
            echo "   python examples/train_kitti_3d.py"
            echo ""

        else
            echo -e "${RED}✗ Dataset structure verification failed${NC}"
            echo "Please check the extracted files manually"
        fi
    else
        echo "Extraction cancelled."
    fi
else
    echo -e "${YELLOW}⚠ Missing $(( 3 - FILES_FOUND )) required file(s)${NC}"
    echo ""
    echo "Please download the missing files manually and run this script again."
fi

echo ""
echo "================================================================================"
