#!/bin/bash
################################################################################
# KITTI 3D Object Detection Dataset Download Script
################################################################################
#
# This script downloads the KITTI 3D object detection dataset files.
# Dataset source: http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d
#
# Usage:
#   ./download_kitti.sh [download_dir] [output_dir]
#
# Arguments:
#   download_dir - Directory to store downloaded zip files (default: ./downloads)
#   output_dir   - Directory to extract dataset to (default: ./datasets/kitti)
#
################################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
DOWNLOAD_DIR="${1:-./downloads}"
OUTPUT_DIR="${2:-./datasets/kitti}"

# KITTI dataset URLs
KITTI_BASE_URL="https://s3.eu-central-1.amazonaws.com/avg-kitti"
FILES=(
    "data_object_image_2.zip"
    "data_object_label_2.zip"
    "data_object_calib.zip"
)

# Print header
echo "================================================================================"
echo "KITTI 3D Object Detection Dataset Downloader"
echo "================================================================================"
echo ""

# Create download directory
mkdir -p "$DOWNLOAD_DIR"

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Determine download tool
if command_exists wget; then
    DOWNLOAD_CMD="wget -c -P"
elif command_exists curl; then
    DOWNLOAD_CMD="curl -C - -o"
else
    echo -e "${RED}Error: Neither wget nor curl is installed.${NC}"
    echo "Please install wget or curl and try again."
    exit 1
fi

# Function to download file
download_file() {
    local url="$1"
    local filename=$(basename "$url")
    local filepath="$DOWNLOAD_DIR/$filename"

    if [ -f "$filepath" ]; then
        echo -e "${GREEN}✓${NC} File already exists: $filename"
        return 0
    fi

    echo -e "${BLUE}Downloading:${NC} $filename"

    if command_exists wget; then
        wget -c "$url" -P "$DOWNLOAD_DIR"
    else
        curl -C - -o "$filepath" "$url"
    fi

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓${NC} Downloaded: $filename"
        return 0
    else
        echo -e "${RED}✗${NC} Failed to download: $filename"
        return 1
    fi
}

# Function to verify file exists
verify_file() {
    local filepath="$1"
    local filename=$(basename "$filepath")

    if [ -f "$filepath" ]; then
        local size=$(du -h "$filepath" | cut -f1)
        echo -e "${GREEN}✓${NC} Found: $filename ($size)"
        return 0
    else
        echo -e "${RED}✗${NC} Missing: $filename"
        return 1
    fi
}

# Function to extract zip file
extract_file() {
    local zipfile="$1"
    local filename=$(basename "$zipfile")

    echo -e "${BLUE}Extracting:${NC} $filename"

    if unzip -q -o "$zipfile" -d "$OUTPUT_DIR"; then
        echo -e "${GREEN}✓${NC} Extracted: $filename"
        return 0
    else
        echo -e "${RED}✗${NC} Failed to extract: $filename"
        return 1
    fi
}

# Print download information
echo "Download directory: $DOWNLOAD_DIR"
echo "Output directory: $OUTPUT_DIR"
echo ""
echo "Files to download:"
echo "  - data_object_image_2.zip (12 GB) - Left color images"
echo "  - data_object_label_2.zip (5 MB)  - Training labels"
echo "  - data_object_calib.zip (16 MB)   - Camera calibration"
echo ""
echo "Dataset source: http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d"
echo ""

# Ask for confirmation
read -p "Start download? [y/N] " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Download cancelled."
    exit 0
fi

echo ""
echo "================================================================================"
echo "Downloading KITTI Dataset Files"
echo "================================================================================"
echo ""

# Download files
ALL_DOWNLOADED=true
for file in "${FILES[@]}"; do
    if ! download_file "$KITTI_BASE_URL/$file"; then
        ALL_DOWNLOADED=false
    fi
done

echo ""
echo "================================================================================"
echo "Verifying Downloads"
echo "================================================================================"
echo ""

# Verify all files
ALL_PRESENT=true
for file in "${FILES[@]}"; do
    if ! verify_file "$DOWNLOAD_DIR/$file"; then
        ALL_PRESENT=false
    fi
done

if [ "$ALL_PRESENT" = false ]; then
    echo ""
    echo -e "${RED}Some files are missing or failed to download.${NC}"
    echo "Please check the errors above and try again."
    exit 1
fi

echo ""
echo -e "${GREEN}✓ All files downloaded successfully!${NC}"
echo ""

# Ask to extract
read -p "Extract files now? [y/N] " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Extraction skipped."
    echo ""
    echo "To extract later, run:"
    echo "  ./download_kitti.sh $DOWNLOAD_DIR $OUTPUT_DIR"
    exit 0
fi

echo ""
echo "================================================================================"
echo "Extracting Dataset"
echo "================================================================================"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Extract files
ALL_EXTRACTED=true
for file in "${FILES[@]}"; do
    if ! extract_file "$DOWNLOAD_DIR/$file"; then
        ALL_EXTRACTED=false
    fi
done

if [ "$ALL_EXTRACTED" = false ]; then
    echo ""
    echo -e "${RED}Some files failed to extract.${NC}"
    exit 1
fi

echo ""
echo "================================================================================"
echo "Setup Complete!"
echo "================================================================================"
echo ""
echo -e "${GREEN}✓ KITTI dataset has been downloaded and extracted successfully!${NC}"
echo ""
echo "Dataset location: $OUTPUT_DIR"
echo ""
echo "Dataset structure:"
echo "  $OUTPUT_DIR/"
echo "    ├── training/"
echo "    │   ├── image_2/    (left color camera images)"
echo "    │   ├── label_2/    (training labels)"
echo "    │   └── calib/      (calibration files)"
echo "    └── testing/"
echo "        ├── image_2/"
echo "        └── calib/"
echo ""
echo "Next steps:"
echo "  1. Update dataset path in ultralytics/cfg/datasets/kitti-3d.yaml"
echo "     Set 'path: $OUTPUT_DIR'"
echo "  2. Start training:"
echo "     python train.py --data kitti-3d.yaml --model yolov12-3d.yaml"
echo ""
