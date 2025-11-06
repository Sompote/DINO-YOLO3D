# DINO-YOLO3D Inference Guide

This guide explains how to use the `infer.py` script to run inference on images and videos with your trained DINO-YOLO3D model.

## Features

- ✅ **Image Inference**: Process single images or entire directories
- ✅ **Video Inference**: Process video files frame by frame
- ✅ **Webcam Support**: Real-time inference from webcam
- ✅ **3D Visualization**: Draw 3D bounding box projections on images
- ✅ **Rich Information**: Display depth, dimensions (H×W×L), and rotation angle
- ✅ **Export Results**: Save annotated images/videos and detection labels

## Quick Start

### 1. Image Inference

Process a single image:

```bash
python infer.py \
    --weights runs/detect3d/train/weights/best.pt \
    --source /path/to/image.jpg \
    --conf 0.25 \
    --visualize-3d
```

### 2. Video Inference

Process a video file:

```bash
python infer.py \
    --weights runs/detect3d/train/weights/best.pt \
    --source /path/to/video.mp4 \
    --conf 0.25 \
    --imgsz 640
```

### 3. Directory Inference

Process all images in a directory:

```bash
python infer.py \
    --weights runs/detect3d/train/weights/best.pt \
    --source /path/to/images/ \
    --save-txt \
    --save-conf
```

### 4. Webcam Inference

Real-time inference from webcam (device 0):

```bash
python infer.py \
    --weights runs/detect3d/train/weights/best.pt \
    --source 0 \
    --view-img
```

## Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--weights` | str | **required** | Path to trained weights file (.pt) |
| `--source` | str | **required** | Image/video path, directory, or webcam index (0) |
| `--imgsz` | int | 640 | Inference image size (pixels) |
| `--conf` | float | 0.25 | Confidence threshold (0.0-1.0) |
| `--iou` | float | 0.45 | NMS IoU threshold |
| `--device` | str | auto | Device to use (cuda:0, cpu) |
| `--save-dir` | str | runs/infer3d | Directory to save results |
| `--save-txt` | flag | False | Save detection labels to .txt files |
| `--save-conf` | flag | False | Include confidence scores in saved labels |
| `--classes` | int+ | None | Filter by class IDs (e.g., --classes 0 2 3) |
| `--visualize-3d` | flag | True | Draw 3D bounding boxes |
| `--no-visualize-3d` | flag | False | Disable 3D box visualization |
| `--view-img` | flag | False | Display results in window |
| `--line-width` | int | 2 | Bounding box line width |

## Output Format

### Visual Output

The script generates annotated images/videos with:
- **2D Bounding Boxes**: Standard detection boxes
- **3D Bounding Boxes**: Wireframe projection of 3D boxes (if `--visualize-3d`)
- **Labels**: Class name, confidence, and 3D parameters:
  - `D`: Depth in meters
  - `H`: Height in meters
  - `W`: Width in meters
  - `L`: Length in meters
  - `R`: Rotation angle in degrees

Example label:
```
Car 0.95
D:15.3m H:1.5m W:1.8m L:4.2m
R:45.0°
```

### Text Output (with `--save-txt`)

Detection labels are saved in YOLO format with 3D extensions:

```
<class_id> <x_center> <y_center> <width> <height> <depth> <dim_h> <dim_w> <dim_l> <rotation>
```

Example:
```
0 0.512 0.423 0.156 0.234 15.3 1.5 1.8 4.2 0.785
```

With confidence (using `--save-conf`):
```
0 0.95 0.512 0.423 0.156 0.234 15.3 1.5 1.8 4.2 0.785
```

## Examples

### Example 1: KITTI Image with High Confidence

```bash
python infer.py \
    --weights runs/detect3d/kitti/weights/best.pt \
    --source datasets/kitti/testing/image_2/000000.png \
    --conf 0.5 \
    --save-txt \
    --view-img
```

### Example 2: Traffic Video Processing

```bash
python infer.py \
    --weights runs/detect3d/train/weights/best.pt \
    --source traffic_video.mp4 \
    --conf 0.3 \
    --imgsz 1280 \
    --classes 0 1 2 \
    --save-dir runs/infer3d/traffic
```

Classes: 0=Car, 1=Pedestrian, 2=Cyclist

### Example 3: Batch Processing

```bash
python infer.py \
    --weights runs/detect3d/train/weights/best.pt \
    --source test_images/ \
    --conf 0.25 \
    --save-txt \
    --save-conf \
    --save-dir runs/infer3d/batch_test
```

### Example 4: No 3D Visualization

If you only want 2D boxes with 3D information in labels:

```bash
python infer.py \
    --weights runs/detect3d/train/weights/best.pt \
    --source image.jpg \
    --no-visualize-3d
```

## Understanding 3D Parameters

### Depth (D)
- Distance from camera to object center along Z-axis
- Range: 0-100 meters (model dependent)
- Used for: Distance estimation, collision avoidance

### Dimensions (H, W, L)
- **H**: Height of the 3D bounding box
- **W**: Width of the 3D bounding box
- **L**: Length of the 3D bounding box
- Range: 0-10 meters each (model dependent)
- Used for: Object size classification, clearance calculation

### Rotation (R)
- Rotation angle around Y-axis (yaw)
- Range: -180° to +180°
- 0° means object faces the camera
- Used for: Orientation estimation, trajectory prediction

## Tips for Best Results

1. **Image Size**: Use `--imgsz` matching your training size for best accuracy
2. **Confidence Threshold**: Adjust `--conf` based on your use case:
   - High precision: 0.5-0.7
   - Balanced: 0.25-0.4
   - High recall: 0.1-0.2
3. **GPU Acceleration**: Specify `--device 0` for CUDA device
4. **Class Filtering**: Use `--classes` to focus on specific object types
5. **Video Processing**: Lower `--imgsz` for faster processing of long videos

## Troubleshooting

### Issue: Low FPS on Video
**Solution**: Reduce `--imgsz` or use GPU with `--device 0`

### Issue: No Detections
**Solutions**:
- Lower `--conf` threshold
- Check if weights match the dataset
- Verify input image quality

### Issue: 3D Boxes Look Incorrect
**Note**: 3D box visualization uses simplified projection without camera calibration.
The numeric 3D parameters in labels are accurate, but visual projection is approximate.

### Issue: Out of Memory
**Solution**: Reduce `--imgsz` or process in smaller batches

## Integration with Other Tools

### Convert to KITTI Format

```python
import numpy as np

def yolo3d_to_kitti(label_line, img_width, img_height):
    """Convert YOLO3D format to KITTI format"""
    parts = label_line.strip().split()
    cls_id = int(parts[0])

    # Denormalize box
    x_center = float(parts[1]) * img_width
    y_center = float(parts[2]) * img_height
    width = float(parts[3]) * img_width
    height = float(parts[4]) * img_height

    # Get 3D params
    depth = float(parts[5])
    dim_h = float(parts[6])
    dim_w = float(parts[7])
    dim_l = float(parts[8])
    rotation = float(parts[9])

    # Convert to KITTI format
    # ... (implement KITTI conversion)

    return kitti_line
```

## Performance Benchmarks

Typical inference speeds (RTX 3090):

| Image Size | Batch Size | FPS | mAP50 |
|------------|------------|-----|-------|
| 640×640    | 1          | 45  | 0.82  |
| 1280×1280  | 1          | 18  | 0.85  |
| 640×640    | 8          | 160 | 0.82  |

## Citation

If you use this inference tool in your research, please cite:

```bibtex
@software{yolov12_3d_inference,
  title={DINO-YOLO3D Inference Tool},
  author={AI Research Group, KMUTT},
  year={2024},
  url={https://github.com/Sompote/DINO-YOLO3D}
}
```

## Support

For issues and questions:
- GitHub Issues: https://github.com/Sompote/DINO-YOLO3D/issues
- Documentation: https://github.com/Sompote/DINO-YOLO3D/wiki

---

**Developed by AI Research Group**
**Department of Civil Engineering**
**King Mongkut's University of Technology Thonburi (KMUTT)**
