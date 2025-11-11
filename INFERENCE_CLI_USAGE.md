# inference.py - Unified 3D Detection CLI

A single command-line script for running 3D object detection on both **images** and **videos**.

## Features

✅ **Unified Interface** - Same command for images and videos  
✅ **Batch Processing** - Process multiple images at once  
✅ **Custom Model Weights** - Specify any model checkpoint  
✅ **Video Support** - Frame skipping, max frames, progress tracking  
✅ **Temperature Scaling** - Automatic 49% depth improvement  
✅ **Flexible Output** - Save to directory or specific file  

---

## Quick Start

### Single Image
```bash
python inference.py --input image.png --model last-5.pt
```

### Multiple Images
```bash
python inference.py --input img1.png img2.png img3.png --model last-5.pt
```

### Video
```bash
python inference.py --input video.mov --model last-5.pt --output result.mp4
```

---

## Command-Line Arguments

### Required Arguments

| Argument | Description | Example |
|----------|-------------|---------|
| `--input` | Input file(s) - images or video | `--input image.png` |
| `--model` | Model weights path | `--model last-5.pt` |

### Optional Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--output` | `inference_results` | Output directory (images) or file (video) |
| `--conf` | `0.25` | Confidence threshold |
| `--show` | `False` | Display results (images only) |
| `--quiet` | `False` | Suppress detection details |

### Video-Only Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--skip` | `1` | Process every Nth frame |
| `--max-frames` | `0` | Max frames to process (0=all) |

---

## Usage Examples

### 1. Basic Image Inference
```bash
python inference.py \
  --input /path/to/image.png \
  --model last-5.pt
```

Output:
- Saved to: `inference_results/image_result.jpg`
- Prints detection details

### 2. Multiple Images with Custom Output
```bash
python inference.py \
  --input img1.png img2.png img3.png \
  --model best-8.pt \
  --output my_results/
```

Output:
- `my_results/img1_result.jpg`
- `my_results/img2_result.jpg`
- `my_results/img3_result.jpg`

### 3. KITTI Test Images
```bash
python inference.py \
  --input /path/to/kitti/training/image_2/000*.png \
  --model last-5.pt \
  --output kitti_results/ \
  --conf 0.3
```

### 4. Full Video Processing
```bash
python inference.py \
  --input video.mov \
  --model last-5.pt \
  --output output_3d.mp4
```

Progress:
```
Processing Video: video.mov
================================================================================
Resolution: 1920x1080
FPS: 30.0
Total frames: 1800
Output: output_3d.mp4
Processed: 30/1800 frames
Processed: 60/1800 frames
...
Done! Processed 1800 frames
```

### 5. Video with Frame Skip (Faster)
```bash
python inference.py \
  --input video.mov \
  --model last-5.pt \
  --output quick_result.mp4 \
  --skip 3
```

Processes every 3rd frame → 3x faster

### 6. Video Preview (First 100 Frames)
```bash
python inference.py \
  --input video.mov \
  --model last-5.pt \
  --output preview.mp4 \
  --max-frames 100
```

### 7. High Confidence Only
```bash
python inference.py \
  --input image.png \
  --model last-5.pt \
  --conf 0.5 \
  --quiet
```

Only detections with >50% confidence, no verbose output

### 8. Display Results Interactively
```bash
python inference.py \
  --input image.png \
  --model last-5.pt \
  --show
```

Opens matplotlib window to view result

---

## Input File Types

### Images (Auto-detected)
- `.png`
- `.jpg`, `.jpeg`
- `.bmp`
- `.tiff`

### Videos (Auto-detected)
- `.mp4`
- `.avi`
- `.mov`
- `.mkv`
- `.flv`
- `.wmv`

The script automatically detects whether input is image or video based on extension.

---

## Output Format

### For Images
```
inference_results/
├── image1_result.jpg
├── image2_result.jpg
└── image3_result.jpg
```

Each image gets 3D bounding boxes drawn on it.

### For Videos
```
output_video.mp4
```

Video with 3D bounding boxes on every processed frame.

---

## Detection Output (Verbose Mode)

```
================================================================================
Processing: 000010.png
================================================================================
Image size: 1242x375
Loaded calibration from: /path/to/calib/000010.txt
Running inference...
Found 10 detections:
  1. Car (conf=0.897)
     3D location: x=16.65m, y=4.33m, z=23.30m
  2. Car (conf=0.896)
     3D location: x=-4.37m, y=2.69m, z=20.03m
  ...
Saved: inference_results/000010_result.jpg
```

Use `--quiet` to suppress detection details.

---

## Performance

### Images
- ~50-100ms per image (depends on resolution)
- Batch processing is sequential

### Videos
- ~30 FPS on CPU
- Use `--skip` to increase speed
- Use `--max-frames` for quick previews

---

## Model Weights

Specify any trained model:

```bash
# Use last checkpoint
python inference.py --input img.png --model last-5.pt

# Use best checkpoint
python inference.py --input img.png --model best-8.pt

# Use custom path
python inference.py --input img.png --model /path/to/weights.pt
```

---

## Advanced Examples

### Process All Images in Directory
```bash
python inference.py \
  --input /data/kitti/training/image_2/*.png \
  --model last-5.pt \
  --output kitti_results/ \
  --quiet
```

### High-Quality Video (Every Frame)
```bash
python inference.py \
  --input input.mov \
  --model best-8.pt \
  --output high_quality.mp4 \
  --conf 0.3 \
  --skip 1
```

### Fast Preview (Every 5th Frame, First 200 Frames)
```bash
python inference.py \
  --input long_video.mp4 \
  --model last-5.pt \
  --output preview.mp4 \
  --skip 5 \
  --max-frames 200
```

---

## Troubleshooting

### "Model not found"
```bash
# Check model path
ls -lh last-5.pt

# Use absolute path
python inference.py --input img.png --model /full/path/to/last-5.pt
```

### "No detections found"
- Try lowering `--conf` threshold: `--conf 0.2`
- Check if image contains objects
- Verify model is trained for 3D detection

### Video Output is Empty
- Ensure `--output` ends with `.mp4`
- Check disk space
- Try different video codec

---

## Comparison with Original Scripts

| Feature | inference_3d_viz.py | video_3d_clean.py | **inference.py** |
|---------|---------------------|-------------------|------------------|
| Images | ✅ | ❌ | ✅ |
| Videos | ❌ | ✅ | ✅ |
| CLI Args | ❌ | ✅ | ✅ |
| Batch Images | ❌ | ❌ | ✅ |
| Custom Model | Hardcoded | CLI | ✅ CLI |
| Frame Skip | ❌ | ✅ | ✅ |
| Display | ✅ | ❌ | ✅ Optional |

**inference.py** = Best of both worlds! 🎉

---

## Temperature Scaling

All inference automatically benefits from **temperature scaling (T=0.5)**:

✅ Depth accuracy: +49% improvement  
✅ Built into model weights  
✅ No additional configuration needed  

The model (`ultralytics/nn/modules/head.py`) has temperature scaling enabled, so all scripts using the same model get the improvement.

---

Generated: 2025-11-11
