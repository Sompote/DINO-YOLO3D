# DINO-YOLO3D Command Line Syntax Guide

## The Problem: Arguments Not Being Recognized

If you see output like this when you specify `--batch 200 --epochs 400`:
```
Epochs:          100    ← Should be 400!
Batch Size:      16     ← Should be 200!
```

The issue is **NOT with the code** - it's with your **bash command syntax**.

## Common Mistake: Missing Backslashes

### ❌ WRONG (Missing backslashes)

```bash
python yolo3d.py train \
    --model m \
    --data /workspace/yolor2/DINO-YOLO3D/kitti-3d.yaml
    --epochs 400 \        # ← Missing \ on line above!
    --batch 200 \
    --imgsz 640 \
        --device 0        # ← Missing \ !
    --name yolov12s-kitti
```

**What happens:**
Bash interprets this as **TWO separate commands**:

**Command 1** (this is what runs):
```bash
python yolo3d.py train --model m --data /workspace/yolor2/DINO-YOLO3D/kitti-3d.yaml
```
↑ Uses default values: epochs=100, batch=16

**Command 2** (this fails silently):
```bash
--epochs 400 --batch 200 --imgsz 640 --device 0 --name yolov12s-kitti
```
↑ Bash tries to run this as a command, fails

### ✅ CORRECT (All backslashes present)

```bash
python yolo3d.py train \
    --model m \
    --data /workspace/yolor2/DINO-YOLO3D/kitti-3d.yaml \
    --epochs 400 \
    --batch 200 \
    --imgsz 640 \
    --device 0 \
    --name yolov12m-kitti
```

**Note:** The backslash `\` tells bash to continue the command on the next line.

## Solution 1: Use Backslashes Correctly

**Rule:** Every line except the last MUST end with `\`

```bash
python yolo3d.py train \              # ← backslash
    --model m \                       # ← backslash
    --data kitti-3d.yaml \            # ← backslash
    --epochs 400 \                    # ← backslash
    --batch 200 \                     # ← backslash
    --device 0 \                      # ← backslash
    --name my-experiment              # ← NO backslash (last line)
```

## Solution 2: Single Line Command (Recommended for Copy-Paste)

Put everything on one line:

```bash
python yolo3d.py train --model m --data kitti-3d.yaml --epochs 400 --batch 200 --imgsz 640 --device 0 --name yolov12m-kitti
```

## Solution 3: Verify Your Command Before Running

Use echo to test:

```bash
echo python yolo3d.py train \
    --model m \
    --data kitti-3d.yaml \
    --epochs 400 \
    --batch 200 \
    --device 0
```

If it prints as a single line, you're good! If it breaks into multiple lines, you're missing backslashes.

## Correct Training Commands

### For RTX 5090 (32GB) - Model Size M

```bash
python yolo3d.py train --model m --data kitti-3d.yaml --epochs 400 --batch 200 --nbs 200 --imgsz 640 --device 0 --name yolov12m-kitti-400ep
```

**Or with backslashes:**
```bash
python yolo3d.py train \
    --model m \
    --data kitti-3d.yaml \
    --epochs 400 \
    --batch 200 \
    --nbs 200 \
    --imgsz 640 \
    --device 0 \
    --name yolov12m-kitti-400ep
```

### For RTX 5090 (32GB) - Model Size L

```bash
python yolo3d.py train --model l --data kitti-3d.yaml --epochs 400 --batch 128 --nbs 128 --imgsz 640 --device 0 --name yolov12l-kitti-400ep
```

### For RTX 5090 (32GB) - Model Size S (Faster Training)

```bash
python yolo3d.py train --model s --data kitti-3d.yaml --epochs 400 --batch 256 --nbs 256 --imgsz 640 --device 0 --name yolov12s-kitti-400ep
```

## How to Verify Arguments Are Being Read

After running the command, check the "Training Configuration" output:

```
Training Configuration
================================================================================
Model:           ultralytics/cfg/models/v12/yolov12-3d.yaml
Scale:           m
Dataset:         kitti-3d.yaml
Epochs:          400    ← Check this matches your --epochs
Batch Size:      200    ← Check this matches your --batch
Nominal Batch:   200    ← Should match --nbs (or show "64 (auto)")
Image Size:      640
Device:          0
Name:            yolov12m-kitti-400ep
```

**If Epochs or Batch Size don't match what you specified, you have a syntax error in your command!**

## Common Shell Syntax Errors

### 1. Missing Backslash
```bash
python yolo3d.py train \
    --model m
    --batch 200     # ← Will fail! Missing \ above
```

### 2. Space After Backslash
```bash
python yolo3d.py train \
    --model m \     # ← Extra space after \, will fail!
    --batch 200
```

### 3. Tab Instead of Backslash
```bash
python yolo3d.py train
    --model m       # ← Looks like continuation but isn't!
    --batch 200
```

## Testing Your Command Syntax

### Method 1: Echo Test
```bash
# Add 'echo' before your command
echo python yolo3d.py train \
    --model m \
    --batch 200

# Should print:
# python yolo3d.py train --model m --batch 200
```

### Method 2: History Check
```bash
# After running, check what bash saw:
history 1

# Should show as single command:
# python yolo3d.py train --model m --batch 200 ...
```

## Best Practices

### ✅ DO

1. **Copy full commands** from this guide
2. **Use single-line commands** when copy-pasting to avoid errors
3. **Verify the "Training Configuration" output** shows your values
4. **Use --nbs equal to --batch** to disable gradient accumulation for maximum speed

### ❌ DON'T

1. **Manually type multi-line commands** with backslashes (easy to make mistakes)
2. **Assume default values** will work for your use case
3. **Ignore the configuration output** - always verify your arguments were read

## Quick Reference: Key Arguments

| Argument | Description | Default | Your Value |
|----------|-------------|---------|------------|
| `--model` | Model size (n/s/m/l/x) or path | n | m |
| `--data` | Dataset YAML file | required | kitti-3d.yaml |
| `--epochs` | Training epochs | 100 | 400 |
| `--batch` | Batch size per GPU | 16 | 200 |
| `--nbs` | Nominal batch (grad accum) | 64 | 200 (disable accum) |
| `--imgsz` | Input image size | 640 | 640 |
| `--device` | GPU device | 0 | 0 |
| `--name` | Experiment name | train | yolov12m-kitti-400ep |

## Summary

**Your code is fine! The issue is bash command syntax.**

✅ **The Fix:**
1. Use single-line commands (safest)
2. OR ensure every line except the last ends with `\`
3. OR verify with echo before running

✅ **Correct Command for Your Setup (RTX 5090):**
```bash
python yolo3d.py train --model m --data /workspace/yolor2/DINO-YOLO3D/kitti-3d.yaml --epochs 400 --batch 200 --nbs 200 --imgsz 640 --device 0 --name yolov12m-kitti-400ep
```

This will give you:
- Model: YOLOv12-m (medium)
- Epochs: 400
- Batch: 200 (effective batch = 200, no accumulation)
- Dataset: KITTI 3D
- Device: GPU 0
- Output: runs/detect3d/yolov12m-kitti-400ep/
