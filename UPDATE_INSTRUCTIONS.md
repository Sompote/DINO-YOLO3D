# Update Instructions - Fix Validation Error

## Problem

You're getting this error during validation:
```
✗ Training failed: 'tuple' object has no attribute 'shape'
```

## Root Cause

**You need to pull the latest code!** The fix was committed to GitHub but your server still has the old code.

## Solution: Update Your Code on the Server

Run these commands on your server (`root@C.27487975:/workspace/yolo8/YOLOv12-3D`):

### Step 1: Pull Latest Code

```bash
cd /workspace/yolo8/YOLOv12-3D
git pull origin main
```

### Step 2: Verify the Fix is Present

Check that the file contains the tuple/list fix:

```bash
grep -A 5 "Check if batch\[\"img\"\] is a tuple" ultralytics/models/yolo/detect3d/val.py
```

You should see:
```python
        # Check if batch["img"] is a tuple or list (should be a tensor)
        if "img" in batch:
            if isinstance(batch["img"], (tuple, list)):
                # If it's a tuple/list, convert to tensor by stacking
                batch["img"] = torch.stack(list(batch["img"]), 0)
```

### Step 3: Restart Training

Now run your training command with **CORRECT syntax** (all backslashes):

```bash
python yolo3d.py train \
    --model m \
    --data /workspace/yolor2/YOLOv12-3D/kitti-3d.yaml \
    --epochs 400 \
    --batch 200 \
    --nbs 200 \
    --imgsz 640 \
    --device 0 \
    --name yolov12m-kitti-400ep
```

**Or use single line to avoid syntax errors:**

```bash
python yolo3d.py train --model m --data /workspace/yolor2/YOLOv12-3D/kitti-3d.yaml --epochs 400 --batch 200 --nbs 200 --imgsz 640 --device 0 --name yolov12m-kitti-400ep
```

## What Changed

The fix (commit 9b38e65) handles the case where `batch["img"]` is a tuple/list instead of a tensor:

**File**: `ultralytics/models/yolo/detect3d/val.py`

**Change**: Lines 38-42 now check and convert:
```python
# Check if batch["img"] is a tuple or list (should be a tensor)
if "img" in batch:
    if isinstance(batch["img"], (tuple, list)):
        # If it's a tuple/list, convert to tensor by stacking
        batch["img"] = torch.stack(list(batch["img"]), 0)
```

This ensures `batch["img"]` is always a tensor before the parent class tries to access `.shape`.

## Verification

After pulling and restarting training, you should see:

1. ✅ Training completes epoch 1
2. ✅ Validation runs without errors
3. ✅ Metrics displayed:
   ```
   Class     Images  Instances  Box(P  R  mAP50  mAP50-95)  Depth  Dim  Rot
   all       7481    25000      0.xxx  ...                   N/A    N/A  N/A
   ```

## Troubleshooting

### If git pull fails with "local changes"

```bash
# Stash your changes
git stash

# Pull latest
git pull origin main

# Reapply your changes (if needed)
git stash pop
```

### If validation still fails

Check the exact error message and file/line number. The error should no longer be about `'tuple' object has no attribute 'shape'`.

### If you modified code locally

Your local modifications may conflict with the update. Either:
1. Commit your changes first, then pull
2. Or use `git stash` to save them temporarily

## Recent Commits on GitHub

These fixes are now available:

- **d436946**: Command syntax guide (bash backslash documentation)
- **711cea8**: Validation status documentation
- **9b38e65**: Fix validation tuple error ← **YOU NEED THIS!**
- **672cbb3**: Gradient accumulation control with --nbs
- **4e4635f**: Validation batch type checking

## Summary

1. **Run `git pull origin main`** on your server
2. **Verify the fix is present** with grep command above
3. **Restart training** with correct syntax
4. **Training should now complete** without validation errors!

## Quick Commands

```bash
# On your server
cd /workspace/yolo8/YOLOv12-3D
git pull origin main

# Verify fix
grep -A 5 "Check if batch\[\"img\"\] is a tuple" ultralytics/models/yolo/detect3d/val.py

# Train with corrected command
python yolo3d.py train --model m --data /workspace/yolor2/YOLOv12-3D/kitti-3d.yaml --epochs 400 --batch 200 --nbs 200 --imgsz 640 --device 0 --name yolov12m-kitti-400ep
```

That's it! The validation error will be fixed after you pull the latest code.
