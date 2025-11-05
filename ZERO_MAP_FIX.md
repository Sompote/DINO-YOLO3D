# Fix for Zero mAP2D and mAP3D During Training

## Problem

During training validation, `map2d` and `map3D` were always showing as **0.0000**, even though the model was learning and loss was decreasing.

## Root Cause

The bug was in `ultralytics/models/yolo/detect3d/val.py` at the `postprocess()` method (line ~234).

### Technical Details

When calling `non_max_suppression()`, the code did not pass the `nc` (number of classes) parameter. This caused NMS to miscalculate the tensor structure:

1. **Input tensor shape**: `(batch, 4+nc+7, num_anchors)`
   - 4 bbox coordinates
   - nc class probabilities (e.g., 4 for KITTI)
   - 7 3D parameters (x, y, z, h, w, l, rotation_y)

2. **Without `nc` parameter**, NMS calculates:
   ```python
   nc = prediction.shape[1] - 4 = (4 + nc + 7) - 4 = nc + 7  # WRONG!
   ```
   - NMS thinks there are `nc + 7` classes instead of `nc` classes
   - The 7 3D parameters are treated as **class channels**
   - Since decoded 3D params have values like 10.0, 30.0, 50.0 (much larger than conf threshold 0.25)
   - NMS treats them as high-confidence "classes" and creates bogus detections
   - Real class predictions get corrupted
   - Result: **Zero mAP because detections are nonsense**

3. **With `nc` parameter** (the fix), NMS calculates:
   ```python
   nc = nc  # Correctly specified as 4 for KITTI
   nm = prediction.shape[1] - nc - 4 = 7  # Correctly identifies 7 extra channels
   ```
   - NMS knows there are only `nc` classes
   - The 7 additional channels are treated as "masks" (extra data to preserve)
   - NMS output: `(num_detections, 6 + nm)` = `(num_detections, 13)` ✓
   - 3D parameters are correctly preserved!

## The Fix

**File**: `ultralytics/models/yolo/detect3d/val.py`

**Change**: Add `nc=self.nc` parameter to `non_max_suppression()` call:

```python
outputs = non_max_suppression(
    preds_2d_for_nms,
    self.args.conf,
    self.args.iou,
    labels=self.lb,
    multi_label=True,
    agnostic=self.args.single_cls or self.args.agnostic_nms,
    max_det=self.args.max_det,
    nc=self.nc,  # ← FIX: Specify number of classes to preserve 3D params
)
```

## Verification

Run the test script to verify the fix:

```bash
python test_nms_fix.py
```

**Expected output**:
- Without `nc`: 21+ bogus detections with only 6 channels
- With `nc`: 2 correct detections with 13 channels (6 standard + 7 3D params)

## Impact

After this fix:
- ✅ NMS correctly preserves 3D parameters during validation
- ✅ 3D IoU can be computed correctly for mAP calculation
- ✅ map2d and map3D will show non-zero values during training
- ✅ KITTI mAP metrics (easy/moderate/hard) will be computed correctly

## Testing on Cloud

Since you train on the cloud, after applying this fix:

1. Start a new training run or resume existing training
2. Watch the validation metrics during training
3. You should now see non-zero values for:
   - `metrics/mAP50(B)` - 2D mAP at IoU 0.5
   - `metrics/mAP50-95(B)` - 2D mAP averaged over IoU 0.5:0.95
   - `kitti/moderate/mAP` - KITTI 3D mAP for moderate difficulty
   - `kitti/easy/mAP` - KITTI 3D mAP for easy difficulty
   - `kitti/hard/mAP` - KITTI 3D mAP for hard difficulty

## Related Files

- `ultralytics/models/yolo/detect3d/val.py` - Main fix location
- `ultralytics/utils/ops.py` - NMS implementation
- `test_nms_fix.py` - Verification test script

## Commit Message

```
Fix zero mAP by specifying nc parameter in NMS

- Add nc=self.nc parameter to non_max_suppression() call
- Without this, NMS miscalculates and treats 3D params as classes
- This caused all detections to be corrupted and mAP to be zero
- With the fix, 3D params are correctly preserved as extra channels
- Verified with test_nms_fix.py showing 2 correct detections vs 21 bogus

Fixes #zero-map-during-training
```

## Credits

Fixed by Claude Code on 2025-11-05
Issue reported by: User (training on cloud)
