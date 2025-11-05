# --valpercent Fix Documentation

## Problem

The `--valpercent` argument was not working during training. When users tried to use `--valpercent=10` to validate on only 10% of the validation data (for faster validation during training), the argument was being ignored.

## Root Causes

1. **Missing from CFG_FLOAT_KEYS**: The `valpercent` parameter was not registered in the configuration validation keys in `ultralytics/cfg/__init__.py`

2. **Missing from default.yaml**: The `valpercent` parameter was not defined in the default configuration file `ultralytics/cfg/default.yaml`

3. **Incorrect args access**: The train.py code was using `hasattr()` and `.get()` which didn't properly handle the args object

## Solution

### Files Modified:

#### 1. `ultralytics/cfg/__init__.py` (Line 160)
Added `valpercent` to CFG_FLOAT_KEYS:
```python
CFG_FLOAT_KEYS = {  # integer or float arguments, i.e. x=2 and x=2.0
    "warmup_epochs",
    "box",
    "cls",
    "dfl",
    "degrees",
    "shear",
    "time",
    "workspace",
    "batch",
    "valpercent",  # percentage of validation data to use (0-100)
}
```

#### 2. `ultralytics/cfg/default.yaml` (Line 49)
Added default value in Val/Test settings section:
```yaml
# Val/Test settings
val: True # (bool) validate/test during training
split: val # (str) dataset split to use for validation, i.e. 'val', 'test' or 'train'
valpercent: 100.0 # (float) percentage of validation data to use (1-100), useful for faster validation during training
```

#### 3. `ultralytics/models/yolo/detect3d/train.py` (Lines 45-53)
Fixed the valpercent handling:
```python
# Handle valpercent for validation mode
val_fraction = 1.0
if mode == "val":
    # Get valpercent from args (default to 100 if not set)
    valpercent = getattr(self.args, "valpercent", 100.0)
    if valpercent < 100.0:
        val_fraction = valpercent / 100.0
        if RANK in {-1, 0}:  # Log for single-GPU or rank 0 in DDP
            LOGGER.info(f"Using {valpercent}% ({val_fraction:.2f}) of validation data for validation")
```

## Usage

Now you can use `--valpercent` (or `valpercent=`) to speed up validation during training:

### Command Line:
```bash
# Use only 10% of validation data
yolo detect3d train data=kitti-3d.yaml model=yolov12n-3d.yaml epochs=100 valpercent=10

# Use 25% of validation data
yolo detect3d train data=kitti-3d.yaml model=yolov12n-3d.yaml epochs=100 valpercent=25

# Use 50% of validation data
yolo detect3d train data=kitti-3d.yaml model=yolov12n-3d.yaml epochs=100 valpercent=50

# Use all validation data (default)
yolo detect3d train data=kitti-3d.yaml model=yolov12n-3d.yaml epochs=100 valpercent=100
# or simply omit valpercent (defaults to 100)
yolo detect3d train data=kitti-3d.yaml model=yolov12n-3d.yaml epochs=100
```

### Python API:
```python
from ultralytics import YOLO

# Create model
model = YOLO("yolov12n-3d.yaml")

# Train with 10% validation data
model.train(
    data="kitti-3d.yaml",
    epochs=100,
    valpercent=10  # Only validate on 10% of val data
)

# Train with 25% validation data
model.train(
    data="kitti-3d.yaml",
    epochs=100,
    valpercent=25
)
```

## Benefits

### Speed Improvements

With KITTI 3D dataset (~3700 validation images):

| valpercent | Images Used | Val Time (approx) | Speedup |
|------------|-------------|-------------------|---------|
| 100 (default) | 3700 | 100% | 1x |
| 50 | 1850 | ~50% | ~2x faster |
| 25 | 925 | ~25% | ~4x faster |
| 10 | 370 | ~10% | ~10x faster |

### Use Cases

1. **Quick iterations during development**: Use `valpercent=10` or `valpercent=25` for rapid experimentation
2. **Hyperparameter tuning**: Validate on a subset to quickly evaluate different settings
3. **Early training monitoring**: Check if model is learning correctly without full validation
4. **Final training**: Use `valpercent=100` (default) for the actual training run to get accurate metrics

## Validation Behavior

- The `valpercent` parameter **only affects validation**, not training
- Training always uses all training data (controlled by the `fraction` parameter if needed)
- Validation images are selected from the beginning of the validation set (first N%)
- The subset is consistent across epochs (same images each time)

## Examples

### Fast Development Cycle:
```bash
# Quick test with 10% validation
yolo detect3d train data=kitti-3d.yaml model=yolov12n-3d.yaml epochs=10 valpercent=10

# Once hyperparameters look good, train with full validation
yolo detect3d train data=kitti-3d.yaml model=yolov12n-3d.yaml epochs=100 valpercent=100
```

### Hyperparameter Search:
```bash
# Test different learning rates with fast validation
yolo detect3d train data=kitti-3d.yaml model=yolov12n-3d.yaml epochs=50 lr0=0.001 valpercent=25
yolo detect3d train data=kitti-3d.yaml model=yolov12n-3d.yaml epochs=50 lr0=0.01 valpercent=25
yolo detect3d train data=kitti-3d.yaml model=yolov12n-3d.yaml epochs=50 lr0=0.1 valpercent=25
```

### Retraining with Inverse Sigmoid Depth:
```bash
# Fast retraining with 10% validation to check if it's working
yolo detect3d train data=kitti-3d.yaml model=yolov12n-3d.yaml epochs=10 valpercent=10

# Full retraining with 100% validation for final model
yolo detect3d train data=kitti-3d.yaml model=yolov12n-3d.yaml epochs=100 valpercent=100
```

## Notes

- **Default value**: 100.0 (use all validation data)
- **Valid range**: 1.0 to 100.0 (percentage)
- **Type**: Float (can use integers like `valpercent=10` or floats like `valpercent=12.5`)
- **Works with**: All YOLO modes that use validation (train, val)

## Testing

The fix has been verified to:
1. ✅ Accept valpercent argument without errors
2. ✅ Parse valpercent values correctly (10, 25, 50, 100)
3. ✅ Pass valpercent through to dataset creation
4. ✅ Create validation datasets with correct size based on percentage

You can verify the fix is working by checking the training logs for:
```
Using 10% (0.10) of validation data for validation
```

## Technical Details

### How it works:

1. User passes `valpercent=10` argument
2. Argument is validated by `check_cfg()` using `CFG_FLOAT_KEYS`
3. Trainer's `build_dataset()` method receives the valpercent value
4. For validation mode, `val_fraction = valpercent / 100.0` is calculated
5. KITTIDataset receives `val_fraction` parameter
6. BaseDataset's `get_img_files()` method uses val_fraction to select subset:
   ```python
   data_fraction = self.val_fraction if self.val_fraction is not None else self.fraction
   if data_fraction < 1:
       im_files = im_files[: round(len(im_files) * data_fraction)]
   ```

## Summary

The `--valpercent` feature is now fully functional and can be used to speed up validation during training by using only a percentage of the validation dataset. This is especially useful for quick iterations, hyperparameter tuning, and development work.

**Recommended workflow:**
1. Development: `valpercent=10` or `valpercent=25`
2. Hyperparameter tuning: `valpercent=25` or `valpercent=50`
3. Final training: `valpercent=100` (default) or omit the argument
