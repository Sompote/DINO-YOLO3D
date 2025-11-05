# Fast Validation During Training

## Overview

The `--valpercent` argument allows you to validate on only a percentage of your validation data during training, significantly speeding up the validation phase while still providing meaningful metrics.

## Usage

```bash
# Use 10% of validation data for faster validation
python yolo3d.py train --model yolov12s-3d.yaml --data kitti-3d.yaml --epochs 100 --valpercent 10

# Use 25% of validation data
python yolo3d.py train --model yolov12s-3d.yaml --data kitti-3d.yaml --epochs 100 --valpercent 25

# Use full validation data (default)
python yolo3d.py train --model yolov12s-3d.yaml --data kitti-3d.yaml --epochs 100 --valpercent 100
# Or simply omit the argument:
python yolo3d.py train --model yolov12s-3d.yaml --data kitti-3d.yaml --epochs 100
```

## When to Use

### ✅ Good Use Cases

1. **Rapid Prototyping**
   - Testing model architectures quickly
   - Experimenting with hyperparameters
   - Early training stages to check if model is learning

2. **Development Iteration**
   - Debugging training pipeline
   - Quick sanity checks
   - Faster feedback during development

3. **Limited Resources**
   - Training on machines with limited memory
   - When validation takes too long
   - Cloud training with hourly costs

### ❌ When NOT to Use

1. **Final Training Runs**
   - Use 100% for production models
   - When reporting final metrics
   - For model comparison/benchmarking

2. **Model Selection**
   - When choosing best checkpoint
   - Critical decision points
   - Publishing results

## Recommended Values

| Percentage | Use Case | Speed Up | Reliability |
|------------|----------|----------|-------------|
| 100% | Production training, final metrics | 1x (baseline) | ⭐⭐⭐⭐⭐ |
| 50% | Intermediate testing | ~2x faster | ⭐⭐⭐⭐ |
| 25% | Development iteration | ~4x faster | ⭐⭐⭐ |
| 10% | Rapid prototyping | ~10x faster | ⭐⭐ |
| <10% | Quick sanity check | >10x faster | ⭐ (not reliable) |

## Example Speedup

With KITTI 3D validation set (7481 images):

| Percentage | Images Used | Approx. Time | Use Case |
|------------|-------------|--------------|----------|
| 100% | 7481 | ~15 minutes | Final training |
| 50% | 3740 | ~7-8 minutes | Testing |
| 25% | 1870 | ~4 minutes | Development |
| 10% | 748 | ~1.5 minutes | Quick check |

*Times are approximate and depend on hardware, batch size, and model size.*

## How It Works

The `--valpercent` parameter is converted to a fraction and passed to the ultralytics trainer:

```python
# User specifies: --valpercent 10
# Internally converts to: fraction=0.1
# Ultralytics randomly samples 10% of validation data each epoch
```

**Important**: The sampling is **random** each epoch, so you get different subsets for variety.

## Configuration Display

The training configuration will show your validation percentage:

```
Training Configuration
================================================================================
  Model:           yolov12s-3d.yaml
  Dataset:         kitti-3d.yaml
  Epochs:          100
  Batch Size:      16
  Val Data:        10%          ← Shows percentage
  ...
```

**No Dataset Modification Required!** The `--valpercent` option works with your existing dataset configuration:

```yaml
# kitti-3d.yaml - No changes needed!
path: /path/to/kitti
train: training/image_2
val: training/image_2  ← Works with same directory as train
nc: 8
names: [...]
```

## Warnings

### Low Percentage Warning

If you use less than 10%, you'll see a warning:

```
⚠ Using only 5% of validation data - metrics may not be representative!
ℹ Recommended minimum: --valpercent 10
```

### Invalid Value Error

Values must be between 1 and 100:

```bash
# ERROR: Invalid value
python yolo3d.py train --model s --data kitti-3d.yaml --valpercent 0
python yolo3d.py train --model s --data kitti-3d.yaml --valpercent 150
```

## Best Practices

1. **Start Fast, Finish Slow**
   ```bash
   # Phase 1: Quick iteration (epochs 1-50)
   python yolo3d.py train --model s --data kitti-3d.yaml --epochs 50 --valpercent 10

   # Phase 2: Fine-tune with full validation (epochs 51-100)
   python yolo3d.py train --model last.pt --data kitti-3d.yaml --epochs 100 --valpercent 100
   ```

2. **Progressive Validation**
   - First 30 epochs: --valpercent 10
   - Next 40 epochs: --valpercent 25
   - Final 30 epochs: --valpercent 100

3. **Development vs Production**
   - **Dev**: Use 10-25% for faster iteration
   - **Prod**: Always use 100% for final model

4. **No Dataset Preparation Needed**
   - Works with existing kitti-3d.yaml (even if train/val point to same directory)
   - No need to create separate val/ directory
   - No need to copy or split data files
   - Just use --valpercent and it works!

## Comparison with Other Speedup Methods

| Method | Speedup | Impact on Metrics | When to Use |
|--------|---------|-------------------|-------------|
| `--valpercent 10` | ~10x validation | Slight variance | Frequent validation |
| Reduce `--batch` | Minor | None | Memory constrained |
| Skip validation | Infinite | No metrics | Not recommended |
| Reduce `--epochs` | Linear | None | Quick testing |

## Technical Details

### Implementation

The `--valpercent` parameter is passed through the dataset loader without copying data:

```python
# In detect3d/train.py - build_dataset()
if mode == "val" and hasattr(self.args, "valpercent"):
    if self.args.valpercent < 100.0:
        val_fraction = self.args.valpercent / 100.0

# In ultralytics/data/base.py - get_img_files()
data_fraction = self.val_fraction if self.val_fraction is not None else self.fraction
if data_fraction < 1:
    im_files = im_files[: round(len(im_files) * data_fraction)]
```

### Data Sampling

- **No Data Copying**: Loads only subset directly from training directory
- **Subset Selection**: Uses first N% of validation directory
- **Reproducible**: Set `--seed` for consistent subset selection
- **Memory Efficient**: Only loads subset into memory, not full dataset

### Metrics Computation

- All metrics (mAP, precision, recall) computed on sampled data
- 3D IoU, location error, etc. use sampled subset
- Results extrapolated from sample (not scaled)

## Examples

### Quick Development Iteration
```bash
python yolo3d.py train \
  --model yolov12n-3d.yaml \
  --data kitti-3d.yaml \
  --epochs 10 \
  --batch 8 \
  --valpercent 10 \
  --name quick_test
```

### Production Training
```bash
python yolo3d.py train \
  --model yolov12m-3d.yaml \
  --data kitti-3d.yaml \
  --epochs 200 \
  --batch 16 \
  --valpercent 100 \
  --patience 50 \
  --name production_run
```

### Balanced Approach
```bash
python yolo3d.py train \
  --model yolov12s-3d.yaml \
  --data kitti-3d.yaml \
  --epochs 100 \
  --batch 16 \
  --valpercent 25 \
  --name balanced
```

## FAQ

**Q: Will this affect training quality?**
A: No, only validation speed. Training uses full dataset.

**Q: Are metrics less accurate with lower percentages?**
A: Yes, but trends are still visible. Use ≥10% for reliability.

**Q: Can I change percentage mid-training?**
A: No, but you can resume with different percentage.

**Q: Does this save GPU memory?**
A: Slightly, but main benefit is speed.

**Q: What's the default?**
A: 100% (full validation data)

**Q: Do I need to create a separate val/ directory?**
A: No! The val_fraction parameter loads only a subset directly from your existing dataset. Works even if train and val point to the same directory in your YAML.

**Q: Does it copy data files?**
A: No! It loads only the first N% of files directly from the directory. No disk space wasted, no copying time.

**Q: What if I want different subsets each epoch?**
A: Set a random seed with --seed for reproducibility, or omit it for different subsets each time.

## See Also

- [CLI Guide](CLI_GUIDE.md) - Full command reference
- [Training Guide](TRAINING_READY.md) - Complete training documentation
- [Quick Start](QUICK_START.md) - Getting started guide
