# GPU Memory Management and Batch Size

## The Issue: Batch Size Doesn't Affect GPU Memory

When you change the `--batch` parameter but GPU memory usage stays the same, this is because of **gradient accumulation**.

## What is Gradient Accumulation?

Gradient accumulation is a technique that allows training with a large **effective batch size** while using a smaller **actual batch size** per forward pass.

### How It Works

Instead of updating weights after every batch, gradients are accumulated over multiple batches:

```
For 4 batches:
  batch 1: forward → loss → backward (accumulate gradients)
  batch 2: forward → loss → backward (accumulate gradients)
  batch 3: forward → loss → backward (accumulate gradients)
  batch 4: forward → loss → backward (accumulate gradients)
  → optimizer step (update weights)
  → zero gradients
```

This simulates training with `4 × batch_size` while only loading `batch_size` images in GPU memory at once.

## Why YOLOv12-3D Uses Gradient Accumulation

The code automatically calculates accumulation steps to maintain a **nominal batch size (nbs)** of **64**:

```python
# In ultralytics/engine/trainer.py line 301
accumulate = max(round(nbs / batch_size), 1)
```

### Examples:

| Batch Size | Nominal Batch (nbs) | Accumulation Steps | Effective Batch | GPU Memory Use |
|------------|---------------------|-------------------|-----------------|----------------|
| 2          | 64                  | 32                | 64              | Low (2 images) |
| 4          | 64                  | 16                | 64              | Low (4 images) |
| 8          | 64                  | 8                 | 64              | Medium (8 images) |
| 16         | 64                  | 4                 | 64              | High (16 images) |
| 32         | 64                  | 2                 | 64              | Very High (32 images) |
| 64         | 64                  | 1                 | 64              | Maximum (64 images) |

**Key Point:** All configurations above train with the **same effective batch size of 64**, which is why:
- Training convergence is similar
- Loss values are comparable
- But GPU memory usage differs

## Solutions

### Solution 1: Disable Gradient Accumulation (Maximize GPU Usage)

Set `--nbs` equal to your batch size to disable accumulation:

```bash
# Batch size = 8, no accumulation
python yolo3d.py train \
    --model s \
    --data ultralytics/cfg/datasets/kitti-3d.yaml \
    --batch 8 \
    --nbs 8 \
    --epochs 100 \
    --device 0
```

**Effect:**
- Accumulation = 8 / 8 = 1 (disabled)
- GPU memory scales with batch size
- Faster training (fewer backward passes)
- Different training dynamics (smaller effective batch)

### Solution 2: Set Custom Nominal Batch Size

Use a larger `--nbs` to control accumulation manually:

```bash
# Batch size = 8, effective batch = 32
python yolo3d.py train \
    --model s \
    --data ultralytics/cfg/datasets/kitti-3d.yaml \
    --batch 8 \
    --nbs 32 \
    --epochs 100 \
    --device 0
```

**Effect:**
- Accumulation = 32 / 8 = 4 steps
- Effective batch size = 32
- GPU memory for 8 images only

### Solution 3: Use Default (Recommended)

Keep the default `nbs=64` for stable training:

```bash
python yolo3d.py train \
    --model s \
    --data ultralytics/cfg/datasets/kitti-3d.yaml \
    --batch 8 \
    --epochs 100 \
    --device 0
```

**Effect:**
- Automatic accumulation for effective batch = 64
- Proven stable for YOLO training
- Good convergence characteristics

## Choosing the Right Batch Size

### For Limited GPU Memory (e.g., 4GB-8GB)

```bash
--batch 2 --nbs 64  # or --nbs 2 to disable accumulation
```

- Use small batch (2-4)
- Keep `nbs=64` for stable training
- Or set `nbs=batch` to use all available memory

### For Medium GPU Memory (e.g., 12GB-16GB)

```bash
--batch 8 --nbs 64
```

- Batch size 8-16 recommended
- Default `nbs=64` works well

### For Large GPU Memory (e.g., 24GB+)

```bash
--batch 16 --nbs 64
# or disable accumulation
--batch 32 --nbs 32
```

- Use larger batches (16-32)
- Can disable accumulation with `--nbs` = `--batch`

## Monitoring GPU Memory

### Check Current Memory Usage

```bash
# On Linux/Mac
watch nvidia-smi

# On Windows
nvidia-smi -l 1
```

### Expected Memory Usage (YOLOv12s on 640×640 images)

| Batch Size | Approximate GPU Memory |
|------------|------------------------|
| 1          | ~2-3 GB                |
| 2          | ~3-4 GB                |
| 4          | ~5-6 GB                |
| 8          | ~8-10 GB               |
| 16         | ~14-16 GB              |
| 32         | ~24-28 GB              |

*Note: Actual memory usage depends on model size (n/s/m/l/x), image size, and 3D head complexity.*

## Training Speed Trade-offs

### With Gradient Accumulation (Default)
- **Pros:**
  - Stable training with large effective batch
  - Works on limited GPU memory
  - Consistent results
- **Cons:**
  - Slower (more backward passes before optimizer step)
  - More iterations per epoch

### Without Gradient Accumulation
- **Pros:**
  - Faster training (optimizer updates every batch)
  - Simpler gradient flow
- **Cons:**
  - Requires more GPU memory
  - May need to adjust learning rate for smaller batch
  - Different convergence behavior

## Recommended Settings

### For Maximum Training Speed (if you have enough GPU memory)

```bash
python yolo3d.py train \
    --model s \
    --data ultralytics/cfg/datasets/kitti-3d.yaml \
    --batch 16 \
    --nbs 16 \
    --epochs 100 \
    --device 0
```

### For Maximum GPU Memory Efficiency

```bash
python yolo3d.py train \
    --model s \
    --data ultralytics/cfg/datasets/kitti-3d.yaml \
    --batch 2 \
    --nbs 64 \
    --epochs 100 \
    --device 0
```

### For Balanced Approach (Recommended)

```bash
python yolo3d.py train \
    --model s \
    --data ultralytics/cfg/datasets/kitti-3d.yaml \
    --batch 8 \
    --nbs 64 \
    --epochs 100 \
    --device 0
```

## Summary

**Why doesn't batch size affect GPU memory?**
- Because gradient accumulation keeps the effective batch constant at `nbs=64`

**How to make batch size affect GPU memory?**
- Set `--nbs` equal to `--batch` to disable accumulation
- Or adjust `--nbs` to control effective batch size

**What's the best setting?**
- Default (`nbs=64`) is recommended for stable YOLO training
- Adjust `--batch` based on your GPU memory
- Only disable accumulation if you need maximum speed and have enough memory

## Reference

- Gradient Accumulation: `ultralytics/engine/trainer.py:301`
- Batch Size Parameter: `yolo3d.py:415`
- Nominal Batch Size: `yolo3d.py:416`
