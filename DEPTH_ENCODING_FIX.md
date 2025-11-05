# Depth Encoding Fix for YOLOv12-3D

## Problem Analysis

Current implementation uses: `depth = sigmoid(x) * 100`

This causes:
- Mean error: -1.54m
- **High variance: ±8.61m std dev**
- Over-prediction for close objects (<15m): +50-160% error
- Under-prediction for far objects (>30m): -10-26% error

## Research-Based Solutions

### Method 1: Inverse Sigmoid (MonoFlex-style) ⭐ RECOMMENDED

Based on MonoFlex (CVPR 2021):
```python
# Encoding: d = 1/sigmoid(x) - 1
# Better distribution for [0, ∞) range

# In head.py (line 256 and decode_bboxes_3d line 290):
# Change from:
loc_z = params_3d[:, 2:3, :].sigmoid() * 100

# To:
loc_z = 1.0 / (params_3d[:, 2:3, :].sigmoid() + 1e-6) - 1.0  # Inverse sigmoid
# Clip to reasonable range
loc_z = loc_z.clamp(0, 100)

# In loss.py (line 892):
# Change from:
pred_depth = params_3d[:, :, 2:3].sigmoid() * 100

# To:
pred_depth = 1.0 / (params_3d[:, :, 2:3].sigmoid() + 1e-6) - 1.0
pred_depth = pred_depth.clamp(0, 100)
```

**Advantages:**
- Better for depth distribution (not biased toward middle)
- More sensitive to close objects
- Used in CVPR 2021 state-of-the-art method

**Disadvantages:**
- Requires retraining (different depth distribution)
- Need to update loss normalization

---

### Method 2: Log Depth (FCOS3D-style)

Based on FCOS3D (ICCV 2021):
```python
# Predicts log(depth), loss in linear space

# In head.py:
# Change from:
loc_z = params_3d[:, 2:3, :].sigmoid() * 100

# To:
loc_z = torch.exp(params_3d[:, 2:3, :].sigmoid() * 4.6)  # exp(sigmoid*ln(100))
# Range: exp(0) = 1m to exp(4.6) = 100m

# In loss.py:
# Change from:
pred_depth = params_3d[:, :, 2:3].sigmoid() * 100

# To:
pred_depth = torch.exp(params_3d[:, :, 2:3].sigmoid() * 4.6)
# Loss still computed in linear space (no change needed)
```

**Advantages:**
- More uniform distribution across log scale
- Better for wide depth ranges (1-100m)
- Proven effective in ICCV 2021

**Disadvantages:**
- Requires retraining
- Log space can be harder to debug

---

### Method 3: Monodepth-style Constrained Inverse

Based on Monodepth2 (ICCV 2019):
```python
# D = 1 / (a*sigmoid(x) + b) where a,b constrain range

# For KITTI depth range [1, 80] meters:
min_depth = 1.0
max_depth = 80.0
a = 1.0 / min_depth - 1.0 / max_depth  # = 0.9875
b = 1.0 / max_depth  # = 0.0125

# In head.py:
loc_z = 1.0 / (a * params_3d[:, 2:3, :].sigmoid() + b)

# In loss.py:
pred_depth = 1.0 / (a * params_3d[:, :, 2:3].sigmoid() + b)
```

**Advantages:**
- Mathematically constrains depth to [min, max]
- Better distribution than linear sigmoid
- No need for clamping

**Disadvantages:**
- Requires retraining
- Two hyperparameters to tune

---

### Method 4: Quick Fix - Reduced Range (NO RETRAINING)

```python
# Simply reduce the range to match KITTI distribution better

# In head.py (line 256):
# Change from:
loc_z = params_3d[:, 2:3, :].sigmoid() * 100

# To:
loc_z = params_3d[:, 2:3, :].sigmoid() * 70  # Better matches KITTI range

# In loss.py (line 892):
# Change from:
pred_depth = params_3d[:, :, 2:3].sigmoid() * 100

# To:
pred_depth = params_3d[:, :, 2:3].sigmoid() * 70
```

**Advantages:**
- ✅ NO RETRAINING NEEDED
- Simple one-line change
- Reduces bias toward 50m → now biased toward 35m

**Disadvantages:**
- Still has sigmoid S-curve bias
- Not optimal, but better than current

---

## Implementation Priority

1. **Immediate**: Apply **Method 4** (no retraining) to test if range adjustment helps
2. **Short-term**: Implement **Method 1** (Inverse Sigmoid) and retrain for best results
3. **Alternative**: Try **Method 2** (Log Depth) if Method 1 doesn't work well

## Testing

After applying any fix:
```bash
python check_depth_multiple.py
```

Expected improvements:
- Mean error: <1m
- Std deviation: <5m
- Close object error: <20%
- Far object error: <15%
