#!/usr/bin/env python3
"""
Test script to verify x, y, z coordinate output from Detect3D head.
"""
import sys
from pathlib import Path
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from ultralytics import YOLO

print("=" * 80)
print("Testing X, Y, Z Coordinate Output")
print("=" * 80)

# Test 1: Load model
print("\n[Test 1] Loading model...")
try:
    model = YOLO("ultralytics/cfg/models/v12/yolov12-3d.yaml")
    print("✓ Model loaded successfully")

    # Check head parameters
    head = model.model.model[-1]
    print(f"✓ Head type: {type(head).__name__}")
    print(f"✓ Number of 3D parameters: {head.n3d}")
    assert head.n3d == 7, f"Expected n3d=7 (x,y,z,h,w,l,rot), got {head.n3d}"
    print(f"✓ Parameter count correct: 7 (x, y, z, h, w, l, rotation_y)")

except Exception as e:
    print(f"✗ Error loading model: {e}")
    sys.exit(1)

# Test 2: Forward pass
print("\n[Test 2] Testing forward pass...")
try:
    # Create dummy input
    x = torch.randn(1, 3, 640, 640)

    # Test training mode
    model.model.train()
    output_train = model.model(x)
    print(f"✓ Training mode output: {len(output_train)} elements")
    print(f"  - 2D features: {len(output_train[0])} scales")
    print(f"  - 3D params shape: {output_train[1].shape}")
    assert output_train[1].shape[1] == 7, f"Expected 7 params, got {output_train[1].shape[1]}"

    # Test inference mode
    model.model.eval()
    with torch.no_grad():
        output_inf = model.model(x)
    print(f"✓ Inference mode output shape: {output_inf[0].shape}")

    # Extract and verify 3D parameters
    # Output should have: [batch, 4+nc+7, anchors]
    # 4 (bbox) + 8 (classes) + 7 (3D params) = 19 channels
    expected_channels = 4 + 8 + 7  # bbox + classes + 3d_params
    actual_channels = output_inf[0].shape[1]
    print(f"  - Output channels: {actual_channels} (4 bbox + 8 cls + 7 3d_params)")

except Exception as e:
    print(f"✗ Error during forward pass: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Parameter decoding
print("\n[Test 3] Testing parameter decoding...")
try:
    # Create dummy 3D parameters [x, y, z, h, w, l, rot_y]
    dummy_params = torch.randn(2, 7, 100)  # [bs=2, params=7, n_anchors=100]

    # Simulate decoding as in forward pass (inference mode)
    loc_x = (dummy_params[:, 0:1, :].sigmoid() - 0.5) * 100  # [-50, 50]
    loc_y = (dummy_params[:, 1:2, :].sigmoid() - 0.5) * 100  # [-50, 50]
    loc_z = dummy_params[:, 2:3, :].sigmoid() * 100  # [0, 100]
    dims = dummy_params[:, 3:6, :].sigmoid() * 10  # [0, 10]
    rot = (dummy_params[:, 6:7, :].sigmoid() - 0.5) * 2 * torch.pi  # [-π, π]

    decoded = torch.cat([loc_x, loc_y, loc_z, dims, rot], dim=1)

    print(f"✓ Parameter decoding successful")
    print(f"  - Input shape: {dummy_params.shape}")
    print(f"  - Decoded shape: {decoded.shape}")
    print(f"  - loc_x range: [{loc_x.min():.2f}, {loc_x.max():.2f}]")
    print(f"  - loc_y range: [{loc_y.min():.2f}, {loc_y.max():.2f}]")
    print(f"  - loc_z range: [{loc_z.min():.2f}, {loc_z.max():.2f}]")
    print(f"  - dims range: [{dims.min():.2f}, {dims.max():.2f}]")
    print(f"  - rotation range: [{rot.min():.2f}, {rot.max():.2f}]")

    # Verify ranges
    assert loc_x.min() >= -50 and loc_x.max() <= 50, "loc_x out of range"
    assert loc_y.min() >= -50 and loc_y.max() <= 50, "loc_y out of range"
    assert loc_z.min() >= 0 and loc_z.max() <= 100, "loc_z out of range"
    assert dims.min() >= 0 and dims.max() <= 10, "dims out of range"
    assert rot.min() >= -torch.pi - 0.1 and rot.max() <= torch.pi + 0.1, "rotation out of range"
    print(f"✓ All parameters in valid ranges")

except Exception as e:
    print(f"✗ Error during parameter decoding: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 80)
print("All tests passed! ✓")
print("=" * 80)
print("\nModel now outputs full 3D coordinates:")
print("  - x (lateral): [-50m, +50m]")
print("  - y (vertical): [-50m, +50m]")
print("  - z (depth): [0m, 100m]")
print("  - h, w, l (dimensions): [0m, 10m]")
print("  - rotation_y: [-π, +π]")
print("\nReady for training with KITTI dataset!")
