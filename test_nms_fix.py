#!/usr/bin/env python3
"""Test that NMS correctly preserves 3D parameters when nc is specified."""

import torch
from ultralytics.utils.ops import non_max_suppression


def test_nms_preserves_3d_params():
    """Test that NMS preserves 3D parameters when nc parameter is specified."""

    # Simulate predictions with 3D params
    # Shape: (batch=1, channels=4+nc+7, anchors=10)
    nc = 4  # KITTI has 4 classes
    n3d = 7  # 7 3D parameters
    n_anchors = 10
    batch_size = 1

    # Create fake predictions
    # Format: [bbox_dist(4*reg_max), cls_probs(nc), 3d_params(7)]
    # For simplicity, let's assume reg_max=16, so bbox = 64 channels
    # But after DFL decoding, bbox becomes 4 channels
    # So input to NMS: [bbox_xyxy(4), cls_probs(nc), 3d_params(7)]

    preds = torch.zeros((batch_size, 4 + nc + n3d, n_anchors))

    # Set up some fake detections
    # Detection 1: high confidence for class 0
    preds[0, 0:4, 0] = torch.tensor([100, 100, 200, 200])  # bbox (xywh format)
    preds[0, 4, 0] = 0.9  # class 0 confidence
    preds[0, 4+nc:4+nc+n3d, 0] = torch.tensor([10, 20, 30, 2, 1.5, 4, 1.57])  # 3D params

    # Detection 2: high confidence for class 1
    preds[0, 0:4, 1] = torch.tensor([300, 300, 400, 400])  # bbox
    preds[0, 5, 1] = 0.8  # class 1 confidence
    preds[0, 4+nc:4+nc+n3d, 1] = torch.tensor([15, 25, 35, 1.8, 1.2, 3.5, -1.2])  # 3D params

    # Detection 3: low confidence (should be filtered)
    preds[0, 0:4, 2] = torch.tensor([500, 500, 600, 600])  # bbox
    preds[0, 6, 2] = 0.1  # class 2 confidence (below threshold)
    preds[0, 4+nc:4+nc+n3d, 2] = torch.tensor([5, 10, 15, 1, 1, 1, 0])  # 3D params

    print("Input predictions shape:", preds.shape)
    print(f"  batch={batch_size}, channels={4+nc+n3d} (4 bbox + {nc} classes + {n3d} 3D params), anchors={n_anchors}")

    # Test WITHOUT nc parameter (BUG)
    print("\n" + "="*60)
    print("TEST 1: WITHOUT nc parameter (BUG - treats 3D params as classes)")
    print("="*60)
    outputs_bug = non_max_suppression(
        preds.clone(),
        conf_thres=0.25,
        iou_thres=0.45,
        multi_label=True,
        # nc parameter NOT specified
    )

    if len(outputs_bug) > 0 and outputs_bug[0].shape[0] > 0:
        print(f"Output shape: {outputs_bug[0].shape}")
        print(f"  Expected: (n_detections, 6 + {n3d}) = (n_detections, 13)")
        print(f"  Got: {outputs_bug[0].shape}")
        print(f"Number of detections: {outputs_bug[0].shape[0]}")
        print(f"First detection: {outputs_bug[0][0].tolist()}")
    else:
        print("No detections! NMS filtered everything out due to bug.")

    # Test WITH nc parameter (FIX)
    print("\n" + "="*60)
    print("TEST 2: WITH nc parameter (FIX - correctly preserves 3D params)")
    print("="*60)
    outputs_fix = non_max_suppression(
        preds.clone(),
        conf_thres=0.25,
        iou_thres=0.45,
        multi_label=True,
        nc=nc,  # FIX: Specify number of classes
    )

    if len(outputs_fix) > 0 and outputs_fix[0].shape[0] > 0:
        print(f"Output shape: {outputs_fix[0].shape}")
        print(f"  Expected: (n_detections, 6 + {n3d}) = (n_detections, 13)")
        print(f"  Got: {outputs_fix[0].shape}")
        print(f"Number of detections: {outputs_fix[0].shape[0]}")

        for i, det in enumerate(outputs_fix[0]):
            print(f"\nDetection {i+1}:")
            print(f"  BBox (xyxy): {det[0:4].tolist()}")
            print(f"  Confidence: {det[4].item():.3f}")
            print(f"  Class: {int(det[5].item())}")
            print(f"  3D params: {det[6:13].tolist()}")

        # Verify 3D params are preserved correctly
        if outputs_fix[0].shape[1] == 13:
            print("\n✅ SUCCESS: 3D parameters are preserved (13 channels total)")
        else:
            print(f"\n❌ FAIL: Expected 13 channels, got {outputs_fix[0].shape[1]}")
    else:
        print("❌ FAIL: No detections found")

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    bug_dets = outputs_bug[0].shape[0] if len(outputs_bug) > 0 else 0
    fix_dets = outputs_fix[0].shape[0] if len(outputs_fix) > 0 else 0

    print(f"Without nc parameter: {bug_dets} detections")
    print(f"With nc parameter: {fix_dets} detections")

    if fix_dets > 0 and outputs_fix[0].shape[1] == 13:
        print("\n✅ FIX VERIFIED: NMS now correctly preserves 3D parameters!")
        return True
    else:
        print("\n❌ FIX FAILED: 3D parameters not preserved correctly")
        return False


if __name__ == "__main__":
    success = test_nms_preserves_3d_params()
    exit(0 if success else 1)
