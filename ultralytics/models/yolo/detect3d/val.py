# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import math
from collections import defaultdict

import numpy as np
import torch

from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import LOGGER
from ultralytics.utils.metrics import DetMetrics, box_iou, compute_ap
from ultralytics.utils.ops import scale_boxes


class Detection3DValidator(DetectionValidator):
    """
    A class extending the DetectionValidator class for validation based on a 3D detection model.

    Example:
        ```python
        from ultralytics.models.yolo.detect3d import Detection3DValidator

        args = dict(model="yolov12n-3d.pt", data="kitti.yaml")
        validator = Detection3DValidator(args=args)
        validator()
        ```
    """

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        """Initialize Detection3DValidator with 3D detection specific settings."""
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        self.args.task = "detect3d"
        self.difficulties = ("easy", "moderate", "hard")
        # KITTI official IoU thresholds: 0.7 for Car/Truck, 0.5 for Pedestrian/Cyclist
        self.kitti_iou_thresholds = {0: 0.7, 1: 0.7, 2: 0.5, 3: 0.5}
        self.n3d = 7  # Updated: x, y, z, h, w, l, rotation_y (was 5: z, h, w, l, rotation_y)
        # KITTI uses 40 recall positions (updated 2019) instead of 11 from Pascal VOC
        self.kitti_recall_positions = 40
        self._reset_kitti_metrics()

    def _reset_kitti_metrics(self):
        """Reset containers used for KITTI-style evaluation."""
        self.depth_errors = defaultdict(list)
        self.dim_errors = defaultdict(list)
        self.rot_errors = defaultdict(list)
        self.kitti_stats = None
        self.kitti_gt_counts = None

    def _prepare_pred(self, pred, pbatch):
        """
        Override to preserve 3D parameters (not just first 4 bbox channels).

        Args:
            pred: Prediction tensor with shape (n, 13) - [x, y, w, h, conf, cls, x_3d, y_3d, z_3d, h_3d, w_3d, l_3d, rot_y]
            pbatch: Prepared batch dict with image info

        Returns:
            predn: Scaled predictions preserving all channels
        """
        predn = pred.clone()
        # Scale only the 2D bbox coordinates (first 4 channels)
        # The 3D parameters (channels 6-12) don't need scaling as they're in world coordinates
        scale_boxes(
            pbatch["imgsz"], predn[:, :4], pbatch["ori_shape"], ratio_pad=pbatch["ratio_pad"]
        )  # native-space pred
        return predn

    def preprocess(self, batch):
        """Preprocesses batch of images for YOLO training."""
        if not isinstance(batch, dict):
            raise TypeError(f"Expected batch to be a dict, got {type(batch)}")
        if "img" not in batch:
            raise KeyError("Batch is missing required 'img' key for validation.")

        def _flatten_items(items):
            for item in items:
                if isinstance(item, (list, tuple)):
                    yield from _flatten_items(item)
                else:
                    yield item

        def _ensure_tensor(value, *, stack=False, cat=False):
            if value is None:
                return None
            if isinstance(value, torch.Tensor):
                if stack and value.dim() == 3:
                    return value.unsqueeze(0)
                return value
            if isinstance(value, (list, tuple)):
                flattened = list(_flatten_items(value))
                if not flattened:
                    return torch.empty(0)
                tensors = [
                    item if isinstance(item, torch.Tensor) else torch.as_tensor(item)
                    for item in flattened
                ]
                if stack:
                    stacked = torch.stack(tensors, dim=0)
                    if stacked.dim() == 3:
                        stacked = stacked.unsqueeze(0)
                    return stacked
                if cat:
                    try:
                        return torch.cat(tensors, dim=0)
                    except RuntimeError:
                        return torch.stack(tensors, dim=0)
                return torch.as_tensor(tensors)
            tensor = torch.as_tensor(value)
            if stack and tensor.dim() == 3:
                tensor = tensor.unsqueeze(0)
            return tensor

        batch["img"] = _ensure_tensor(batch["img"], stack=True)
        if not isinstance(batch["img"], torch.Tensor):
            raise TypeError(f"Unable to convert batch['img'] to tensor (got {type(batch['img'])}).")
        if batch["img"].dim() == 3:
            batch["img"] = batch["img"].unsqueeze(0)
        elif batch["img"].dim() > 4:
            batch["img"] = batch["img"].reshape(-1, *batch["img"].shape[-3:])

        for key in ["batch_idx", "cls", "bboxes"]:
            if key in batch:
                batch[key] = _ensure_tensor(batch[key], cat=True)

        batch["img"] = batch["img"].to(self.device, non_blocking=True)
        batch["img"] = (batch["img"].half() if self.args.half else batch["img"].float()) / 255

        for key in ["batch_idx", "cls", "bboxes"]:
            if key in batch and isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(self.device)

        if self.args.save_hybrid and "bboxes" in batch and "cls" in batch:
            height, width = batch["img"].shape[2:]
            nb = len(batch["img"])
            bboxes = batch["bboxes"] * torch.tensor((width, height, width, height), device=self.device)
            self.lb = [
                torch.cat([batch["cls"][batch["batch_idx"] == i], bboxes[batch["batch_idx"] == i]], dim=-1)
                for i in range(nb)
            ]

        for key in ["dimensions_3d", "location_3d", "rotation_y", "alpha", "truncation", "occlusion", "bbox_height"]:
            if key in batch and batch[key] is not None:
                batch[key] = _ensure_tensor(batch[key], cat=True)
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(self.device, non_blocking=True).float()

        return batch

    def postprocess(self, preds):
        """Apply Non-maximum suppression to prediction outputs."""
        # Handle 3D predictions
        preds_3d = None
        if isinstance(preds, (list, tuple)):
            preds_2d = preds[0]
            extra = preds[1] if len(preds) > 1 else None

            # During training mode (validation within training):
            # preds[0] is a list of tensors (not concatenated yet)
            # We need to concatenate them first
            if isinstance(preds_2d, list):
                # Concatenate all detection layers along anchor dimension
                # This is what _inference() does, but we need to do it manually
                # because we're in training mode
                # Each layer: (batch, channels, h, w) -> (batch, channels, h*w)
                # Then concat along dim=2 (anchor dimension)
                shape = preds_2d[0].shape  # BCHW
                x_cat = torch.cat([xi.view(shape[0], shape[1], -1) for xi in preds_2d], dim=2)
                # Now x_cat has shape (batch, channels, total_anchors)
                # This matches what _inference() returns
                preds_2d = x_cat

            if self.args.verbose:
                LOGGER.info(f"DEBUG postprocess: preds is tuple with {len(preds)} elements")
                LOGGER.info(f"DEBUG postprocess: preds[0] type: {type(preds_2d)}, shape: {preds_2d.shape}")
                LOGGER.info(f"DEBUG postprocess: extra type: {type(extra)}")

            if isinstance(extra, (list, tuple)):
                # Expect tuple like (features, params_3d)
                if self.args.verbose:
                    LOGGER.info(f"DEBUG postprocess: extra is tuple with {len(extra)} elements")
                    if len(extra) > 0:
                        LOGGER.info(f"DEBUG postprocess: extra[0] shape: {extra[0].shape if isinstance(extra[0], torch.Tensor) else 'N/A'}")
                    if len(extra) > 1:
                        LOGGER.info(f"DEBUG postprocess: extra[1] shape: {extra[1].shape if isinstance(extra[1], torch.Tensor) else 'N/A'}")
                if extra and isinstance(extra[-1], torch.Tensor):
                    preds_3d = extra[-1]
            elif isinstance(extra, torch.Tensor):
                preds_3d = extra
        else:
            preds_2d = preds

        if self.args.verbose:
            LOGGER.info(f"DEBUG postprocess: Final preds_2d shape: {preds_2d.shape if preds_2d is not None else 'None'}")
            LOGGER.info(f"DEBUG postprocess: Final preds_3d shape: {preds_3d.shape if preds_3d is not None else 'None'}")

        # Extract and append 3D params based on mode
        if preds_3d is not None and isinstance(preds_2d, torch.Tensor):
            # preds_3d has shape (batch, n3d, num_anchors) from Detect3D.forward()
            # We need to append these to preds_2d before NMS

            # Check if we're in inference mode (where 3D params are already in preds_2d)
            if preds_2d.shape[1] > 6:
                # 3D params are already in preds_2d (decoded format from Detect3D.forward())
                # During inference, Detect3D.forward() already concatenated decoded 3D params
                # So we don't need to do anything - just use preds_2d as is!
                preds_2d_for_nms = preds_2d
                if self.args.verbose:
                    LOGGER.info(f"DEBUG postprocess: Using existing decoded 3D params in preds_2d")
                    LOGGER.info(f"DEBUG postprocess: Shape: {preds_2d_for_nms.shape}")
            else:
                # No 3D params in preds_2d (training mode)
                # We need to append raw 3D params
                preds_2d_for_nms = torch.cat([preds_2d, preds_3d], dim=1)
                if self.args.verbose:
                    LOGGER.info(f"DEBUG postprocess: Appending RAW 3D to 2D (training mode)")
                    LOGGER.info(f"DEBUG postprocess: New shape: {preds_2d_for_nms.shape}")
        else:
            preds_2d_for_nms = preds_2d
            if self.args.verbose:
                LOGGER.warning(f"DEBUG postprocess: No 3D params found!")

        # Apply NMS while preserving 3D params
        from ultralytics.utils.ops import non_max_suppression

        # CRITICAL FIX: Pass nc parameter so NMS knows there are only self.nc classes,
        # not self.nc + 7. Without this, NMS miscalculates and treats 3D params as class channels!
        # With nc specified correctly, NMS will automatically preserve the 3D params as extra channels
        # (treating them as "masks" internally), so output will be (n_detections, 6 + 7).
        outputs = non_max_suppression(
            preds_2d_for_nms,
            self.args.conf,
            self.args.iou,
            labels=self.lb,
            multi_label=True,
            agnostic=self.args.single_cls or self.args.agnostic_nms,
            max_det=self.args.max_det,
            nc=self.nc,  # FIX: Specify number of classes to preserve 3D params
        )

        if self.args.verbose and outputs:
            LOGGER.info(f"DEBUG postprocess: NMS output shape for batch[0]: {outputs[0].shape if len(outputs) > 0 else 'empty'}")
            if len(outputs) > 0 and outputs[0].shape[1] > 6:
                LOGGER.info(f"DEBUG postprocess: NMS output has {outputs[0].shape[1]} channels (expected > 6)")
        return outputs

    def init_metrics(self, model):
        """Initialize standard and KITTI-specific metrics."""
        super().init_metrics(model)
        self.kitti_stats = {
            diff: {cls_id: {"conf": [], "tp": []} for cls_id in range(self.nc)} for diff in self.difficulties
        }
        self.kitti_gt_counts = {diff: [0 for _ in range(self.nc)] for diff in self.difficulties}
        # ensure overall buckets exist
        self.depth_errors = defaultdict(list)
        self.dim_errors = defaultdict(list)
        self.rot_errors = defaultdict(list)

    @staticmethod
    def _difficulty_flags(truncation, occlusion, bbox_height):
        """Return dictionary indicating which KITTI difficulty levels an object satisfies."""
        flags = {"easy": False, "moderate": False, "hard": False}
        if bbox_height is None:
            return flags
        height = float(bbox_height)
        trunc = float(truncation) if truncation is not None else 0.0
        occ = float(occlusion) if occlusion is not None else 0
        if height >= 40 and occ <= 0 and trunc <= 0.15:
            flags["easy"] = True
        if height >= 25 and occ <= 1 and trunc <= 0.3:
            flags["moderate"] = True
        if height >= 25 and occ <= 2 and trunc <= 0.5:
            flags["hard"] = True
        return flags

    @staticmethod
    def _angle_difference(pred, target):
        """Return absolute angle difference normalized to [-pi, pi]."""
        diff = (pred - target + math.pi) % (2 * math.pi) - math.pi
        return abs(diff)

    @staticmethod
    def _boxes3d_to_corners(location, dimensions, rotation_y):
        """
        Convert 3D bounding boxes to 8 corner points in camera coordinates.
        Args:
            location: (N, 3) - (x, y, z) center in camera coords
            dimensions: (N, 3) - (h, w, l) height, width, length
            rotation_y: (N,) - rotation around Y axis
        Returns:
            corners: (N, 8, 3) - 8 corner points for each box
        """
        if location is None or dimensions is None or rotation_y is None:
            return None
        if location.shape[0] == 0:
            return torch.zeros((0, 8, 3), device=location.device, dtype=location.dtype)

        # Dimensions: h, w, l
        h, w, l = dimensions[:, 0:1], dimensions[:, 1:2], dimensions[:, 2:3]

        # Create corner template (8 corners of a unit box centered at origin)
        # Order: rear face (clockwise from top-left), then front face
        x_corners = torch.cat([l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2], dim=1)
        y_corners = torch.cat([-h/2, h/2, h/2, -h/2, -h/2, h/2, h/2, -h/2], dim=1)
        z_corners = torch.cat([w/2, w/2, w/2, w/2, -w/2, -w/2, -w/2, -w/2], dim=1)

        corners = torch.stack([x_corners, y_corners, z_corners], dim=2)  # (N, 8, 3)

        # Apply rotation around Y axis
        # Reshape rotation angles to (N, 1) for broadcasting
        cos_ry = torch.cos(rotation_y).reshape(-1, 1)  # (N, 1)
        sin_ry = torch.sin(rotation_y).reshape(-1, 1)  # (N, 1)

        # Rotation matrix around Y axis
        corners_rotated = torch.zeros_like(corners)  # (N, 8, 3)
        corners_rotated[:, :, 0] = cos_ry * corners[:, :, 0] + sin_ry * corners[:, :, 2]
        corners_rotated[:, :, 1] = corners[:, :, 1]
        corners_rotated[:, :, 2] = -sin_ry * corners[:, :, 0] + cos_ry * corners[:, :, 2]

        # Translate to location
        corners_rotated += location.unsqueeze(1)

        return corners_rotated

    @staticmethod
    def _bev_iou(corners1, corners2):
        """
        Compute Bird's Eye View IoU between two sets of 3D boxes.
        Args:
            corners1: (N, 8, 3) corners of N boxes
            corners2: (M, 8, 3) corners of M boxes
        Returns:
            iou: (N, M) IoU matrix
        """
        if corners1 is None or corners2 is None:
            return torch.zeros((0, 0))
        if corners1.shape[0] == 0 or corners2.shape[0] == 0:
            return torch.zeros((corners1.shape[0], corners2.shape[0]), device=corners1.device)

        # Extract ground plane coordinates (X-Z plane)
        # Use bottom 4 corners (indices 0-3 are one face)
        bev1 = corners1[:, :4, [0, 2]]  # (N, 4, 2) - x, z coords
        bev2 = corners2[:, :4, [0, 2]]  # (M, 4, 2)

        # Compute axis-aligned bounding boxes in BEV
        min1 = bev1.min(dim=1)[0]  # (N, 2)
        max1 = bev1.max(dim=1)[0]  # (N, 2)
        min2 = bev2.min(dim=1)[0]  # (M, 2)
        max2 = bev2.max(dim=1)[0]  # (M, 2)

        # Compute areas
        area1 = (max1[:, 0] - min1[:, 0]) * (max1[:, 1] - min1[:, 1])  # (N,)
        area2 = (max2[:, 0] - min2[:, 0]) * (max2[:, 1] - min2[:, 1])  # (M,)

        # Compute intersection
        lt = torch.maximum(min1[:, None, :], min2[None, :, :])  # (N, M, 2)
        rb = torch.minimum(max1[:, None, :], max2[None, :, :])  # (N, M, 2)
        wh = (rb - lt).clamp(min=0)  # (N, M, 2)
        inter = wh[:, :, 0] * wh[:, :, 1]  # (N, M)

        # Compute union and IoU
        union = area1[:, None] + area2[None, :] - inter
        iou = inter / (union + 1e-7)

        return iou

    @staticmethod
    def _box3d_iou(corners1, corners2):
        """
        Compute 3D IoU between two sets of 3D boxes (simplified axis-aligned approximation).
        Args:
            corners1: (N, 8, 3) corners of N boxes
            corners2: (M, 8, 3) corners of M boxes
        Returns:
            iou: (N, M) IoU matrix
        """
        if corners1 is None or corners2 is None:
            return torch.zeros((0, 0))
        if corners1.shape[0] == 0 or corners2.shape[0] == 0:
            return torch.zeros((corners1.shape[0], corners2.shape[0]), device=corners1.device)

        # Compute axis-aligned bounding boxes in 3D
        min1 = corners1.min(dim=1)[0]  # (N, 3)
        max1 = corners1.max(dim=1)[0]  # (N, 3)
        min2 = corners2.min(dim=1)[0]  # (M, 3)
        max2 = corners2.max(dim=1)[0]  # (M, 3)

        # Compute volumes
        vol1 = ((max1 - min1).prod(dim=1))  # (N,)
        vol2 = ((max2 - min2).prod(dim=1))  # (M,)

        # Compute intersection
        lt = torch.maximum(min1[:, None, :], min2[None, :, :])  # (N, M, 3)
        rb = torch.minimum(max1[:, None, :], max2[None, :, :])  # (N, M, 3)
        whl = (rb - lt).clamp(min=0)  # (N, M, 3)
        inter = whl.prod(dim=2)  # (N, M)

        # Compute union and IoU
        union = vol1[:, None] + vol2[None, :] - inter
        iou = inter / (union + 1e-7)

        return iou

    def _compute_kitti_ap(self, recall, precision):
        """
        Compute AP using KITTI methodology (40 recall positions).

        KITTI evaluation updated in 2019 to use 40 evenly-spaced recall positions
        instead of 11 from Pascal VOC. AP is the mean precision at these positions.

        Args:
            recall: (N,) recall values sorted in ascending order
            precision: (N,) precision values

        Returns:
            ap: Average Precision using KITTI's 40-position sampling
        """
        # Sample precision at 40 evenly-spaced recall positions [0, 1]
        recall_positions = np.linspace(0, 1, self.kitti_recall_positions)

        # Interpolate precision at each recall position
        # For each position, find max precision at recall >= position (monotonic decreasing)
        precisions = np.zeros(self.kitti_recall_positions)
        for i, r_pos in enumerate(recall_positions):
            # Find all recalls >= this position
            mask = recall >= r_pos
            if np.any(mask):
                # Take maximum precision at or above this recall
                precisions[i] = np.max(precision[mask])
            else:
                precisions[i] = 0.0

        # AP is mean of precision at these 40 positions
        ap = np.mean(precisions)
        return ap

    def _convert_pred_params(self, params):
        """Convert raw network outputs into physical location/dimension/rotation values."""
        if params is None or params.shape[1] < self.n3d:
            return None, None, None, None, None
        # Decode 7 parameters: x, y, z, h, w, l, rotation_y
        loc_x = (torch.sigmoid(params[:, 0]) - 0.5) * 100.0  # [-50, 50]m
        loc_y = (torch.sigmoid(params[:, 1]) - 0.5) * 100.0  # [-50, 50]m
        # Depth with inverse sigmoid encoding (MonoFlex-style): d = 1/sigmoid(x) - 1
        depth = 1.0 / (torch.sigmoid(params[:, 2]) + 1e-6) - 1.0
        depth = depth.clamp(0, 100)  # [0, 100]m
        dims = torch.sigmoid(params[:, 3:6]) * 10.0  # [0, 10]m (h, w, l)
        rot = (torch.sigmoid(params[:, 6]) - 0.5) * 2 * math.pi  # [-π, π]

        # DEBUG: Print first prediction to check values
        if self.args.verbose and params.shape[0] > 0:
            LOGGER.info(f"DEBUG: First prediction values:")
            LOGGER.info(f"  loc_x: {loc_x[0].item():.3f}, loc_y: {loc_y[0].item():.3f}, depth: {depth[0].item():.3f}")
            LOGGER.info(f"  dims: {dims[0].detach().cpu().numpy()}")
            LOGGER.info(f"  rot: {rot[0].item():.3f} rad ({math.degrees(rot[0].item()):.1f} deg)")

        return loc_x, loc_y, depth, dims, rot

    def update_metrics(self, preds, batch):
        """Update 2D metrics and accumulate KITTI-style 3D statistics."""
        # DEBUG: Check if we're getting called
        if self.args.verbose:
            LOGGER.info(f"DEBUG update_metrics: Called with {len(preds)} predictions")

        super().update_metrics(preds, batch)

        batch_idx = batch["batch_idx"]
        gt_dims_all = batch.get("dimensions_3d")
        gt_loc_all = batch.get("location_3d")
        gt_rot_all = batch.get("rotation_y")
        gt_trunc = batch.get("truncation")
        gt_occlusion = batch.get("occlusion")
        gt_height = batch.get("bbox_height")

        for si, pred in enumerate(preds):
            pbatch = self._prepare_batch(si, batch)
            gt_cls = pbatch["cls"]
            gt_bbox = pbatch["bbox"]

            if self.args.verbose and si < 3:  # Only log first few
                LOGGER.info(f"DEBUG update_metrics: Image {si}, GT objects: {gt_cls.numel()}")

            if gt_cls.numel() == 0:
                # Still record false positives for KITTI metrics
                for diff in self.difficulties:
                    for det in pred:
                        cls_id = int(det[5].item())
                        if cls_id < self.nc:
                            self.kitti_stats[diff][cls_id]["conf"].append(float(det[4].item()))
                            self.kitti_stats[diff][cls_id]["tp"].append(0.0)
                continue

            mask = batch_idx == si
            gt_dims = gt_dims_all[mask] if gt_dims_all is not None else None
            gt_loc = gt_loc_all[mask] if gt_loc_all is not None else None
            gt_rot = gt_rot_all[mask] if gt_rot_all is not None else None
            gt_tr = gt_trunc[mask] if gt_trunc is not None else None
            gt_occ = gt_occlusion[mask] if gt_occlusion is not None else None
            gt_h = gt_height[mask] if gt_height is not None else None

            difficulty_indices = {diff: [] for diff in self.difficulties}
            for gi in range(len(gt_cls)):
                trunc_val = gt_tr[gi].item() if gt_tr is not None else 0.0
                occ_val = gt_occ[gi].item() if gt_occ is not None else 0.0
                height_val = gt_h[gi].item() if gt_h is not None else None
                flags = self._difficulty_flags(trunc_val, occ_val, height_val)
                for diff, active in flags.items():
                    if active:
                        difficulty_indices[diff].append(gi)
                        cls_int = int(gt_cls[gi].item())
                        if cls_int < self.nc:
                            self.kitti_gt_counts[diff][cls_int] += 1

            predn = self._prepare_pred(pred, pbatch)
            pred_boxes = predn[:, :4]
            confidences = pred[:, 4]
            pred_classes = pred[:, 5].to(torch.int64)

            # Extract 3D parameters
            pred_loc_x = pred_loc_y = pred_depth = pred_dims = pred_rot = None
            if pred.shape[1] > 6:
                # Extract 3D params from pred
                params_3d = pred[:, 6:6+self.n3d]

                # Check if params are decoded (values > 1) or raw (values in [0,1])
                # Strategy: Check dimension channel (ch 3-6). If max < 0.5, likely raw (sigmoid range)
                # After decoding dims should be [0, 10], so if max(dim) < 0.5, it's still raw
                dim_max = params_3d[:, 3:6].max().item()

                if dim_max > 0.5:
                    # Already decoded - use directly
                    pred_loc_x = params_3d[:, 0:1]
                    pred_loc_y = params_3d[:, 1:2]
                    pred_depth = params_3d[:, 2:3]
                    pred_dims = params_3d[:, 3:6]
                    pred_rot = params_3d[:, 6:7]
                    if self.args.verbose:
                        LOGGER.info(f"DEBUG: Using DECODED 3D params (dim_max={dim_max:.3f})")
                else:
                    # Raw params - decode them
                    if self.args.verbose:
                        LOGGER.info(f"DEBUG: Decoding RAW 3D params (dim_max={dim_max:.3f})")

                    # Decode using same logic as Detect3D.forward()
                    pred_loc_x = (torch.sigmoid(params_3d[:, 0:1]) - 0.5) * 100.0
                    pred_loc_y = (torch.sigmoid(params_3d[:, 1:2]) - 0.5) * 100.0
                    # Depth with inverse sigmoid encoding (MonoFlex-style)
                    pred_depth = 1.0 / (torch.sigmoid(params_3d[:, 2:3]) + 1e-6) - 1.0
                    pred_depth = pred_depth.clamp(0, 100)
                    pred_dims = torch.sigmoid(params_3d[:, 3:6]) * 10.0
                    pred_rot = (torch.sigmoid(params_3d[:, 6:7]) - 0.5) * 2 * math.pi

                if self.args.verbose:
                    LOGGER.info(f"DEBUG: 3D values:")
                    LOGGER.info(f"  loc_x: {pred_loc_x[0].item():.3f}, loc_y: {pred_loc_y[0].item():.3f}, depth: {pred_depth[0].item():.3f}")
                    LOGGER.info(f"  dims: {pred_dims[0].detach().cpu().numpy()}")
                    LOGGER.info(f"  rot: {pred_rot[0].item():.3f} rad ({math.degrees(pred_rot[0].item()):.1f} deg)")

            # Compute 3D IoU for KITTI matching (instead of 2D IoU)
            pred_corners = None
            gt_corners = None
            iou_3d = None
            iou_bev = None

            if pred_loc_x is not None and pred_loc_y is not None and pred_depth is not None and pred_dims is not None and pred_rot is not None:
                # Build predicted 3D locations from network predictions
                pred_loc = torch.zeros((pred.shape[0], 3), device=pred.device, dtype=pred.dtype)
                pred_loc[:, 0:1] = pred_loc_x  # X (lateral position in camera coords)
                pred_loc[:, 1:2] = pred_loc_y  # Y (vertical position in camera coords)
                pred_loc[:, 2:3] = pred_depth  # Z (depth from camera)

                pred_corners = self._boxes3d_to_corners(pred_loc, pred_dims, pred_rot)

            if gt_loc is not None and gt_dims is not None and gt_rot is not None:
                # Convert gt_rot to proper shape if needed
                gt_rot_reshaped = gt_rot if gt_rot.dim() == 1 else gt_rot.squeeze(-1)
                gt_corners = self._boxes3d_to_corners(gt_loc, gt_dims, gt_rot_reshaped)

            # Compute 3D and BEV IoU
            if pred_corners is not None and gt_corners is not None:
                iou_3d = self._box3d_iou(pred_corners, gt_corners)
                iou_bev = self._bev_iou(pred_corners, gt_corners)

            # Use 3D IoU for matching if available, otherwise fall back to 2D
            if iou_3d is not None:
                ious = iou_3d
            else:
                ious = box_iou(pred_boxes, gt_bbox) if gt_bbox.numel() else torch.zeros(
                    (pred_boxes.shape[0], 0), device=pred_boxes.device
                )

            # DEBUG: Print IoU statistics
            if self.args.verbose and ious.numel() > 0:
                max_iou = torch.max(ious).item()
                mean_iou = torch.mean(ious).item()
                LOGGER.info(f"DEBUG: IoU stats - max: {max_iou:.4f}, mean: {mean_iou:.4f}, shape: {ious.shape}")
                # Find a matching prediction
                best_pred_iou = -1
                best_gt_iou = -1
                for p_idx in range(min(5, ious.shape[0])):
                    for g_idx in range(min(5, ious.shape[1])):
                        if ious[p_idx, g_idx] > best_pred_iou:
                            best_pred_iou = ious[p_idx, g_idx].item()
                            best_gt_iou = g_idx
                LOGGER.info(f"DEBUG: Best IoU: {best_pred_iou:.4f} (pred {0} vs gt {best_gt_iou})")
                LOGGER.info(f"DEBUG: Using 3D IoU: {iou_3d is not None}")

            order = torch.argsort(confidences, descending=True)

            for diff in self.difficulties:
                gt_index_list = difficulty_indices[diff]
                used = set()
                if not gt_index_list:
                    continue

                for idx in order.tolist():
                    cls_id = int(pred_classes[idx].item())
                    if cls_id >= self.nc:
                        continue
                    conf_val = float(confidences[idx].item())
                    threshold = self.kitti_iou_thresholds.get(cls_id, 0.5)
                    best_iou = 0.0
                    best_local = -1
                    for local_i, gt_global in enumerate(gt_index_list):
                        if local_i in used:
                            continue
                        if int(gt_cls[gt_global].item()) != cls_id:
                            continue
                        iou = float(ious[idx, gt_global].item()) if ious.numel() else 0.0
                        if iou >= threshold and iou > best_iou:
                            best_iou = iou
                            best_local = local_i

                    matched = best_local >= 0
                    self.kitti_stats[diff][cls_id]["conf"].append(conf_val)
                    self.kitti_stats[diff][cls_id]["tp"].append(1.0 if matched else 0.0)

                    if matched:
                        used.add(best_local)
                        gt_global = gt_index_list[best_local]
                        if pred_depth is not None and gt_loc is not None:
                            depth_gt = float(gt_loc[gt_global, 2].item())
                            self.depth_errors["all"].append(abs(float(pred_depth[idx].item()) - depth_gt))
                            self.depth_errors[diff].append(abs(float(pred_depth[idx].item()) - depth_gt))
                        if pred_dims is not None and gt_dims is not None:
                            dims_gt = gt_dims[gt_global].detach().cpu().numpy()
                            dims_pred = pred_dims[idx].detach().cpu().numpy()
                            dim_err = float(np.mean(np.abs(dims_pred - dims_gt)))
                            self.dim_errors["all"].append(dim_err)
                            self.dim_errors[diff].append(dim_err)
                        if pred_rot is not None and gt_rot is not None:
                            angle_gt = float(gt_rot[gt_global].item())
                            angle_pred = float(pred_rot[idx].item())
                            rot_err = self._angle_difference(angle_pred, angle_gt)
                            self.rot_errors["all"].append(rot_err)
                            self.rot_errors[diff].append(rot_err)

        # DEBUG: Summary of KITTI stats
        if self.args.verbose:
            total_conf = sum(len(self.kitti_stats['moderate'][cls_id]['conf']) for cls_id in range(self.nc))
            total_tp = sum(sum(self.kitti_stats['moderate'][cls_id]['tp']) for cls_id in range(self.nc))
            LOGGER.info(f"DEBUG update_metrics: Total detections recorded: {total_conf}, TPs: {total_tp}")
            total_gt = sum(self.kitti_gt_counts['moderate'])
            LOGGER.info(f"DEBUG update_metrics: Total GT objects (moderate): {total_gt}")

    def get_stats(self):
        """Returns metrics statistics and results dictionary."""
        stats = super().get_stats()

        # DEBUG: Check if we have KITTI data
        if self.args.verbose:
            LOGGER.info(f"DEBUG get_stats: kitti_stats exists: {self.kitti_stats is not None}")
            if self.kitti_stats:
                LOGGER.info(f"DEBUG get_stats: kitti_stats['moderate'][0] conf: {len(self.kitti_stats['moderate'][0]['conf'])} items")
            LOGGER.info(f"DEBUG get_stats: kitti_gt_counts exists: {self.kitti_gt_counts is not None}")
            if self.kitti_gt_counts:
                LOGGER.info(f"DEBUG get_stats: kitti_gt_counts['moderate']: {self.kitti_gt_counts['moderate']}")

        # Compute KITTI mAP first
        self.kitti_summary = {diff: {} for diff in self.difficulties}

        for diff in self.difficulties:
            ap_values = []
            for cls_id in range(self.nc):
                class_name = self.names[cls_id]
                total_gt = self.kitti_gt_counts[diff][cls_id] if self.kitti_gt_counts else 0
                stats[f"kitti/{diff}/gt/{class_name}"] = total_gt

                class_stats = self.kitti_stats[diff][cls_id] if self.kitti_stats else {"conf": [], "tp": []}
                conf_list = class_stats["conf"]
                tp_list = class_stats["tp"]

                if total_gt <= 0:
                    continue

                if not conf_list:
                    stats[f"kitti/{diff}/AP/{class_name}"] = 0.0
                    stats[f"kitti/{diff}/precision/{class_name}"] = 0.0
                    stats[f"kitti/{diff}/recall/{class_name}"] = 0.0
                    continue

                conf = np.asarray(conf_list)
                tp = np.asarray(tp_list)
                order = np.argsort(-conf)
                tp = tp[order]
                fp = 1.0 - tp
                tp_cum = np.cumsum(tp)
                fp_cum = np.cumsum(fp)
                recall = tp_cum / (total_gt + 1e-16)
                precision = tp_cum / (tp_cum + fp_cum + 1e-16)
                # Use KITTI-specific AP computation (40 recall positions)
                ap = self._compute_kitti_ap(recall, precision)

                stats[f"kitti/{diff}/AP/{class_name}"] = float(ap)
                stats[f"kitti/{diff}/precision/{class_name}"] = float(precision[-1])
                stats[f"kitti/{diff}/recall/{class_name}"] = float(recall[-1])
                ap_values.append(ap)
                self.kitti_summary[diff][class_name] = float(ap)

            if ap_values:
                mean_ap = float(np.mean(ap_values))
                stats[f"kitti/{diff}/mAP"] = mean_ap
                self.kitti_summary[diff]["mAP"] = mean_ap

        # Replace standard mAP with KITTI mAP for display
        # Use 'moderate' difficulty as the primary metric (standard in KITTI)
        if "moderate" in self.kitti_summary and "mAP" in self.kitti_summary["moderate"]:
            kitti_map = self.kitti_summary["moderate"]["mAP"]
            stats["metrics/mAP50-95(B)"] = kitti_map  # Override standard mAP with KITTI mAP
            stats["metrics/mAP50(B)"] = kitti_map  # Also set mAP50 to same value
            # Also update self.metrics.results_dict so it shows in the table
            if hasattr(self, 'metrics') and hasattr(self.metrics, 'results_dict'):
                self.metrics.results_dict["metrics/mAP50-95(B)"] = kitti_map
                self.metrics.results_dict["metrics/mAP50(B)"] = kitti_map

        for diff_key in ["all", *self.difficulties]:
            depth_vals = self.depth_errors.get(diff_key, [])
            dim_vals = self.dim_errors.get(diff_key, [])
            rot_vals = self.rot_errors.get(diff_key, [])
            if depth_vals:
                stats[f"kitti/{diff_key}/depth_mae"] = float(np.mean(depth_vals))
            if dim_vals:
                stats[f"kitti/{diff_key}/dim_mae"] = float(np.mean(dim_vals))
            if rot_vals:
                stats[f"kitti/{diff_key}/rot_mae"] = float(np.mean(rot_vals))

        if self.depth_errors.get("all"):
            stats["metrics/depth_error"] = float(np.mean(self.depth_errors["all"]))
        if self.dim_errors.get("all"):
            stats["metrics/dim_error"] = float(np.mean(self.dim_errors["all"]))
        if self.rot_errors.get("all"):
            stats["metrics/rot_error"] = float(np.mean(self.rot_errors["all"]))

        return stats

    def get_desc(self):
        """Return a formatted string summarizing class metrics of YOLO model."""
        # Extend with 3D metrics
        # Format: Class | Images | Instances | Box(P) | R | map50 | KmAP50 | KmAP
        # That's 1 + 7 = 8 values total
        return ("%22s" + "%11s" * 7) % (
            "Class",
            "Images",
            "Instances",
            "Box(P",
            "R",
            "map50",
            "KmAP50",
            "KmAP",
        )

    def print_results(self):
        """Log base results alongside KITTI-specific summaries."""
        super().print_results()
        if not getattr(self, "kitti_summary", None):
            return
        for diff in self.difficulties:
            summary = self.kitti_summary.get(diff, {})
            if not summary:
                continue
            map_val = summary.get("mAP")
            if map_val is not None:
                LOGGER.info(f"KITTI {diff.capitalize()} mAP: {map_val:.3f}")
