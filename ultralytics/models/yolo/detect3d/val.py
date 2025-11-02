# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import math
from collections import defaultdict

import numpy as np
import torch

from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import LOGGER
from ultralytics.utils.metrics import DetMetrics, box_iou, compute_ap


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
        self.kitti_iou_thresholds = {0: 0.7, 1: 0.7, 2: 0.5, 3: 0.5}
        self.n3d = 5
        self._reset_kitti_metrics()

    def _reset_kitti_metrics(self):
        """Reset containers used for KITTI-style evaluation."""
        self.depth_errors = defaultdict(list)
        self.dim_errors = defaultdict(list)
        self.rot_errors = defaultdict(list)
        self.kitti_stats = None
        self.kitti_gt_counts = None

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

        for key in ["dimensions_3d", "location_3d", "rotation_y", "alpha"]:
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
            if isinstance(extra, (list, tuple)):
                # Expect tuple like (features, params_3d)
                if extra and isinstance(extra[-1], torch.Tensor):
                    preds_3d = extra[-1]
            elif isinstance(extra, torch.Tensor):
                preds_3d = extra
        else:
            preds_2d = preds

        if preds_3d is not None and isinstance(preds_2d, torch.Tensor):
            # Append raw 3D params as additional channels so NMS keeps them
            params = preds_3d.permute(0, 2, 1).contiguous()
            if params.shape[0] == preds_2d.shape[0] and params.shape[1] == preds_2d.shape[1]:
                preds_2d = torch.cat([preds_2d, params], dim=2)

        # Apply NMS on 2D predictions
        outputs = super().postprocess(preds_2d)
        return outputs
*** End Patch
*** Insert After: Line 75
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

    def _convert_pred_params(self, params):
        """Convert raw network outputs into physical depth/dimension/rotation values."""
        if params is None or params.shape[1] < self.n3d:
            return None, None, None
        depth = torch.sigmoid(params[:, 0]) * 100.0  # 0-100m
        dims = torch.sigmoid(params[:, 1:4]) * 10.0  # 0-10m
        rot = (torch.sigmoid(params[:, 4]) - 0.5) * 2 * math.pi  # -pi to pi
        return depth, dims, rot

    def update_metrics(self, preds, batch):
        """Update 2D metrics and accumulate KITTI-style 3D statistics."""
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
            extra_params = pred[:, 6:] if pred.shape[1] > 6 else None
            pred_depth, pred_dims, pred_rot = self._convert_pred_params(extra_params)

            ious = box_iou(pred_boxes, gt_bbox) if gt_bbox.numel() else torch.zeros(
                (pred_boxes.shape[0], 0), device=pred_boxes.device
            )
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

    def get_stats(self):
        """Returns metrics statistics and results dictionary."""
        stats = super().get_stats()

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
                ap, _, _ = compute_ap(recall, precision)

                stats[f"kitti/{diff}/AP/{class_name}"] = float(ap)
                stats[f"kitti/{diff}/precision/{class_name}"] = float(precision[-1])
                stats[f"kitti/{diff}/recall/{class_name}"] = float(recall[-1])
                ap_values.append(ap)
                self.kitti_summary[diff][class_name] = float(ap)

            if ap_values:
                mean_ap = float(np.mean(ap_values))
                stats[f"kitti/{diff}/mAP"] = mean_ap
                self.kitti_summary[diff]["mAP"] = mean_ap

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
        return ("%22s" + "%11s" * 9) % (
            "Class",
            "Images",
            "Instances",
            "Box(P",
            "R",
            "mAP50",
            "mAP50-95)",
            "Depth",
            "Dim",
            "Rot",
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
