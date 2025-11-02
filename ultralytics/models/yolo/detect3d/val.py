# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import torch

from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils.metrics import DetMetrics, box_iou


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
        # Add 3D-specific metrics tracking
        self.depth_errors = []
        self.dim_errors = []
        self.rot_errors = []

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
        if isinstance(preds, tuple):
            preds_2d, preds_3d = preds[0], preds[1]
        else:
            preds_2d = preds
            preds_3d = None

        # Apply NMS on 2D predictions
        return super().postprocess(preds_2d)

    def update_metrics(self, preds, batch):
        """Metrics."""
        # Update 2D metrics using parent class
        super().update_metrics(preds, batch)

        # TODO: Add 3D-specific metrics
        # - Average Precision for 3D IoU
        # - Depth error (absolute/relative)
        # - Dimension error
        # - Rotation error

    def get_stats(self):
        """Returns metrics statistics and results dictionary."""
        stats = super().get_stats()

        # Add 3D metrics to stats if available
        if self.depth_errors:
            stats["metrics/depth_error"] = sum(self.depth_errors) / len(self.depth_errors)
        if self.dim_errors:
            stats["metrics/dim_error"] = sum(self.dim_errors) / len(self.dim_errors)
        if self.rot_errors:
            stats["metrics/rot_error"] = sum(self.rot_errors) / len(self.rot_errors)

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
