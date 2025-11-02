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
        # Ensure batch is a dict (not a tuple or other type)
        if not isinstance(batch, dict):
            raise TypeError(f"Expected batch to be a dict, got {type(batch)}")

        # Check if batch["img"] is a tuple or list (should be a tensor)
        if "img" in batch:
            if isinstance(batch["img"], (tuple, list)):
                # If it's a tuple/list, convert to tensor by stacking
                batch["img"] = torch.stack(list(batch["img"]), 0)
            # Also check if it's already a tensor but needs to be stacked
            elif isinstance(batch["img"], torch.Tensor) and batch["img"].dim() == 3:
                # Single image tensor [C, H, W] -> add batch dimension [1, C, H, W]
                batch["img"] = batch["img"].unsqueeze(0)

        # Ensure all batch items that should be tensors are tensors
        for key in ["batch_idx", "cls", "bboxes"]:
            if key in batch and not isinstance(batch[key], torch.Tensor):
                if isinstance(batch[key], (list, tuple)):
                    batch[key] = torch.cat([x if isinstance(x, torch.Tensor) else torch.tensor(x) for x in batch[key]], 0)

        batch = super().preprocess(batch)

        # Move 3D annotations to device (check they exist and are tensors)
        for key in ["dimensions_3d", "location_3d", "rotation_y", "alpha"]:
            if key in batch and batch[key] is not None and hasattr(batch[key], 'to'):
                batch[key] = batch[key].to(self.device, non_blocking=True)

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
