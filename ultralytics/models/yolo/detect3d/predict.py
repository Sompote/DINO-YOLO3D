# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import torch

from ultralytics.engine.results import Results
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.utils import ops


class Detection3DPredictor(DetectionPredictor):
    """
    A class extending the DetectionPredictor class for prediction based on a 3D detection model.

    Example:
        ```python
        from ultralytics.utils import ASSETS
        from ultralytics.models.yolo.detect3d import Detection3DPredictor

        args = dict(model="yolov12n-3d.pt", source=ASSETS)
        predictor = Detection3DPredictor(overrides=args)
        predictor.predict_cli()
        ```
    """

    def postprocess(self, preds, img, orig_imgs):
        """Post-processes predictions and returns a list of Results objects."""
        # Handle 3D predictions
        if isinstance(preds, tuple) and len(preds) == 2:
            preds_2d, preds_3d_dict = preds
            # Extract 3D parameters if available
            if isinstance(preds_3d_dict, tuple):
                preds_3d = preds_3d_dict[0]  # Get the concatenated 3D parameters
            else:
                preds_3d = None
        else:
            preds_2d = preds
            preds_3d = None

        # Apply NMS to 2D predictions
        preds_2d = ops.non_max_suppression(
            preds_2d,
            self.args.conf,
            self.args.iou,
            agnostic=self.args.agnostic_nms,
            max_det=self.args.max_det,
            classes=self.args.classes,
        )

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        for i, (pred, orig_img, img_path) in enumerate(zip(preds_2d, orig_imgs, self.batch[0])):
            # Scale boxes back to original image size
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)

            # Create Results object
            # For 3D detection, we would need to extend Results class to include 3D parameters
            # For now, store 2D results
            results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=pred))

        return results
