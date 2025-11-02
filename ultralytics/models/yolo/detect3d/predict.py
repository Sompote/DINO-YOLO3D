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
        """Post-processes predictions and returns a list of Results objects with 3D parameters."""
        # Handle 3D predictions - the head already concatenates 3D params to predictions
        # Expected format: [batch, num_preds, 4+1+nc+5]
        # where 4=xyxy, 1=conf, nc=classes, 5=3D params (depth, h, w, l, rotation)

        if isinstance(preds, tuple) and len(preds) == 2:
            # preds[0] already has 3D params concatenated
            preds_with_3d = preds[0]
        else:
            preds_with_3d = preds

        # Apply NMS to predictions (NMS will preserve all columns including 3D params)
        preds_nms = ops.non_max_suppression(
            preds_with_3d,
            self.args.conf,
            self.args.iou,
            agnostic=self.args.agnostic_nms,
            max_det=self.args.max_det,
            classes=self.args.classes,
            nc=len(self.model.names),  # Specify number of classes
        )

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        for i, (pred, orig_img, img_path) in enumerate(zip(preds_nms, orig_imgs, self.batch[0])):
            if len(pred) == 0:
                # No detections
                results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=pred))
                continue

            # Scale boxes back to original image size (only the first 4 columns)
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)

            # pred now contains: [x1, y1, x2, y2, conf, cls, depth, h_3d, w_3d, l_3d, rotation_y]
            # The Results object will store all of this in boxes.data
            results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=pred))

        return results
