"""
Example training script for YOLOv12-3D on KITTI dataset.

Developed by AI Research Group
Department of Civil Engineering
King Mongkut's University of Technology Thonburi (KMUTT)
Bangkok, Thailand

This script demonstrates how to train a 3D object detection model using YOLOv12-3D
on the KITTI dataset.

Requirements:
    - KITTI dataset downloaded and extracted
    - ultralytics package installed
    - GPU recommended for training

Usage:
    python examples/train_kitti_3d.py
"""

from ultralytics import YOLO


def train_yolov12_3d():
    """Train YOLOv12-3D model on KITTI dataset."""

    # Load a model
    # Option 1: Start from scratch
    model = YOLO("ultralytics/cfg/models/v12/yolov12-3d.yaml")

    # Option 2: Start from a pretrained 2D model (transfer learning)
    # model = YOLO('yolov12n.pt')  # load a pretrained model
    # model.model = Detection3DModel('ultralytics/cfg/models/v12/yolov12-3d.yaml')

    # Train the model
    results = model.train(
        data="ultralytics/cfg/datasets/kitti-3d.yaml",  # path to dataset config
        epochs=100,  # number of epochs
        imgsz=640,  # image size
        batch=16,  # batch size (adjust based on GPU memory)
        name="yolov12n-3d-kitti",  # experiment name
        device=0,  # GPU device (0, 1, 2, etc.) or 'cpu'
        workers=8,  # number of dataloader workers
        # Learning rate settings
        lr0=0.01,  # initial learning rate
        lrf=0.01,  # final learning rate (lr0 * lrf)
        momentum=0.937,  # SGD momentum
        weight_decay=0.0005,  # optimizer weight decay
        warmup_epochs=3.0,  # warmup epochs
        warmup_momentum=0.8,  # warmup initial momentum
        warmup_bias_lr=0.1,  # warmup initial bias lr
        # Data augmentation (conservative for 3D detection)
        hsv_h=0.015,  # HSV-Hue augmentation
        hsv_s=0.7,  # HSV-Saturation augmentation
        hsv_v=0.4,  # HSV-Value augmentation
        degrees=0.0,  # rotation (disabled for 3D)
        translate=0.1,  # translation
        scale=0.5,  # scale
        shear=0.0,  # shear (disabled for 3D)
        perspective=0.0,  # perspective (disabled for 3D)
        flipud=0.0,  # flip up-down (disabled)
        fliplr=0.5,  # flip left-right
        mosaic=1.0,  # mosaic augmentation
        mixup=0.0,  # mixup (disabled for 3D)
        copy_paste=0.0,  # copy-paste (disabled)
        # Loss weights (can be tuned)
        box=7.5,  # 2D box loss gain
        cls=0.5,  # class loss gain
        dfl=1.5,  # DFL loss gain
        # Training settings
        patience=50,  # early stopping patience
        save=True,  # save checkpoints
        save_period=-1,  # save checkpoint every x epochs (-1 = disabled)
        cache=False,  # cache images for faster training
        exist_ok=False,  # overwrite existing experiment
        pretrained=False,  # use pretrained model
        optimizer="SGD",  # optimizer (SGD, Adam, AdamW)
        verbose=True,  # verbose output
        seed=0,  # random seed
        deterministic=True,  # deterministic mode
        single_cls=False,  # train as single-class dataset
        rect=False,  # rectangular training
        cos_lr=False,  # cosine learning rate scheduler
        close_mosaic=10,  # disable mosaic for final X epochs
        amp=True,  # automatic mixed precision training
        fraction=1.0,  # dataset fraction to train on
        profile=False,  # profile ONNX and TensorRT speeds
        freeze=None,  # freeze layers (e.g., [0, 1, 2])
        multi_scale=False,  # multi-scale training
        # Validation settings
        val=True,  # validate during training
        plots=True,  # save plots
    )

    # Evaluate on validation set
    metrics = model.val()
    print(f"\nValidation Results:")
    print(f"mAP50: {metrics.box.map50:.3f}")
    print(f"mAP50-95: {metrics.box.map:.3f}")

    return model, results


def resume_training():
    """Resume training from a checkpoint."""
    model = YOLO("runs/detect/yolov12n-3d-kitti/weights/last.pt")
    results = model.train(resume=True)
    return model, results


if __name__ == "__main__":
    # Train from scratch
    model, results = train_yolov12_3d()

    # Optionally, test the model
    # results = model.predict(source='path/to/test/images', save=True, conf=0.25)

    print("\n" + "=" * 50)
    print("Training completed!")
    print(f"Model saved to: runs/detect/yolov12n-3d-kitti/weights/best.pt")
    print("=" * 50)
