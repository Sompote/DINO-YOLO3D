#!/usr/bin/env python3
"""
YOLOv12-3D CLI Tool

Developed by AI Research Group
Department of Civil Engineering
King Mongkut's University of Technology Thonburi (KMUTT)

A unified command-line interface for YOLOv12-3D operations.

Usage:
    yolo3d train --help
    yolo3d val --help
    yolo3d predict --help
    yolo3d export --help
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

# Add current directory to Python path to use local ultralytics package
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    from ultralytics import YOLO
    from ultralytics.utils import LOGGER
except ImportError as e:
    print("Error: ultralytics package not found or failed to import.")
    print(f"Details: {e}")
    print("\nSolutions:")
    print("  1. Make sure you're running from the project root directory")
    print("  2. Install ultralytics: pip install ultralytics")
    print("  3. Check that the local ultralytics/ directory exists")
    sys.exit(1)


class Colors:
    """ANSI color codes."""

    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def print_banner():
    """Print CLI banner."""
    banner = f"""
{Colors.BOLD}{Colors.OKCYAN}╔══════════════════════════════════════════════════════════════════════════════╗
║                        YOLOv12-3D Command Line Tool                          ║
║                                                                              ║
║              AI Research Group, Civil Engineering, KMUTT                     ║
╚══════════════════════════════════════════════════════════════════════════════╝{Colors.ENDC}
"""
    print(banner)


def print_success(msg: str):
    """Print success message."""
    print(f"{Colors.OKGREEN}✓ {msg}{Colors.ENDC}")


def print_error(msg: str):
    """Print error message."""
    print(f"{Colors.FAIL}✗ {msg}{Colors.ENDC}")


def print_warning(msg: str):
    """Print warning message."""
    print(f"{Colors.WARNING}⚠ {msg}{Colors.ENDC}")


def print_info(msg: str):
    """Print info message."""
    print(f"{Colors.OKBLUE}ℹ {msg}{Colors.ENDC}")


def print_section(title: str):
    """Print section header."""
    print(f"\n{Colors.BOLD}{Colors.HEADER}{'=' * 80}\n{title}\n{'=' * 80}{Colors.ENDC}")


def cmd_train(args):
    """Train YOLOv12-3D model."""
    print_banner()
    print_section("Training Configuration")

    # Validate inputs
    if not Path(args.data).exists():
        print_error(f"Dataset config not found: {args.data}")
        print_info("Example: ultralytics/cfg/datasets/kitti-3d.yaml")
        return 1

    # Validate valpercent
    if args.valpercent < 1 or args.valpercent > 100:
        print_error(f"Invalid --valpercent value: {args.valpercent}")
        print_info("Must be between 1 and 100")
        return 1

    # Warn if using very low validation percentage
    if args.valpercent < 10:
        print_warning(f"Using only {args.valpercent}% of validation data - metrics may not be representative!")
        print_info("Recommended minimum: --valpercent 10")

    # Determine model and scale
    model_scale = None
    if args.model.endswith(".pt"):
        print_info(f"Loading pretrained model: {args.model}")
        model_path = args.model
    elif args.model.endswith(".yaml"):
        print_info(f"Creating model from config: {args.model}")
        model_path = args.model
    else:
        # Assume it's a model size (n, s, m, l, x)
        model_scale = args.model
        model_path = f"ultralytics/cfg/models/v12/yolov12{args.model}-3d.yaml"
        if not Path(model_path).exists():
            model_path = f"ultralytics/cfg/models/v12/yolov12-3d.yaml"
        print_info(f"Using model: {model_path} with scale: {model_scale}")

    # Print configuration
    nbs_display = args.nbs if args.nbs > 0 else "64 (auto)"
    accumulate = max(round((args.nbs if args.nbs > 0 else 64) / args.batch), 1)

    # Validation info
    val_info = f"{args.valpercent}%" if args.valpercent < 100.0 else "100% (full)"

    config_info = f"""
  Model:           {Colors.BOLD}{model_path}{Colors.ENDC}
  Scale:           {Colors.BOLD}{model_scale if model_scale else 'default'}{Colors.ENDC}
  Dataset:         {Colors.BOLD}{args.data}{Colors.ENDC}
  Epochs:          {Colors.BOLD}{args.epochs}{Colors.ENDC}
  Batch Size:      {Colors.BOLD}{args.batch}{Colors.ENDC}
  Nominal Batch:   {Colors.BOLD}{nbs_display}{Colors.ENDC}
  Grad Accum:      {Colors.BOLD}{accumulate} step(s){Colors.ENDC}
  Image Size:      {Colors.BOLD}{args.imgsz}{Colors.ENDC}
  Device:          {Colors.BOLD}{args.device}{Colors.ENDC}
  Workers:         {Colors.BOLD}{args.workers}{Colors.ENDC}
  Val Data:        {Colors.BOLD}{val_info}{Colors.ENDC}
  Experiment Name: {Colors.BOLD}{args.name}{Colors.ENDC}
"""
    print(config_info)

    # Show info about gradient accumulation
    if accumulate > 1:
        print_info(f"Using gradient accumulation: {accumulate} steps × {args.batch} batch = {accumulate * args.batch} effective batch")
        print_info(f"To disable gradient accumulation, set --nbs {args.batch}")

    print_section("Starting Training")

    try:
        # Load model
        # For YAML files with specified scale, we need to modify the config
        tmp_yaml_path = None
        if model_scale and model_path.endswith(".yaml"):
            import yaml
            import tempfile
            import os

            # Load YAML config
            with open(model_path, "r") as f:
                cfg = yaml.safe_load(f)

            # Add scale to config
            cfg["scale"] = model_scale

            # Create temporary YAML file with scale in filename for proper detection
            # Use pattern like yolov12s-3d.yaml so guess_model_scale can extract 's'
            tmp_fd, tmp_yaml_path = tempfile.mkstemp(suffix=f"-yolov12{model_scale}-3d.yaml", text=True)
            with os.fdopen(tmp_fd, "w") as tmp_file:
                yaml.dump(cfg, tmp_file)

            # Load model from temporary file (don't delete yet, model needs to read it)
            model = YOLO(tmp_yaml_path)
        else:
            model = YOLO(model_path)

        # Train
        train_kwargs = {
            "data": args.data,
            "epochs": args.epochs,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "name": args.name,
            "device": args.device,
            "workers": args.workers,
            "patience": args.patience,
            "save": args.save,
            "pretrained": args.pretrained,
            "optimizer": args.optimizer,
            "verbose": args.verbose,
            "seed": args.seed,
            "lr0": args.lr0,
            "lrf": args.lrf,
            "momentum": args.momentum,
            "weight_decay": args.weight_decay,
            "warmup_epochs": args.warmup_epochs,
            "box": args.box,
            "cls": args.cls,
            "dfl": args.dfl,
            "plots": args.plots,
            "val": args.val,
        }

        # Add nbs if specified (controls gradient accumulation)
        if args.nbs > 0:
            train_kwargs["nbs"] = args.nbs

        results = model.train(**train_kwargs)

        print_section("Training Complete!")
        print_success(f"Model saved to: runs/detect/{args.name}/weights/best.pt")

        if args.val:
            print_info("Running validation...")
            # Apply validation fraction if less than 100%
            if args.valpercent < 100.0:
                val_fraction = args.valpercent / 100.0
                print_info(f"Using {args.valpercent}% of validation data for faster validation")
                metrics = model.val(fraction=val_fraction)
            else:
                metrics = model.val()
            print(f"\n  mAP50-95: {Colors.BOLD}{metrics.box.map:.3f}{Colors.ENDC}")
            print(f"  mAP50:    {Colors.BOLD}{metrics.box.map50:.3f}{Colors.ENDC}")

        return 0

    except Exception as e:
        import traceback

        traceback.print_exc()
        print_error(f"Training failed: {e}")
        return 1
    finally:
        # Clean up temporary YAML file if created
        if tmp_yaml_path and Path(tmp_yaml_path).exists():
            import os
            os.unlink(tmp_yaml_path)


def cmd_val(args):
    """Validate YOLOv12-3D model."""
    print_banner()
    print_section("Validation Configuration")

    # Validate inputs
    if not Path(args.model).exists():
        print_error(f"Model not found: {args.model}")
        return 1

    if not Path(args.data).exists():
        print_error(f"Dataset config not found: {args.data}")
        return 1

    # Print configuration
    print(f"""
  Model:      {Colors.BOLD}{args.model}{Colors.ENDC}
  Dataset:    {Colors.BOLD}{args.data}{Colors.ENDC}
  Batch Size: {Colors.BOLD}{args.batch}{Colors.ENDC}
  Image Size: {Colors.BOLD}{args.imgsz}{Colors.ENDC}
  Device:     {Colors.BOLD}{args.device}{Colors.ENDC}
""")

    print_section("Running Validation")

    try:
        model = YOLO(args.model)
        metrics = model.val(
            data=args.data,
            batch=args.batch,
            imgsz=args.imgsz,
            device=args.device,
            plots=args.plots,
            save_json=args.save_json,
            verbose=args.verbose,
        )

        print_section("Validation Results")
        print(f"""
  Metrics (2D Detection):
    mAP50-95: {Colors.BOLD}{metrics.box.map:.4f}{Colors.ENDC}
    mAP50:    {Colors.BOLD}{metrics.box.map50:.4f}{Colors.ENDC}
    mAP75:    {Colors.BOLD}{metrics.box.map75:.4f}{Colors.ENDC}

  Per Class:
    Precision: {Colors.BOLD}{metrics.box.mp:.4f}{Colors.ENDC}
    Recall:    {Colors.BOLD}{metrics.box.mr:.4f}{Colors.ENDC}
""")

        print_success("Validation complete!")
        return 0

    except Exception as e:
        print_error(f"Validation failed: {e}")
        return 1


def cmd_predict(args):
    """Run inference with YOLOv12-3D model."""
    print_banner()
    print_section("Inference Configuration")

    # Validate inputs
    if not Path(args.model).exists():
        print_error(f"Model not found: {args.model}")
        return 1

    if not Path(args.source).exists():
        print_error(f"Source not found: {args.source}")
        return 1

    # Print configuration
    print(f"""
  Model:       {Colors.BOLD}{args.model}{Colors.ENDC}
  Source:      {Colors.BOLD}{args.source}{Colors.ENDC}
  Confidence:  {Colors.BOLD}{args.conf}{Colors.ENDC}
  IoU:         {Colors.BOLD}{args.iou}{Colors.ENDC}
  Image Size:  {Colors.BOLD}{args.imgsz}{Colors.ENDC}
  Device:      {Colors.BOLD}{args.device}{Colors.ENDC}
  Save:        {Colors.BOLD}{args.save}{Colors.ENDC}
""")

    print_section("Running Inference")

    try:
        model = YOLO(args.model)

        results = model.predict(
            source=args.source,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=args.device,
            save=args.save,
            save_txt=args.save_txt,
            save_conf=args.save_conf,
            show=args.show,
            verbose=args.verbose,
            project=args.project,
            name=args.name,
        )

        # Count detections
        total_detections = sum(len(r.boxes) for r in results)

        print_section("Inference Results")
        print(f"""
  Images processed: {Colors.BOLD}{len(results)}{Colors.ENDC}
  Total detections: {Colors.BOLD}{total_detections}{Colors.ENDC}
  Average per image: {Colors.BOLD}{total_detections / len(results):.1f}{Colors.ENDC}
""")

        if args.save:
            save_dir = Path(args.project) / args.name if args.project else Path("runs/detect") / args.name
            print_success(f"Results saved to: {save_dir}")

        return 0

    except Exception as e:
        print_error(f"Inference failed: {e}")
        return 1


def cmd_export(args):
    """Export YOLOv12-3D model to different formats."""
    print_banner()
    print_section("Export Configuration")

    # Validate inputs
    if not Path(args.model).exists():
        print_error(f"Model not found: {args.model}")
        return 1

    # Print configuration
    print(f"""
  Model:       {Colors.BOLD}{args.model}{Colors.ENDC}
  Format:      {Colors.BOLD}{args.format}{Colors.ENDC}
  Image Size:  {Colors.BOLD}{args.imgsz}{Colors.ENDC}
  Half:        {Colors.BOLD}{args.half}{Colors.ENDC}
  Optimize:    {Colors.BOLD}{args.optimize}{Colors.ENDC}
""")

    print_section(f"Exporting to {args.format.upper()}")

    try:
        model = YOLO(args.model)

        exported_model = model.export(
            format=args.format,
            imgsz=args.imgsz,
            half=args.half,
            optimize=args.optimize,
            int8=args.int8,
            dynamic=args.dynamic,
            simplify=args.simplify,
        )

        print_success(f"Model exported to: {exported_model}")
        return 0

    except Exception as e:
        print_error(f"Export failed: {e}")
        return 1


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="yolo3d",
        description="YOLOv12-3D Command Line Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train model
  yolo3d train --data kitti-3d.yaml --epochs 100 --batch 16

  # Validate model
  yolo3d val --model runs/detect/train/weights/best.pt --data kitti-3d.yaml

  # Run inference
  yolo3d predict --model best.pt --source images/ --conf 0.25

  # Export model
  yolo3d export --model best.pt --format onnx

Developed by AI Research Group, Civil Engineering, KMUTT
For more information: python yolo3d.py <command> --help
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # ============================================================================
    # TRAIN COMMAND
    # ============================================================================
    parser_train = subparsers.add_parser("train", help="Train YOLOv12-3D model")

    # Model & Data
    parser_train.add_argument(
        "--model", type=str, default="n", help="Model config or weights (n/s/m/l/x, .yaml, or .pt)"
    )
    parser_train.add_argument("--data", type=str, required=True, help="Dataset config file (.yaml)")

    # Training parameters
    parser_train.add_argument("--epochs", type=int, default=100, help="Number of epochs (default: 100)")
    parser_train.add_argument("--batch", type=int, default=16, help="Batch size per GPU (default: 16)")
    parser_train.add_argument("--nbs", type=int, default=-1, help="Nominal batch size for gradient accumulation. Set to batch size to disable accumulation. Default: -1 (auto=64)")
    parser_train.add_argument("--imgsz", type=int, default=640, help="Image size (default: 640)")
    parser_train.add_argument("--device", type=str, default="0", help="Device (0, 1, 2, etc. or cpu)")
    parser_train.add_argument("--workers", type=int, default=8, help="Number of workers (default: 8)")
    parser_train.add_argument("--name", type=str, default="train", help="Experiment name (default: train)")

    # Optimization
    parser_train.add_argument(
        "--optimizer", type=str, default="SGD", choices=["SGD", "Adam", "AdamW"], help="Optimizer (default: SGD)"
    )
    parser_train.add_argument("--lr0", type=float, default=0.01, help="Initial learning rate (default: 0.01)")
    parser_train.add_argument("--lrf", type=float, default=0.01, help="Final learning rate factor (default: 0.01)")
    parser_train.add_argument("--momentum", type=float, default=0.937, help="SGD momentum (default: 0.937)")
    parser_train.add_argument("--weight-decay", type=float, default=0.0005, help="Weight decay (default: 0.0005)")
    parser_train.add_argument("--warmup-epochs", type=float, default=3.0, help="Warmup epochs (default: 3.0)")

    # Loss weights
    parser_train.add_argument("--box", type=float, default=7.5, help="Box loss weight (default: 7.5)")
    parser_train.add_argument("--cls", type=float, default=0.5, help="Class loss weight (default: 0.5)")
    parser_train.add_argument("--dfl", type=float, default=1.5, help="DFL loss weight (default: 1.5)")

    # Other options
    parser_train.add_argument("--patience", type=int, default=50, help="Early stopping patience (default: 50)")
    parser_train.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")
    parser_train.add_argument("--pretrained", action="store_true", help="Use pretrained weights")
    parser_train.add_argument("--save", action="store_true", default=True, help="Save checkpoints")
    parser_train.add_argument("--plots", action="store_true", default=True, help="Save training plots")
    parser_train.add_argument("--val", action="store_true", default=True, help="Validate during training")
    parser_train.add_argument("--valpercent", type=float, default=100.0, help="Percentage of validation data to use (1-100, default: 100). Use lower values for faster validation.")
    parser_train.add_argument("--verbose", action="store_true", help="Verbose output")
    parser_train.add_argument("-y", "--yes", action="store_true", help="Skip confirmation")

    parser_train.set_defaults(func=cmd_train)

    # ============================================================================
    # VAL COMMAND
    # ============================================================================
    parser_val = subparsers.add_parser("val", help="Validate YOLOv12-3D model")

    parser_val.add_argument("--model", type=str, required=True, help="Model weights (.pt)")
    parser_val.add_argument("--data", type=str, required=True, help="Dataset config file (.yaml)")
    parser_val.add_argument("--batch", type=int, default=16, help="Batch size (default: 16)")
    parser_val.add_argument("--imgsz", type=int, default=640, help="Image size (default: 640)")
    parser_val.add_argument("--device", type=str, default="0", help="Device (0, 1, 2, etc. or cpu)")
    parser_val.add_argument("--plots", action="store_true", help="Save validation plots")
    parser_val.add_argument("--save-json", action="store_true", help="Save results to JSON")
    parser_val.add_argument("--verbose", action="store_true", help="Verbose output")

    parser_val.set_defaults(func=cmd_val)

    # ============================================================================
    # PREDICT COMMAND
    # ============================================================================
    parser_predict = subparsers.add_parser("predict", help="Run inference")

    parser_predict.add_argument("--model", type=str, required=True, help="Model weights (.pt)")
    parser_predict.add_argument("--source", type=str, required=True, help="Source (image, folder, video)")
    parser_predict.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (default: 0.25)")
    parser_predict.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold (default: 0.45)")
    parser_predict.add_argument("--imgsz", type=int, default=640, help="Image size (default: 640)")
    parser_predict.add_argument("--device", type=str, default="0", help="Device (0, 1, 2, etc. or cpu)")
    parser_predict.add_argument("--save", action="store_true", default=True, help="Save results")
    parser_predict.add_argument("--save-txt", action="store_true", help="Save results as txt")
    parser_predict.add_argument("--save-conf", action="store_true", help="Save confidence scores")
    parser_predict.add_argument("--show", action="store_true", help="Show results")
    parser_predict.add_argument("--project", type=str, default="runs/detect", help="Project directory")
    parser_predict.add_argument("--name", type=str, default="predict", help="Experiment name")
    parser_predict.add_argument("--verbose", action="store_true", help="Verbose output")

    parser_predict.set_defaults(func=cmd_predict)

    # ============================================================================
    # EXPORT COMMAND
    # ============================================================================
    parser_export = subparsers.add_parser("export", help="Export model")

    parser_export.add_argument("--model", type=str, required=True, help="Model weights (.pt)")
    parser_export.add_argument(
        "--format",
        type=str,
        default="onnx",
        choices=["onnx", "torchscript", "tflite", "edgetpu", "tfjs", "coreml", "engine"],
        help="Export format (default: onnx)",
    )
    parser_export.add_argument("--imgsz", type=int, default=640, help="Image size (default: 640)")
    parser_export.add_argument("--half", action="store_true", help="FP16 quantization")
    parser_export.add_argument("--int8", action="store_true", help="INT8 quantization")
    parser_export.add_argument("--dynamic", action="store_true", help="Dynamic axes")
    parser_export.add_argument("--simplify", action="store_true", help="Simplify ONNX model")
    parser_export.add_argument("--optimize", action="store_true", help="Optimize for mobile")

    parser_export.set_defaults(func=cmd_export)

    # Parse arguments
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
