#!/usr/bin/env python3
"""
YOLOv12-3D with DINO ViT-B Integration - CLI Tool

Developed by AI Research Group
Department of Civil Engineering
King Mongkut's University of Technology Thonburi (KMUTT)

A unified command-line interface for YOLOv12-3D operations with DINO ViT-B integration.

Features:
- YOLOv12-3D with DINO ViT-B backbone integration
- Support for P0 and P3 level DINO enhancement
- M (Medium) and L (Large) model variants
- 3D object detection capabilities

Usage:
    yolodio3d train --help
    yolodio3d val --help
    yolodio3d predict --help
    yolodio3d export --help
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
║                  YOLOv12-3D with DINO ViT-B CLI Tool                     ║
║                                                                              ║
║              AI Research Group, Civil Engineering, KMUTT                     ║
║                                                                              ║
║  Model Variants:                                                            ║
║    • YOLOv12m-3d-dino: Medium model with DINO P0 or P0+P3 integration      ║
║    • YOLOv12l-3d-dino: Large model with DINO P0 or P0+P3 integration       ║
║                                                                              ║
║  DINO Integration Options:                                                  ║
║    • Single (P0): Early feature enhancement for lightweight DINO           ║
║    • Dual (P0+P3): Dual-scale enhancement for maximum performance          ║
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


def get_model_config(yolo_size, use_dino=True, dino_integration="dual"):
    """
    Generate model configuration path based on components.

    Args:
        yolo_size: YOLO size (m/l)
        use_dino: Whether to use DINO enhancement
        dino_integration: DINO integration type ('single' for P0 only, 'dual' for P0+P3)

    Returns:
        Path to model YAML file
    """
    base_dir = Path("ultralytics/cfg/models/v12")

    if not use_dino:
        # Base YOLOv12-3D model
        return base_dir / "yolov12-3d.yaml"

    # DINO-enhanced model
    if dino_integration == "single":
        model_name = f"yolov12{yolo_size}-3d-dino-p0.yaml"
    elif dino_integration == "dual":
        model_name = f"yolov12{yolo_size}-3d-dino-p0p3.yaml"
    else:
        # Default to dual
        model_name = f"yolov12{yolo_size}-3d-dino-p0p3.yaml"

    model_path = base_dir / model_name

    # Check if model exists
    if not model_path.exists():
        print_warning(f"Model config not found: {model_path}")
        print_info(f"Falling back to base yolov12-3d.yaml")
        return base_dir / "yolov12-3d.yaml"

    return model_path


def cmd_train(args):
    """Train YOLOv12-3D-DINO model."""
    print_banner()
    print_section("Training Configuration")

    # Validate inputs
    if not Path(args.data).exists():
        print_error(f"Dataset config not found: {args.data}")
        print_info("Example: ultralytics/cfg/datasets/kitti-3d.yaml")
        sys.exit(1)

    # Get model config
    model_config = get_model_config(args.yolo_size, args.use_dino, args.dino_integration)
    print_info(f"Model config: {model_config}")

    # Validate model size
    if args.yolo_size not in ['m', 'l']:
        print_error(f"Invalid YOLO size: {args.yolo_size}. Must be 'm' or 'l'")
        sys.exit(1)

    # Validate valpercent
    if args.valpercent < 1 or args.valpercent > 100:
        print_error(f"Invalid --valpercent value: {args.valpercent}")
        print_info("Must be between 1 and 100")
        sys.exit(1)

    # Warn if using very low validation percentage
    if args.valpercent < 10:
        print_warning(f"Using only {args.valpercent}% of validation data - metrics may not be representative!")
        print_info("Recommended minimum: --valpercent 10")

    # Print DINO info if enabled
    if args.use_dino:
        integration_type = "Dual (P0+P3)" if args.dino_integration == "dual" else "Single (P0 only)"
        print_success(f"DINO ViT-B integration enabled: {integration_type}")
        print_info(f"Model: YOLOv12{args.yolo_size}-3D-DINO")
    else:
        print_info(f"Model: YOLOv12{args.yolo_size}-3D (Base)")

    # Print training settings
    print_section("Training Settings")
    print_info(f"Dataset: {args.data}")
    print_info(f"Model size: {args.yolo_size}")
    print_info(f"Epochs: {args.epochs}")
    print_info(f"Batch size: {args.batch_size}")
    print_info(f"Image size: {args.imgsz}")
    print_info(f"Device: {args.device}")
    print_info(f"Workers: {args.workers}")
    print_info(f"Output directory: {args.project}/{args.name}")


    # Validation data percentage
    val_info = f"{args.valpercent}%" if args.valpercent < 100.0 else "100% (full)"
    print_info(f"Validation data: {val_info}")
    if args.freeze_dino:
        print_info("DINO backbone: FROZEN (recommended for faster training)")

    # Build training arguments
    train_args = {
        'data': args.data,
        'epochs': args.epochs,
        'batch': args.batch_size,
        'imgsz': args.imgsz,
        'device': args.device,
        'workers': args.workers,
        'project': args.project,
        'name': args.name,
        'exist_ok': args.exist_ok,
        'resume': args.resume,
        'pretrained': args.pretrained,
        'valpercent': args.valpercent,
    }

    # Add DINO-specific args
    if args.use_dino and args.freeze_dino:
        train_args['freeze'] = 10  # Freeze first 10 layers including DINO
        print_info("Freezing backbone layers for faster training")

    print_section("Starting Training")
    print_banner()

    # Load model and start training
    try:
        model = YOLO(str(model_config))
        model.train(**train_args)
        print_success("Training completed successfully!")
    except Exception as e:
        print_error(f"Training failed: {e}")
        sys.exit(1)


def cmd_val(args):
    """Validate YOLOv12-3D-DINO model."""
    print_banner()
    print_section("Validation Configuration")

    # Get model config
    model_config = get_model_config(args.yolo_size, args.use_dino, args.dino_integration)
    print_info(f"Model config: {model_config}")

    # Print DINO info if enabled
    if args.use_dino:
        integration_type = "Dual (P0+P3)" if args.dino_integration == "dual" else "Single (P0 only)"
        print_success(f"DINO ViT-B integration: {integration_type}")
    else:
        print_info("Base YOLOv12-3D (no DINO)")

    # Build validation arguments
    val_args = {
        'model': args.model if args.model else str(model_config),
        'data': args.data,
        'batch': args.batch_size,
        'imgsz': args.imgsz,
        'device': args.device,
        'workers': args.workers,
        'project': args.project,
        'name': args.name,
        'save_json': args.save_json,
        'save_txt': args.save_txt,
    }

    print_section("Starting Validation")

    try:
        model = YOLO(val_args['model'])
        model.val(**val_args)
        print_success("Validation completed successfully!")
    except Exception as e:
        print_error(f"Validation failed: {e}")
        sys.exit(1)


def cmd_predict(args):
    """Predict using YOLOv12-3D-DINO model."""
    print_banner()
    print_section("Prediction Configuration")

    # Build prediction arguments
    pred_args = {
        'model': args.model,
        'source': args.source,
        'imgsz': args.imgsz,
        'batch': args.batch_size,
        'device': args.device,
        'conf': args.conf,
        'iou': args.iou,
        'save': args.save,
        'save_txt': args.save_txt,
        'save_conf': args.save_conf,
        'project': args.project,
        'name': args.name,
        'exist_ok': args.exist_ok,
    }

    print_info(f"Model: {args.model}")
    print_info(f"Source: {args.source}")
    print_info(f"Confidence threshold: {args.conf}")
    print_info(f"IOU threshold: {args.iou}")

    print_section("Starting Prediction")

    try:
        model = YOLO(args.model)
        model.predict(**pred_args)
        print_success("Prediction completed successfully!")
    except Exception as e:
        print_error(f"Prediction failed: {e}")
        sys.exit(1)


def cmd_export(args):
    """Export YOLOv12-3D-DINO model."""
    print_banner()
    print_section("Export Configuration")

    # Get model config
    model_config = get_model_config(args.yolo_size, args.use_dino, args.dino_integration)

    # Print DINO info if enabled
    if args.use_dino:
        integration_type = "Dual (P0+P3)" if args.dino_integration == "dual" else "Single (P0 only)"
        print_success(f"DINO ViT-B integration: {integration_type}")
    else:
        print_info("Base YOLOv12-3D (no DINO)")

    # Build export arguments
    export_args = {
        'model': args.model if args.model else str(model_config),
        'format': args.format,
        'imgsz': args.imgsz,
        'dynamic': args.dynamic,
        'simplify': args.simplify,
        'opset': args.opset,
        'save_dir': args.save_dir,
        'project': args.project,
        'name': args.name,
    }

    print_info(f"Model: {export_args['model']}")
    print_info(f"Format: {args.format}")
    print_info(f"Image size: {args.imgsz}")

    print_section("Starting Export")

    try:
        model = YOLO(export_args['model'])
        model.export(**export_args)
        print_success("Export completed successfully!")
    except Exception as e:
        print_error(f"Export failed: {e}")
        sys.exit(1)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="YOLOv12-3D with DINO ViT-B Integration - CLI Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train YOLOv12m-3D with DINO (dual P0+P3) on KITTI dataset
  python yolodio3d.py train --data ultralytics/cfg/datasets/kitti-3d.yaml --yolo-size m --dino-integration dual --epochs 100 --batch-size 16

  # Train YOLOv12l-3D with DINO (single P0 only) on KITTI dataset
  python yolodio3d.py train --data ultralytics/cfg/datasets/kitti-3d.yaml --yolo-size l --dino-integration single --epochs 100 --batch-size 8

  # Train YOLOv12m-3D with DINO (dual P0+P3) on KITTI dataset
  python yolodio3d.py train --data ultralytics/cfg/datasets/kitti-3d.yaml --yolo-size m --dino-integration dual --epochs 100 --batch-size 16

  # Train without DINO (base YOLOv12-3D)
  python yolodio3d.py train --data ultralytics/cfg/datasets/kitti-3d.yaml --yolo-size m --no-dino --epochs 100

  # Validate model
  python yolodio3d.py val --data ultralytics/cfg/datasets/kitti-3d.yaml --yolo-size m

  # Predict on images
  python yolodio3d.py predict --model runs/train/exp/weights/best.pt --source images/

  # Export to ONNX
  python yolodio3d.py export --format onnx --yolo-size m
        """
    )

    # Global arguments
    parser.add_argument('--version', action='version', version='%(prog)s 1.0.0')

    # Create subparsers
    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Train command
    parser_train = subparsers.add_parser('train', help='Train a model')
    parser_train.add_argument('--data', required=True, help='Path to dataset YAML file')
    parser_train.add_argument('--yolo-size', default='m', choices=['m', 'l'],
                              help='YOLO model size (default: m)')
    parser_train.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser_train.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser_train.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser_train.add_argument('--device', default='', help='CUDA device (e.g., 0,0)')
    parser_train.add_argument('--workers', type=int, default=8, help='DataLoader workers')
    parser_train.add_argument('--project', default='runs/train', help='Project name')
    parser_train.add_argument('--name', default='yolov12-3d-dino', help='Experiment name')
    parser_train.add_argument('--exist-ok', action='store_true', help='Overwrite existing experiment')
    parser_train.add_argument('--resume', action='store_true', help='Resume training')
    parser_train.add_argument('--pretrained', action='store_true', help='Use pretrained weights')
    parser_train.add_argument('--no-dino', dest='use_dino', action='store_false',
                              help='Disable DINO integration (use base YOLOv12-3D)')
    parser_train.add_argument('--dino-integration', default='dual', choices=['single', 'dual'],
                              help='DINO integration type: single=P0 only, dual=P0+P3 (default: dual)')
    parser_train.add_argument('--freeze-dino', action='store_true', default=True,
                              help='Freeze DINO backbone weights (recommended)')
    parser_train.add_argument('--valpercent', type=float, default=100.0,
                              help='Percentage of validation data to use (1-100, default: 100)')
    parser_train.set_defaults(func=cmd_train, use_dino=True)

    # Val command
    parser_val = subparsers.add_parser('val', help='Validate a model')
    parser_val.add_argument('--model', help='Path to model file')
    parser_val.add_argument('--data', required=True, help='Path to dataset YAML file')
    parser_val.add_argument('--yolo-size', default='m', choices=['m', 'l'],
                            help='YOLO model size (default: m)')
    parser_val.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser_val.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser_val.add_argument('--device', default='', help='CUDA device (e.g., 0,0)')
    parser_val.add_argument('--workers', type=int, default=8, help='DataLoader workers')
    parser_val.add_argument('--project', default='runs/val', help='Project name')
    parser_val.add_argument('--name', default='val', help='Experiment name')
    parser_val.add_argument('--save-json', action='store_true', help='Save results to JSON')
    parser_val.add_argument('--save-txt', action='store_true', help='Save results to TXT')
    parser_val.add_argument('--no-dino', dest='use_dino', action='store_false',
                            help='Disable DINO integration')
    parser_val.add_argument('--dino-integration', default='dual', choices=['single', 'dual'],
                            help='DINO integration type: single=P0 only, dual=P0+P3 (default: dual)')
    parser_val.set_defaults(func=cmd_val, use_dino=True)

    # Predict command
    parser_pred = subparsers.add_parser('predict', help='Predict using a model')
    parser_pred.add_argument('--model', required=True, help='Path to model file')
    parser_pred.add_argument('--source', required=True, help='Source (image/video/directory)')
    parser_pred.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser_pred.add_argument('--batch-size', type=int, default=1, help='Batch size')
    parser_pred.add_argument('--device', default='', help='CUDA device (e.g., 0,0)')
    parser_pred.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser_pred.add_argument('--iou', type=float, default=0.7, help='IOU threshold')
    parser_pred.add_argument('--save', action='store_true', help='Save results')
    parser_pred.add_argument('--save-txt', action='store_true', help='Save results to TXT')
    parser_pred.add_argument('--save-conf', action='store_true', help='Save confidence scores')
    parser_pred.add_argument('--project', default='runs/predict', help='Project name')
    parser_pred.add_argument('--name', default='predict', help='Experiment name')
    parser_pred.add_argument('--exist-ok', action='store_true', help='Overwrite existing results')
    parser_pred.set_defaults(func=cmd_predict)

    # Export command
    parser_export = subparsers.add_parser('export', help='Export a model')
    parser_export.add_argument('--model', help='Path to model file')
    parser_export.add_argument('--yolo-size', default='m', choices=['m', 'l'],
                               help='YOLO model size (default: m)')
    parser_export.add_argument('--format', default='onnx',
                               choices=['onnx', 'torchscript', 'openvino', 'engine', 'coreml', 'mlmodel'],
                               help='Export format')
    parser_export.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser_export.add_argument('--dynamic', action='store_true', help='Dynamic ONNX axes')
    parser_export.add_argument('--simplify', action='store_true', help='Simplify ONNX model')
    parser_export.add_argument('--opset', type=int, default=11, help='ONNX opset version')
    parser_export.add_argument('--save-dir', default='exports', help='Output directory')
    parser_export.add_argument('--project', default='runs/export', help='Project name')
    parser_export.add_argument('--name', default='export', help='Experiment name')
    parser_export.add_argument('--no-dino', dest='use_dino', action='store_false',
                               help='Disable DINO integration')
    parser_export.add_argument('--dino-integration', default='dual', choices=['single', 'dual'],
                               help='DINO integration type: single=P0 only, dual=P0+P3 (default: dual)')
    parser_export.set_defaults(func=cmd_export, use_dino=True)

    # Parse arguments
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Run the command
    args.func(args)


if __name__ == '__main__':
    main()
