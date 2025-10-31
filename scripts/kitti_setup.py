#!/usr/bin/env python3
"""
KITTI 3D Object Detection Dataset Setup CLI

Developed by AI Research Group
Department of Civil Engineering
King Mongkut's University of Technology Thonburi (KMUTT)

A command-line tool for downloading and preparing KITTI dataset.

Usage:
    kitti_setup download --help
    kitti_setup extract --help
    kitti_setup verify --help
    kitti_setup split --help
"""

import argparse
import sys
import zipfile
from pathlib import Path
from typing import Optional, Tuple, List
import random

try:
    from tqdm import tqdm

    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("Warning: tqdm not installed. Install for progress bars: pip install tqdm")


class Colors:
    """ANSI color codes for terminal output."""

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
║                    KITTI 3D Object Detection Setup Tool                      ║
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


class KITTISetup:
    """KITTI dataset setup manager."""

    REQUIRED_FILES = {
        "images": "data_object_image_2.zip",
        "labels": "data_object_label_2.zip",
        "calib": "data_object_calib.zip",
    }

    FILE_SIZES = {
        "images": 12.0,  # GB
        "labels": 0.005,  # GB
        "calib": 0.016,  # GB
    }

    KITTI_CLASSES = ["Car", "Truck", "Pedestrian", "Cyclist", "Misc", "Van", "Tram", "Person_sitting"]

    def __init__(self, data_dir: str = "./datasets/kitti", download_dir: str = "./downloads"):
        """Initialize KITTI setup."""
        self.data_dir = Path(data_dir)
        self.download_dir = Path(download_dir)
        self.training_dir = self.data_dir / "training"
        self.testing_dir = self.data_dir / "testing"

    def check_downloads(self) -> Tuple[dict, list]:
        """Check if required files are downloaded."""
        found_files = {}
        missing_files = []

        print_section("Checking Downloaded Files")

        for key, filename in self.REQUIRED_FILES.items():
            filepath = self.download_dir / filename
            if filepath.exists():
                size_gb = filepath.stat().st_size / (1024**3)
                found_files[key] = filepath
                print_success(f"{filename} ({size_gb:.2f} GB)")
            else:
                missing_files.append(filename)
                print_error(f"{filename} - Not found")

        return found_files, missing_files

    def print_download_instructions(self):
        """Print download instructions."""
        print_section("Manual Download Required")

        print(f"""
{Colors.BOLD}KITTI dataset requires accepting a license agreement.{Colors.ENDC}

{Colors.OKCYAN}Steps to download:{Colors.ENDC}

  1. Visit: {Colors.UNDERLINE}http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d{Colors.ENDC}

  2. Register/Login to your KITTI account

  3. Download these files:
""")

        for key, filename in self.REQUIRED_FILES.items():
            size = self.FILE_SIZES[key]
            print(f"     • {Colors.BOLD}{filename}{Colors.ENDC} ({size:.2f} GB)")

        print(f"""
  4. Place downloaded files in: {Colors.OKCYAN}{self.download_dir.resolve()}{Colors.ENDC}

  5. Run: {Colors.BOLD}python scripts/kitti_setup.py extract{Colors.ENDC}
""")

    def extract_files(self, found_files: dict) -> bool:
        """Extract downloaded files."""
        print_section("Extracting Files")

        # Create data directory
        self.data_dir.mkdir(parents=True, exist_ok=True)

        for key, filepath in found_files.items():
            print_info(f"Extracting {filepath.name}...")

            try:
                with zipfile.ZipFile(filepath, "r") as zip_ref:
                    members = zip_ref.namelist()

                    if TQDM_AVAILABLE:
                        with tqdm(
                            total=len(members),
                            desc=f"  {filepath.name}",
                            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}",
                        ) as pbar:
                            for member in members:
                                zip_ref.extract(member, self.data_dir)
                                pbar.update(1)
                    else:
                        for i, member in enumerate(members):
                            zip_ref.extract(member, self.data_dir)
                            if (i + 1) % 100 == 0:
                                print(f"    Extracted {i + 1}/{len(members)} files...", end="\r")
                        print()

                print_success(f"{filepath.name} extracted successfully")

            except Exception as e:
                print_error(f"Failed to extract {filepath.name}: {e}")
                return False

        return True

    def verify_structure(self) -> bool:
        """Verify dataset structure."""
        print_section("Verifying Dataset Structure")

        required_dirs = [self.training_dir / "image_2", self.training_dir / "label_2", self.training_dir / "calib"]

        all_exist = True
        for dir_path in required_dirs:
            if dir_path.exists():
                print_success(f"{dir_path.relative_to(self.data_dir)}")
            else:
                print_error(f"{dir_path.relative_to(self.data_dir)} - Missing!")
                all_exist = False

        if not all_exist:
            return False

        # Count files
        n_images = len(list((self.training_dir / "image_2").glob("*.png")))
        n_labels = len(list((self.training_dir / "label_2").glob("*.txt")))
        n_calib = len(list((self.training_dir / "calib").glob("*.txt")))

        print_section("Dataset Statistics")
        print(f"""
  Training Images:      {Colors.BOLD}{n_images:>6}{Colors.ENDC}
  Training Labels:      {Colors.BOLD}{n_labels:>6}{Colors.ENDC}
  Calibration Files:    {Colors.BOLD}{n_calib:>6}{Colors.ENDC}
""")

        if n_images == n_labels == n_calib == 7481:
            print_success("Dataset verification passed!")
            return True
        else:
            print_warning("File counts don't match expected values (7481)")
            return True  # Still return True as some variation is acceptable

    def create_splits(self, val_split: float = 0.2, seed: int = 42) -> bool:
        """Create train/val splits."""
        print_section(f"Creating Train/Val Split ({int((1 - val_split) * 100)}% / {int(val_split * 100)}%)")

        image_dir = self.training_dir / "image_2"
        if not image_dir.exists():
            print_error("Training images not found!")
            return False

        # Get all images
        image_files = sorted(list(image_dir.glob("*.png")))
        n_total = len(image_files)
        n_val = int(n_total * val_split)
        n_train = n_total - n_val

        # Shuffle and split
        random.seed(seed)
        indices = list(range(n_total))
        random.shuffle(indices)

        train_indices = sorted(indices[:n_train])
        val_indices = sorted(indices[n_train:])

        # Create splits directory
        splits_dir = self.data_dir / "ImageSets"
        splits_dir.mkdir(exist_ok=True)

        # Write split files
        with open(splits_dir / "train.txt", "w") as f:
            for idx in train_indices:
                f.write(f"{image_files[idx].stem}\n")

        with open(splits_dir / "val.txt", "w") as f:
            for idx in val_indices:
                f.write(f"{image_files[idx].stem}\n")

        with open(splits_dir / "trainval.txt", "w") as f:
            for img in image_files:
                f.write(f"{img.stem}\n")

        print(f"""
  Training samples:     {Colors.BOLD}{n_train:>6}{Colors.ENDC}  ({int((1 - val_split) * 100)}%)
  Validation samples:   {Colors.BOLD}{n_val:>6}{Colors.ENDC}  ({int(val_split * 100)}%)
  Total samples:        {Colors.BOLD}{n_total:>6}{Colors.ENDC}

  Split files saved to: {Colors.OKCYAN}{splits_dir}{Colors.ENDC}
""")

        print_success("Train/val split created successfully")
        return True

    def create_yaml(self, output_path: Optional[str] = None) -> bool:
        """Create dataset YAML configuration."""
        print_section("Creating Dataset Configuration")

        if output_path is None:
            output_path = self.data_dir / "kitti-3d.yaml"
        else:
            output_path = Path(output_path)

        yaml_content = f"""# KITTI 3D Object Detection Dataset
# Auto-generated by kitti_setup.py

path: {self.data_dir.resolve()}
train: training/image_2
val: training/image_2
test: testing/image_2

names:
  0: Car
  1: Truck
  2: Pedestrian
  3: Cyclist
  4: Misc
  5: Van
  6: Tram
  7: Person_sitting

nc: 8
task: detect3d

# Train/val splits
train_split: ImageSets/train.txt
val_split: ImageSets/val.txt
"""

        with open(output_path, "w") as f:
            f.write(yaml_content)

        print_success(f"Dataset config created: {Colors.OKCYAN}{output_path}{Colors.ENDC}")
        return True


def cmd_download(args):
    """Download command - shows instructions."""
    print_banner()
    setup = KITTISetup(args.data_dir, args.download_dir)

    # Create download directory
    setup.download_dir.mkdir(parents=True, exist_ok=True)

    # Check existing downloads
    found_files, missing_files = setup.check_downloads()

    if missing_files:
        setup.print_download_instructions()
        print_warning(f"Missing {len(missing_files)} file(s). Please download manually.")
        return 1
    else:
        print_success("All files downloaded!")
        print_info("Run 'python scripts/kitti_setup.py extract' to continue")
        return 0


def cmd_extract(args):
    """Extract command - extracts downloaded files."""
    print_banner()
    setup = KITTISetup(args.data_dir, args.download_dir)

    # Check downloads
    found_files, missing_files = setup.check_downloads()

    if missing_files:
        print_error(f"Missing {len(missing_files)} file(s)")
        setup.print_download_instructions()
        return 1

    # Extract files
    if not setup.extract_files(found_files):
        print_error("Extraction failed")
        return 1

    print_success("Extraction completed successfully!")
    print_info("Run 'python scripts/kitti_setup.py verify' to verify")
    return 0


def cmd_verify(args):
    """Verify command - verifies dataset structure."""
    print_banner()
    setup = KITTISetup(args.data_dir, args.download_dir)

    if not setup.verify_structure():
        print_error("Verification failed")
        return 1

    print_success("Dataset verified successfully!")
    print_info("Run 'python scripts/kitti_setup.py split' to create train/val splits")
    return 0


def cmd_split(args):
    """Split command - creates train/val splits."""
    print_banner()
    setup = KITTISetup(args.data_dir, args.download_dir)

    if not setup.create_splits(args.val_split, args.seed):
        print_error("Split creation failed")
        return 1

    print_success("Splits created successfully!")

    if args.create_yaml:
        setup.create_yaml()

    print_section("Setup Complete!")
    print(f"""
{Colors.OKGREEN}✓ KITTI dataset is ready for training!{Colors.ENDC}

{Colors.BOLD}Next steps:{Colors.ENDC}

  1. Update config if needed:
     {Colors.OKCYAN}ultralytics/cfg/datasets/kitti-3d.yaml{Colors.ENDC}

  2. Start training:
     {Colors.BOLD}python examples/train_kitti_3d.py{Colors.ENDC}
""")
    return 0


def cmd_all(args):
    """All command - runs complete setup."""
    print_banner()
    setup = KITTISetup(args.data_dir, args.download_dir)

    # Check downloads
    found_files, missing_files = setup.check_downloads()
    if missing_files:
        print_error(f"Missing {len(missing_files)} file(s)")
        setup.print_download_instructions()
        return 1

    # Extract
    if not setup.extract_files(found_files):
        return 1

    # Verify
    if not setup.verify_structure():
        return 1

    # Split
    if not setup.create_splits(args.val_split, args.seed):
        return 1

    # Create YAML
    if args.create_yaml:
        setup.create_yaml()

    print_section("Setup Complete!")
    print(f"""
{Colors.OKGREEN}✓ KITTI dataset is ready for training!{Colors.ENDC}

{Colors.BOLD}Dataset location:{Colors.ENDC} {Colors.OKCYAN}{setup.data_dir.resolve()}{Colors.ENDC}

{Colors.BOLD}Next steps:{Colors.ENDC}

  1. Start training:
     {Colors.BOLD}python examples/train_kitti_3d.py{Colors.ENDC}
""")
    return 0


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="KITTI 3D Object Detection Dataset Setup Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check downloads and show instructions
  python scripts/kitti_setup.py download

  # Extract downloaded files
  python scripts/kitti_setup.py extract

  # Verify dataset structure
  python scripts/kitti_setup.py verify

  # Create train/val splits
  python scripts/kitti_setup.py split --val-split 0.2

  # Run complete setup (extract + verify + split)
  python scripts/kitti_setup.py all --create-yaml

Developed by AI Research Group, Civil Engineering, KMUTT
        """,
    )

    # Global arguments
    parser.add_argument(
        "--data-dir", type=str, default="./datasets/kitti", help="Dataset directory (default: ./datasets/kitti)"
    )
    parser.add_argument(
        "--download-dir", type=str, default="./downloads", help="Download directory (default: ./downloads)"
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Download command
    parser_download = subparsers.add_parser("download", help="Check downloads and show instructions")
    parser_download.set_defaults(func=cmd_download)

    # Extract command
    parser_extract = subparsers.add_parser("extract", help="Extract downloaded files")
    parser_extract.set_defaults(func=cmd_extract)

    # Verify command
    parser_verify = subparsers.add_parser("verify", help="Verify dataset structure")
    parser_verify.set_defaults(func=cmd_verify)

    # Split command
    parser_split = subparsers.add_parser("split", help="Create train/val splits")
    parser_split.add_argument("--val-split", type=float, default=0.2, help="Validation split ratio (default: 0.2)")
    parser_split.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser_split.add_argument("--create-yaml", action="store_true", help="Create dataset YAML config")
    parser_split.set_defaults(func=cmd_split)

    # All command
    parser_all = subparsers.add_parser("all", help="Run complete setup")
    parser_all.add_argument("--val-split", type=float, default=0.2, help="Validation split ratio (default: 0.2)")
    parser_all.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser_all.add_argument("--create-yaml", action="store_true", help="Create dataset YAML config")
    parser_all.set_defaults(func=cmd_all)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
