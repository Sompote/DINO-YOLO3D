"""
Simple test to verify --valpercent argument is recognized and dataset size is correct.
"""
from ultralytics.cfg import get_cfg
from ultralytics.models.yolo.detect3d.train import Detection3DTrainer

def test_valpercent_config():
    """Test that valpercent is properly recognized in config."""
    print("Testing valpercent in configuration...")

    # Test with different valpercent values
    for valpercent in [10, 25, 50, 100]:
        try:
            cfg = get_cfg(overrides={"valpercent": valpercent, "data": "kitti-3d.yaml"})
            assert hasattr(cfg, "valpercent"), f"Config doesn't have valpercent attribute"
            assert cfg.valpercent == valpercent, f"Expected {valpercent}, got {cfg.valpercent}"
            print(f"✅ valpercent={valpercent} - Config OK")
        except Exception as e:
            print(f"❌ valpercent={valpercent} - Failed: {e}")
            return False

    return True

def test_dataset_size():
    """Test that dataset size changes with valpercent."""
    print("\nTesting dataset size with valpercent...")

    # Create a trainer to test dataset building
    trainer = Detection3DTrainer(overrides={
        "model": "yolov12n-3d.yaml",
        "data": "kitti-3d.yaml",
        "epochs": 1,
        "valpercent": 100.0
    })

    # Build validation dataset with 100%
    dataset_100 = trainer.build_dataset(
        trainer.data["val"],
        mode="val",
        batch=8
    )
    size_100 = len(dataset_100)
    print(f"Dataset size with valpercent=100: {size_100} images")

    # Build validation dataset with 10%
    trainer.args.valpercent = 10.0
    dataset_10 = trainer.build_dataset(
        trainer.data["val"],
        mode="val",
        batch=8
    )
    size_10 = len(dataset_10)
    print(f"Dataset size with valpercent=10: {size_10} images")

    # Check that 10% is approximately 1/10 of 100%
    expected_size = int(size_100 * 0.1)
    tolerance = 5  # Allow some tolerance

    if abs(size_10 - expected_size) <= tolerance:
        print(f"✅ Dataset size is correct: {size_10} ≈ {expected_size} (10% of {size_100})")
        return True
    else:
        print(f"❌ Dataset size mismatch: {size_10} != {expected_size} (10% of {size_100})")
        return False

if __name__ == "__main__":
    print("="*60)
    print("VALPERCENT FUNCTIONALITY TEST")
    print("="*60)

    # Test 1: Configuration
    config_ok = test_valpercent_config()

    # Test 2: Dataset size
    if config_ok:
        dataset_ok = test_dataset_size()
    else:
        dataset_ok = False
        print("\n⚠️ Skipping dataset test due to config failure")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Configuration test: {'✅ PASS' if config_ok else '❌ FAIL'}")
    print(f"Dataset size test: {'✅ PASS' if dataset_ok else '❌ FAIL'}")

    if config_ok and dataset_ok:
        print("\n🎉 All tests passed! --valpercent is working correctly.")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
