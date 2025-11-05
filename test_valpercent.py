"""
Test script to verify --valpercent works correctly.
This script tests validation with different percentages of data.
"""
from ultralytics import YOLO
import sys

def test_valpercent(valpercent):
    """Test validation with a specific percentage of data."""
    print(f"\n{'='*60}")
    print(f"Testing valpercent={valpercent}")
    print(f"{'='*60}\n")

    try:
        # Load model
        model = YOLO("last-3.pt")

        # Run validation with valpercent
        results = model.val(
            data="kitti-3d.yaml",
            valpercent=valpercent,
            verbose=True
        )

        print(f"\n✅ Success! Validation completed with valpercent={valpercent}")
        print(f"Results: {results}")
        return True

    except Exception as e:
        print(f"\n❌ Failed with valpercent={valpercent}")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Test different valpercent values
    test_values = [10, 25, 50, 100]

    if len(sys.argv) > 1:
        # If a specific value is provided, test only that
        test_values = [float(sys.argv[1])]

    results = {}
    for val in test_values:
        success = test_valpercent(val)
        results[val] = "✅ PASS" if success else "❌ FAIL"
        print(f"\nResult for valpercent={val}: {results[val]}")

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for val, status in results.items():
        print(f"valpercent={val:>3}: {status}")
