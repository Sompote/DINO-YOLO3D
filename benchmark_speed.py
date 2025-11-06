#!/usr/bin/env python3
"""
Benchmark YOLOv12-3D model inference speed
"""

import time
import torch
from ultralytics import YOLO
import numpy as np
from pathlib import Path
import cv2

def benchmark_model(model_path, warmup_runs=10, test_runs=100):
    """Benchmark model inference speed."""

    print(f"Loading model: {model_path}")
    model = YOLO(model_path, task="detect3d")
    print("Model loaded!\n")

    # Create a dummy frame
    test_image = np.random.randint(0, 255, (650, 1920, 3), dtype=np.uint8)

    # Warmup runs
    print(f"Warming up model with {warmup_runs} runs...")
    for i in range(warmup_runs):
        _ = model.predict(test_image, save=False, verbose=False)
    print("Warmup complete!\n")

    # Benchmark
    print(f"Running benchmark with {test_runs} inference runs...")
    times = []

    for i in range(test_runs):
        start_time = time.time()
        results = model.predict(test_image, save=False, verbose=False)
        end_time = time.time()

        inference_time = end_time - start_time
        times.append(inference_time)

        if (i + 1) % 20 == 0:
            print(f"  Completed {i + 1}/{test_runs} runs")

    # Calculate statistics
    times = np.array(times)
    mean_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)
    fps = 1.0 / mean_time

    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Test runs: {test_runs}")
    print(f"Image size: {test_image.shape[1]}x{test_image.shape[0]}")
    print(f"Device: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}")
    print()
    print(f"Mean inference time: {mean_time:.4f} seconds")
    print(f"Std deviation:       {std_time:.4f} seconds")
    print(f"Min inference time:  {min_time:.4f} seconds")
    print(f"Max inference time:  {max_time:.4f} seconds")
    print()
    print(f"Average FPS:         {fps:.2f}")
    print(f"Min FPS:             {1/max_time:.2f}")
    print(f"Max FPS:             {1/min_time:.2f}")
    print("=" * 60)

    return {
        'mean_time': mean_time,
        'std_time': std_time,
        'min_time': min_time,
        'max_time': max_time,
        'fps': fps
    }

def benchmark_with_video(model_path, video_path, num_frames=100):
    """Benchmark with actual video frames."""

    print(f"\nBenchmarking with video: {video_path}")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video")
        return None

    model = YOLO(model_path, task="detect3d")

    # Read frames
    frames = []
    for _ in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()

    print(f"Loaded {len(frames)} frames")
    print(f"Running inference on video frames...")

    times = []
    for i, frame in enumerate(frames):
        start_time = time.time()
        results = model.predict(frame, save=False, verbose=False)
        end_time = time.time()

        inference_time = end_time - start_time
        times.append(inference_time)

        if (i + 1) % 20 == 0:
            print(f"  Processed {i + 1}/{len(frames)} frames")

    times = np.array(times)
    mean_time = np.mean(times)
    fps = 1.0 / mean_time

    print("\n" + "=" * 60)
    print("VIDEO BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Video frames processed: {len(frames)}")
    print(f"Mean inference time: {mean_time:.4f} seconds")
    print(f"Average FPS: {fps:.2f}")
    print("=" * 60)

    return {
        'mean_time': mean_time,
        'fps': fps
    }

if __name__ == "__main__":
    model_path = "/Users/sompoteyouwai/env/yolo3d/yolov12/last-4.pt"
    video_path = "/Users/sompoteyouwai/Downloads/1106.mov"

    print("YOLOv12-3D Model Inference Speed Benchmark")
    print("=" * 60)

    # CPU Info
    print(f"PyTorch version: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("CUDA: Not available (using CPU)")
    print()

    # Benchmark with dummy data
    benchmark_model(model_path, warmup_runs=10, test_runs=50)

    # Benchmark with video
    benchmark_with_video(model_path, video_path, num_frames=50)
