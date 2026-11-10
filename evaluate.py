"""
Counting Accuracy Evaluator
---------------------------
Runs YOLOv8 on a folder of test images and compares detected object counts
against a ground-truth JSON file. Outputs an overall counting accuracy %
that you can put on your resume.

Usage:
    python evaluate.py --images test_images/ --ground_truth ground_truth.json
    python evaluate.py --images test_images/ --ground_truth ground_truth.json --verbose

Ground-truth JSON format:
{
    "image1.jpg": {"person": 3, "car": 2},
    "image2.jpg": {"dog": 1}
}

Only classes listed in the ground truth for each image are evaluated
(misses and false positives are both counted as errors).
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
from ultralytics import YOLO


def load_ground_truth(path):
    with open(path, "r") as f:
        return json.load(f)


def detect_counts(model, image_path, conf=0.25):
    """Run YOLO and return a dict of {class_name: count}."""
    img = Image.open(image_path).convert("RGB")
    img_array = np.array(img)
    results = model(img_array, conf=conf, imgsz=1024, verbose=False)[0]

    counts = {}
    for box in results.boxes:
        class_name = model.names[int(box.cls)]
        counts[class_name] = counts.get(class_name, 0) + 1
    return counts


def evaluate(ground_truth, images_dir, model, conf=0.25, verbose=False):
    total_classes = 0
    correct_classes = 0
    per_image_results = []

    for filename, expected in ground_truth.items():
        image_path = os.path.join(images_dir, filename)
        if not os.path.exists(image_path):
            print(f"⚠️  Skipping {filename} (not found in {images_dir})")
            continue

        detected = detect_counts(model, image_path, conf=conf)

        img_correct = 0
        img_total = 0
        details = []

        # Check every expected class
        for cls, true_count in expected.items():
            pred_count = detected.get(cls, 0)
            correct = pred_count == true_count
            total_classes += 1
            img_total += 1
            if correct:
                correct_classes += 1
                img_correct += 1
            details.append(f"  {cls}: expected {true_count}, got {pred_count} "
                           f"{'✓' if correct else '✗'}")

        # Penalize false positives (detected classes not in ground truth)
        for cls, pred_count in detected.items():
            if cls not in expected:
                total_classes += 1
                img_total += 1
                details.append(f"  {cls}: false positive, got {pred_count} ✗")

        acc = (img_correct / img_total * 100) if img_total else 0.0
        per_image_results.append((filename, acc))

        if verbose:
            print(f"\n{filename} — {acc:.1f}%")
            for line in details:
                print(line)

    overall = (correct_classes / total_classes * 100) if total_classes else 0.0
    return overall, per_image_results


def main():
    parser = argparse.ArgumentParser(description="Evaluate YOLO counting accuracy.")
    parser.add_argument("--images", required=True, help="Folder containing test images")
    parser.add_argument("--ground_truth", required=True, help="JSON file with ground-truth counts")
    parser.add_argument("--model", default="yolov8m.pt", help="YOLO model weights")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--verbose", action="store_true", help="Show per-class details")
    args = parser.parse_args()

    ground_truth = load_ground_truth(args.ground_truth)
    print(f"Loaded ground truth for {len(ground_truth)} images.")

    print(f"Loading {args.model}...")
    model = YOLO(args.model)

    overall, per_image = evaluate(ground_truth, args.images, model,
                                  conf=args.conf, verbose=args.verbose)

    print("\n" + "=" * 50)
    print(f"OVERALL COUNTING ACCURACY: {overall:.1f}%")
    print("=" * 50)
    print("\nPer-image breakdown:")
    for name, acc in per_image:
        print(f"  {name}: {acc:.1f}%")

    # Save a simple report
    report = {
        "model": args.model,
        "confidence_threshold": args.conf,
        "images_evaluated": len(per_image),
        "overall_counting_accuracy_percent": round(overall, 1),
        "per_image": {n: round(a, 1) for n, a in per_image},
    }
    report_path = Path(args.images).parent / "evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to {report_path}")


if __name__ == "__main__":
    main()
