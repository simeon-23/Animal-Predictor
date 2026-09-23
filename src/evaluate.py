"""Evaluate a saved classifier and enforce the accuracy quality gate."""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets

from model_utils import get_transform, load_model, log_metric


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--test_data", required=True)
    parser.add_argument("--metrics_out", required=True)
    parser.add_argument("--min_accuracy", type=float, default=0.70)
    args = parser.parse_args()

    model, classes = load_model(args.model_dir)
    dataset = datasets.ImageFolder(args.test_data, transform=get_transform())
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    matrix = [[0 for _ in classes] for _ in classes]
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            predictions = model(images).argmax(1)
            for real, predicted in zip(labels.tolist(), predictions.tolist()):
                matrix[real][predicted] += 1
                correct += int(real == predicted)
                total += 1
    accuracy = correct / total if total else 0.0
    per_animal = {}
    for index, name in enumerate(classes):
        row_total = sum(matrix[index])
        per_animal[name] = matrix[index][index] / row_total if row_total else 0.0

    print(f"Overall accuracy: {accuracy:.4f}")
    print("Per-animal accuracy:")
    for name, value in per_animal.items():
        print(f"  {name}: {value:.4f}")
    print("Confusion matrix (rows = real, columns = predicted):")
    print("             " + " ".join(f"{name:>9}" for name in classes))
    for name, row in zip(classes, matrix):
        print(f"{name:>13} " + " ".join(f"{value:9}" for value in row))

    metrics = {"accuracy": accuracy, "per_animal_accuracy": per_animal,
               "classes": classes, "confusion_matrix": matrix,
               "min_accuracy": args.min_accuracy}
    output = Path(args.metrics_out)
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    log_metric("test_accuracy", accuracy)
    if accuracy < args.min_accuracy:
        print(f"QUALITY GATE FAILED: {accuracy:.4f} < {args.min_accuracy:.4f}")
        raise SystemExit(1)
    print(f"QUALITY GATE PASSED: {accuracy:.4f} >= {args.min_accuracy:.4f}")


if __name__ == "__main__":
    main()