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
    matrix = [[0 for _ in classes] for _ in classes]
    with torch.no_grad():
        for images, labels in DataLoader(dataset, batch_size=32):
            predictions = model(images).argmax(1)
            for real, predicted in zip(labels.tolist(), predictions.tolist()):
                matrix[real][predicted] += 1
    total = sum(map(sum, matrix))
    accuracy = sum(matrix[i][i] for i in range(len(classes))) / total if total else 0.0
    per_animal = {name: (matrix[i][i] / sum(matrix[i]) if sum(matrix[i]) else 0.0)
                  for i, name in enumerate(classes)}
    print(f"Overall accuracy: {accuracy:.4f}")
    print("Per-animal accuracy:")
    for name, value in per_animal.items():
        print(f"  {name}: {value:.4f}")
    print("Confusion matrix (rows=real, columns=predicted):")
    print("     " + " ".join(f"{name:>8}" for name in classes))
    for name, row in zip(classes, matrix):
        print(f"{name:>5} " + " ".join(f"{value:8}" for value in row))
    output = Path(args.metrics_out)
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps({"accuracy": accuracy,
        "per_animal_accuracy": per_animal, "classes": classes,
        "confusion_matrix": matrix}, indent=2), encoding="utf-8")
    log_metric("test_accuracy", accuracy)
    if accuracy < args.min_accuracy:
        print(f"QUALITY GATE FAILED: {accuracy:.4f} < {args.min_accuracy:.4f}")
        raise SystemExit(1)
    print(f"QUALITY GATE PASSED: {accuracy:.4f} >= {args.min_accuracy:.4f}")


if __name__ == "__main__":
    main()