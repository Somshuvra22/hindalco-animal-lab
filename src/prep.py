"""Validate and split raw class-folder images."""
import argparse
import random
import shutil
from pathlib import Path
from PIL import Image


def find_class_root(raw_data):
    root = Path(raw_data)
    folders = [p for p in root.iterdir() if p.is_dir()]
    if len(folders) == 1 and not any(p.is_file() for p in root.iterdir()):
        root = folders[0]
    return root


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_data", required=True)
    parser.add_argument("--train_out", required=True)
    parser.add_argument("--test_out", required=True)
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--min_images", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    root = find_class_root(args.raw_data)
    classes = sorted(p.name for p in root.iterdir() if p.is_dir())
    if len(classes) < 2:
        raise ValueError("Need at least two animal folders")
    randomizer = random.Random(args.seed)
    rows = []
    for animal in classes:
        files = sorted(p for p in (root / animal).iterdir() if p.is_file())
        if len(files) < args.min_images:
            raise ValueError(f"{animal} has {len(files)} images; need {args.min_images}")
        valid = []
        for file in files:
            try:
                with Image.open(file) as image:
                    image.verify()
                valid.append(file)
            except Exception as error:
                raise ValueError(f"Unreadable image: {file} ({error})") from error
        randomizer.shuffle(valid)
        test_count = max(1, int(len(valid) * args.test_ratio))
        test_files, train_files = valid[:test_count], valid[test_count:]
        for output, selected in [(args.train_out, train_files), (args.test_out, test_files)]:
            destination = Path(output) / animal
            destination.mkdir(parents=True, exist_ok=True)
            for file in selected:
                shutil.copy2(file, destination / file.name)
        rows.append((animal, len(train_files), len(test_files)))
    print("animal | train | test")
    print("-------|-------|------")
    for row in rows:
        print(f"{row[0]} | {row[1]} | {row[2]}")


if __name__ == "__main__":
    main()