#!/usr/bin/env python3

import argparse
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASETS_ROOT = ROOT / "datasets"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "yolo_unified_dataset"
DEFAULT_PROJECT_DIR = ROOT / "volleyball_model"

MASTER_CLASSES = {
    0: "volleyball",
    1: "player",
    2: "block",
    3: "defense",
    4: "serve",
    5: "set",
    6: "spike",
}

DATASET_CONFIGS = {
    "Players Dataset.yolo26": {
        "prefix": "players",
        "class_map": {
            0: 0,
            1: 1,
        },
    },
    "Volleyball_v2.v1-original.yolo26": {
        "prefix": "ball",
        "class_map": {
            0: 0,
        },
    },
    "Volleyball Actions.v5-original.yolo26": {
        "prefix": "actions",
        "class_map": {
            0: 2,
            1: 3,
            2: 4,
            3: 5,
            4: 6,
        },
    },
}

SPLIT_MAP = {
    "train": "train",
    "valid": "val",
    "val": "val",
    "test": "test",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge the three .yolo26 volleyball datasets and train one YOLO detector."
    )
    parser.add_argument("--model", default="yolo26x.pt", help="Base YOLO weights")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--imgsz", type=int, default=1920, help="Training image size")
    parser.add_argument("--batch", type=int, default=2, help="Batch size")
    parser.add_argument("--device", default="0", help="Training device, e.g. 0 or cpu")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Where to write the unified YOLO dataset",
    )
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=DEFAULT_PROJECT_DIR,
        help="Ultralytics project output directory",
    )
    parser.add_argument(
        "--run-name",
        default="train_unified",
        help="Ultralytics run name inside the project directory",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Only build the unified dataset and data.yaml",
    )
    return parser.parse_args()


def ensure_clean_output(output_dir):
    if output_dir.exists():
        shutil.rmtree(output_dir)

    for split in {"train", "val", "test"}:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)


def discover_datasets():
    discovered = []

    for dataset_dir in sorted(DATASETS_ROOT.glob("*.yolo26")):
        config = DATASET_CONFIGS.get(dataset_dir.name)
        if config is None:
            continue
        discovered.append((dataset_dir, config))

    missing = sorted(set(DATASET_CONFIGS) - {path.name for path, _ in discovered})
    if missing:
        missing_str = ", ".join(missing)
        raise FileNotFoundError(f"Missing expected datasets: {missing_str}")

    return discovered


def remap_label_lines(label_path, class_map):
    remapped_lines = []

    if not label_path.exists():
        return remapped_lines

    for raw_line in label_path.read_text().splitlines():
        parts = raw_line.strip().split()
        if len(parts) < 5:
            continue

        source_class = int(parts[0])
        target_class = class_map.get(source_class)
        if target_class is None:
            continue

        parts[0] = str(target_class)
        remapped_lines.append(" ".join(parts))

    return remapped_lines


def copy_dataset_split(dataset_dir, config, source_split, target_split, output_dir, stats):
    images_dir = dataset_dir / source_split / "images"
    labels_dir = dataset_dir / source_split / "labels"

    if not images_dir.exists():
        return

    image_paths = sorted(
        path for path in images_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    for image_path in image_paths:
        label_path = labels_dir / f"{image_path.stem}.txt"
        remapped_lines = remap_label_lines(label_path, config["class_map"])

        dest_stem = f"{config['prefix']}_{dataset_dir.stem}_{image_path.stem}"
        dest_image = output_dir / "images" / target_split / f"{dest_stem}{image_path.suffix.lower()}"
        dest_label = output_dir / "labels" / target_split / f"{dest_stem}.txt"

        shutil.copy2(image_path, dest_image)
        dest_label.write_text("\n".join(remapped_lines) + ("\n" if remapped_lines else ""))

        stats[target_split]["images"] += 1
        stats[target_split]["labels"] += len(remapped_lines)


def write_data_yaml(output_dir):
    yaml_path = output_dir / "data.yaml"
    names = ", ".join(f"'{name}'" for _, name in sorted(MASTER_CLASSES.items()))
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {output_dir.resolve().as_posix()}",
                "train: images/train",
                "val: images/val",
                "test: images/test",
                f"nc: {len(MASTER_CLASSES)}",
                f"names: [{names}]",
                "",
            ]
        )
    )
    return yaml_path


def prepare_unified_dataset(output_dir):
    ensure_clean_output(output_dir)
    datasets = discover_datasets()
    stats = {
        "train": {"images": 0, "labels": 0},
        "val": {"images": 0, "labels": 0},
        "test": {"images": 0, "labels": 0},
    }

    print(f"Building unified dataset in {output_dir}")

    for dataset_dir, config in datasets:
        print(f"- Merging {dataset_dir.name}")
        for source_split, target_split in SPLIT_MAP.items():
            copy_dataset_split(dataset_dir, config, source_split, target_split, output_dir, stats)

    yaml_path = write_data_yaml(output_dir)

    print("Unified dataset summary:")
    for split in ("train", "val", "test"):
        split_stats = stats[split]
        print(
            f"  {split}: {split_stats['images']} images, {split_stats['labels']} total annotations"
        )
    print(f"  classes: {list(MASTER_CLASSES.values())}")
    print(f"  yaml: {yaml_path}")

    return yaml_path


def train_yolo_model(args, yaml_path):
    from ultralytics import YOLO

    print("Starting YOLO training")
    model = YOLO(args.model)
    model.train(
        data=str(yaml_path.resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(args.project_dir.resolve()),
        name=args.run_name,
        box=10.0,
    )
    print(
        "Training complete. Best weights should be at "
        f"{args.project_dir.resolve() / args.run_name / 'weights' / 'best.pt'}"
    )


def main():
    args = parse_args()
    yaml_path = prepare_unified_dataset(args.output_dir)

    if args.skip_train:
        print("Skipping training because --skip-train was set.")
        return

    train_yolo_model(args, yaml_path)


if __name__ == "__main__":
    main()
