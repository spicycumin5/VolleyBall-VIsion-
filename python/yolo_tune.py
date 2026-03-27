#!/usr/bin/env python3

import argparse
import logging
import multiprocessing
import random
import shutil
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from queue import Empty

from PIL import Image
from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)


ROOT = Path(__file__).resolve().parents[1]
DATASETS_ROOT = ROOT / "datasets"
DEFAULT_PREPARED_ROOT = ROOT / "data" / "yolo_model_datasets"
DEFAULT_PROJECT_DIR = ROOT / "volleyball_model"
DEFAULT_SEED = 42

TASK_CONFIGS = {
    "players": {
        "dataset_dir": DATASETS_ROOT / "Players Dataset.yolo26",
        "names": ["player"],
        "class_map": {1: 0},
        "target_splits": None,
    },
    "actions": {
        "dataset_dir": DATASETS_ROOT / "Volleyball Actions.v5-original.yolo26",
        "names": ["block", "defense", "serve", "set", "spike"],
        "class_map": {0: 0, 1: 1, 2: 2, 3: 3, 4: 4},
        "target_splits": None,
    },
    "ball": {
        "dataset_dir": DATASETS_ROOT / "Volleyball_v2.v1-original.yolo26",
        "names": ["volleyball"],
        "class_map": {0: 0},
        "target_splits": None,
    },
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SOURCE_TO_TARGET_SPLIT = {"train": "train", "valid": "val", "val": "val", "test": "test"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare and train separate YOLO models for players, actions, and volleyball."
    )
    parser.add_argument("--model", default="yolo26x.pt", help="Base YOLO weights for all tasks")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--device", default="0", help="Training device, e.g. 0, 0,1, or cpu")
    parser.add_argument(
        "--prepared-root",
        type=Path,
        default=DEFAULT_PREPARED_ROOT,
        help="Where to write prepared per-task YOLO datasets",
    )
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=DEFAULT_PROJECT_DIR,
        help="Ultralytics project output directory",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=sorted(TASK_CONFIGS.keys()),
        default=sorted(TASK_CONFIGS.keys()),
        help="Which task-specific models to prepare/train",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Dataset sampling seed")
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Train separate tasks concurrently, assigning one device per task when possible",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Only prepare per-task datasets and data.yaml files",
    )
    return parser.parse_args()


def ensure_clean_dir(path):
    if path.exists():
        shutil.rmtree(path)
    for split in ("train", "val", "test"):
        (path / "images" / split).mkdir(parents=True, exist_ok=True)
        (path / "labels" / split).mkdir(parents=True, exist_ok=True)


def list_split_images(dataset_dir, split_name):
    images_dir = dataset_dir / split_name / "images"
    if not images_dir.exists():
        return []

    return sorted(
        path for path in images_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def build_split_sources(dataset_dir, seed):
    sources = {"train": [], "val": [], "test": []}
    for source_split, target_split in SOURCE_TO_TARGET_SPLIT.items():
        sources[target_split].extend(list_split_images(dataset_dir, source_split))

    if sources["train"] and not sources["val"] and not sources["test"]:
        images = list(sources["train"])
        random.Random(seed).shuffle(images)
        total = len(images)
        train_end = max(1, int(total * 0.8))
        val_end = max(train_end + 1, int(total * 0.9))
        val_end = min(val_end, total)
        sources = {
            "train": images[:train_end],
            "val": images[train_end:val_end],
            "test": images[val_end:],
        }
        if not sources["val"] and sources["train"]:
            sources["val"] = sources["train"][-1:]
        if not sources["test"] and sources["train"]:
            sources["test"] = sources["train"][-1:]

    return sources


def remap_label_lines(label_path, class_map):
    if not label_path.exists():
        return []

    remapped = []
    for raw_line in label_path.read_text().splitlines():
        parts = raw_line.strip().split()
        if len(parts) < 5:
            continue
        source_class = int(parts[0])
        target_class = class_map.get(source_class)
        if target_class is None:
            continue
        parts[0] = str(target_class)
        remapped.append(" ".join(parts))
    return remapped


def prepare_task_dataset(task_name, config, prepared_root, seed):
    dataset_dir = config["dataset_dir"]
    output_dir = prepared_root / task_name
    ensure_clean_dir(output_dir)

    split_sources = build_split_sources(dataset_dir, seed)
    stats = {"train": {"images": 0, "labels": 0}, "val": {"images": 0, "labels": 0}, "test": {"images": 0, "labels": 0}}

    for split_name, image_paths in split_sources.items():
        for image_path in image_paths:
            label_path = image_path.parent.parent / "labels" / f"{image_path.stem}.txt"
            remapped = remap_label_lines(label_path, config["class_map"])
            if not remapped:
                continue

            dest_image = output_dir / "images" / split_name / image_path.name
            dest_label = output_dir / "labels" / split_name / f"{image_path.stem}.txt"
            shutil.copy2(image_path, dest_image)
            dest_label.write_text("\n".join(remapped) + "\n")
            stats[split_name]["images"] += 1
            stats[split_name]["labels"] += len(remapped)

    yaml_path = output_dir / "data.yaml"
    names = ", ".join(f"'{name}'" for name in config["names"])
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {output_dir.resolve().as_posix()}",
                "train: images/train",
                "val: images/val",
                "test: images/test",
                f"nc: {len(config['names'])}",
                f"names: [{names}]",
                "",
            ]
        )
    )

    imgsz = infer_imgsz(output_dir / "images" / "train")
    return {
        "task_name": task_name,
        "output_dir": output_dir,
        "yaml_path": yaml_path,
        "imgsz": imgsz,
        "stats": stats,
    }


def round_to_stride(value, stride=32):
    return int(max(stride, round(value / stride) * stride))


def infer_imgsz(images_dir):
    dims = []
    for image_path in images_dir.iterdir():
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        with Image.open(image_path) as img:
            dims.append(max(img.size))

    if not dims:
        return 640

    dims.sort()
    median_dim = dims[len(dims) // 2]
    return round_to_stride(median_dim)


def format_metric_value(value):
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if numeric >= 100:
        return f"{numeric:.1f}"
    if numeric >= 10:
        return f"{numeric:.2f}"
    return f"{numeric:.3f}"


def build_epoch_status(trainer):
    status_parts = []
    if trainer.tloss is not None:
        try:
            loss_items = trainer.tloss.tolist()
        except AttributeError:
            loss_items = [trainer.tloss]
        if not isinstance(loss_items, list):
            loss_items = [loss_items]
        loss_text = ", ".join(str(format_metric_value(item)) for item in loss_items)
        status_parts.append(f"loss {loss_text}")

    metric_keys = (
        "metrics/mAP50-95(B)",
        "metrics/mAP50(B)",
        "metrics/precision(B)",
        "metrics/recall(B)",
    )
    for key in metric_keys:
        metric_value = trainer.metrics.get(key) if trainer.metrics else None
        if metric_value is not None:
            label = key.split("/")[-1].replace("(B)", "")
            status_parts.append(f"{label} {format_metric_value(metric_value)}")
            break

    if trainer.best_fitness is not None:
        status_parts.append(f"best {format_metric_value(trainer.best_fitness)}")

    return " | ".join(status_parts) if status_parts else "epoch complete"


class TrainingProgressDisplay:
    def __init__(self, prepared, epochs):
        self.epochs = epochs
        self.trackers = {}
        self.progress_order = []
        self.live = None

        for dataset_info in prepared:
            task_name = dataset_info["task_name"]
            progress = Progress(
                SpinnerColumn(),
                TextColumn("[bold]{task.description}[/bold]"),
                BarColumn(bar_width=None),
                TaskProgressColumn(),
                TextColumn("{task.completed:>3.0f}/{task.total:.0f} epochs"),
                TextColumn("{task.fields[status]}"),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                expand=True,
            )
            task_id = progress.add_task(task_name, total=epochs, status="waiting")
            self.trackers[task_name] = {"progress": progress, "task_id": task_id}
            self.progress_order.append(task_name)

    def __enter__(self):
        for tracker in self.trackers.values():
            tracker["progress"].start()
        self.live = Live(self.renderable, refresh_per_second=8)
        self.live.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.live is not None:
            self.live.__exit__(exc_type, exc, tb)
        for tracker in self.trackers.values():
            tracker["progress"].stop()

    @property
    def renderable(self):
        return Group(
            *[
                Panel(self.trackers[task_name]["progress"], title=task_name, border_style="blue")
                for task_name in self.progress_order
            ]
        )

    def update(self, event):
        tracker = self.trackers[event["task_name"]]
        progress = tracker["progress"]
        task_id = tracker["task_id"]
        event_type = event["event"]

        if event_type == "start":
            status = f"device {event['device']} | imgsz {event['imgsz']}"
            progress.update(task_id, completed=0, total=event["epochs"], status=status)
        elif event_type == "epoch":
            progress.update(task_id, completed=event["epoch"], total=event["epochs"], status=event["status"])
        elif event_type == "done":
            progress.update(task_id, completed=progress.tasks[task_id].total, status=event["status"])
        elif event_type == "error":
            progress.update(task_id, status=f"[red]{event['status']}[/red]")

        if self.live is not None:
            self.live.update(self.renderable, refresh=True)


def emit_progress_event(progress_sink, event):
    if progress_sink is not None:
        progress_sink(event)


def build_training_callbacks(task_name, device, imgsz, project_dir, progress_sink):
    def on_train_start(trainer):
        emit_progress_event(
            progress_sink,
            {
                "event": "start",
                "task_name": task_name,
                "device": device,
                "imgsz": imgsz,
                "epochs": trainer.epochs,
            },
        )

    def on_fit_epoch_end(trainer):
        emit_progress_event(
            progress_sink,
            {
                "event": "epoch",
                "task_name": task_name,
                "epoch": trainer.epoch + 1,
                "epochs": trainer.epochs,
                "status": build_epoch_status(trainer),
            },
        )

    def on_train_end(trainer):
        emit_progress_event(
            progress_sink,
            {
                "event": "done",
                "task_name": task_name,
                "status": f"best {project_dir.resolve() / task_name / 'weights' / 'best.pt'}",
            },
        )

    return {
        "on_train_start": on_train_start,
        "on_fit_epoch_end": on_fit_epoch_end,
        "on_train_end": on_train_end,
    }


@contextmanager
def quiet_ultralytics_progress():
    from ultralytics.engine import trainer as trainer_module
    from ultralytics.utils import LOGGER as ultralytics_logger

    original_tqdm = getattr(trainer_module, "TQDM")
    original_level = ultralytics_logger.level
    setattr(trainer_module, "TQDM", partial(original_tqdm, disable=True))
    ultralytics_logger.setLevel(max(original_level, logging.WARNING))
    try:
        yield
    finally:
        setattr(trainer_module, "TQDM", original_tqdm)
        ultralytics_logger.setLevel(original_level)


def train_task_model_on_device(model_path, dataset_info, device, epochs, batch, project_dir, progress_sink=None):
    from ultralytics.models import YOLO

    task_name = dataset_info["task_name"]
    model = YOLO(model_path)
    imgsz = dataset_info["imgsz"]
    callbacks = build_training_callbacks(task_name, device, imgsz, project_dir, progress_sink)
    for event_name, callback in callbacks.items():
        model.add_callback(event_name, callback)

    try:
        with quiet_ultralytics_progress():
            model.train(
                data=str(dataset_info["yaml_path"].resolve()),
                epochs=epochs,
                imgsz=imgsz,
                batch=batch,
                device=device,
                project=str(project_dir.resolve()),
                name=task_name,
            )
    except Exception as exc:
        emit_progress_event(
            progress_sink,
            {"event": "error", "task_name": task_name, "status": str(exc)},
        )
        raise


def parse_devices(device_arg):
    device_text = str(device_arg).strip()
    if not device_text:
        return ["0"]
    if device_text.lower() == "cpu":
        return ["cpu"]
    return [part.strip() for part in device_text.split(",") if part.strip()]


def train_task_worker(model_path, dataset_info, device, epochs, batch, project_dir, progress_queue):
    train_task_model_on_device(
        model_path=model_path,
        dataset_info=dataset_info,
        device=device,
        epochs=epochs,
        batch=batch,
        project_dir=project_dir,
        progress_sink=progress_queue.put,
    )


def train_prepared_models(args, prepared):
    devices = parse_devices(args.device)
    with TrainingProgressDisplay(prepared, args.epochs) as display:
        if not args.parallel or len(prepared) == 1 or len(devices) == 1 or devices == ["cpu"]:
            for index, dataset_info in enumerate(prepared):
                assigned_device = devices[min(index, len(devices) - 1)]
                train_task_model_on_device(
                    model_path=args.model,
                    dataset_info=dataset_info,
                    device=assigned_device,
                    epochs=args.epochs,
                    batch=args.batch,
                    project_dir=args.project_dir,
                    progress_sink=display.update,
                )
            return

        multiprocessing.set_start_method("spawn", force=True)
        max_parallel = min(len(devices), len(prepared))
        progress_queue = multiprocessing.Queue()
        for start in range(0, len(prepared), max_parallel):
            chunk = prepared[start:start + max_parallel]
            processes = []
            for dataset_info, device in zip(chunk, devices):
                process = multiprocessing.Process(
                    target=train_task_worker,
                    args=(args.model, dataset_info, device, args.epochs, args.batch, args.project_dir, progress_queue),
                )
                process.start()
                processes.append((dataset_info["task_name"], process, device))

            active = {task_name: (process, device) for task_name, process, device in processes}
            while active:
                try:
                    display.update(progress_queue.get(timeout=0.2))
                except Empty:
                    pass

                finished = []
                for task_name, (process, device) in active.items():
                    if process.is_alive():
                        continue
                    process.join()
                    if process.exitcode != 0:
                        raise RuntimeError(
                            f"Training failed for {task_name} on device {device} with exit code {process.exitcode}"
                        )
                    finished.append(task_name)

                for task_name in finished:
                    active.pop(task_name)

            while True:
                try:
                    display.update(progress_queue.get_nowait())
                except Empty:
                    break


def print_dataset_summary(dataset_info):
    print(f"Prepared {dataset_info['task_name']} dataset")
    for split_name in ("train", "val", "test"):
        split_stats = dataset_info["stats"][split_name]
        print(
            f"  {split_name}: {split_stats['images']} images, {split_stats['labels']} annotations"
        )
    print(f"  yaml: {dataset_info['yaml_path']}")
    print(f"  inferred imgsz: {dataset_info['imgsz']}")


def main():
    args = parse_args()
    prepared = []

    for task_name in args.tasks:
        seed = args.seed + sum(ord(ch) for ch in task_name)
        dataset_info = prepare_task_dataset(task_name, TASK_CONFIGS[task_name], args.prepared_root, seed)
        print_dataset_summary(dataset_info)
        prepared.append(dataset_info)

    if args.skip_train:
        print("Skipping training because --skip-train was set.")
        return

    train_prepared_models(args, prepared)


if __name__ == "__main__":
    main()
