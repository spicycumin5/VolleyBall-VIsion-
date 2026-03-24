"""
A basic ahh test file.

I'll make it prettier later I promise (:
"""
import sys
import collections
import collections.abc
import types
import math

# 1. Patch for FastReID on Python 3.10+
if not hasattr(collections, "Mapping"):
    collections.Mapping = collections.abc.Mapping

# 2. Patch for FastReID on PyTorch 1.13+
if 'torch._six' not in sys.modules:
    torch_six = types.ModuleType('torch._six')
    torch_six.string_classes = (str,)
    torch_six.inf = math.inf
    torch_six.nan = math.nan
    torch_six.with_metaclass = lambda metaclass, *bases: metaclass("NewClass", bases, {})
    sys.modules['torch._six'] = torch_six
    
import argparse
import numpy as np
import os
import cv2
from rich.progress import track as track
from ultralytics import YOLO
import torch
from reiddatabase import ReIDDatabase, TrackMemory
from collections import deque

parser = argparse.ArgumentParser()

parser.add_argument("--model", help="Model to load. If none, defaults to 'yolo26x.pt'",
                    default='yolo26x.pt')
parser.add_argument("--input", required=True, help="Input video path")
parser.add_argument("--output", required=True, help="Output video path")
parser.add_argument('--show_conf', default=False, action='store_true',
                    help='Whether to show the confidence scores')
parser.add_argument('--show_labels', default=False,
                    action='store_true', help='Whether to show the labels')
parser.add_argument('--conf', type=float, default=0.2,
                    help='Object confidence threshold for detection')
parser.add_argument('--classes', nargs='+', default=None,
                    help='List of classes to detect')
parser.add_argument('--line_width', type=int, default=2,
                    help='Line width for bounding box visualization')
parser.add_argument('--font_size', type=float, default=2,
                    help='Font size for label visualization')
parser.add_argument('--imgsz', type=int, default=640,
                    help='Image size for YOLO. 640, 1280, and 1920 are good.')

args = parser.parse_args()


def load_reid_model():
    from fastreid.config import get_cfg
    from fastreid.engine import DefaultPredictor

    cfg = get_cfg()
    cfg.merge_from_file("python/weights/sbs_R101-ibn.yml")
    cfg.MODEL.WEIGHTS = "python/weights/msmt_sbs_R101-ibn.pth"
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    predictor = DefaultPredictor(cfg)
    return predictor


def get_embedding(model, crop):
    crop = cv2.resize(crop, (256, 512))  # standard ReID size
    crop = crop[:, :, ::-1]
    crop = crop.astype("float32") / 255.0

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    crop = (crop - mean) / std

    crop = crop.transpose(2, 0, 1)
    tensor = torch.as_tensor(crop.astype("float32")).unsqueeze(0)

    with torch.no_grad():
        outputs = model(tensor)

    emb = outputs.cpu().numpy()[0]

    norm = np.linalg.norm(emb)
    if norm > 0:
        emb /= norm

    return emb.astype("float32")


def is_blurry(img, threshold=50):
    return cv2.Laplacian(img, cv2.CV_64F).var() < threshold


def main():
    """Do the thing."""
    model = YOLO(args.model)
    reid_model = load_reid_model()

    db = ReIDDatabase(dim=2048)
    track_memory = TrackMemory()

    cap = cv2.VideoCapture(args.input)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_size = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    os.makedirs(os.path.split(args.output)[0], exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, frame_size)

    yolo_to_global = {}

    for _ in track(range(total_frames)):
        ret, frame = cap.read()
        if not ret:
            break
        results = model.track(
            frame,
            conf=args.conf,
            classes=[0],
            tracker='botsort.yaml',
            persist=True,
            verbose=False,
            imgsz=args.imgsz,
        )
        ball_track = model.track(
            frame,
            classes=[32],
            imgsz=args.imgsz,
            verbose=False,
            conf=0.2,
        )

        annotated_frame = frame.copy()

        if results[0].boxes.id is None:
            out.write(frame)
            continue

        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.cpu().numpy().astype(int)

        for box, track_id in zip(boxes, track_ids):
            x1, y1, x2, y2 = map(int, box)

            if track_id in yolo_to_global:
                match_id = yolo_to_global[track_id]
            else:
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0 or is_blurry(crop):
                    continue

                emb = get_embedding(reid_model, crop)

                track_memory.add(track_id, emb)
                agg_emb = track_memory.get(track_id)

                if agg_emb is None:
                    match_id = "Pending..."
                else:
                    if db.index.ntotal == 0:
                        db = ReIDDatabase(dim=len(agg_emb))
                    match_id = db.match(agg_emb)
                    if match_id is None:
                        match_id = db.add(agg_emb)
                    else:
                        db.update(match_id, agg_emb)

            cv2.putText(
                annotated_frame,
                f"ID {match_id}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            cv2.rectangle(
                annotated_frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

        if ball_track[0].boxes is not None and len(ball_track[0].boxes) > 0:
            boxes = ball_track[0].boxes.xyxy.cpu().numpy()
            confs = ball_track[0].boxes.conf.cpu().numpy()

            box = boxes[np.argmax(confs)]
            x1, y1, x2, y2 = map(int, box)

            cv2.putText(
                annotated_frame,
                f"Ball",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 0, 0),
                2
            )

            cv2.rectangle(
                annotated_frame,
                (x1, y1),
                (x2, y2),
                (255, 0, 0),
                2
            )

        out.write(annotated_frame)

    cap.release()
    out.release()


if __name__ == "__main__":
    main()
