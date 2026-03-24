"""
A basic ahh test file.

I'll make it prettier later I promise (:
"""
import sys
import collections
import collections.abc
import types

# 1. Patch for FastReID on Python 3.10+
collections.Mapping = collections.abc.Mapping 

# 2. Patch for FastReID on PyTorch 1.13+
if 'torch._six' not in sys.modules:
    torch_six = types.ModuleType('torch._six')
    torch_six.string_classes = (str,)
    sys.modules['torch._six'] = torch_six
    
import argparse
import numpy as np
import os
import cv2
from rich.progress import track as track
from ultralytics import YOLO
import torch
from reiddatabase import ReIDDatabase, TrackMemory

parser = argparse.ArgumentParser()

parser.add_argument("--model", help="Model to load. If none, defaults to 'yolo26x.pt'",
                    default='yolo26x.pt')
parser.add_argument("--input", required=True, help="Input video path")
parser.add_argument("--output", required=True, help="Output video path")
parser.add_argument('--show_conf', default=False, action='store_true',
                    help='Whether to show the confidence scores')
parser.add_argument('--show_labels', default=False,
                    action='store_true', help='Whether to show the labels')
parser.add_argument('--conf', type=float, default=0.5,
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
    cfg.merge_from_file("fast_reid/configs/MSMT17/vit_transreid.yml")  # ViT config
    cfg.MODEL.WEIGHTS = "vit_transreid.pth"  # pretrained weights
    cfg.MODEL.DEVICE = "cuda"

    predictor = DefaultPredictor(cfg)
    return predictor


def get_embedding(model, crop):
    crop = cv2.resize(crop, (256, 512))  # standard ReID size

    with torch.no_grad():
        outputs = model(crop)
    emb = outputs["feat"].cpu().numpy().flatten()

    emb /= np.linalg.norm(emb)
    return emb


def is_blurry(img, threshold=50):
    return cv2.Laplacian(img, cv2.CV_64F).var() < threshold


def main():
    """Do the thing."""
    model = YOLO(args.model)
    reid_model = load_reid_model()

    db = ReIDDatabase(dim=768)
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

        out.write(annotated_frame)

    cap.release()
    out.release()


if __name__ == "__main__":
    main()
