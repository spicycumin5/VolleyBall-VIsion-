"""
A basic ahh test file.

I'll make it prettier later I promise (:
"""
import sys
import collections
import collections.abc
import types
import math
import torch
import torchvision.transforms as T

# 1. Patch for FastReID on Python 3.10+
if not hasattr(collections, "Mapping"):
    collections.Mapping = collections.abc.Mapping

# 2. Patch for FastReID on PyTorch 1.13+
if not hasattr(torch, "_six"):
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
from PIL import Image


from reiddatabase import ReIDDatabase, TrackMemory
from tracknet import PyTorchTrackNetTracker, draw_heatmap_overlay
from ball_tracker import KalmanBallTracker

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
                    help='Image size for YOLO. 640, 1280, and 1920 are good')
parser.add_argument('--heatmap-conf', type=int, default=0.5,
                    help='Confidence for the ball tracker')

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


def is_on_player(bx, by, player_boxes):
    for (x1, y1, x2, y2) in player_boxes:
        # Calculate the y-coordinate for the bottom of the "head zone"
        head_bottom = y1 + (y2 - y1) * 0.25 
        
        # Check if the ball's (bx, by) is inside this upper rectangle
        if x1 <= bx <= x2 and y1 <= by <= head_bottom:
            return True
            
    return False


def load_dinov2_model():
    # Load the small DINOv2 model (fast and accurate)
    # Output dimension: 768
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval() # Set to evaluation mode

    return model


def get_dinov2_embedding(model, crop_bgr):
    # 1. Convert OpenCV BGR to RGB, then to PIL Image
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(crop_rgb)

    # 2. DINOv2 Standard Preprocessing
    transform = T.Compose([
        T.Resize((224, 224)), # Force square for the ViT
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    tensor = transform(pil_image).unsqueeze(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tensor = tensor.to(device)

    # 3. Extract Features
    with torch.no_grad():
        # DINOv2 returns a tensor of shape (1, embedding_dim)
        features = model(tensor)

    # 4. Normalize and return as numpy array
    emb = features.cpu().numpy()[0]
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb /= norm

    return emb.astype("float32")


def main():
    """Do the thing."""
    model = YOLO(args.model)
    reid_model = load_dinov2_model()

    tracknet = PyTorchTrackNetTracker(
        weights_path="python/weights/tracknet-v4_best-model.pth",
        threshold=args.heatmap_conf,
    )
    kalman = KalmanBallTracker()

    missing_frames = 0
    MAX_COAST_FRAMES = 10

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

        annotated_frame = frame.copy()

        results = model.track(
            frame,
            conf=args.conf,
            classes=[0],
            tracker='botsort.yaml',
            persist=True,
            verbose=False,
            imgsz=args.imgsz,
        )
        current_player_boxes = []

        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)

            for box, track_id in zip(boxes, track_ids):
                current_player_boxes.append((map(int, box)))
                x1, y1, x2, y2 = map(int, box)

                if track_id in yolo_to_global:
                    match_id = yolo_to_global[track_id]
                else:
                    crop = frame[y1:y2, x1:x2]
                    if crop.size == 0 or is_blurry(crop):
                        match_id = "Blurry"
                    else:
                        emb = get_dinov2_embedding(reid_model, crop)

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
                            yolo_to_global[track_id] = match_id 

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

        ball_pos = tracknet.predict(frame)
        if hasattr(tracknet, 'last_heatmap'):
            annotated_frame = draw_heatmap_overlay(annotated_frame, tracknet.last_heatmap)
        final_ball_pos = None
        is_predicted = False

        if ball_pos is not None:
            bx, by = ball_pos
            if not kalman.is_tracking and is_on_player(bx, by, current_player_boxes):
                pass
            else:
                kalman.correct(bx, by)
                final_ball_pos = kalman.predict()
                missing_frames = 0

        if final_ball_pos is None:
            missing_frames += 1
            if missing_frames <= MAX_COAST_FRAMES:
                final_ball_pos = kalman.predict()
                is_predicted = True
            else:
                kalman.reset()

        if final_ball_pos is not None:
            bx, by = final_ball_pos
            color = (255, 0, 255) if is_predicted else (0, 165, 255)

            cv2.circle(annotated_frame, (bx, by), 6, color, -1)

            cv2.putText(
                annotated_frame,
                "Ball (Physics)" if is_predicted else "Ball",
                (bx - 15, by - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

        out.write(annotated_frame)

    cap.release()
    out.release()


if __name__ == "__main__":
    main()
