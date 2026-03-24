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
parser.add_argument('--heatmap_conf', type=int, default=0.5,
                    help='Confidence for the ball tracker')
parser.add_argument('--heatmap_alpha', type=float, default=0.3,
                    help='Alpha for heatmap overlay')

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
        features = model(tensor)

    # 4. Normalize and return as numpy array
    emb = features.cpu().numpy()[0]
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb /= norm

    return emb.astype("float32")


def resolve_player_id(yolo_id, box, frame, frame_idx, reid_model, db, track_memory, 
                      yolo_to_global, last_seen, reid_interval=5):
    """
    Tiers of identity resolution to ensure IDs are stubborn and unique.
    Returns: (match_id, confidence_score)
    """
    x1, y1, x2, y2 = map(int, box)
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

    match_id = yolo_to_global.get(yolo_id)
    score = 1.0  # Default confidence for established tracks

    # Trigger ReID if it's a new track OR we've hit our refresh interval
    should_reid = (match_id is None) or (frame_idx % reid_interval == 0)

    if should_reid:
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0 or is_blurry(crop):
            return (match_id if match_id else "Blurry"), 0

        # Get DINOv2 Embedding
        emb = get_dinov2_embedding(reid_model, crop)
        track_memory.add(yolo_id, emb)
        agg_emb = track_memory.get(yolo_id)

        if agg_emb is not None:
            # TIER 1: Spatial Recovery (Only for brand new tracks)
            if match_id is None:
                best_spatial_id = None
                min_dist = 120  # Max pixels a player could move between frames
                for gid, (lx, ly, l_frame) in last_seen.items():
                    # If player was seen within last 2 seconds (60 frames)
                    if frame_idx - l_frame < 60:
                        dist = math.sqrt((cx - lx)**2 + (cy - ly)**2)
                        if dist < min_dist:
                            best_spatial_id = gid
                            min_dist = dist

                if best_spatial_id is not None:
                    match_id = best_spatial_id

            # TIER 2: Stubborn ReID Match
            # We use a very low threshold (0.45) to prevent new ID creation
            # Search the database for the closest visual match
            matched_info = db.match_with_score(agg_emb, threshold=0.45)

            if matched_info:
                id_from_db, sim_score = matched_info
                # If spatial and ReID disagree, ReID (Visuals) wins
                match_id = id_from_db
                score = sim_score
                db.update(match_id, agg_emb)
            elif match_id is None:
                # TIER 3: New ID (Absolute last resort)
                match_id = db.add(agg_emb)
                score = 1.0

            # Lock the result into the tracker mapping
            yolo_to_global[yolo_id] = match_id

    return match_id, score


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
    track_memory = TrackMemory(history=60)

    cap = cv2.VideoCapture(args.input)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_size = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    os.makedirs(os.path.split(args.output)[0], exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, frame_size)

    yolo_to_global = {}

    REID_INTERVAL = 5
    last_seen = {}

    for frame_idx in track(range(total_frames)):
        ret, frame = cap.read()
        if not ret:
            break

        annotated_frame = frame.copy()

        results = model.track(
            frame,
            conf=args.conf,
            classes=[0],
            tracker='bytetrack.yaml',
            iou=0.5,
            persist=True,
            verbose=False,
            imgsz=args.imgsz,
        )
        frame_candidates = []
        current_player_boxes = []

        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)

            for box, yolo_id in zip(boxes, track_ids):
                # 1. Get identity from our stubborn resolver
                m_id, conf = resolve_player_id(
                    yolo_id, box, frame, frame_idx, reid_model, 
                    db, track_memory, yolo_to_global, last_seen, 
                    reid_interval=5
                )

                # 2. Store candidate for duplicate checking
                x1, y1, x2, y2 = map(int, box)
                frame_candidates.append({
                    'm_id': m_id,
                    'conf': conf,
                    'box': (x1, y1, x2, y2),
                    'center': ((x1+x2)//2, (y1+y2)//2)
                })
                current_player_boxes.append((x1, y1, x2, y2))

        # 3. Constraint Check: No duplicates in one frame
        # Sort by confidence so the "best" Player 1 keeps the ID
        frame_candidates.sort(key=lambda x: x['conf'] if isinstance(x['m_id'], int) else -1, reverse=True)
        
        used_ids_this_frame = set()
        for cand in frame_candidates:
            final_id = cand['m_id']
            
            if isinstance(final_id, int):
                if final_id in used_ids_this_frame:
                    final_id = "Duplicate"
                else:
                    used_ids_this_frame.add(final_id)
                    # Update global "Last Seen" for spatial tracking in next frames
                    last_seen[final_id] = (cand['center'][0], cand['center'][1], frame_idx)

            # 4. Final Drawing
            x1, y1, x2, y2 = cand['box']
            color = (0, 255, 0) if isinstance(final_id, int) else (0, 165, 255)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated_frame, f"P-{final_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        ball_pos = tracknet.predict(frame)
        if tracknet.current_heatmap is not None:
            annotated_frame = draw_heatmap_overlay(
                annotated_frame,
                tracknet.current_heatmap,
                alpha=args.heatmap_alpha,
            )
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
