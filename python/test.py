"""
A basic ahh test file.

I'll make it prettier later I promise (:

Testing if Mutagen is working (again)
"""
import sys
import collections
from collections import defaultdict, deque
import collections.abc
import types
import math
from pathlib import Path
import json
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
from boxmot import BotSort, StrongSort
import faiss

from ball_tracker import MultiBallTracker 
from dino_tracker import dino_track_players
from utils import get_center, is_motion_consistent, track_history, is_on_player

# --- Appearance memory ---
appearance_history = defaultdict(lambda: deque(maxlen=30))

# --- Long-term FAISS DB ---
EMBED_DIM = 512
faiss_index = faiss.IndexFlatL2(EMBED_DIM)

faiss_ids = []  # maps index -> track_id
FAISS_THRESHOLD = 0.7  # tune this

# --- Track lifecycle ---
lost_tracks = {}  # track_id -> last_seen_frame
FRAME_BUFFER = 100

parser = argparse.ArgumentParser()

parser.add_argument("--model", help="Model to load. If none, defaults to 'yolo26x.pt'",
                    default='yolo26x.pt')
parser.add_argument("--input", required=True, help="Input video path")
parser.add_argument("--output", required=True, help="Output video path")
parser.add_argument('--show_conf', default=False, action='store_true',
                    help='Whether to show the confidence scores')
parser.add_argument('--conf', type=float, default=0.2,
                    help='Object confidence threshold for detection')
parser.add_argument('--imgsz', type=int, default=1920,
                    help='Image size for YOLO. 640, 1280, and 1920 are good')
parser.add_argument('--heatmap_conf', type=int, default=0.5,
                    help='Confidence for the ball tracker')
parser.add_argument('--heatmap_alpha', type=float, default=0.0,
                    help='Alpha for heatmap overlay')
parser.add_argument('--json_output', default=None, help="Path to output json")
parser.add_argument('--dino', action='store_true', default=False,
                    help='Use Dino for tracking, else BoxMOT. Much slower. Warning: Dino sucks')
parser.add_argument('--device', default=0,
                    help='Device to use')

args = parser.parse_args()


def write_json_frame(file_obj, frame_idx, ball_pos, players, ball_conf):
    if file_obj is None:
        return
    frame_data = {
        "frame": frame_idx,
        "ball": {"x": ball_pos[0], "y": ball_pos[1], "conf": ball_conf} if ball_pos else None,
        "players": players
    }

    file_obj.write(json.dumps(frame_data) + '\n')


def get_embeddings(tracker):
    """Extract embeddings from StrongSORT in a version-safe way."""
    feats = []

    if hasattr(tracker, "tracker") and hasattr(tracker.tracker, "tracks"):
        for t in tracker.tracker.tracks:
            emb = None

            # Try all possible attribute names
            if hasattr(t, "smooth_feat") and t.smooth_feat is not None:
                emb = t.smooth_feat
            elif hasattr(t, "curr_feat") and t.curr_feat is not None:
                emb = t.curr_feat
            elif hasattr(t, "features") and len(t.features) > 0:
                emb = t.features[-1]  # most recent feature

            if emb is not None:
                feats.append(emb)

    return feats


def match_long_term(embedding):
    if len(faiss_ids) == 0:
        return None

    D, I = faiss_index.search(np.array([embedding]).astype("float32"), k=1)

    if D[0][0] < FAISS_THRESHOLD:
        return faiss_ids[I[0][0]]

    return None


def add_to_faiss(track_id, embedding):
    faiss_index.add(np.array([embedding]).astype("float32"))
    faiss_ids.append(track_id)


def track_players(frame, player_dets, tracker, annotated_frame, frame_idx=0):
    current_player_boxes = []
    frame_players_data = []

    if len(player_dets) > 0:
        tracks = tracker.update(player_dets, frame)
    else:
        tracks = np.empty((0, 8))

    embeddings = get_embeddings(tracker)

    if len(tracks) > 0:
        boxes = tracks[:, :4]
        track_ids = tracks[:, 4].astype(int)
        confs = tracks[:, 5]

        for i, (box, track_id, conf) in enumerate(zip(boxes, track_ids, confs)):
            x1, y1, x2, y2 = map(int, box)
            center = get_center((x1, y1, x2, y2))

            # --- Motion gating ---
            if not is_motion_consistent(track_id, center):
                continue

            # --- Appearance smoothing ---
            if i < len(embeddings):
                emb = embeddings[i]
                appearance_history[track_id].append(emb)

                # average embedding
                avg_emb = np.mean(appearance_history[track_id], axis=0)

                # --- Long-term matching ---
                matched_id = match_long_term(avg_emb)

                if matched_id is not None and matched_id != track_id:
                    track_id = matched_id

                add_to_faiss(track_id, avg_emb)

            # --- Track lifecycle ---
            lost_tracks[track_id] = frame_idx

            # --- Draw ---
            color = (0, 255, 0)
            label = f"P-{track_id}"

            track_history[track_id].append(center)

            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated_frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            current_player_boxes.append((x1, y1, x2, y2))

            frame_players_data.append({
                "tid": int(track_id),
                "box": [x1, y1, x2, y2],
                "conf": float(conf)
            })

    # --- Cleanup old tracks ---
    to_delete = []
    for tid, last_seen in lost_tracks.items():
        if frame_idx - last_seen > FRAME_BUFFER:
            to_delete.append(tid)

    for tid in to_delete:
        lost_tracks.pop(tid, None)
        appearance_history.pop(tid, None)

    return current_player_boxes, frame_players_data


def track_ball(ball_dets, tracker, annotated_frame):
    primary_id, tracks = tracker.update(ball_dets)
    
    final_ball_pos = None
    ball_conf = 0.0

    for tid, data in tracks.items():
        bx, by = int(data['pos'][0]), int(data['pos'][1])
        is_predicted = data['is_predicted']
        is_primary = (tid == primary_id)
        
        if is_primary:
            final_ball_pos = [bx, by]
            ball_conf = data['conf']
            color = (255, 0, 255) if is_predicted else (0, 165, 255)
            label = "Primary (Physics)" if is_predicted else "Primary Ball"
            
            cv2.circle(annotated_frame, (bx, by), 6, color, -1)
            cv2.putText(annotated_frame, label, (bx - 15, by - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        else:
            # Draw stationary/secondary balls as hollow gray circles so they don't distract
            color = (150, 150, 150) 
            cv2.circle(annotated_frame, (bx, by), 4, color, 2)

    return final_ball_pos, ball_conf


def main():
    """Do the thing."""
    device = 0 if torch.cuda.is_available() else "cpu"
    # https://github.com/mikel-brostrom/boxmot/blob/59784c49eeec19736b48e034382d393d764d611d/boxmot/trackers/botsort/botsort.py#L21
    # tracker = BotSort(
    #     reid_weights=Path('osnet_ibn_x1_0_msmt17.pt'),
    #     device=device,
    #     half=False,
    #     match_thresh=0.95,
    #     proximity_thresh=0.86,
    #     appearance_thresh=0.7,
    # )
    tracker = StrongSort(
        reid_weights=Path('osnet_ain_x1_0_msmt17.pt'),
        device=device,
        half=False,

        max_dist=0.4,              # appearance similarity (VERY important)
        max_iou_distance=0.8,      # spatial matching
        max_age=50,                # frames to keep lost tracks
        nn_budget=200,
    )
    model = YOLO(args.model)

    # tracknet = PyTorchTrackNetTracker(
    #     weights_path="python/weights/tracknet-v4_best-model.pth",
    #     threshold=args.heatmap_conf,
    # )

    MAX_COAST_FRAMES = 20

    ball_tracker = MultiBallTracker(max_coast_frames=MAX_COAST_FRAMES)

    cap = cv2.VideoCapture(args.input)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_size = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    os.makedirs(os.path.split(args.output)[0], exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, frame_size)

    json_file = None
    if args.json_output:
        os.makedirs(os.path.split(args.json_output)[0], exist_ok=True)
        json_file = open(args.json_output, 'w')

    for frame_idx in track(range(total_frames)):
        ret, frame = cap.read()
        if not ret:
            break

        annotated_frame = frame.copy()

        # 1. Run YOLO ONCE to find BOTH Players (0) and Ball (1)
        results = model.predict(
            frame,
            conf=args.conf,
            classes=[0, 1],
            verbose=False,
            imgsz=args.imgsz,
            iou=0.95,
            device=args.device
        )

        # 2. Split the YOLO detections into two arrays based on class
        dets = results[0].boxes.data.cpu().numpy()
        player_dets = dets[dets[:, 5] == 0] if len(dets) > 0 else np.empty((0, 6))
        ball_dets = dets[dets[:, 5] == 1] if len(dets) > 0 else np.empty((0, 6))

        # 3. Track Players via BoxMOT
        current_player_boxes, frame_players_data = dino_track_players(
            frame, player_dets, tracker, annotated_frame
        ) if args.dino else track_players(
            frame, player_dets, tracker, annotated_frame, frame_idx
        )

        # 4. Track Ball via Kalman Physics
        final_ball_pos, ball_conf = track_ball(
            ball_dets, ball_tracker, annotated_frame
        )

        # 5. Output
        out.write(annotated_frame)
        write_json_frame(json_file, frame_idx, final_ball_pos, frame_players_data, ball_conf)

    cap.release()
    out.release()
    if json_file:
        json_file.close()


if __name__ == "__main__":
    main()
