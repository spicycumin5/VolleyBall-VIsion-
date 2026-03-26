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

from ball_tracker import MultiBallTracker 
from dino_tracker import dino_track_players
from utils import get_center, is_motion_consistent, track_history, is_on_player

# --- Appearance memory ---
appearance_history = defaultdict(lambda: deque(maxlen=30))

# --- Unified YOLO classes ---
BALL_CLASS_ID = 0
PLAYER_CLASS_ID = 1
ACTION_CLASS_IDS = {2, 3, 4, 5, 6}
PLAYER_LIKE_CLASS_IDS = {PLAYER_CLASS_ID, *ACTION_CLASS_IDS}
CLASS_ID_TO_STATE = {
    PLAYER_CLASS_ID: "player",
    2: "block",
    3: "defense",
    4: "serve",
    5: "set",
    6: "spike",
}

# --- Long-term identity DB ---
canonical_gallery = {}
tracker_to_canonical = {}
canonical_states = {}
next_canonical_id = 1
GALLERY_THRESHOLD = 0.7
SPATIAL_MATCH_IOU = 0.6
SPATIAL_MATCH_DIST = 45
SPATIAL_MATCH_MAX_GAP = 3

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


def normalize_embedding(embedding):
    embedding = np.asarray(embedding, dtype="float32")
    norm = np.linalg.norm(embedding)
    if norm == 0:
        return None
    return embedding / norm


def bbox_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter_area

    if union <= 0:
        return 0.0

    return inter_area / union


def detection_state_name(class_id):
    return CLASS_ID_TO_STATE.get(int(class_id), "player")


def match_detection_to_track(box, detections, used_detection_indices):
    best_idx = None
    best_score = (-1, -1.0, -1.0)

    for idx, det in enumerate(detections):
        if idx in used_detection_indices:
            continue

        det_box = tuple(map(int, det[:4]))
        det_conf = float(det[4]) if len(det) > 4 else 0.0
        det_class = int(det[5]) if len(det) > 5 else PLAYER_CLASS_ID
        iou = bbox_iou(box, det_box)

        if iou <= 0:
            continue

        is_action = int(det_class in ACTION_CLASS_IDS)
        score = (is_action, iou, det_conf)
        if score > best_score:
            best_score = score
            best_idx = idx

    return best_idx


def match_long_term(embedding, used_ids=None):
    if embedding is None or len(canonical_gallery) == 0:
        return None

    used_ids = used_ids or set()
    best_id = None
    best_dist = float("inf")

    for canonical_id, gallery_embedding in canonical_gallery.items():
        if canonical_id in used_ids:
            continue

        dist = np.linalg.norm(embedding - gallery_embedding)
        if dist < best_dist:
            best_dist = dist
            best_id = canonical_id

    if best_dist < GALLERY_THRESHOLD:
        return best_id

    return None


def update_gallery(canonical_id, embedding, momentum=0.9):
    if embedding is None:
        return

    if canonical_id in canonical_gallery:
        blended = momentum * canonical_gallery[canonical_id] + (1 - momentum) * embedding
        canonical_gallery[canonical_id] = normalize_embedding(blended)
    else:
        canonical_gallery[canonical_id] = embedding


def match_recent_canonical(box, center, used_ids, frame_idx):
    best_id = None
    best_score = (-1.0, float("inf"))

    for canonical_id, state in canonical_states.items():
        if canonical_id in used_ids:
            continue

        if frame_idx - state["last_seen"] > SPATIAL_MATCH_MAX_GAP:
            continue

        prev_box = state["box"]
        prev_center = state["center"]
        iou = bbox_iou(box, prev_box)
        center_dist = np.linalg.norm(center - prev_center)

        if iou < SPATIAL_MATCH_IOU and center_dist > SPATIAL_MATCH_DIST:
            continue

        score = (iou, -center_dist)
        if score > best_score:
            best_score = score
            best_id = canonical_id

    return best_id


def assign_canonical_id(raw_track_id, box, center, embedding, used_ids, frame_idx):
    global next_canonical_id

    canonical_id = tracker_to_canonical.get(raw_track_id)
    if canonical_id is not None and canonical_id not in used_ids:
        return canonical_id

    matched_id = match_recent_canonical(box, center, used_ids=used_ids, frame_idx=frame_idx)
    if matched_id is not None:
        tracker_to_canonical[raw_track_id] = matched_id
        return matched_id

    appearance_id = match_long_term(embedding, used_ids=used_ids)
    if appearance_id is not None:
        tracker_to_canonical[raw_track_id] = appearance_id
        return appearance_id

    canonical_id = next_canonical_id
    next_canonical_id += 1
    tracker_to_canonical[raw_track_id] = canonical_id
    return canonical_id


def track_players(frame, player_dets, tracker, annotated_frame, frame_idx=0):
    current_player_boxes = []
    frame_players_data = []
    used_canonical_ids = set()
    used_detection_indices = set()

    tracker_dets = player_dets.copy()
    if len(tracker_dets) > 0:
        tracker_dets[:, 5] = PLAYER_CLASS_ID

    if len(tracker_dets) > 0:
        tracks = tracker.update(tracker_dets, frame)
    else:
        tracks = np.empty((0, 8))

    embedding_map = {}

    if hasattr(tracker, "tracker") and hasattr(tracker.tracker, "tracks"):
        for t in tracker.tracker.tracks:
            emb = None

            if hasattr(t, "smooth_feat") and t.smooth_feat is not None:
                emb = t.smooth_feat
            elif hasattr(t, "curr_feat") and t.curr_feat is not None:
                emb = t.curr_feat
            elif hasattr(t, "features") and len(t.features) > 0:
                emb = t.features[-1]

            tid = None
            if hasattr(t, "id"):
                tid = t.id
            elif hasattr(t, "track_id") and not callable(t.track_id):
                tid = t.track_id
            elif hasattr(t, "track_id") and callable(t.track_id):
                tid = t.track_id()

            if tid is not None and emb is not None:
                if isinstance(tid, (int, np.integer, str)):
                    try:
                        embedding_map[int(tid)] = emb
                    except ValueError:
                        continue

    if len(tracks) > 0:
        boxes = tracks[:, :4]
        track_ids = tracks[:, 4].astype(int)
        confs = tracks[:, 5]

        for box, track_id, conf in zip(boxes, track_ids, confs):
            raw_track_id = int(track_id)
            x1, y1, x2, y2 = map(int, box)
            track_box = (x1, y1, x2, y2)
            center = get_center((x1, y1, x2, y2))

            avg_emb = None
            if raw_track_id in embedding_map:
                emb = normalize_embedding(embedding_map[raw_track_id])
                if emb is not None:
                    appearance_history[raw_track_id].append(emb)
                    avg_emb = normalize_embedding(np.mean(appearance_history[raw_track_id], axis=0))

            canonical_id = assign_canonical_id(
                raw_track_id,
                track_box,
                center,
                avg_emb,
                used_canonical_ids,
                frame_idx,
            )

            if not is_motion_consistent(canonical_id, center):
                continue

            used_canonical_ids.add(canonical_id)

            update_gallery(canonical_id, avg_emb)
            canonical_states[canonical_id] = {
                "box": (x1, y1, x2, y2),
                "center": center,
                "last_seen": frame_idx,
            }

            # --- Track lifecycle ---
            lost_tracks[raw_track_id] = frame_idx

            # --- Draw ---
            color = (0, 255, 0)
            matched_det_idx = match_detection_to_track(track_box, player_dets, used_detection_indices)
            state = "player"
            state_conf = float(conf)
            if matched_det_idx is not None:
                used_detection_indices.add(matched_det_idx)
                matched_det = player_dets[matched_det_idx]
                state = detection_state_name(matched_det[5])
                state_conf = float(matched_det[4])

            label = f"P-{canonical_id} {state}"

            track_history[canonical_id].append(center)

            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated_frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            current_player_boxes.append((x1, y1, x2, y2))

            frame_players_data.append({
                "tid": int(canonical_id),
                "box": [x1, y1, x2, y2],
                "conf": float(conf),
                "state": state,
                "state_conf": state_conf,
            })

    # --- Cleanup old tracks ---
    to_delete = []
    for tid, last_seen in lost_tracks.items():
        if frame_idx - last_seen > FRAME_BUFFER:
            to_delete.append(tid)

    for tid in to_delete:
        lost_tracks.pop(tid, None)
        appearance_history.pop(tid, None)
        tracker_to_canonical.pop(tid, None)

    stale_canonical_ids = [
        canonical_id
        for canonical_id, state in canonical_states.items()
        if frame_idx - state["last_seen"] > FRAME_BUFFER
    ]

    for canonical_id in stale_canonical_ids:
        canonical_states.pop(canonical_id, None)
        canonical_gallery.pop(canonical_id, None)
        track_history.pop(canonical_id, None)

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

    cap = cv2.VideoCapture(args.input)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_size = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    MAX_COAST_FRAMES = 30
    ball_tracker = MultiBallTracker(max_coast_frames=MAX_COAST_FRAMES, fps=fps)

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

        # 1. Run YOLO once for volleyball, players, and player actions.
        results = model.predict(
            frame,
            conf=args.conf,
            classes=[BALL_CLASS_ID, *sorted(PLAYER_LIKE_CLASS_IDS)],
            verbose=False,
            imgsz=args.imgsz,
            iou=0.95,
            device=args.device
        )

        # 2. Split detections into volleyball vs. player-like detections.
        dets = results[0].boxes.data.cpu().numpy()
        player_dets = dets[np.isin(dets[:, 5].astype(int), list(PLAYER_LIKE_CLASS_IDS))] if len(dets) > 0 else np.empty((0, 6))
        ball_dets = dets[dets[:, 5].astype(int) == BALL_CLASS_ID] if len(dets) > 0 else np.empty((0, 6))

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
