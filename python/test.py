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
import warnings
from rich.progress import track as track
from ultralytics import YOLO
from PIL import Image
from boxmot import BotSort, StrongSort
from scipy.optimize import linear_sum_assignment

from ball_tracker import MultiBallTracker 
from player_action_db import PlayerActionDatabase
from utils import get_center, is_motion_consistent, track_history, is_on_player

# --- Appearance memory ---
appearance_history = defaultdict(lambda: deque(maxlen=30))

# --- Separate YOLO classes ---
BALL_CLASS_ID = 0
PLAYER_TRACK_CLASS_ID = 0
ACTION_ID_TO_STATE = {
    0: "block",
    1: "defense",
    2: "serve",
    3: "set",
    4: "spike",
}

# --- Long-term identity DB ---
canonical_gallery = {}
tracker_to_canonical = {}
canonical_states = {}
canonical_center_history = defaultdict(lambda: deque(maxlen=10))
canonical_bottom_history = defaultdict(lambda: deque(maxlen=10))
next_canonical_id = 1
GALLERY_THRESHOLD = 0.7
SPATIAL_MATCH_IOU = 0.6
SPATIAL_MATCH_DIST = 45
SPATIAL_MATCH_MAX_GAP = 3
RECENT_RECLAIM_MAX_GAP = 30
RECENT_RECLAIM_IOU = 0.75
MOTION_MATCH_MAX_GAP = 30
PREDICTED_CENTER_DISTANCE_SCALE = 1.75
PREDICTED_BOTTOM_DISTANCE_SCALE = 1.25
MAX_ASSIGNMENT_COST = 1e6

# --- Track lifecycle ---
lost_tracks = {}  # track_id -> last_seen_frame
FRAME_BUFFER = 100

parser = argparse.ArgumentParser()

parser.add_argument("--model", default=None,
                    help="Legacy unified model path. Prefer --player_model/--action_model/--ball_model")
parser.add_argument("--player_model", default=None,
                    help="Path to player detector weights")
parser.add_argument("--action_model", default=None,
                    help="Path to action detector weights")
parser.add_argument("--ball_model", default=None,
                    help="Path to ball detector weights")
parser.add_argument("--input", required=True, help="Input video path")
parser.add_argument("--output", required=True, help="Output video path")
parser.add_argument('--show_conf', default=False, action='store_true',
                    help='Whether to show the confidence scores')
parser.add_argument('--conf', '--confs', dest='conf', type=float, default=0.2,
                    help='Default confidence threshold for player, action, and ball detection')
parser.add_argument('--player_conf', type=float, default=None,
                    help='Confidence threshold override for player detection')
parser.add_argument('--action_conf', type=float, default=None,
                    help='Confidence threshold override for action detection')
parser.add_argument('--ball_conf', type=float, default=None,
                    help='Confidence threshold override for ball detection')
parser.add_argument('--imgsz', type=int, default=1920,
                    help='Image size for YOLO. 640, 1280, and 1920 are good')
parser.add_argument('--player_imgsz', type=int, default=None,
                    help='Optional inference size override for player model')
parser.add_argument('--action_imgsz', type=int, default=None,
                    help='Optional inference size override for action model')
parser.add_argument('--ball_imgsz', type=int, default=None,
                    help='Optional inference size override for ball model')
parser.add_argument('--ball_gravity', type=float, default=600.0,
                    help='Ball-tracker gravity in pixels/sec^2')
parser.add_argument('--heatmap_conf', type=int, default=0.5,
                    help='Confidence for the ball tracker')
parser.add_argument('--heatmap_alpha', type=float, default=0.0,
                    help='Alpha for heatmap overlay')
parser.add_argument('--json_output', default=None, help="Path to output json")
parser.add_argument('--action_json_output', default=None,
                    help='Path to output player action table json')
parser.add_argument('--db_output', default=None, help="Path to output sqlite database")
parser.add_argument('--dino', action='store_true', default=False,
                    help='Deprecated and ignored. Player tracking always uses BoxMOT')
parser.add_argument('--device', default=0,
                    help='Device to use')

args = parser.parse_args()


def write_json_frame(file_obj, frame_idx, ball, players):
    if file_obj is None:
        return
    frame_data = {
        "frame": frame_idx,
        "ball": ball,
        "players": players
    }

    file_obj.write(json.dumps(frame_data) + '\n')


def resolve_action_json_output(json_output_path, action_json_output_path):
    if action_json_output_path:
        return action_json_output_path

    if not json_output_path:
        return None

    json_path = Path(json_output_path)
    return str(json_path.with_name(f"{json_path.stem}_actions.json"))


def collect_action_rows(action_rows, frame_idx, players):
    if action_rows is None:
        return

    for player in players:
        action = str(player.get("state", "player"))
        if action == "player":
            continue

        action_rows.append({
            "player_id": int(player["tid"]),
            "action": action,
            "frame": int(frame_idx),
            "action_conf": None if player.get("state_conf") is None else float(player["state_conf"]),
        })


def write_action_table(file_path, action_rows):
    if file_path is None:
        return

    parent = os.path.split(file_path)[0]
    if parent:
        os.makedirs(parent, exist_ok=True)

    with open(file_path, 'w') as file_obj:
        json.dump(action_rows, file_obj)


def warn_if_dino_requested(enabled):
    if enabled:
        warnings.warn("`--dino` is deprecated and ignored; using BoxMOT tracking instead.", stacklevel=2)


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
    return ACTION_ID_TO_STATE.get(int(class_id), "player")


def get_bottom_center(box):
    x1, y1, x2, y2 = box
    return np.array([(x1 + x2) / 2, y2], dtype="float32")


def get_box_size(box):
    x1, y1, x2, y2 = box
    width = max(float(x2 - x1), 1.0)
    height = max(float(y2 - y1), 1.0)
    return width, height


def get_box_diag(box):
    width, height = get_box_size(box)
    return float(np.hypot(width, height))


def estimate_velocity(history):
    if len(history) < 2:
        return np.zeros(2, dtype="float32")

    first_frame, first_point = history[0]
    last_frame, last_point = history[-1]
    frame_gap = max(int(last_frame - first_frame), 1)
    velocity = (np.asarray(last_point, dtype="float32") - np.asarray(first_point, dtype="float32")) / frame_gap
    return velocity.astype("float32")


def predict_point(last_point, velocity, frame_gap):
    return np.asarray(last_point, dtype="float32") + np.asarray(velocity, dtype="float32") * max(frame_gap, 0)


def bbox_from_center(bottom_center, size):
    width, height = size
    cx, by = map(float, bottom_center)
    return (
        cx - (width / 2.0),
        by - height,
        cx + (width / 2.0),
        by,
    )


def expand_box(box, pad_ratio=0.1):
    x1, y1, x2, y2 = map(float, box)
    width = max(x2 - x1, 1.0)
    height = max(y2 - y1, 1.0)
    pad_x = width * pad_ratio
    pad_y = height * pad_ratio
    return (x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y)


def point_in_box(point, box):
    px, py = map(float, point)
    x1, y1, x2, y2 = map(float, box)
    return x1 <= px <= x2 and y1 <= py <= y2


def embedding_distance(embedding, canonical_id):
    gallery_embedding = canonical_gallery.get(canonical_id)
    if embedding is None or gallery_embedding is None:
        return None
    return float(np.linalg.norm(embedding - gallery_embedding))


def build_canonical_candidate(box, center, bottom_center, embedding, raw_track_id, canonical_id, frame_idx):
    state = canonical_states.get(canonical_id)
    if state is None:
        return None

    gap = frame_idx - state["last_seen"]
    if gap < 1 or gap > MOTION_MATCH_MAX_GAP:
        return None

    previous_box = state["box"]
    iou = bbox_iou(box, previous_box)
    width, height = get_box_size(box)
    scale = max(get_box_diag(box), get_box_diag(previous_box), 1.0)

    center_velocity = estimate_velocity(canonical_center_history[canonical_id])
    bottom_velocity = estimate_velocity(canonical_bottom_history[canonical_id])
    predicted_center = predict_point(state["center"], center_velocity, gap)
    predicted_bottom = predict_point(state["bottom_center"], bottom_velocity, gap)
    predicted_box = bbox_from_center(predicted_bottom, state["size"])
    predicted_iou = bbox_iou(box, predicted_box)
    previous_box_padded = expand_box(previous_box)
    predicted_box_padded = expand_box(predicted_box)

    center_dist = float(np.linalg.norm(center - predicted_center))
    bottom_dist = float(np.linalg.norm(bottom_center - predicted_bottom))
    appearance_dist = embedding_distance(embedding, canonical_id)
    sticky_match = tracker_to_canonical.get(raw_track_id) == canonical_id
    center_contained = point_in_box(center, previous_box_padded) or point_in_box(center, predicted_box_padded)
    bottom_contained = point_in_box(bottom_center, previous_box_padded) or point_in_box(bottom_center, predicted_box_padded)
    spatially_contained = center_contained or bottom_contained

    reclaim = gap <= RECENT_RECLAIM_MAX_GAP and (iou >= RECENT_RECLAIM_IOU or predicted_iou >= RECENT_RECLAIM_IOU)
    max_center_dist = max(SPATIAL_MATCH_DIST, scale * PREDICTED_CENTER_DISTANCE_SCALE)
    max_bottom_dist = max(SPATIAL_MATCH_DIST, scale * PREDICTED_BOTTOM_DISTANCE_SCALE)
    strong_appearance = appearance_dist is not None and appearance_dist < GALLERY_THRESHOLD

    if not reclaim:
        if not spatially_contained and not sticky_match:
            return None
        motion_ok = center_dist <= max_center_dist or bottom_dist <= max_bottom_dist
        overlap_ok = iou >= 0.2 or predicted_iou >= 0.2
        if not motion_ok and not overlap_ok and not strong_appearance:
            return None

    cost = 0.0
    cost += bottom_dist / max_bottom_dist
    cost += 0.75 * (center_dist / max_center_dist)
    cost += 0.6 * (1.0 - max(iou, predicted_iou))
    if appearance_dist is not None:
        cost += 0.5 * (appearance_dist / GALLERY_THRESHOLD)
    if sticky_match:
        cost -= 0.2
    if reclaim:
        cost -= 0.35

    return {
        "canonical_id": canonical_id,
        "cost": cost,
        "gap": gap,
        "iou": iou,
        "predicted_iou": predicted_iou,
        "center_dist": center_dist,
        "bottom_dist": bottom_dist,
        "appearance_dist": appearance_dist,
        "reclaim": reclaim,
        "sticky_match": sticky_match,
        "spatially_contained": spatially_contained,
    }


def assign_canonical_ids_for_tracks(track_entries, frame_idx):
    global next_canonical_id

    assignments = {}
    candidate_ids = sorted(
        canonical_id
        for canonical_id, state in canonical_states.items()
        if 0 < frame_idx - state["last_seen"] <= MOTION_MATCH_MAX_GAP
    )

    if candidate_ids and track_entries:
        cost_matrix = np.full((len(track_entries), len(candidate_ids)), MAX_ASSIGNMENT_COST, dtype="float32")
        candidate_details = {}

        for row, entry in enumerate(track_entries):
            for col, canonical_id in enumerate(candidate_ids):
                candidate = build_canonical_candidate(
                    entry["box"],
                    entry["center"],
                    entry["bottom_center"],
                    entry["embedding"],
                    entry["raw_track_id"],
                    canonical_id,
                    frame_idx,
                )
                if candidate is None:
                    continue
                cost_matrix[row, col] = candidate["cost"]
                candidate_details[(row, col)] = candidate

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        used_ids = set()
        for row, col in zip(row_ind, col_ind):
            if cost_matrix[row, col] >= MAX_ASSIGNMENT_COST:
                continue
            candidate = candidate_details[(row, col)]
            canonical_id = candidate["canonical_id"]
            if canonical_id in used_ids:
                continue
            assignments[row] = canonical_id
            used_ids.add(canonical_id)

    used_ids = set(assignments.values())
    for row, entry in enumerate(track_entries):
        if row in assignments:
            continue

        appearance_id = match_long_term(entry["embedding"], used_ids=used_ids)
        if appearance_id is not None:
            assignments[row] = appearance_id
            used_ids.add(appearance_id)
            continue

        canonical_id = next_canonical_id
        next_canonical_id += 1
        assignments[row] = canonical_id
        used_ids.add(canonical_id)

    for row, canonical_id in assignments.items():
        tracker_to_canonical[track_entries[row]["raw_track_id"]] = canonical_id

    return assignments


def update_canonical_state(canonical_id, box, center, bottom_center, frame_idx):
    width, height = get_box_size(box)
    canonical_states[canonical_id] = {
        "box": tuple(map(int, box)),
        "center": np.asarray(center, dtype="float32"),
        "bottom_center": np.asarray(bottom_center, dtype="float32"),
        "size": (width, height),
        "last_seen": frame_idx,
    }
    canonical_center_history[canonical_id].append((frame_idx, np.asarray(center, dtype="float32")))
    canonical_bottom_history[canonical_id].append((frame_idx, np.asarray(bottom_center, dtype="float32")))


def match_detection_to_track(box, detections, used_detection_indices):
    best_idx = None
    best_score = (-1.0, -1.0)

    for idx, det in enumerate(detections):
        if idx in used_detection_indices:
            continue

        det_box = tuple(map(int, det[:4]))
        det_conf = float(det[4]) if len(det) > 4 else 0.0
        iou = bbox_iou(box, det_box)

        if iou <= 0:
            continue

        score = (iou, det_conf)
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


def track_players(frame, player_dets, action_dets, tracker, annotated_frame, frame_idx=0):
    current_player_boxes = []
    frame_players_data = []
    used_detection_indices = set()

    tracker_dets = player_dets.copy()
    if len(tracker_dets) > 0:
        tracker_dets[:, 5] = PLAYER_TRACK_CLASS_ID

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
        track_entries = []

        for box, track_id, conf in zip(boxes, track_ids, confs):
            raw_track_id = int(track_id)
            x1, y1, x2, y2 = map(int, box)
            track_box = (x1, y1, x2, y2)
            center = get_center(track_box)
            bottom_center = get_bottom_center(track_box)

            avg_emb = None
            if raw_track_id in embedding_map:
                emb = normalize_embedding(embedding_map[raw_track_id])
                if emb is not None:
                    appearance_history[raw_track_id].append(emb)
                    avg_emb = normalize_embedding(np.mean(appearance_history[raw_track_id], axis=0))

            track_entries.append(
                {
                    "raw_track_id": raw_track_id,
                    "box": track_box,
                    "center": center,
                    "bottom_center": bottom_center,
                    "conf": float(conf),
                    "embedding": avg_emb,
                }
            )

        assignments = assign_canonical_ids_for_tracks(track_entries, frame_idx)

        for index, entry in enumerate(track_entries):
            canonical_id = assignments[index]
            track_box = entry["box"]
            center = entry["center"]
            raw_track_id = entry["raw_track_id"]
            x1, y1, x2, y2 = track_box

            if not is_motion_consistent(canonical_id, center):
                continue

            update_gallery(canonical_id, entry["embedding"])
            update_canonical_state(canonical_id, track_box, center, entry["bottom_center"], frame_idx)

            # --- Track lifecycle ---
            lost_tracks[raw_track_id] = frame_idx

            # --- Draw ---
            state = "player"
            state_conf = entry["conf"]
            matched_det_idx = match_detection_to_track(track_box, action_dets, used_detection_indices)
            if matched_det_idx is not None:
                used_detection_indices.add(matched_det_idx)
                matched_det = action_dets[matched_det_idx]
                state = detection_state_name(matched_det[5])
                state_conf = float(matched_det[4])

            color = (255, 0, 255) if state != "player" else (0, 255, 0)

            label = f"P-{canonical_id} {state}"

            track_history[canonical_id].append(center)

            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated_frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            current_player_boxes.append((x1, y1, x2, y2))

            frame_players_data.append({
                "tid": int(canonical_id),
                "box": [x1, y1, x2, y2],
                "conf": entry["conf"],
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
        canonical_center_history.pop(canonical_id, None)
        canonical_bottom_history.pop(canonical_id, None)
        track_history.pop(canonical_id, None)

    return current_player_boxes, frame_players_data


def track_ball(ball_dets, tracker, annotated_frame):
    primary_id, tracks = tracker.update(ball_dets)
    
    final_ball = None

    for tid, data in tracks.items():
        bx, by = int(data['pos'][0]), int(data['pos'][1])
        x1, y1, x2, y2 = [int(round(v)) for v in data['box']]
        is_predicted = data['is_predicted']
        is_primary = (tid == primary_id)
        
        if is_primary:
            final_ball = {
                "tid": int(tid),
                "box": [x1, y1, x2, y2],
                "center": [bx, by],
                "conf": float(data['conf']),
                "predicted": bool(is_predicted),
            }
            color = (255, 0, 255) if is_predicted else (0, 165, 255)
            label = "Primary (Physics)" if is_predicted else "Primary Ball"
            
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            cv2.circle(annotated_frame, (bx, by), 4, color, -1)
            cv2.putText(annotated_frame, label, (x1, y1 - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        else:
            # Draw stationary/secondary balls as hollow gray boxes so they don't distract.
            color = (150, 150, 150) 
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 1)

    return final_ball


def predict_detections(model, frame, conf, imgsz, device, classes=None):
    if model is None:
        return np.empty((0, 6))

    results = model.predict(
        frame,
        conf=conf,
        classes=classes,
        verbose=False,
        imgsz=imgsz,
        iou=0.95,
        device=device,
    )
    return results[0].boxes.data.cpu().numpy()


def resolve_imgsz(override, default):
    return override if override is not None else default


def resolve_conf(override, default):
    return override if override is not None else default


def resolve_video_fps(capture, fallback=30.0):
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    if not np.isfinite(fps) or fps <= 1e-3:
        return fallback
    return fps


def filter_detections_by_conf(dets, conf_threshold):
    if len(dets) == 0:
        return dets
    return dets[dets[:, 4] >= conf_threshold]


def load_models():
    unified_model = YOLO(args.model) if args.model else None
    player_model = YOLO(args.player_model) if args.player_model else None
    action_model = YOLO(args.action_model) if args.action_model else None
    ball_model = YOLO(args.ball_model) if args.ball_model else None

    use_separate_models = any(model is not None for model in [player_model, action_model, ball_model])
    if use_separate_models and player_model is None:
        raise ValueError("Separate-model mode requires --player_model")
    if not use_separate_models and unified_model is None:
        raise ValueError("Provide either --model or --player_model")

    return unified_model, player_model, action_model, ball_model, use_separate_models


def main():
    """Do the thing."""
    warn_if_dino_requested(args.dino)
    device = 0 if torch.cuda.is_available() else "cpu"
    player_conf = resolve_conf(args.player_conf, args.conf)
    action_conf = resolve_conf(args.action_conf, args.conf)
    ball_conf = resolve_conf(args.ball_conf, args.conf)
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
    unified_model, player_model, action_model, ball_model, use_separate_models = load_models()

    # tracknet = PyTorchTrackNetTracker(
    #     weights_path="python/weights/tracknet-v4_best-model.pth",
    #     threshold=args.heatmap_conf,
    # )

    cap = cv2.VideoCapture(args.input)
    fps = resolve_video_fps(cap)
    frame_size = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    MAX_COAST_FRAMES = 30
    ball_tracker = MultiBallTracker(
        max_coast_frames=MAX_COAST_FRAMES,
        fps=fps,
        gravity=args.ball_gravity,
    ) if (ball_model or unified_model) else None

    os.makedirs(os.path.split(args.output)[0], exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, frame_size)

    json_file = None
    if args.json_output:
        os.makedirs(os.path.split(args.json_output)[0], exist_ok=True)
        json_file = open(args.json_output, 'w')

    action_json_output = resolve_action_json_output(args.json_output, args.action_json_output)
    action_rows = [] if action_json_output else None

    player_action_db = None
    if args.db_output:
        player_action_db = PlayerActionDatabase(
            db_path=args.db_output,
            video_path=args.input,
            fps=fps,
            total_frames=total_frames,
        )

    for frame_idx in track(range(total_frames)):
        ret, frame = cap.read()
        if not ret:
            break

        annotated_frame = frame.copy()

        if use_separate_models:
            player_dets = predict_detections(
                player_model,
                frame,
                conf=player_conf,
                imgsz=resolve_imgsz(args.player_imgsz, args.imgsz),
                device=args.device,
            )
            action_dets = predict_detections(
                action_model,
                frame,
                conf=action_conf,
                imgsz=resolve_imgsz(args.action_imgsz, args.imgsz),
                device=args.device,
            )
            ball_dets = predict_detections(
                ball_model,
                frame,
                conf=ball_conf,
                imgsz=resolve_imgsz(args.ball_imgsz, args.imgsz),
                device=args.device,
            ) if ball_model is not None else np.empty((0, 6))
        else:
            dets = predict_detections(
                unified_model,
                frame,
                conf=min(player_conf, action_conf, ball_conf),
                imgsz=args.imgsz,
                device=args.device,
                classes=[BALL_CLASS_ID, 1, 2, 3, 4, 5, 6],
            )
            player_dets = dets[np.isin(dets[:, 5].astype(int), [1, 2, 3, 4, 5, 6])] if len(dets) > 0 else np.empty((0, 6))
            player_dets = filter_detections_by_conf(player_dets, player_conf)
            action_dets = dets[np.isin(dets[:, 5].astype(int), [2, 3, 4, 5, 6])] if len(dets) > 0 else np.empty((0, 6))
            action_dets = filter_detections_by_conf(action_dets, action_conf)
            if len(action_dets) > 0:
                action_dets = action_dets.copy()
                action_dets[:, 5] = action_dets[:, 5] - 2
            ball_dets = dets[dets[:, 5].astype(int) == BALL_CLASS_ID] if len(dets) > 0 else np.empty((0, 6))
            ball_dets = filter_detections_by_conf(ball_dets, ball_conf)

        # 3. Track Players via BoxMOT
        current_player_boxes, frame_players_data = track_players(
            frame, player_dets, action_dets, tracker, annotated_frame, frame_idx
        )

        # 4. Track Ball via Kalman Physics
        final_ball = track_ball(ball_dets, ball_tracker, annotated_frame) if ball_tracker is not None else None

        # 5. Output
        out.write(annotated_frame)
        write_json_frame(json_file, frame_idx, final_ball, frame_players_data)
        collect_action_rows(action_rows, frame_idx, frame_players_data)
        if player_action_db is not None:
            player_action_db.add_frame_players(frame_idx, frame_players_data)

    cap.release()
    out.release()
    if json_file:
        json_file.close()
    write_action_table(action_json_output, action_rows)
    if player_action_db is not None:
        player_action_db.close()


if __name__ == "__main__":
    main()
