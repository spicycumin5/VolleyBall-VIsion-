import collections
import collections.abc
import json
import math
import os
import sys
import types
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np
import torch
from boxmot import StrongSort
from rich.progress import track
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO

from utils import get_center, is_motion_consistent, track_history


if not hasattr(collections, "Mapping"):
    collections.Mapping = collections.abc.Mapping

if not hasattr(torch, "_six"):
    torch_six = types.ModuleType("torch._six")
    torch_six.string_classes = (str,)
    torch_six.inf = math.inf
    torch_six.nan = math.nan
    torch_six.with_metaclass = lambda metaclass, *bases: metaclass("NewClass", bases, {})
    sys.modules["torch._six"] = torch_six


BALL_CLASS_ID = 0
PLAYER_TRACK_CLASS_ID = 0
ACTION_ID_TO_STATE = {
    0: "block",
    1: "defense",
    2: "serve",
    3: "set",
    4: "spike",
}

appearance_history = defaultdict(lambda: deque(maxlen=30))
canonical_gallery = {}
tracker_to_canonical = {}
canonical_redirects = {}
canonical_states = {}
canonical_center_history = defaultdict(lambda: deque(maxlen=10))
canonical_bottom_history = defaultdict(lambda: deque(maxlen=10))
next_canonical_id = 1

GALLERY_THRESHOLD = 0.7
SPATIAL_MATCH_IOU = 0.6
SPATIAL_MATCH_DIST = 45
SPATIAL_MATCH_MAX_GAP = 3
MERGE_IOU_THRESHOLD = 0.8
MERGE_AREA_RATIO_THRESHOLD = 0.75
RECENT_RECLAIM_MAX_GAP = 30
RECENT_RECLAIM_IOU = 0.75
MOTION_MATCH_MAX_GAP = 30
PREDICTED_CENTER_DISTANCE_SCALE = 1.75
PREDICTED_BOTTOM_DISTANCE_SCALE = 1.25
MAX_ASSIGNMENT_COST = 1e6
POSE_KEYPOINT_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]

lost_tracks = {}
FRAME_BUFFER = 100


def write_json_frame(file_obj, frame_idx, ball, players):
    if file_obj is None:
        return
    frame_data = {
        "frame": frame_idx,
        "ball": ball,
        "players": players,
    }
    file_obj.write(json.dumps(frame_data) + "\n")


def load_frame_rows(json_output_path):
    if not json_output_path or not os.path.exists(json_output_path):
        return []
    with open(json_output_path, "r") as json_file:
        raw_text = json_file.read().strip()
    if not raw_text:
        return []
    if raw_text.startswith("["):
        data = json.loads(raw_text)
        return data if isinstance(data, list) else []
    return [json.loads(raw_line) for raw_line in raw_text.splitlines() if raw_line.strip()]


def configure_tracking_memory(memory_frames):
    global FRAME_BUFFER
    FRAME_BUFFER = max(int(memory_frames), 1)
    return FRAME_BUFFER


def resolve_canonical_id(canonical_id):
    resolved_id = int(canonical_id)
    seen_ids = set()
    while resolved_id in canonical_redirects and resolved_id not in seen_ids:
        seen_ids.add(resolved_id)
        resolved_id = int(canonical_redirects[resolved_id])
    return resolved_id


def resolve_video_output_path(input_path, output_path, no_mp4):
    if no_mp4:
        return None
    if output_path:
        return output_path
    if not input_path:
        return None
    input_file = Path(input_path)
    return str(input_file.with_name(f"{input_file.stem}_tracked.mp4"))


def resolve_json_output_path(output_path, json_output_path, input_path=None):
    if json_output_path:
        return json_output_path
    if not output_path:
        if not input_path:
            return None
        input_file = Path(input_path)
        return str(input_file.with_name(f"{input_file.stem}_tracked.json"))
    output = Path(output_path)
    return str(output.with_suffix(".json"))


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
        action_rows.append(
            {
                "player_id": int(player["tid"]),
                "action": action,
                "frame": int(frame_idx),
                "action_conf": None if player.get("state_conf") is None else float(player["state_conf"]),
            }
        )


def write_action_table(file_path, action_rows):
    if file_path is None:
        return
    parent = os.path.split(file_path)[0]
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(file_path, "w") as file_obj:
        json.dump(action_rows, file_obj)


def warn_if_dino_requested(enabled):
    if enabled:
        import warnings

        warnings.warn("`--dino` is deprecated and ignored; using BoxMOT tracking instead.", stacklevel=2)


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


def bbox_area_ratio(box_a, box_b):
    area_a = max(0.0, float(box_a[2] - box_a[0])) * max(0.0, float(box_a[3] - box_a[1]))
    area_b = max(0.0, float(box_b[2] - box_b[0])) * max(0.0, float(box_b[3] - box_b[1]))
    max_area = max(area_a, area_b, 1e-6)
    min_area = min(area_a, area_b)
    return min_area / max_area


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
    return (cx - (width / 2.0), by - height, cx + (width / 2.0), by)


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
    return {"canonical_id": canonical_id, "cost": cost}


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
            canonical_id = candidate_details[(row, col)]["canonical_id"]
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
    assignments = merge_duplicate_assignments(track_entries, assignments)
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


def merge_canonical_ids(primary_id, duplicate_id):
    primary_id = resolve_canonical_id(primary_id)
    duplicate_id = resolve_canonical_id(duplicate_id)
    if primary_id == duplicate_id:
        return primary_id
    if primary_id not in canonical_states or duplicate_id not in canonical_states:
        return primary_id

    primary_state = canonical_states[primary_id]
    duplicate_state = canonical_states[duplicate_id]
    if duplicate_state["last_seen"] > primary_state["last_seen"]:
        canonical_states[primary_id] = duplicate_state

    primary_embedding = canonical_gallery.get(primary_id)
    duplicate_embedding = canonical_gallery.get(duplicate_id)
    if duplicate_embedding is not None:
        if primary_embedding is not None:
            canonical_gallery[primary_id] = normalize_embedding((primary_embedding + duplicate_embedding) / 2.0)
        else:
            canonical_gallery[primary_id] = duplicate_embedding

    canonical_center_history[primary_id].extend(canonical_center_history.get(duplicate_id, []))
    canonical_bottom_history[primary_id].extend(canonical_bottom_history.get(duplicate_id, []))
    track_history[primary_id].extend(track_history.get(duplicate_id, []))

    for raw_track_id, canonical_id in list(tracker_to_canonical.items()):
        if canonical_id == duplicate_id:
            tracker_to_canonical[raw_track_id] = primary_id

    canonical_redirects[duplicate_id] = primary_id
    for stale_id, redirected_id in list(canonical_redirects.items()):
        if resolve_canonical_id(redirected_id) != redirected_id:
            canonical_redirects[stale_id] = resolve_canonical_id(redirected_id)

    canonical_states.pop(duplicate_id, None)
    canonical_gallery.pop(duplicate_id, None)
    canonical_center_history.pop(duplicate_id, None)
    canonical_bottom_history.pop(duplicate_id, None)
    track_history.pop(duplicate_id, None)
    return primary_id


def merge_duplicate_assignments(track_entries, assignments):
    if len(track_entries) < 2:
        return assignments

    merged_assignments = dict(assignments)
    for left_idx in range(len(track_entries)):
        left_id = merged_assignments.get(left_idx)
        if left_id is None:
            continue
        left_box = track_entries[left_idx]["box"]
        for right_idx in range(left_idx + 1, len(track_entries)):
            right_id = merged_assignments.get(right_idx)
            if right_id is None or right_id == left_id:
                continue

            right_box = track_entries[right_idx]["box"]
            if bbox_iou(left_box, right_box) < MERGE_IOU_THRESHOLD:
                continue
            if bbox_area_ratio(left_box, right_box) < MERGE_AREA_RATIO_THRESHOLD:
                continue

            left_canonical_id = int(left_id)
            right_canonical_id = int(right_id)
            primary_id = min(left_canonical_id, right_canonical_id)
            duplicate_id = max(left_canonical_id, right_canonical_id)
            merged_id = merge_canonical_ids(primary_id, duplicate_id)
            for assignment_idx, assigned_id in list(merged_assignments.items()):
                if assigned_id == duplicate_id:
                    merged_assignments[assignment_idx] = merged_id
            left_id = merged_assignments.get(left_idx)

    return merged_assignments


def track_players(frame, player_dets, action_dets, tracker, annotated_frame, frame_idx=0, player_callback=None):
    current_player_boxes = []
    frame_players_data = []
    emitted_ids = set()
    used_detection_indices = set()
    tracker_dets = player_dets.copy()
    if len(tracker_dets) > 0:
        tracker_dets[:, 5] = PLAYER_TRACK_CLASS_ID
    tracks = tracker.update(tracker_dets, frame) if len(tracker_dets) > 0 else np.empty((0, 8))
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
                try:
                    embedding_map[int(str(tid))] = emb
                except (TypeError, ValueError):
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
            canonical_id = resolve_canonical_id(assignments[index])
            track_box = entry["box"]
            center = entry["center"]
            raw_track_id = entry["raw_track_id"]
            x1, y1, x2, y2 = track_box
            if canonical_id in emitted_ids:
                continue
            if not is_motion_consistent(canonical_id, center):
                continue
            update_gallery(canonical_id, entry["embedding"])
            update_canonical_state(canonical_id, track_box, center, entry["bottom_center"], frame_idx)
            lost_tracks[raw_track_id] = frame_idx
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
            cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            player_data = {
                "tid": int(canonical_id),
                "box": [x1, y1, x2, y2],
                "conf": entry["conf"],
                "state": state,
                "state_conf": state_conf,
            }
            if player_callback is not None:
                extra = player_callback(
                    frame=frame,
                    annotated_frame=annotated_frame,
                    track_box=track_box,
                    canonical_id=canonical_id,
                    entry=entry,
                    state=state,
                    state_conf=state_conf,
                    color=color,
                )
                if extra:
                    player_data.update(extra)
            current_player_boxes.append((x1, y1, x2, y2))
            frame_players_data.append(player_data)
            emitted_ids.add(canonical_id)
    to_delete = [tid for tid, last_seen in lost_tracks.items() if frame_idx - last_seen > FRAME_BUFFER]
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


def track_ball(ball_dets, tracker, annotated_frame, include_predicted_output=False, return_track_snapshot=False):
    primary_id, tracks = tracker.update(ball_dets)
    final_ball = None
    track_snapshot = []
    for tid, data in tracks.items():
        bx, by = int(data["pos"][0]), int(data["pos"][1])
        x1, y1, x2, y2 = [int(round(v)) for v in data["box"]]
        is_predicted = data["is_predicted"]
        is_primary = tid == primary_id
        track_snapshot.append(
            {
                "tid": int(tid),
                "center": [bx, by],
                "stagnant_frames": int(data.get("stagnant_frames", 0)),
                "is_predicted": bool(is_predicted),
                "is_primary": bool(is_primary),
            }
        )
        if is_primary:
            if is_predicted and not include_predicted_output:
                continue
            final_ball = {
                "tid": int(tid),
                "box": [x1, y1, x2, y2],
                "center": [bx, by],
                "conf": float(data["conf"]),
                "predicted": bool(is_predicted),
            }
            color = (255, 0, 255) if is_predicted else (0, 165, 255)
            label = "Primary (Physics)" if is_predicted else "Primary Ball"
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            cv2.circle(annotated_frame, (bx, by), 4, color, -1)
            cv2.putText(annotated_frame, label, (x1, y1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        else:
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (150, 150, 150), 1)
    if return_track_snapshot:
        return final_ball, track_snapshot
    return final_ball, None


def draw_pose_keypoints(annotated_frame, keypoints, threshold=0.35):
    visible = {}
    for keypoint in keypoints or []:
        keypoint_id = keypoint.get("id")
        if keypoint_id is None:
            continue
        conf = keypoint.get("conf")
        if conf is not None and float(conf) < threshold:
            continue
        visible[int(keypoint_id)] = (int(round(float(keypoint["x"]))), int(round(float(keypoint["y"]))))
    for start_idx, end_idx in POSE_KEYPOINT_CONNECTIONS:
        if start_idx in visible and end_idx in visible:
            cv2.line(annotated_frame, visible[start_idx], visible[end_idx], (255, 255, 0), 2)
    for point in visible.values():
        cv2.circle(annotated_frame, point, 3, (255, 255, 0), -1)


def draw_mask_polygons(annotated_frame, polygons, color, alpha=0.2):
    if not polygons:
        return
    overlay = annotated_frame.copy()
    for polygon in polygons:
        if len(polygon) < 3:
            continue
        contour = np.array([[int(point[0]), int(point[1])] for point in polygon], dtype=np.int32)
        cv2.fillPoly(overlay, [contour], color)
        cv2.polylines(annotated_frame, [contour], True, color, 2)
    cv2.addWeighted(overlay, alpha, annotated_frame, 1.0 - alpha, 0, annotated_frame)


def draw_player_annotations(annotated_frame, players):
    for player in players or []:
        box = player.get("box")
        if not box or len(box) != 4:
            continue
        x1, y1, x2, y2 = [int(round(float(value))) for value in box]
        state = str(player.get("state", "player"))
        color = (255, 0, 255) if state != "player" else (0, 255, 0)
        draw_mask_polygons(annotated_frame, player.get("mask_polygons"), color)
        draw_pose_keypoints(annotated_frame, player.get("keypoints", []))
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            annotated_frame,
            f"P-{int(player['tid'])} {state}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
        )


def draw_ball_annotation(annotated_frame, ball):
    if not ball:
        return
    x1, y1, x2, y2 = [int(round(float(value))) for value in ball["box"]]
    bx, by = [int(round(float(value))) for value in ball.get("center", [0, 0])]
    is_predicted = bool(ball.get("predicted", False))
    color = (255, 0, 255) if is_predicted else (0, 165, 255)
    label = "Primary (Physics)" if is_predicted else "Primary Ball"
    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
    cv2.circle(annotated_frame, (bx, by), 4, color, -1)
    cv2.putText(annotated_frame, label, (x1, y1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def render_frame_annotations(annotated_frame, frame_data):
    draw_player_annotations(annotated_frame, frame_data.get("players", []))
    draw_ball_annotation(annotated_frame, frame_data.get("ball"))
    frame_label = f"Frame {int(frame_data.get('frame', 0))}"
    text_size, _ = cv2.getTextSize(frame_label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    text_x = max(10, annotated_frame.shape[1] - text_size[0] - 12)
    text_y = 28
    cv2.putText(
        annotated_frame,
        frame_label,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        3,
    )
    cv2.putText(
        annotated_frame,
        frame_label,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 0),
        1,
    )
    return annotated_frame


def render_video_from_json(input_path, json_output_path, output_path, fps, frame_size):
    if not output_path or not json_output_path:
        return
    output_parent = os.path.split(output_path)[0]
    if output_parent:
        os.makedirs(output_parent, exist_ok=True)
    cap = cv2.VideoCapture(input_path)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, frame_size)
    try:
        frame_rows = load_frame_rows(json_output_path)
        for frame_data in track(frame_rows, description="Rendering MP4..."):
            ret, frame = cap.read()
            if not ret:
                break
            annotated_frame = render_frame_annotations(frame.copy(), frame_data)
            out.write(annotated_frame)
    finally:
        cap.release()
        out.release()


def interpolate_ball_gaps(frame_rows, fps, gravity, max_gap_frames=120, show_progress=False, interpolation_mode="physics"):
    if not frame_rows:
        return frame_rows

    fps_value = max(float(fps or 0.0), 1e-6)
    gravity_value = float(gravity or 0.0)
    last_seen_index = None

    def get_ball_center(frame_index):
        ball = frame_rows[frame_index].get("ball")
        if ball is None:
            return None
        center = ball.get("center")
        if center is None or len(center) < 2:
            return None
        return float(center[0]), float(center[1])

    def has_ball(frame_index):
        return frame_rows[frame_index].get("ball") is not None

    def is_observed_ball(frame_index):
        ball = frame_rows[frame_index].get("ball")
        return ball is not None and not bool(ball.get("predicted", False))

    def estimate_velocity_from_frames(left_index, right_index):
        if left_index is None or right_index is None or right_index <= left_index:
            return None
        left_center = get_ball_center(left_index)
        right_center = get_ball_center(right_index)
        if left_center is None or right_center is None:
            return None
        dt = max((right_index - left_index) / fps_value, 1e-6)
        return (
            (right_center[0] - left_center[0]) / dt,
            (right_center[1] - left_center[1]) / dt,
        )

    def collect_anchor_indices(anchor_index, direction, prefer_observed=True):
        indices = []
        cursor = anchor_index
        predicate = is_observed_ball if prefer_observed else has_ball
        while 0 <= cursor < len(frame_rows) and len(indices) < 3:
            if predicate(cursor):
                indices.append(cursor)
            cursor += direction
        if len(indices) < 2 and prefer_observed:
            return collect_anchor_indices(anchor_index, direction, prefer_observed=False)
        return indices

    def estimate_anchor_velocity(anchor_index, direction):
        anchor_indices = collect_anchor_indices(anchor_index, direction, prefer_observed=True)
        if len(anchor_indices) < 2:
            return None
        if direction < 0:
            anchor_indices = list(reversed(anchor_indices))
        return estimate_velocity_from_frames(anchor_indices[0], anchor_indices[-1])

    def estimate_local_gravity(previous_index, next_index):
        gravity_samples = []
        for anchor_index, direction in [(previous_index, -1), (next_index, 1)]:
            anchor_velocity = estimate_anchor_velocity(anchor_index, direction)
            if anchor_velocity is None:
                continue
            neighbor_indices = collect_anchor_indices(anchor_index, direction, prefer_observed=True)
            if len(neighbor_indices) < 3:
                continue
            if direction < 0:
                neighbor_indices = list(reversed(neighbor_indices))
            first_velocity = estimate_velocity_from_frames(neighbor_indices[0], neighbor_indices[1])
            second_velocity = estimate_velocity_from_frames(neighbor_indices[1], neighbor_indices[2])
            if first_velocity is None or second_velocity is None:
                continue
            dt = max((neighbor_indices[2] - neighbor_indices[0]) / fps_value / 2.0, 1e-6)
            gravity_samples.append((second_velocity[1] - first_velocity[1]) / dt)
        if gravity_samples:
            return float(np.clip(np.median(gravity_samples), 50.0, 4000.0))
        return gravity_value

    def estimate_bridge_velocity(start_pos, end_pos, total_time, _local_gravity):
        if total_time <= 0:
            return 0.0, 0.0
        vx = (end_pos[0] - start_pos[0]) / total_time
        vy = (end_pos[1] - start_pos[1]) / total_time
        return vx, vy

    def cubic_hermite_point(start_value, end_value, start_slope, end_slope, alpha, total_time):
        alpha2 = alpha * alpha
        alpha3 = alpha2 * alpha
        h00 = (2.0 * alpha3) - (3.0 * alpha2) + 1.0
        h10 = alpha3 - (2.0 * alpha2) + alpha
        h01 = (-2.0 * alpha3) + (3.0 * alpha2)
        h11 = alpha3 - alpha2
        return (
            (h00 * start_value)
            + (h10 * total_time * start_slope)
            + (h01 * end_value)
            + (h11 * total_time * end_slope)
        )

    def projectile_y(start_y, initial_vy, gravity_accel, t):
        return start_y + (initial_vy * t) + (0.5 * gravity_accel * t * t)

    def solve_projectile_vy(start_y, end_y, gravity_accel, total_time):
        if total_time <= 0:
            return 0.0
        return (end_y - start_y - (0.5 * gravity_accel * total_time * total_time)) / total_time

    def build_gap_path(previous_ball, next_ball, gap_frames):
        if gap_frames <= 0:
            return []
        start_center = previous_ball.get("center")
        end_center = next_ball.get("center")
        start_box = previous_ball.get("box")
        end_box = next_ball.get("box")
        if start_center is None or end_center is None or start_box is None or end_box is None:
            return []

        total_frames = gap_frames + 1
        total_time = total_frames / fps_value
        if total_time <= 0:
            return []

        start_pos = (float(start_center[0]), float(start_center[1]))
        end_pos = (float(end_center[0]), float(end_center[1]))
        local_gravity = estimate_local_gravity(last_seen_index, next_seen_index)
        fallback_vx, fallback_vy = estimate_bridge_velocity(start_pos, end_pos, total_time, local_gravity)
        start_velocity = estimate_anchor_velocity(last_seen_index, -1)
        if start_velocity is None:
            start_velocity = (fallback_vx, fallback_vy)
        end_velocity = estimate_anchor_velocity(next_seen_index, 1)
        if end_velocity is None:
            end_velocity = (fallback_vx, fallback_vy)

        if interpolation_mode == "cubic":
            path = []
            for frame_offset in range(1, gap_frames + 1):
                alpha = frame_offset / total_frames
                cx = cubic_hermite_point(start_pos[0], end_pos[0], float(start_velocity[0]), float(end_velocity[0]), alpha, total_time)
                cy = cubic_hermite_point(start_pos[1], end_pos[1], float(start_velocity[1]), float(end_velocity[1]), alpha, total_time)
                path.append((cx, cy, alpha))
            return path

        target_start_vx, target_start_vy = float(start_velocity[0]), float(start_velocity[1])
        target_end_vx, target_end_vy = float(end_velocity[0]), float(end_velocity[1])

        bridged_vx = 0.5 * (target_start_vx + target_end_vx)
        if not np.isfinite(bridged_vx):
            bridged_vx = fallback_vx
        bridged_vx = 0.5 * bridged_vx + 0.5 * fallback_vx

        no_bump_vy = solve_projectile_vy(start_pos[1], end_pos[1], local_gravity, total_time)
        no_bump_end_vy = no_bump_vy + (local_gravity * total_time)
        no_bump_score = abs(no_bump_vy - target_start_vy) + abs(no_bump_end_vy - target_end_vy)

        best_candidate = {
            "mode": "projectile",
            "score": no_bump_score,
            "initial_vy": no_bump_vy,
            "bump_time": None,
            "post_bump_vy": None,
        }

        likely_bump = target_start_vy > 0.0 or target_end_vy < 0.0 or (target_end_vy - target_start_vy) < (-0.75 * abs(local_gravity) * total_time)
        if likely_bump and total_frames >= 3:
            for bump_frame in range(1, total_frames):
                bump_time = bump_frame / fps_value
                remaining_time = total_time - bump_time
                if remaining_time <= 0:
                    continue

                pre_bump_y = projectile_y(start_pos[1], target_start_vy, local_gravity, bump_time)
                post_from_position = solve_projectile_vy(pre_bump_y, end_pos[1], local_gravity, remaining_time)
                post_from_velocity = target_end_vy - (local_gravity * remaining_time)
                post_bump_vy = 0.5 * (post_from_position + post_from_velocity)

                predicted_end_y = projectile_y(pre_bump_y, post_bump_vy, local_gravity, remaining_time)
                predicted_end_vy = post_bump_vy + (local_gravity * remaining_time)
                bump_impulse = post_bump_vy - (target_start_vy + (local_gravity * bump_time))

                score = 0.0
                score += abs(predicted_end_y - end_pos[1])
                score += 0.35 * abs(predicted_end_vy - target_end_vy)
                if bump_impulse > 0.0:
                    score -= min(bump_impulse * 0.01, 12.0)
                if bump_impulse < 20.0:
                    score += 20.0

                if score < best_candidate["score"]:
                    best_candidate = {
                        "mode": "bump",
                        "score": score,
                        "initial_vy": target_start_vy,
                        "bump_time": bump_time,
                        "post_bump_vy": post_bump_vy,
                    }

        path = []
        for frame_offset in range(1, gap_frames + 1):
            t = frame_offset / fps_value
            alpha = frame_offset / total_frames
            cx = start_pos[0] + (bridged_vx * t)
            if best_candidate["mode"] == "bump" and best_candidate["bump_time"] is not None and best_candidate["post_bump_vy"] is not None and t > best_candidate["bump_time"]:
                y_at_bump = (
                    start_pos[1]
                    + (best_candidate["initial_vy"] * best_candidate["bump_time"])
                    + (0.5 * local_gravity * best_candidate["bump_time"] * best_candidate["bump_time"])
                )
                dt_after_bump = t - best_candidate["bump_time"]
                cy = y_at_bump + (best_candidate["post_bump_vy"] * dt_after_bump) + (0.5 * local_gravity * dt_after_bump * dt_after_bump)
            else:
                cy = projectile_y(start_pos[1], best_candidate["initial_vy"], local_gravity, t)
            path.append((cx, cy, alpha))
        return path

    frame_iterator = enumerate(frame_rows)
    if show_progress:
        frame_iterator = enumerate(track(frame_rows, description="Filling ball gaps..."))

    for next_seen_index, frame_data in frame_iterator:
        ball = frame_data.get("ball")
        if ball is None:
            continue
        if last_seen_index is None:
            last_seen_index = next_seen_index
            continue

        gap = next_seen_index - last_seen_index - 1
        if gap <= 0 or gap > int(max_gap_frames):
            last_seen_index = next_seen_index
            continue

        previous_ball = frame_rows[last_seen_index].get("ball")
        next_ball = frame_data.get("ball")
        if previous_ball is None or next_ball is None:
            last_seen_index = next_seen_index
            continue

        start_center = previous_ball.get("center")
        end_center = next_ball.get("center")
        start_box = previous_ball.get("box")
        end_box = next_ball.get("box")
        if start_center is None or end_center is None or start_box is None or end_box is None:
            last_seen_index = next_seen_index
            continue
        if previous_ball.get("tid") != next_ball.get("tid") and gap > max(3, int(max_gap_frames) // 6):
            last_seen_index = next_seen_index
            continue

        start_width = float(start_box[2]) - float(start_box[0])
        start_height = float(start_box[3]) - float(start_box[1])
        end_width = float(end_box[2]) - float(end_box[0])
        end_height = float(end_box[3]) - float(end_box[1])
        interpolated_tid = int(previous_ball.get("tid", next_ball.get("tid", 1)))
        interpolated_conf = min(float(previous_ball.get("conf", 0.0)), float(next_ball.get("conf", 0.0)))
        gap_path = build_gap_path(previous_ball, next_ball, gap)

        for frame_offset, point in enumerate(gap_path, start=1):
            missing_index = last_seen_index + frame_offset
            if frame_rows[missing_index].get("ball") is not None:
                continue
            cx, cy, alpha = point
            width = (1.0 - alpha) * start_width + alpha * end_width
            height = (1.0 - alpha) * start_height + alpha * end_height
            frame_rows[missing_index]["ball"] = {
                "tid": interpolated_tid,
                "box": [
                    int(round(cx - (width / 2.0))),
                    int(round(cy - (height / 2.0))),
                    int(round(cx + (width / 2.0))),
                    int(round(cy + (height / 2.0))),
                ],
                "center": [int(round(cx)), int(round(cy))],
                "conf": interpolated_conf,
                "predicted": True,
            }

        last_seen_index = next_seen_index

    return frame_rows


def apply_ball_blacklist_to_frames(frame_rows, stagnant_frame_limit=5, stagnant_distance_thresh=4.0, show_progress=False, initial_blacklist_points=None):
    if not frame_rows:
        return frame_rows

    blacklist_points = [np.asarray(point, dtype="float32") for point in (initial_blacklist_points or [])]
    stagnant_run = []
    stagnant_anchor = None

    def is_blacklisted(center):
        point = np.asarray(center, dtype="float32")
        for blacklisted_point in blacklist_points:
            if np.linalg.norm(point - blacklisted_point) <= float(stagnant_distance_thresh):
                return True
        return False

    frame_iterable = frame_rows
    if show_progress:
        frame_iterable = track(frame_rows, description="Applying ball blacklist...")

    for frame_data in frame_iterable:
        ball = frame_data.get("ball")
        if ball is None:
            stagnant_run = []
            stagnant_anchor = None
            continue

        center = ball.get("center")
        if center is None or len(center) < 2:
            frame_data["ball"] = None
            stagnant_run = []
            stagnant_anchor = None
            continue

        center_point = [float(center[0]), float(center[1])]
        if is_blacklisted(center_point):
            frame_data["ball"] = None
            stagnant_run = []
            stagnant_anchor = None
            continue

        if stagnant_anchor is None:
            stagnant_anchor = center_point
            stagnant_run = [frame_data]
            continue

        distance_from_anchor = float(np.linalg.norm(np.asarray(center_point) - np.asarray(stagnant_anchor)))
        if distance_from_anchor <= float(stagnant_distance_thresh):
            stagnant_run.append(frame_data)
            if len(stagnant_run) >= int(stagnant_frame_limit):
                blacklist_points.append(np.asarray(stagnant_anchor, dtype="float32"))
                for stagnant_frame in stagnant_run:
                    stagnant_frame["ball"] = None
                stagnant_run = []
                stagnant_anchor = None
            continue

        stagnant_anchor = center_point
        stagnant_run = [frame_data]

    return frame_rows


def derive_ball_blacklist_points(ball_track_snapshots, stagnant_frame_limit=5, stagnant_distance_thresh=4.0):
    if not ball_track_snapshots:
        return []
    blacklist_points = []
    for frame_snapshot in ball_track_snapshots:
        for track_info in frame_snapshot or []:
            if int(track_info.get("stagnant_frames", 0)) < int(stagnant_frame_limit):
                continue
            center = track_info.get("center")
            if center is None or len(center) < 2:
                continue
            point = np.asarray([float(center[0]), float(center[1])], dtype="float32")
            if any(np.linalg.norm(point - existing) <= float(stagnant_distance_thresh) for existing in blacklist_points):
                continue
            blacklist_points.append(point)
    return blacklist_points


def rewrite_json_outputs(json_output_path, action_json_output_path=None, fps=None, ball_gravity=None, ball_gap_fill_frames=120, ball_interpolation_mode="physics", ball_blacklist_frames=5, ball_track_snapshots=None):
    if json_output_path and os.path.exists(json_output_path):
        rewritten_rows = load_frame_rows(json_output_path)
        for frame_data in track(rewritten_rows, description="Rewriting frame JSON..."):
            for player in frame_data.get("players", []):
                if "tid" in player:
                    player["tid"] = resolve_canonical_id(player["tid"])
            ball = frame_data.get("ball")
            if ball is not None:
                ball.pop("velocity", None)
                ball.pop("gravity", None)
        blacklist_points = derive_ball_blacklist_points(
            ball_track_snapshots,
            stagnant_frame_limit=ball_blacklist_frames,
        )
        rewritten_rows = apply_ball_blacklist_to_frames(
            rewritten_rows,
            stagnant_frame_limit=ball_blacklist_frames,
            show_progress=True,
            initial_blacklist_points=blacklist_points,
        )
        rewritten_rows = interpolate_ball_gaps(
            rewritten_rows,
            fps=fps,
            gravity=ball_gravity,
            max_gap_frames=ball_gap_fill_frames,
            show_progress=True,
            interpolation_mode=ball_interpolation_mode,
        )
        with open(json_output_path, "w") as json_file:
            if rewritten_rows:
                json.dump(rewritten_rows, json_file)
            else:
                json_file.write("[]")

    if action_json_output_path and os.path.exists(action_json_output_path):
        with open(action_json_output_path, "r") as action_file:
            action_rows = json.load(action_file)
        for row in track(action_rows, description="Rewriting action JSON..."):
            if "player_id" in row:
                row["player_id"] = resolve_canonical_id(row["player_id"])
        with open(action_json_output_path, "w") as action_file:
            json.dump(action_rows, action_file)


def predict_detections(model, frame, conf, imgsz, device, classes=None):
    if model is None:
        return np.empty((0, 6))
    results = model.predict(frame, conf=conf, classes=classes, verbose=False, imgsz=imgsz, iou=0.95, device=device)
    return results[0].boxes.data.cpu().numpy()


def resolve_imgsz(override, default):
    return override if override is not None else default


def resolve_conf(override, default):
    return override if override is not None else default


def resolve_device(device_arg, masked_device=None):
    if masked_device is not None:
        return masked_device
    if device_arg is None:
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if isinstance(device_arg, str):
        normalized = device_arg.strip().lower()
        if normalized == "auto":
            return "cuda:0" if torch.cuda.is_available() else "cpu"
        if normalized.isdigit():
            return f"cuda:{int(normalized)}"
        if normalized.startswith("cuda:"):
            return normalized
        return normalized
    return device_arg


def configure_runtime_device(device):
    if not isinstance(device, str) or not device.startswith("cuda:") or "," in device:
        return
    try:
        device_index = int(device.split(":", 1)[1])
    except (IndexError, ValueError):
        return
    torch.cuda.set_device(device_index)


def resolve_video_fps(capture, fallback=30.0):
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    if not np.isfinite(fps) or fps <= 1e-3:
        return fallback
    return fps


def filter_detections_by_conf(dets, conf_threshold):
    if len(dets) == 0:
        return dets
    return dets[dets[:, 4] >= conf_threshold]


def move_model_to_device(model, device):
    if model is None:
        return None
    model.to(device if device != "cpu" else "cpu")
    model.predictor = None
    return model


def load_yolo_models(device, model=None, player_model=None, action_model=None, ball_model=None, require_player_model=True):
    unified_model = move_model_to_device(YOLO(model), device) if model else None
    player_model_obj = move_model_to_device(YOLO(player_model), device) if player_model else None
    action_model_obj = move_model_to_device(YOLO(action_model), device) if action_model else None
    ball_model_obj = move_model_to_device(YOLO(ball_model), device) if ball_model else None
    use_separate_models = any(model_obj is not None for model_obj in [player_model_obj, action_model_obj, ball_model_obj])
    if use_separate_models and require_player_model and player_model_obj is None:
        raise ValueError("Separate-model mode requires --player_model")
    if not use_separate_models and unified_model is None:
        raise ValueError("Provide either --model or --player_model")
    return unified_model, player_model_obj, action_model_obj, ball_model_obj, use_separate_models


def detect_standard_frame(frame, args, device, unified_model, player_model, action_model, ball_model, use_separate_models):
    player_conf = resolve_conf(args.player_conf, args.conf)
    action_conf = resolve_conf(args.action_conf, args.conf)
    ball_conf = resolve_conf(args.ball_conf, args.conf)
    if use_separate_models:
        player_dets = predict_detections(player_model, frame, conf=player_conf, imgsz=resolve_imgsz(args.player_imgsz, args.imgsz), device=device)
        action_dets = predict_detections(action_model, frame, conf=action_conf, imgsz=resolve_imgsz(args.action_imgsz, args.imgsz), device=device)
        ball_dets = predict_detections(ball_model, frame, conf=ball_conf, imgsz=resolve_imgsz(args.ball_imgsz, args.imgsz), device=device) if ball_model is not None else np.empty((0, 6))
    else:
        dets = predict_detections(unified_model, frame, conf=min(player_conf, action_conf, ball_conf), imgsz=args.imgsz, device=device, classes=[BALL_CLASS_ID, 1, 2, 3, 4, 5, 6])
        player_dets = dets[np.isin(dets[:, 5].astype(int), [1, 2, 3, 4, 5, 6])] if len(dets) > 0 else np.empty((0, 6))
        player_dets = filter_detections_by_conf(player_dets, player_conf)
        action_dets = dets[np.isin(dets[:, 5].astype(int), [2, 3, 4, 5, 6])] if len(dets) > 0 else np.empty((0, 6))
        action_dets = filter_detections_by_conf(action_dets, action_conf)
        if len(action_dets) > 0:
            action_dets = action_dets.copy()
            action_dets[:, 5] = action_dets[:, 5] - 2
        ball_dets = dets[dets[:, 5].astype(int) == BALL_CLASS_ID] if len(dets) > 0 else np.empty((0, 6))
        ball_dets = filter_detections_by_conf(ball_dets, ball_conf)
    return player_dets, action_dets, ball_dets


def create_strongsort(device, reid_weights="osnet_ain_x1_0_msmt17.pt"):
    return StrongSort(
        reid_weights=Path(reid_weights),
        device=device,
        half=False,
        max_dist=0.4,
        max_iou_distance=0.8,
        max_age=50,
        nn_budget=200,
    )
