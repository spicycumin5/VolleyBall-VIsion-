#!/usr/bin/env python3

import numpy as np
import cv2
from PIL import Image

import torch
import torchvision.transforms as T

from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from utils import get_center, is_motion_consistent, track_history

# --- Global Tracking State for DINOv2 ---
dino_tracks = {}
next_dino_id = 1
dino_model = None
dino_transform = None


def _initialize_dino_model():
    """Lazily loads the DINOv2 model and transform on the first frame."""
    global dino_model, dino_transform
    
    if dino_model is not None:
        return

    print("\n[INFO] Loading DINOv2 ViT-Giant...")
    dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14_reg')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dino_model.to(device)
    dino_model.eval()

    dino_transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def _extract_features(frame, player_dets):
    """Crops detections and processes them through DINOv2 to get embeddings."""
    global dino_model, dino_transform
    
    det_embs = []
    valid_dets = []
    det_centers = []
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for det in player_dets:
        x1, y1, x2, y2, conf, cls = det
        ix1, iy1, ix2, iy2 = map(int, [x1, y1, x2, y2])
        
        # Prevent out-of-bounds crops
        ix1, iy1 = max(0, ix1), max(0, iy1)
        ix2, iy2 = min(frame.shape[1], ix2), min(frame.shape[0], iy2)
        
        if ix2 - ix1 < 10 or iy2 - iy1 < 10:
            continue

        # Extract features
        crop_rgb = cv2.cvtColor(frame[iy1:iy2, ix1:ix2], cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(crop_rgb)
        tensor = dino_transform(pil_image).unsqueeze(0).to(device)

        with torch.no_grad():
            emb = dino_model(tensor).cpu().numpy()[0]
        
        # Normalize the embedding (Critical for cosine similarity)
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb /= norm
            
        det_embs.append(emb)
        valid_dets.append((ix1, iy1, ix2, iy2, conf))
        det_centers.append([(ix1 + ix2) / 2, (iy1 + iy2) / 2])

    return np.array(det_embs), valid_dets, np.array(det_centers)


def _match_tracks(active_track_ids, det_embs, det_centers, frame_width, cost_threshold=0.6):
    """Calculates cost matrix and runs the Hungarian algorithm to match identities."""
    track_embs = np.array([dino_tracks[tid]["emb"] for tid in active_track_ids])
    track_centers = np.array([get_center(dino_tracks[tid]["box"]) for tid in active_track_ids])
    
    # Calculate Distances
    appearance_cost = cdist(track_embs, det_embs, metric='cosine')
    spatial_cost = cdist(track_centers, det_centers, metric='euclidean') / frame_width
    
    # Blended Cost Matrix: 85% Appearance, 15% Motion
    cost_matrix = appearance_cost + (spatial_cost * 0.15)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # Filter valid matches below threshold
    matched_indices = []
    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] < cost_threshold:
            matched_indices.append((r, c))
            
    return matched_indices


def _update_tracking_state(active_track_ids, matched_indices, valid_dets, det_embs):
    """Updates bounding boxes, EMA embeddings, spawns new tracks, and deletes dead ones."""
    global dino_tracks, next_dino_id
    
    unmatched_dets = set(range(len(valid_dets)))
    matched_tracks = set()

    # 1. Update Matched Tracks
    for r, c in matched_indices:
        tid = active_track_ids[r]
        dino_tracks[tid]["box"] = valid_dets[c][:4]
        
        # Exponential moving average (update embedding slightly)
        dino_tracks[tid]["emb"] = 0.9 * dino_tracks[tid]["emb"] + 0.1 * det_embs[c]
        dino_tracks[tid]["missed"] = 0
        
        unmatched_dets.remove(c)
        matched_tracks.add(tid)

    # 2. Increment Missed for Unmatched Tracks
    for tid in active_track_ids:
        if tid not in matched_tracks:
            dino_tracks[tid]["missed"] += 1

    # 3. Spawn New Tracks
    for c in unmatched_dets:
        dino_tracks[next_dino_id] = {
            "box": valid_dets[c][:4],
            "emb": det_embs[c],
            "missed": 0
        }
        next_dino_id += 1

    # 4. Cleanup Dead Tracks
    dead_tracks = [tid for tid, data in dino_tracks.items() if data["missed"] > 30]
    for tid in dead_tracks:
        del dino_tracks[tid]


def _render_and_format(annotated_frame):
    """Draws boxes on the frame and formats the JSON-friendly output data."""
    current_player_boxes = []
    frame_players_data = []

    for tid, data in dino_tracks.items():
        if data["missed"] > 0:
            continue # Skip drawing if player is occluded this frame

        x1, y1, x2, y2 = data["box"]
        center = get_center((x1, y1, x2, y2))
        
        # Retain original motion consistency check
        if not is_motion_consistent(tid, center):
            continue
            
        track_history[tid].append(center)
        current_player_boxes.append((x1, y1, x2, y2))

        # Draw
        color = (255, 105, 180) # Hot Pink for DINO tracks
        label = f"P-{tid} (DINO)"
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated_frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Format Data
        frame_players_data.append({
            "tid": int(tid),
            "box": [x1, y1, x2, y2],
            "conf": 1.0 
        })

    return current_player_boxes, frame_players_data


def dino_track_players(frame, player_dets, tracker, annotated_frame):
    """
    Drop-in replacement using DINOv2 Giant (ViT-g14) for pure appearance-based ReID.
    Note: The 'tracker' argument is ignored as we use our native DINO engine.
    """
    # 1. Initialize
    _initialize_dino_model()

    if len(player_dets) == 0:
        return [], []

    # 2. Extract Features
    det_embs, valid_dets, det_centers = _extract_features(frame, player_dets)

    if not valid_dets:
        return [], []
        
    # 3. Match Tracks
    active_track_ids = list(dino_tracks.keys())
    matched_indices = []
    
    if active_track_ids:
        matched_indices = _match_tracks(active_track_ids, det_embs, det_centers, frame.shape[1])

    # 4. Update State
    _update_tracking_state(active_track_ids, matched_indices, valid_dets, det_embs)

    # 5. Output Results
    return _render_and_format(annotated_frame)
