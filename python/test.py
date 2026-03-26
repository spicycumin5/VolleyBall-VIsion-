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
from boxmot import BotSort

from ball_tracker import KalmanBallTracker

from dino_tracker import dino_track_players

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

args = parser.parse_args()


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


def write_json_frame(file_obj, frame_idx, ball_pos, players, ball_conf):
    if file_obj is None:
        return
    frame_data = {
        "frame": frame_idx,
        "ball": {"x": ball_pos[0], "y": ball_pos[1], "conf": ball_conf} if ball_pos else None,
        "players": players
    }

    file_obj.write(json.dumps(frame_data) + '\n')


track_history = defaultdict(lambda: deque(maxlen=10))


def get_center(box):
    x1, y1, x2, y2 = box
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2])


def box_area(box):
    x1, y1, x2, y2 = box
    return (x2 - x1) * (y2 - y1)


def motion_distance(prev_center, new_center):
    return np.linalg.norm(prev_center - new_center)


def is_far_player(box, threshold=5000):
    return box_area(box) < threshold


def is_motion_consistent(track_id, new_center, max_jump=150):
    """
    Reject matches where the object teleports unrealistically.
    """
    if len(track_history[track_id]) == 0:
        return True

    prev_center = track_history[track_id][-1]
    dist = motion_distance(prev_center, new_center)

    return dist < max_jump


def track_players(frame, player_dets, tracker, annotated_frame):
    """Processes YOLO player detections through BoxMOT for ID tracking."""
    current_player_boxes = []
    frame_players_data = []

    if len(player_dets) > 0:
        # BoxMOT expects array of [x1, y1, x2, y2, conf, cls]
        tracks = tracker.update(player_dets, frame)
    else:
        tracks = np.empty((0, 8))

    if len(tracks) > 0:
        boxes = tracks[:, :4]
        track_ids = tracks[:, 4].astype(int)
        confs = tracks[:, 5]

        for box, track_id, conf in zip(boxes, track_ids, confs):
            x1, y1, x2, y2 = map(int, box)
            current_player_boxes.append((x1, y1, x2, y2))
            center = get_center((x1, y1, x2, y2))

            # Motion consistency checks
            if not is_motion_consistent(track_id, center):
                continue

            color = (0, 255, 0)
            label = f"P-{track_id}"

            # Log history and draw
            track_history[track_id].append(center)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated_frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            frame_players_data.append({
                "tid": int(track_id),
                "box": [x1, y1, x2, y2],
                "conf": float(conf)
            })

    return current_player_boxes, frame_players_data


def track_ball(ball_dets, kalman, current_player_boxes, annotated_frame, missing_frames, max_coast_frames):
    """Processes YOLO ball detections, applying Kalman physics for dropped frames."""
    final_ball_pos = None
    is_predicted = False
    ball_conf = 0.0
    bx, by = None, None

    # 1. Extract the best YOLO ball detection
    if len(ball_dets) > 0:
        # Sort by confidence descending and grab the best one
        ball_dets = ball_dets[ball_dets[:, 4].argsort()[::-1]]
        x1, y1, x2, y2, ball_conf, cls = ball_dets[0]
        bx, by = int((x1 + x2) / 2), int((y1 + y2) / 2)

    # 2. Apply Kalman Smoothing / Coasting
    if bx is not None and by is not None:
        if not kalman.is_tracking and is_on_player(bx, by, current_player_boxes):
            pass
        else:
            kalman.correct(bx, by)
            final_ball_pos = kalman.predict()
            missing_frames = 0
    else:
        missing_frames += 1
        if missing_frames <= max_coast_frames:
            final_ball_pos = kalman.predict()
            is_predicted = True
        else:
            kalman.reset()

    # 3. Draw Results
    if final_ball_pos is not None:
        bx, by = int(final_ball_pos[0]), int(final_ball_pos[1])
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

    return final_ball_pos, ball_conf, missing_frames


def main():
    """Do the thing."""
    device = 0 if torch.cuda.is_available() else "cpu"
    # https://github.com/mikel-brostrom/boxmot/blob/59784c49eeec19736b48e034382d393d764d611d/boxmot/trackers/botsort/botsort.py#L21
    tracker = BotSort(
        reid_weights=Path('osnet_ibn_x1_0_msmt17.pt'),
        device=device,
        half=False,
        match_thresh=0.95,
        proximity_thresh=0.86,
        appearance_thresh=0.7,
    )
    model = YOLO(args.model)

    # tracknet = PyTorchTrackNetTracker(
    #     weights_path="python/weights/tracknet-v4_best-model.pth",
    #     threshold=args.heatmap_conf,
    # )
    kalman = KalmanBallTracker()

    missing_frames = 0
    MAX_COAST_FRAMES = 20

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
        )

        # 2. Split the YOLO detections into two arrays based on class
        dets = results[0].boxes.data.cpu().numpy()
        player_dets = dets[dets[:, 5] == 0] if len(dets) > 0 else np.empty((0, 6))
        ball_dets = dets[dets[:, 5] == 1] if len(dets) > 0 else np.empty((0, 6))

        # 3. Track Players via BoxMOT
        current_player_boxes, frame_players_data = dino_track_players(
            frame, player_dets, tracker, annotated_frame
        )

        # 4. Track Ball via Kalman Physics
        final_ball_pos, ball_conf, missing_frames = track_ball(
            ball_dets, kalman, current_player_boxes, annotated_frame, missing_frames, MAX_COAST_FRAMES
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
