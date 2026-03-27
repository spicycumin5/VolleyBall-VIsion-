"""SAM 3 hybrid tracker built on the regular pipeline."""

import argparse
import os
import sys


def preparse_device_arg(argv):
    for idx, arg in enumerate(argv):
        if arg == "--device" and idx + 1 < len(argv):
            return argv[idx + 1]
        if arg.startswith("--device="):
            return arg.split("=", 1)[1]
    return None


def configure_cuda_visibility(raw_device):
    if raw_device is None:
        return None
    normalized = str(raw_device).strip().lower()
    if normalized in {"", "auto"}:
        return None
    if normalized == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        return "cpu"
    if normalized.startswith("cuda:"):
        normalized = normalized.split(":", 1)[1]
    if normalized.isdigit():
        os.environ["CUDA_VISIBLE_DEVICES"] = normalized
        return "cuda:0"
    return None


MASKED_DEVICE = configure_cuda_visibility(preparse_device_arg(sys.argv))

import cv2
import numpy as np
from rich.progress import track
from ultralytics import SAM

from ball_tracker import MultiBallTracker
from player_action_db import PlayerActionDatabase
from tracking_shared import (
    collect_action_rows,
    configure_runtime_device,
    create_strongsort,
    detect_standard_frame,
    load_yolo_models,
    move_model_to_device,
    resolve_action_json_output,
    resolve_device,
    resolve_video_fps,
    track_ball,
    track_players,
    warn_if_dino_requested,
    write_action_table,
    write_json_frame,
)


parser = argparse.ArgumentParser()
parser.add_argument("--model", default=None, help="Legacy unified model path. Prefer --player_model/--action_model/--ball_model")
parser.add_argument("--player_model", default=None, help="Path to player detector weights")
parser.add_argument("--action_model", default=None, help="Path to action detector weights")
parser.add_argument("--ball_model", default=None, help="Path to ball detector weights")
parser.add_argument("--input", required=True, help="Input video path")
parser.add_argument("--output", required=True, help="Output video path")
parser.add_argument("--show_conf", default=False, action="store_true", help="Whether to show the confidence scores")
parser.add_argument("--conf", "--confs", dest="conf", type=float, default=0.2, help="Default confidence threshold for player, action, and ball detection")
parser.add_argument("--player_conf", type=float, default=None, help="Confidence threshold override for player detection")
parser.add_argument("--action_conf", type=float, default=None, help="Confidence threshold override for action detection")
parser.add_argument("--ball_conf", type=float, default=None, help="Confidence threshold override for ball detection")
parser.add_argument("--imgsz", type=int, default=1920, help="Image size for YOLO. 640, 1280, and 1920 are good")
parser.add_argument("--player_imgsz", type=int, default=None, help="Optional inference size override for player model")
parser.add_argument("--action_imgsz", type=int, default=None, help="Optional inference size override for action model")
parser.add_argument("--ball_imgsz", type=int, default=None, help="Optional inference size override for ball model")
parser.add_argument("--ball_gravity", type=float, default=600.0, help="Ball-tracker gravity in pixels/sec^2")
parser.add_argument("--heatmap_conf", type=int, default=0.5, help="Confidence for the ball tracker")
parser.add_argument("--heatmap_alpha", type=float, default=0.0, help="Alpha for heatmap overlay")
parser.add_argument("--json_output", default=None, help="Path to output json")
parser.add_argument("--action_json_output", default=None, help="Path to output player action table json")
parser.add_argument("--db_output", default=None, help="Path to output sqlite database")
parser.add_argument("--dino", action="store_true", default=False, help="Deprecated and ignored. Player tracking always uses BoxMOT")
parser.add_argument("--sam_model", default="sam3.pt", help="Path to SAM 3 weights used to refine tracked player masks")
parser.add_argument("--sam_alpha", type=float, default=0.2, help="Mask overlay alpha for SAM 3 player masks")
parser.add_argument("--sam_min_area", type=int, default=200, help="Minimum SAM mask area in pixels before drawing")
parser.add_argument("--device", default="auto", help="Device to use (for example: auto, cpu, 0, or 0,1)")


def segment_box_with_sam(frame, box, sam_model, device):
    if sam_model is None:
        return None
    results = sam_model.predict(source=frame, bboxes=[list(map(float, box))], verbose=False, device=device)
    if not results:
        return None
    result = results[0]
    if result.masks is None or len(result.masks.data) == 0:
        return None
    return result.masks.data[0].detach().cpu().numpy() > 0.5


def mask_to_polygons(mask):
    if mask is None:
        return []
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        if len(contour) < 3:
            continue
        polygons.append([[int(point[0][0]), int(point[0][1])] for point in contour])
    return polygons


def draw_sam_mask(annotated_frame, mask, color, alpha, min_area):
    if mask is None:
        return []
    mask_uint8 = mask.astype(np.uint8)
    if int(mask_uint8.sum()) < int(min_area):
        return []
    overlay = annotated_frame.copy()
    overlay[mask] = color
    cv2.addWeighted(overlay, alpha, annotated_frame, 1.0 - alpha, 0, annotated_frame)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(annotated_frame, contours, -1, color, 2)
    return [contour for contour in mask_to_polygons(mask) if len(contour) >= 3]


def make_sam_callback(args, sam_model, device):
    def callback(frame, annotated_frame, track_box, color, **_kwargs):
        polygons = draw_sam_mask(
            annotated_frame,
            segment_box_with_sam(frame, track_box, sam_model, device),
            color=color,
            alpha=args.sam_alpha,
            min_area=args.sam_min_area,
        )
        return {"mask_polygons": polygons}

    return callback


def main():
    args = parser.parse_args()
    warn_if_dino_requested(args.dino)
    device = resolve_device(args.device, MASKED_DEVICE)
    configure_runtime_device(device)

    tracker = create_strongsort(device)
    unified_model, player_model, action_model, ball_model, use_separate_models = load_yolo_models(
        device,
        model=args.model,
        player_model=args.player_model,
        action_model=args.action_model,
        ball_model=args.ball_model,
    )
    sam_model = move_model_to_device(SAM(args.sam_model), device) if args.sam_model else None
    player_callback = make_sam_callback(args, sam_model, device)

    cap = cv2.VideoCapture(args.input)
    fps = resolve_video_fps(cap)
    frame_size = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    ball_tracker = MultiBallTracker(max_coast_frames=30, fps=fps, gravity=args.ball_gravity) if (ball_model or unified_model) else None

    os.makedirs(os.path.split(args.output)[0], exist_ok=True)
    out = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, frame_size)

    json_file = None
    if args.json_output:
        os.makedirs(os.path.split(args.json_output)[0], exist_ok=True)
        json_file = open(args.json_output, "w")

    action_json_output = resolve_action_json_output(args.json_output, args.action_json_output)
    action_rows = [] if action_json_output else None

    player_action_db = None
    if args.db_output:
        player_action_db = PlayerActionDatabase(db_path=args.db_output, video_path=args.input, fps=fps, total_frames=total_frames)

    for frame_idx in track(range(total_frames)):
        ret, frame = cap.read()
        if not ret:
            break
        annotated_frame = frame.copy()
        player_dets, action_dets, ball_dets = detect_standard_frame(
            frame,
            args,
            device,
            unified_model,
            player_model,
            action_model,
            ball_model,
            use_separate_models,
        )
        _, frame_players_data = track_players(
            frame,
            player_dets,
            action_dets,
            tracker,
            annotated_frame,
            frame_idx,
            player_callback=player_callback,
        )
        final_ball = track_ball(ball_dets, ball_tracker, annotated_frame) if ball_tracker is not None else None
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
