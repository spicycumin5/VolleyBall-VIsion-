"""Pose tracker experiment built on the regular pipeline."""

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
from ultralytics import YOLO

from ball_tracker import MultiBallTracker
from player_action_db import PlayerActionDatabase
from tracking_shared import (
    collect_action_rows,
    configure_tracking_memory,
    configure_runtime_device,
    create_strongsort,
    load_yolo_models,
    move_model_to_device,
    predict_detections,
    render_video_from_json,
    rewrite_json_outputs,
    resolve_json_output_path,
    resolve_video_output_path,
    resolve_action_json_output,
    resolve_conf,
    resolve_device,
    resolve_imgsz,
    resolve_video_fps,
    track_ball,
    track_players,
    warn_if_dino_requested,
    write_action_table,
    write_json_frame,
    bbox_iou,
)


KEYPOINT_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]

parser = argparse.ArgumentParser()
parser.add_argument("--model", default=None, help="Optional pose model path. Defaults to yolo26x-pose when omitted")
parser.add_argument("--player_model", default="yolo26x-pose", help="Path or model name for the player pose detector")
parser.add_argument("--action_model", default=None, help="Path to action detector weights")
parser.add_argument("--ball_model", default=None, help="Path to ball detector weights")
parser.add_argument("--reid_weights", default="osnet_ain_x1_0_msmt17.pt", help="ReID weights for StrongSORT")
parser.add_argument("--input", required=True, help="Input video path")
parser.add_argument("--output", default=None, help="Output video path")
parser.add_argument("--show_conf", default=False, action="store_true", help="Whether to show the confidence scores")
parser.add_argument("--conf", "--confs", dest="conf", type=float, default=0.2, help="Default confidence threshold for player, action, and ball detection")
parser.add_argument("--player_conf", type=float, default=None, help="Confidence threshold override for player detection")
parser.add_argument("--action_conf", type=float, default=None, help="Confidence threshold override for action detection")
parser.add_argument("--ball_conf", type=float, default=0.8, help="Confidence threshold override for ball detection")
parser.add_argument("--imgsz", type=int, default=1920, help="Image size for YOLO. 640, 1280, and 1920 are good")
parser.add_argument("--player_imgsz", type=int, default=None, help="Optional inference size override for player model")
parser.add_argument("--action_imgsz", type=int, default=None, help="Optional inference size override for action model")
parser.add_argument("--ball_imgsz", type=int, default=None, help="Optional inference size override for ball model")
parser.add_argument("--ball_gravity", type=float, default=600.0, help="Ball-tracker gravity in pixels/sec^2")
parser.add_argument("--ball_blacklist", type=int, default=5, help="Blacklist a ball location after this many stagnant frames")
parser.add_argument("--ball_memory_frames", type=int, default=120, help="Live ball tracker coast length in frames")
parser.add_argument("--ball_gap_fill_frames", type=int, default=480, help="Maximum post-process ball gap fill length in frames")
parser.add_argument("--cubic", action="store_true", default=False, help="Use cubic interpolation instead of physics for post-process ball gap filling")
parser.add_argument("--memory_frames", type=int, default=360, help="Canonical ID memory window in frames")
parser.add_argument("--heatmap_conf", type=int, default=0.5, help="Confidence for the ball tracker")
parser.add_argument("--heatmap_alpha", type=float, default=0.0, help="Alpha for heatmap overlay")
parser.add_argument("--json_output", default=None, help="Path to output json")
parser.add_argument("--action_json_output", default=None, help="Path to output player action table json")
parser.add_argument("--db_output", default=None, help="Path to output sqlite database")
parser.add_argument("--dino", action="store_true", default=False, help="Deprecated and ignored. Player tracking always uses BoxMOT")
parser.add_argument("--keypoint_conf", type=float, default=0.35, help="Minimum pose keypoint confidence to draw")
parser.add_argument("--no-mp4", action="store_true", default=False, help="Skip writing the MP4 output")
parser.add_argument("--device", default="auto", help="Device to use (for example: auto, cpu, 0, or 0,1)")


def pose_keypoints_to_serializable(points, confs):
    serialized = []
    for idx, point in enumerate(points):
        score = None if confs is None else float(confs[idx])
        serialized.append({"id": int(idx), "x": float(point[0]), "y": float(point[1]), "conf": score})
    return serialized


def visible_pose_keypoints(points, confs, threshold):
    visible = {}
    for idx, point in enumerate(points):
        score = 1.0 if confs is None else float(confs[idx])
        if score < threshold:
            continue
        visible[idx] = (int(point[0]), int(point[1]))
    return visible


def draw_pose_skeleton(annotated_frame, keypoints, keypoint_confs, threshold):
    visible = visible_pose_keypoints(keypoints, keypoint_confs, threshold)
    for start_idx, end_idx in KEYPOINT_CONNECTIONS:
        if start_idx in visible and end_idx in visible:
            cv2.line(annotated_frame, visible[start_idx], visible[end_idx], (255, 255, 0), 2)
    for point in visible.values():
        cv2.circle(annotated_frame, point, 3, (255, 255, 0), -1)


def match_pose_to_track(box, pose_detections, used_pose_indices):
    best_idx = None
    best_score = (-1.0, -1.0)
    for idx, pose_det in enumerate(pose_detections):
        if idx in used_pose_indices:
            continue
        pose_box = tuple(map(int, pose_det["box"]))
        score = (bbox_iou(box, pose_box), float(pose_det["conf"]))
        if score[0] <= 0:
            continue
        if score > best_score:
            best_score = score
            best_idx = idx
    return best_idx


def predict_pose_detections(model, frame, conf, imgsz, device):
    results = model.predict(frame, conf=conf, classes=[0], verbose=False, imgsz=imgsz, iou=0.95, device=device)
    result = results[0]
    detections = result.boxes.data.cpu().numpy() if result.boxes is not None else np.empty((0, 6))
    pose_detections = []
    if result.boxes is None or result.keypoints is None:
        return detections, pose_detections
    boxes_xyxy = result.boxes.xyxy.detach().cpu().numpy().astype(int)
    confs = result.boxes.conf.detach().cpu().numpy()
    classes = result.boxes.cls.detach().cpu().numpy().astype(int)
    keypoints_xy = result.keypoints.xy.detach().cpu().numpy()
    keypoints_conf = result.keypoints.conf.detach().cpu().numpy() if result.keypoints.conf is not None else None
    for idx, class_id in enumerate(classes):
        if class_id != 0:
            continue
        pose_detections.append(
            {
                "box": boxes_xyxy[idx].tolist(),
                "conf": float(confs[idx]),
                "keypoints_xy": keypoints_xy[idx],
                "keypoints_conf": None if keypoints_conf is None else keypoints_conf[idx],
            }
        )
    return detections, pose_detections


def make_pose_callback(args, pose_detections):
    used_pose_indices = set()

    def callback(annotated_frame, track_box, **_kwargs):
        matched_pose_idx = match_pose_to_track(track_box, pose_detections, used_pose_indices)
        if matched_pose_idx is None:
            return {"keypoints": []}
        used_pose_indices.add(matched_pose_idx)
        pose_det = pose_detections[matched_pose_idx]
        draw_pose_skeleton(annotated_frame, pose_det["keypoints_xy"], pose_det["keypoints_conf"], args.keypoint_conf)
        return {"keypoints": pose_keypoints_to_serializable(pose_det["keypoints_xy"], pose_det["keypoints_conf"])}

    return callback


def main():
    args = parser.parse_args()
    warn_if_dino_requested(args.dino)
    device = resolve_device(args.device, MASKED_DEVICE)
    configure_runtime_device(device)

    tracker = create_strongsort(device, reid_weights=args.reid_weights)
    pose_model_name = args.player_model or args.model or "yolo26x-pose"
    player_model = move_model_to_device(YOLO(pose_model_name), device)
    _, _, action_model, ball_model, _ = load_yolo_models(
        device,
        action_model=args.action_model,
        ball_model=args.ball_model,
        require_player_model=False,
    )

    cap = cv2.VideoCapture(args.input)
    fps = resolve_video_fps(cap)
    configure_tracking_memory(args.memory_frames)
    frame_size = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    ball_tracker = MultiBallTracker(
        max_coast_frames=max(int(args.ball_memory_frames), 1),
        fps=fps,
        gravity=args.ball_gravity,
        stagnant_frame_limit=max(int(args.ball_blacklist), 1),
    ) if ball_model is not None else None

    output_path = resolve_video_output_path(args.input, args.output, args.no_mp4)
    json_output_path = resolve_json_output_path(output_path, args.json_output, args.input)

    json_file = None
    if json_output_path:
        json_parent = os.path.split(json_output_path)[0]
        if json_parent:
            os.makedirs(json_parent, exist_ok=True)
        json_file = open(json_output_path, "w")

    action_json_output = resolve_action_json_output(json_output_path, args.action_json_output)
    action_rows = [] if action_json_output else None
    ball_track_snapshots = []

    player_action_db = None
    if args.db_output:
        player_action_db = PlayerActionDatabase(db_path=args.db_output, video_path=args.input, fps=fps, total_frames=total_frames)

    player_conf = resolve_conf(args.player_conf, args.conf)
    action_conf = resolve_conf(args.action_conf, args.conf)
    ball_conf = resolve_conf(args.ball_conf, args.conf)

    for frame_idx in track(range(total_frames), description="Inference..."):
        ret, frame = cap.read()
        if not ret:
            break
        annotated_frame = frame.copy()
        player_dets, pose_detections = predict_pose_detections(
            player_model,
            frame,
            conf=player_conf,
            imgsz=resolve_imgsz(args.player_imgsz, args.imgsz),
            device=device,
        )
        action_dets = predict_detections(
            action_model,
            frame,
            conf=action_conf,
            imgsz=resolve_imgsz(args.action_imgsz, args.imgsz),
            device=device,
        )
        ball_dets = predict_detections(
            ball_model,
            frame,
            conf=ball_conf,
            imgsz=resolve_imgsz(args.ball_imgsz, args.imgsz),
            device=device,
        ) if ball_model is not None else np.empty((0, 6))
        _, frame_players_data = track_players(
            frame,
            player_dets,
            action_dets,
            tracker,
            annotated_frame,
            frame_idx,
            player_callback=make_pose_callback(args, pose_detections),
        )
        if ball_tracker is not None:
            ball_result = track_ball(ball_dets, ball_tracker, annotated_frame, return_track_snapshot=True)
            final_ball, track_snapshot = ball_result
            ball_track_snapshots.append(track_snapshot)
        else:
            final_ball = None
        write_json_frame(json_file, frame_idx, final_ball, frame_players_data)
        collect_action_rows(action_rows, frame_idx, frame_players_data)
        if player_action_db is not None:
            player_action_db.add_frame_players(frame_idx, frame_players_data)

    cap.release()
    if json_file:
        json_file.close()
    write_action_table(action_json_output, action_rows)
    if player_action_db is not None:
        player_action_db.close()
    rewrite_json_outputs(
        json_output_path,
        action_json_output,
        fps=fps,
        ball_gravity=args.ball_gravity,
        ball_gap_fill_frames=args.ball_gap_fill_frames,
        ball_interpolation_mode="cubic" if args.cubic else "physics",
        ball_blacklist_frames=args.ball_blacklist,
        ball_track_snapshots=ball_track_snapshots,
    )
    render_video_from_json(args.input, json_output_path, output_path, fps, frame_size)


if __name__ == "__main__":
    main()
