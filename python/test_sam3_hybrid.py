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
    detection_state_name,
    collect_action_rows,
    configure_tracking_memory,
    configure_runtime_device,
    create_strongsort,
    detect_standard_frame,
    load_yolo_models,
    move_model_to_device,
    render_video_from_json,
    rewrite_json_outputs,
    resolve_json_output_path,
    resolve_video_output_path,
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
parser.add_argument("--sam_model", default="sam3.pt", help="Path to SAM 3 weights used to refine tracked player masks")
parser.add_argument("--sam_imgsz", type=int, default=None, help="Optional inference size override for SAM; defaults to --imgsz")
parser.add_argument("--sam_crop_pad", type=float, default=0.2, help="Extra padding ratio around each player box before SAM")
parser.add_argument("--sam_alpha", type=float, default=0.2, help="Mask overlay alpha for SAM 3 player masks")
parser.add_argument("--sam_min_area", type=int, default=200, help="Minimum SAM mask area in pixels before drawing")
parser.add_argument("--no-mp4", action="store_true", default=False, help="Skip writing the MP4 output")
parser.add_argument("--device", default="auto", help="Device to use (for example: auto, cpu, 0, or 0,1)")


def crop_frame_for_box(frame, box, pad_ratio):
    frame_h, frame_w = frame.shape[:2]
    x1, y1, x2, y2 = [int(round(float(value))) for value in box]
    width = max(x2 - x1, 1)
    height = max(y2 - y1, 1)
    pad_x = int(round(width * max(float(pad_ratio), 0.0)))
    pad_y = int(round(height * max(float(pad_ratio), 0.0)))
    crop_x1 = max(0, x1 - pad_x)
    crop_y1 = max(0, y1 - pad_y)
    crop_x2 = min(frame_w, x2 + pad_x)
    crop_y2 = min(frame_h, y2 + pad_y)
    if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
        return None, None, None
    cropped_frame = frame[crop_y1:crop_y2, crop_x1:crop_x2]
    local_box = [x1 - crop_x1, y1 - crop_y1, x2 - crop_x1, y2 - crop_y1]
    return cropped_frame, local_box, (crop_x1, crop_y1, crop_x2, crop_y2)


def remap_local_mask_to_frame(local_mask, frame_shape, crop_bounds):
    crop_x1, crop_y1, crop_x2, crop_y2 = crop_bounds
    full_mask = np.zeros(frame_shape[:2], dtype=bool)
    full_mask[crop_y1:crop_y2, crop_x1:crop_x2] = local_mask[: crop_y2 - crop_y1, : crop_x2 - crop_x1]
    return full_mask


def classify_action_from_mask(frame, full_mask, box, action_model, device, args):
    if action_model is None or full_mask is None:
        return None, None
    x1, y1, x2, y2 = [int(round(float(value))) for value in box]
    frame_h, frame_w = frame.shape[:2]
    x1 = max(0, min(frame_w - 1, x1))
    y1 = max(0, min(frame_h - 1, y1))
    x2 = max(x1 + 1, min(frame_w, x2))
    y2 = max(y1 + 1, min(frame_h, y2))
    crop = frame[y1:y2, x1:x2].copy()
    crop_mask = full_mask[y1:y2, x1:x2]
    if crop.size == 0 or crop_mask.size == 0 or not np.any(crop_mask):
        return None, None
    crop[~crop_mask] = 0
    results = action_model.predict(
        crop,
        conf=args.action_conf if args.action_conf is not None else args.conf,
        verbose=False,
        imgsz=args.action_imgsz if args.action_imgsz is not None else args.imgsz,
        iou=0.95,
        device=device,
    )
    if not results:
        return None, None
    dets = results[0].boxes.data.cpu().numpy()
    if len(dets) == 0:
        return None, None
    best_det = dets[np.argmax(dets[:, 4])]
    return detection_state_name(best_det[5]), float(best_det[4])


def pack_crops_on_canvas(crop_entries):
    if not crop_entries:
        return None, None, None

    total_area = sum(crop.shape[0] * crop.shape[1] for _, crop, _, _ in crop_entries)
    max_crop_width = max(crop.shape[1] for _, crop, _, _ in crop_entries)
    target_row_width = max(max_crop_width, int(np.ceil(np.sqrt(max(total_area, 1)))))

    rows = []
    current_row = []
    current_width = 0
    current_height = 0
    for entry in crop_entries:
        _, crop, _, _ = entry
        crop_h, crop_w = crop.shape[:2]
        if current_row and current_width + crop_w > target_row_width:
            rows.append((current_row, current_width, current_height))
            current_row = []
            current_width = 0
            current_height = 0
        current_row.append(entry)
        current_width += crop_w
        current_height = max(current_height, crop_h)
    if current_row:
        rows.append((current_row, current_width, current_height))

    canvas_width = max(row_width for _, row_width, _ in rows)
    canvas_height = sum(row_height for _, _, row_height in rows)
    first_crop = crop_entries[0][1]
    canvas = np.zeros((canvas_height, canvas_width, first_crop.shape[2]), dtype=first_crop.dtype)
    prompt_boxes = []
    prompt_meta = []

    cursor_y = 0
    for row_entries, _, row_height in rows:
        cursor_x = 0
        for player_idx, crop, local_box, crop_bounds in row_entries:
            crop_h, crop_w = crop.shape[:2]
            canvas[cursor_y:cursor_y + crop_h, cursor_x:cursor_x + crop_w] = crop
            atlas_box = [
                float(local_box[0] + cursor_x),
                float(local_box[1] + cursor_y),
                float(local_box[2] + cursor_x),
                float(local_box[3] + cursor_y),
            ]
            atlas_bounds = (cursor_x, cursor_y, cursor_x + crop_w, cursor_y + crop_h)
            prompt_boxes.append(atlas_box)
            prompt_meta.append((player_idx, crop_bounds, atlas_bounds))
            cursor_x += crop_w
        cursor_y += row_height

    return canvas, prompt_boxes, prompt_meta


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


def batch_segment_players_with_sam(frame, players, sam_model, action_model, device, args):
    if sam_model is None or not players:
        return players

    sam_imgsz = args.sam_imgsz if args.sam_imgsz is not None else args.imgsz
    sam_crop_pad = args.sam_crop_pad
    crop_entries = []

    for idx, player in enumerate(players):
        box = player.get("box")
        if not box or len(box) != 4:
            continue
        cropped_frame, local_box, crop_bounds = crop_frame_for_box(frame, box, sam_crop_pad)
        if cropped_frame is None or local_box is None or crop_bounds is None:
            continue
        crop_entries.append((idx, cropped_frame, local_box, crop_bounds))

    if not crop_entries:
        return players

    canvas, prompt_boxes, prompt_meta = pack_crops_on_canvas(crop_entries)
    if canvas is None or prompt_boxes is None or prompt_meta is None or not prompt_boxes:
        return players

    results = sam_model.predict(
        source=canvas,
        bboxes=prompt_boxes,
        verbose=False,
        device=device,
        imgsz=sam_imgsz,
    )
    if not results:
        return players

    result = results[0]
    if result.masks is None or len(result.masks.data) == 0:
        return players
    masks_data = list(result.masks.data)

    for mask_tensor, (player_idx, crop_bounds, atlas_bounds) in zip(masks_data, prompt_meta):
        atlas_mask = mask_tensor.detach().cpu().numpy() > 0.5
        atlas_x1, atlas_y1, atlas_x2, atlas_y2 = atlas_bounds
        local_mask = atlas_mask[atlas_y1:atlas_y2, atlas_x1:atlas_x2]
        full_mask = remap_local_mask_to_frame(local_mask, frame.shape, crop_bounds)
        if int(full_mask.astype(np.uint8).sum()) < int(args.sam_min_area):
            continue
        players[player_idx]["mask_polygons"] = [
            contour for contour in mask_to_polygons(full_mask) if len(contour) >= 3
        ]
        action_state, action_conf = classify_action_from_mask(
            frame,
            full_mask,
            players[player_idx].get("box"),
            action_model,
            device,
            args,
        )
        if action_state is not None:
            players[player_idx]["state"] = action_state
            players[player_idx]["state_conf"] = action_conf

    return players


def make_sam_callback(*_args, **_kwargs):
    return None


def main():
    args = parser.parse_args()
    warn_if_dino_requested(args.dino)
    device = resolve_device(args.device, MASKED_DEVICE)
    configure_runtime_device(device)

    tracker = create_strongsort(device, reid_weights=args.reid_weights)
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
    configure_tracking_memory(args.memory_frames)
    frame_size = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    ball_tracker = MultiBallTracker(
        max_coast_frames=max(int(args.ball_memory_frames), 1),
        fps=fps,
        gravity=args.ball_gravity,
        stagnant_frame_limit=max(int(args.ball_blacklist), 1),
    ) if (ball_model or unified_model) else None

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

    for frame_idx in track(range(total_frames), description="Inference..."):
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
        action_dets = np.empty((0, 6))
        _, frame_players_data = track_players(
            frame,
            player_dets,
            action_dets,
            tracker,
            annotated_frame,
            frame_idx,
            player_callback=player_callback,
        )
        frame_players_data = batch_segment_players_with_sam(frame, frame_players_data, sam_model, action_model, device, args)
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
