#!/usr/bin/env python3
"""
example_pipeline.py — Demonstrates the full ingest workflow.

This is a reference script, NOT a runnable ML pipeline.
Replace the stub functions with your actual model inference code.
"""

import json
from db import volleyvision

# ── Stubs: replace these with your actual ML code ────────────

def detect_rallies(video_path: str) -> list[dict]:
    """
    Your rally-segmentation model returns a list like:
    [{"rally_number": 1, "start_frame": 0, "end_frame": 450}, ...]
    """
    return [
        {"rally_number": 1, "start_frame": 0,   "end_frame": 450},
        {"rally_number": 2, "start_frame": 510,  "end_frame": 980},
    ]


def detect_ball(video_path: str, start: int, end: int) -> list[dict]:
    """
    Your ball detector returns per-frame detections:
    [{"frame_number": 0, "x": 0.45, "y": 0.32, "confidence": 0.92}, ...]
    Missing frames = ball occluded (you can interpolate later).
    """
    import random
    return [
        {
            "frame_number": f,
            "x": 0.3 + 0.001 * (f - start),
            "y": 0.5 - 0.0005 * (f - start) ** 0.8,
            "confidence": round(random.uniform(0.75, 0.99), 2),
        }
        for f in range(start, end + 1, 2)   # simulate 50 % detection rate
    ]


def detect_players(video_path: str, start: int, end: int) -> list[dict]:
    """
    Your player detector + tracker returns:
    [{"frame_number": 0, "player_label": "P1", "team": "home",
      "bbox_x": 0.1, "bbox_y": 0.4, "bbox_w": 0.05, "bbox_h": 0.15,
      "tracker_id": 1, "confidence": 0.88}, ...]
    """
    rows = []
    for f in range(start, end + 1, 3):
        for p in range(1, 7):  # 6 players per side
            rows.append({
                "frame_number": f,
                "player_label": f"P{p}",
                "team": "home",
                "bbox_x": 0.05 + p * 0.12,
                "bbox_y": 0.4,
                "bbox_w": 0.05,
                "bbox_h": 0.15,
                "tracker_id": p,
                "confidence": 0.90,
            })
    return rows


def classify_contacts(ball_path: list[dict]) -> list[dict]:
    """
    Your play-classification algorithm identifies contacts:
    [{"event_order": 1, "contact": "serve", "frame_number": 5,
      "player_label": "P3", "team": "home",
      "ball_x": 0.31, "ball_y": 0.49}, ...]
    """
    return [
        {"event_order": 1, "contact": "serve",  "frame_number": 5,
         "player_label": "P3", "team": "home",
         "ball_x": 0.31, "ball_y": 0.49},
        {"event_order": 2, "contact": "pass",   "frame_number": 48,
         "player_label": "P5", "team": "away",
         "ball_x": 0.55, "ball_y": 0.38},
        {"event_order": 3, "contact": "set",    "frame_number": 72,
         "player_label": "P2", "team": "away",
         "ball_x": 0.60, "ball_y": 0.25},
        {"event_order": 4, "contact": "attack", "frame_number": 95,
         "player_label": "P4", "team": "away",
         "ball_x": 0.48, "ball_y": 0.55},
    ]


def fit_trajectory(ball_path: list[dict]) -> list[dict]:
    """
    Your trajectory fitter returns arc segments:
    [{"segment_index": 0, "start_frame": 5, "end_frame": 48,
      "fit_params": {"type": "quadratic", "cx": [...], "cy": [...]},
      "peak_height": 0.22, "speed_avg": 4.5}, ...]
    """
    return [
        {
            "segment_index": 0,
            "start_frame": 5, "end_frame": 48,
            "fit_params": {"type": "quadratic", "cx": [0.001, 0.31], "cy": [-0.0002, 0.49]},
            "peak_height": 0.22,
            "speed_avg": 4.5,
        },
        {
            "segment_index": 1,
            "start_frame": 48, "end_frame": 95,
            "fit_params": {"type": "quadratic", "cx": [0.002, 0.55], "cy": [-0.0003, 0.38]},
            "peak_height": 0.18,
            "speed_avg": 5.1,
        },
    ]


# ── Main pipeline ────────────────────────────────────────────

def main():
    DSN = "dbname=volleyvision user=postgres host=localhost port=5432"
    db = volleyvision(DSN)

    VIDEO_PATH = "/data/practice_2026-03-20.mp4"
    FPS = 30.0

    # 1. Register the source video
    video_id = db.insert_video(
        file_path=VIDEO_PATH,
        fps=FPS, width=1920, height=1080,
        duration_sec=3600.0,
        metadata={"session": "morning", "gym": "Main Court"},
    )
    print(f"Video registered: {video_id}")

    # 2. Log the model run
    run_id = db.start_model_run(
        video_id, model_name="yolov8-volleyball", model_version="1.2.0",
        run_params={"conf_threshold": 0.5, "device": "cuda:0"},
    )

    # 3. Segment into rallies
    rallies_raw = detect_rallies(VIDEO_PATH)
    for r in rallies_raw:
        rally_id = db.insert_rally(
            video_id,
            rally_number=r["rally_number"],
            start_frame=r["start_frame"],
            end_frame=r["end_frame"],
            fps=FPS,
        )
        print(f"  Rally {r['rally_number']}: {rally_id}")

        # 4. Materialise frames
        frame_ids = db.insert_frames(
            rally_id, r["start_frame"], r["end_frame"], fps=FPS,
        )
        print(f"    Frames inserted: {len(frame_ids)}")

        # 5. Ball detections
        ball_dets = detect_ball(VIDEO_PATH, r["start_frame"], r["end_frame"])
        db.insert_ball_detections(rally_id, ball_dets, model_run_id=run_id)
        print(f"    Ball detections: {len(ball_dets)}")

        # 6. Player detections
        player_dets = detect_players(VIDEO_PATH, r["start_frame"], r["end_frame"])
        db.insert_player_detections(rally_id, player_dets, model_run_id=run_id)
        print(f"    Player detections: {len(player_dets)}")

        # 7. Fit trajectories
        ball_path = db.get_ball_path(rally_id)
        traj_segments = fit_trajectory(ball_path)
        for seg in traj_segments:
            tid = db.insert_trajectory(
                rally_id, model_run_id=run_id, **seg,
            )
            print(f"    Trajectory segment {seg['segment_index']}: {tid}")

        # 8. Classify plays
        contacts = classify_contacts(ball_path)
        for c in contacts:
            db.insert_play_event(rally_id, model_run_id=run_id, **c)
        print(f"    Play events: {len(contacts)}")

    db.finish_model_run(run_id, status="done")
    print("\nPipeline complete.")

    # ── Quick query demo ─────────────────────────────────────
    # (re-fetch the first rally for demonstration)
    with db._cursor(commit=False) as cur:
        cur.execute("SELECT id FROM rallies ORDER BY rally_number LIMIT 1")
        first_rally = cur.fetchone()["id"]

    print("\n--- Ball path (first 5 points) ---")
    for pt in db.get_ball_path(first_rally)[:5]:
        print(f"  frame {pt['frame_number']:>4d}  ({pt['x']:.3f}, {pt['y']:.3f})  "
              f"conf={pt['confidence']}  interp={pt['is_interpolated']}")

    print("\n--- Play summary ---")
    for ev in db.get_play_summary(first_rally):
        print(f"  {ev['event_order']}. {ev['contact']:>10s}  by {ev['player_label']}"
              f"  @ ({ev['ball_x']:.2f}, {ev['ball_y']:.2f})")

    db.close()


if __name__ == "__main__":
    main()