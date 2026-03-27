"""
volleyball_db — Python interface for the volleyball analysis database.

Usage:
    from db import VolleyballDB
    db = VolleyballDB("dbname=volleyball user=postgres")
    vid = db.insert_video("/data/practice_2026-03-20.mp4", fps=30, width=1920, height=1080)
    rally = db.insert_rally(vid, rally_number=1, start_frame=0, end_frame=450, fps=30)
    ...
"""

import json
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from typing import Optional

import psycopg2
import psycopg2.extras

# Register UUID adapter
psycopg2.extras.register_uuid()


# ── Lightweight data containers ──────────────────────────────

@dataclass
class BBox:
    x: float
    y: float
    w: float
    h: float


@dataclass
class Point:
    x: float
    y: float


# ── Database wrapper ─────────────────────────────────────────

class VolleyballDB:
    def __init__(self, dsn: str):
        """
        dsn: a libpq connection string, e.g.
             "dbname=volleyball user=postgres host=localhost port=5432"
        """
        self.dsn = dsn
        self.conn = psycopg2.connect(dsn)
        self.conn.autocommit = False

    # ---- helpers ----

    @contextmanager
    def _cursor(self, commit: bool = True):
        cur = self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        try:
            yield cur
            if commit:
                self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise
        finally:
            cur.close()

    def close(self):
        self.conn.close()

    # ---- videos ----

    def insert_video(
        self,
        file_path: str,
        filename: Optional[str] = None,
        fps: float = 30.0,
        width: Optional[int] = None,
        height: Optional[int] = None,
        duration_sec: Optional[float] = None,
        metadata: Optional[dict] = None,
    ) -> uuid.UUID:
        filename = filename or file_path.rsplit("/", 1)[-1]
        with self._cursor() as cur:
            cur.execute(
                """
                INSERT INTO videos (file_path, filename, fps, width, height,
                                    duration_sec, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (file_path, filename, fps, width, height, duration_sec,
                 json.dumps(metadata or {})),
            )
            row = cur.fetchone()
            if row is None:
                raise RuntimeError("INSERT into videos returned no row")
            return row["id"]

    # ---- rallies ----

    def insert_rally(
        self,
        video_id: uuid.UUID,
        rally_number: int,
        start_frame: int,
        end_frame: int,
        fps: float = 60.0,
        clip_path: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> uuid.UUID:
        with self._cursor() as cur:
            cur.execute(
                """
                INSERT INTO rallies (video_id, rally_number, start_frame, end_frame,
                                     start_sec, end_sec, clip_path, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    video_id, rally_number, start_frame, end_frame,
                    start_frame / fps, end_frame / fps,
                    clip_path, json.dumps(metadata or {}),
                ),
            )
            row = cur.fetchone()
            if row is None:
                raise RuntimeError("INSERT into rallies returned no row")
            return row["id"]

    # ---- frames (batch) ----

    def insert_frames(
        self,
        rally_id: uuid.UUID,
        start_frame: int,
        end_frame: int,
        fps: float = 60.0,
    ) -> list[uuid.UUID]:
        """Bulk-insert every frame in the rally. Returns list of frame UUIDs."""
        rows = [
            (rally_id, f, f - start_frame, f / fps)
            for f in range(start_frame, end_frame + 1)
        ]
        with self._cursor() as cur:
            psycopg2.extras.execute_values(
                cur,
                """
                INSERT INTO frames (rally_id, frame_number, rel_frame, timestamp_sec)
                VALUES %s
                RETURNING id
                """,
                rows,
            )
            return [r["id"] for r in cur.fetchall()]

    # ---- frame lookup (needed to link detections) ----

    def get_frame_id(self, rally_id: uuid.UUID, frame_number: int) -> Optional[uuid.UUID]:
        with self._cursor(commit=False) as cur:
            cur.execute(
                "SELECT id FROM frames WHERE rally_id = %s AND frame_number = %s",
                (rally_id, frame_number),
            )
            row = cur.fetchone()
            return row["id"] if row else None

    def get_frame_ids_for_rally(self, rally_id: uuid.UUID) -> dict[int, uuid.UUID]:
        """Return {frame_number: frame_uuid} for an entire rally."""
        with self._cursor(commit=False) as cur:
            cur.execute(
                "SELECT frame_number, id FROM frames WHERE rally_id = %s",
                (rally_id,),
            )
            return {r["frame_number"]: r["id"] for r in cur.fetchall()}

    # ---- ball detections (batch) ----

    def insert_ball_detections(
        self,
        rally_id: uuid.UUID,
        detections: list[dict],
        model_run_id: Optional[uuid.UUID] = None,
    ):
        """
        detections: list of dicts with keys
            frame_number, x, y, confidence, radius_px (optional)
        """
        frame_map = self.get_frame_ids_for_rally(rally_id)
        rows = []
        for d in detections:
            fid = frame_map.get(d["frame_number"])
            if fid is None:
                continue
            rows.append((
                fid, rally_id,
                d["x"], d["y"],
                d.get("radius_px"),
                d.get("confidence"),
                d.get("is_interpolated", False),
                model_run_id,
            ))
        with self._cursor() as cur:
            psycopg2.extras.execute_values(
                cur,
                """
                INSERT INTO ball_detections
                    (frame_id, rally_id, x, y, radius_px,
                     confidence, is_interpolated, model_run_id)
                VALUES %s
                """,
                rows,
            )

    # ---- player detections (batch) ----

    def insert_player_detections(
        self,
        rally_id: uuid.UUID,
        detections: list[dict],
        model_run_id: Optional[uuid.UUID] = None,
    ):
        """
        detections: list of dicts with keys
            frame_number, player_label, team, bbox_x, bbox_y, bbox_w, bbox_h,
            confidence, tracker_id (optional), pose_keypoints (optional)
        """
        frame_map = self.get_frame_ids_for_rally(rally_id)
        rows = []
        for d in detections:
            fid = frame_map.get(d["frame_number"])
            if fid is None:
                continue
            rows.append((
                fid, rally_id,
                d.get("player_label"),
                d.get("team", "unknown"),
                d["bbox_x"], d["bbox_y"], d["bbox_w"], d["bbox_h"],
                json.dumps(d.get("pose_keypoints")) if d.get("pose_keypoints") else None,
                d.get("confidence"),
                d.get("tracker_id"),
                model_run_id,
            ))
        with self._cursor() as cur:
            psycopg2.extras.execute_values(
                cur,
                """
                INSERT INTO player_detections
                    (frame_id, rally_id, player_label, team,
                     bbox_x, bbox_y, bbox_w, bbox_h,
                     pose_keypoints, confidence, tracker_id, model_run_id)
                VALUES %s
                """,
                rows,
            )

    # ---- trajectories ----

    def insert_trajectory(
        self,
        rally_id: uuid.UUID,
        segment_index: int,
        start_frame: int,
        end_frame: int,
        fit_params: dict,
        peak_height: Optional[float] = None,
        speed_avg: Optional[float] = None,
        model_run_id: Optional[uuid.UUID] = None,
    ) -> uuid.UUID:
        with self._cursor() as cur:
            cur.execute(
                """
                INSERT INTO ball_trajectories
                    (rally_id, segment_index, start_frame, end_frame,
                     fit_params, peak_height, speed_avg, model_run_id)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    rally_id, segment_index, start_frame, end_frame,
                    json.dumps(fit_params), peak_height, speed_avg, model_run_id,
                ),
            )
            row = cur.fetchone()
            if row is None:
                raise RuntimeError("INSERT into ball_trajectories returned no row")
            return row["id"]

    # ---- play events ----

    def insert_play_event(
        self,
        rally_id: uuid.UUID,
        event_order: int,
        contact: str = "unknown",
        frame_number: Optional[int] = None,
        trajectory_id: Optional[uuid.UUID] = None,
        player_label: Optional[str] = None,
        team: Optional[str] = None,
        ball_x: Optional[float] = None,
        ball_y: Optional[float] = None,
        ball_speed: Optional[float] = None,
        confidence: Optional[float] = None,
        model_run_id: Optional[uuid.UUID] = None,
    ) -> uuid.UUID:
        frame_id = None
        if frame_number is not None:
            frame_id = self.get_frame_id(rally_id, frame_number)
        with self._cursor() as cur:
            cur.execute(
                """
                INSERT INTO play_events
                    (rally_id, frame_id, trajectory_id, event_order, contact,
                     player_label, team, ball_x, ball_y, ball_speed,
                     confidence, model_run_id)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    rally_id, frame_id, trajectory_id, event_order, contact,
                    player_label, team, ball_x, ball_y, ball_speed,
                    confidence, model_run_id,
                ),
            )
            row = cur.fetchone()
            if row is None:
                raise RuntimeError("INSERT into play_events returned no row")
            return row["id"]

    # ---- model runs ----

    def start_model_run(
        self,
        video_id: uuid.UUID,
        model_name: str,
        model_version: Optional[str] = None,
        run_params: Optional[dict] = None,
    ) -> uuid.UUID:
        with self._cursor() as cur:
            cur.execute(
                """
                INSERT INTO model_runs (video_id, model_name, model_version, run_params)
                VALUES (%s, %s, %s, %s)
                RETURNING id
                """,
                (video_id, model_name, model_version,
                 json.dumps(run_params or {})),
            )
            row = cur.fetchone()
            if row is None:
                raise RuntimeError("INSERT into model_runs returned no row")
            return row["id"]

    def finish_model_run(self, run_id: uuid.UUID, status: str = "done"):
        with self._cursor() as cur:
            cur.execute(
                "UPDATE model_runs SET finished_at = now(), status = %s WHERE id = %s",
                (status, run_id),
            )

    # ── Query helpers ────────────────────────────────────────

    def get_ball_path(self, rally_id: uuid.UUID) -> list[dict]:
        """Return ordered ball positions for a rally."""
        with self._cursor(commit=False) as cur:
            cur.execute(
                """
                SELECT frame_number, timestamp_sec, x, y, confidence, is_interpolated
                FROM v_ball_path
                WHERE rally_id = %s
                ORDER BY frame_number
                """,
                (rally_id,),
            )
            return [dict(r) for r in cur.fetchall()]

    def get_play_summary(self, rally_id: uuid.UUID) -> list[dict]:
        """Return play-by-play for a rally."""
        with self._cursor(commit=False) as cur:
            cur.execute(
                """
                SELECT event_order, contact, player_label, team,
                       ball_x, ball_y, ball_speed, timestamp_sec
                FROM v_play_summary
                WHERE rally_id = %s
                ORDER BY event_order
                """,
                (rally_id,),
            )
            return [dict(r) for r in cur.fetchall()]

    def get_player_positions_at_frame(
        self, rally_id: uuid.UUID, frame_number: int
    ) -> list[dict]:
        """All player bounding boxes at a specific frame."""
        with self._cursor(commit=False) as cur:
            cur.execute(
                """
                SELECT pd.player_label, pd.team,
                       pd.bbox_x, pd.bbox_y, pd.bbox_w, pd.bbox_h,
                       pd.tracker_id, pd.confidence
                FROM player_detections pd
                JOIN frames f ON f.id = pd.frame_id
                WHERE pd.rally_id = %s AND f.frame_number = %s
                """,
                (rally_id, frame_number),
            )
            return [dict(r) for r in cur.fetchall()]

    def get_trajectories_for_rally(self, rally_id: uuid.UUID) -> list[dict]:
        """All fitted trajectory segments for a rally."""
        with self._cursor(commit=False) as cur:
            cur.execute(
                """
                SELECT segment_index, start_frame, end_frame,
                       fit_params, peak_height, speed_avg
                FROM ball_trajectories
                WHERE rally_id = %s
                ORDER BY segment_index
                """,
                (rally_id,),
            )
            return [dict(r) for r in cur.fetchall()]