"""
schemas.py — Pydantic models defining the shape of every request and response.

Think of these as the "contract" between your frontend and API:
the frontend sends JSON matching a request model, and the API
always responds with JSON matching a response model.

Pydantic validates automatically — if the frontend sends a string
where a number is expected, the request is rejected with a clear
error before your code ever runs.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


# ── Videos ───────────────────────────────────────────────────

class VideoCreate(BaseModel):
    file_path: str
    filename: Optional[str] = None
    fps: float = 30.0
    width: Optional[int] = None
    height: Optional[int] = None
    duration_sec: Optional[float] = None
    metadata: Optional[dict[str, Any]] = None


class VideoOut(BaseModel):
    id: uuid.UUID
    file_path: str
    filename: str
    fps: float
    width: Optional[int]
    height: Optional[int]
    duration_sec: Optional[float]
    ingested_at: datetime
    metadata: Optional[dict[str, Any]]


# ── Rallies ──────────────────────────────────────────────────

class RallyCreate(BaseModel):
    rally_number: int
    start_frame: int
    end_frame: int
    fps: float = 30.0
    clip_path: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None


class RallyOut(BaseModel):
    id: uuid.UUID
    video_id: uuid.UUID
    rally_number: int
    start_frame: int
    end_frame: int
    start_sec: float
    end_sec: float
    clip_path: Optional[str]
    created_at: datetime
    metadata: Optional[dict[str, Any]]


# ── Detections (ball) ────────────────────────────────────────

class BallDetection(BaseModel):
    frame_number: int
    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)
    confidence: Optional[float] = Field(default=None, ge=0, le=1)
    radius_px: Optional[float] = None
    is_interpolated: bool = False


class BallDetectionBatch(BaseModel):
    detections: list[BallDetection]
    model_run_id: Optional[uuid.UUID] = None


class BallPathPoint(BaseModel):
    frame_number: int
    timestamp_sec: float
    x: float
    y: float
    confidence: Optional[float]
    is_interpolated: bool


# ── Detections (players) ─────────────────────────────────────

class PlayerDetection(BaseModel):
    frame_number: int
    player_label: Optional[str] = None
    team: str = "unknown"
    bbox_x: float = Field(ge=0, le=1)
    bbox_y: float = Field(ge=0, le=1)
    bbox_w: float = Field(ge=0, le=1)
    bbox_h: float = Field(ge=0, le=1)
    confidence: Optional[float] = Field(default=None, ge=0, le=1)
    tracker_id: Optional[int] = None
    pose_keypoints: Optional[list] = None


class PlayerDetectionBatch(BaseModel):
    detections: list[PlayerDetection]
    model_run_id: Optional[uuid.UUID] = None


class PlayerPositionOut(BaseModel):
    player_label: Optional[str]
    team: str
    bbox_x: float
    bbox_y: float
    bbox_w: float
    bbox_h: float
    tracker_id: Optional[int]
    confidence: Optional[float]


# ── Trajectories ─────────────────────────────────────────────

class TrajectoryCreate(BaseModel):
    segment_index: int
    start_frame: int
    end_frame: int
    fit_params: dict[str, Any]
    peak_height: Optional[float] = None
    speed_avg: Optional[float] = None
    model_run_id: Optional[uuid.UUID] = None


class TrajectoryOut(BaseModel):
    segment_index: int
    start_frame: int
    end_frame: int
    fit_params: dict[str, Any]
    peak_height: Optional[float]
    speed_avg: Optional[float]


# ── Play events ──────────────────────────────────────────────

class PlayEventCreate(BaseModel):
    event_order: int
    contact: str = "unknown"
    frame_number: Optional[int] = None
    trajectory_id: Optional[uuid.UUID] = None
    player_label: Optional[str] = None
    team: Optional[str] = None
    ball_x: Optional[float] = None
    ball_y: Optional[float] = None
    ball_speed: Optional[float] = None
    confidence: Optional[float] = Field(default=None, ge=0, le=1)
    model_run_id: Optional[uuid.UUID] = None


class PlayEventOut(BaseModel):
    event_order: int
    contact: str
    player_label: Optional[str]
    team: Optional[str]
    ball_x: Optional[float]
    ball_y: Optional[float]
    ball_speed: Optional[float]
    timestamp_sec: Optional[float]


# ── Model runs ───────────────────────────────────────────────

class ModelRunCreate(BaseModel):
    model_name: str
    model_version: Optional[str] = None
    run_params: Optional[dict[str, Any]] = None


class ModelRunOut(BaseModel):
    id: uuid.UUID
    model_name: str
    status: str


# ── Composite: full rally data for frontend rendering ────────

class RallyDetail(BaseModel):
    """Everything the frontend needs to render one rally."""
    rally: RallyOut
    ball_path: list[BallPathPoint]
    trajectories: list[TrajectoryOut]
    play_events: list[PlayEventOut]