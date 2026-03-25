"""
api.py — FastAPI backend for the volleyball analysis database.
 
Run with:
    uvicorn api:app --reload --port 8000
 
Then open http://localhost:8000/docs for interactive Swagger UI.
 
Architecture:
    Frontend  ──HTTP JSON──▶  FastAPI (this file)  ──SQL──▶  PostgreSQL
                                  │
                                  ▼
                              db.py (VolleyballDB wrapper)
"""
 
import os
import uuid
from contextlib import asynccontextmanager
from typing import Optional
 
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
 
from db import VolleyballDB
from api.schemas import (
    VideoCreate, VideoOut,
    RallyCreate, RallyOut, RallyDetail,
    BallDetectionBatch, BallPathPoint,
    PlayerDetectionBatch, PlayerPositionOut,
    TrajectoryCreate, TrajectoryOut,
    PlayEventCreate, PlayEventOut,
    ModelRunCreate, ModelRunOut,
)
 
# ── Database connection ──────────────────────────────────────
 
DSN = os.getenv(
    "DATABASE_URL",
    "dbname=volleyball user=postgres host=localhost port=5432",
)
 
# "lifespan" controls what happens when the server starts and stops.
# We open one database connection at startup and close it at shutdown.
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.db = VolleyballDB(DSN)
    yield
    app.state.db.close()
 
 
app = FastAPI(
    title="Volleyball Video Analysis API",
    version="0.1.0",
    lifespan=lifespan,
)
 
# CORS: allows your frontend (running on a different port) to call this API.
# In production you'd restrict origins to your actual frontend domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
 
 
def db() -> VolleyballDB:
    """Shortcut to access the shared DB instance."""
    return app.state.db
 
 
# ═════════════════════════════════════════════════════════════
# VIDEOS
# ═════════════════════════════════════════════════════════════
 
@app.post("/videos", response_model=dict, status_code=201)
def create_video(body: VideoCreate):
    """Register a new source video."""
    vid = db().insert_video(
        file_path=body.file_path,
        filename=body.filename,
        fps=body.fps,
        width=body.width,
        height=body.height,
        duration_sec=body.duration_sec,
        metadata=body.metadata,
    )
    return {"id": vid}
 
 
@app.get("/videos", response_model=list[VideoOut])
def list_videos():
    """List all registered videos."""
    with db()._cursor(commit=False) as cur:
        cur.execute(
            "SELECT id, file_path, filename, fps, width, height, "
            "duration_sec, ingested_at, metadata FROM videos ORDER BY ingested_at DESC"
        )
        return [dict(r) for r in cur.fetchall()]
 
 
@app.get("/videos/{video_id}", response_model=VideoOut)
def get_video(video_id: uuid.UUID):
    """Get a single video's metadata."""
    with db()._cursor(commit=False) as cur:
        cur.execute(
            "SELECT id, file_path, filename, fps, width, height, "
            "duration_sec, ingested_at, metadata FROM videos WHERE id = %s",
            (video_id,),
        )
        row = cur.fetchone()
        if not row:
            raise HTTPException(404, "Video not found")
        return dict(row)
 
 
# ═════════════════════════════════════════════════════════════
# RALLIES
# ═════════════════════════════════════════════════════════════
 
@app.post("/videos/{video_id}/rallies", response_model=dict, status_code=201)
def create_rally(video_id: uuid.UUID, body: RallyCreate):
    """Create a rally (clip) within a video."""
    rid = db().insert_rally(
        video_id=video_id,
        rally_number=body.rally_number,
        start_frame=body.start_frame,
        end_frame=body.end_frame,
        fps=body.fps,
        clip_path=body.clip_path,
        metadata=body.metadata,
    )
    # Also materialise frames for this rally
    db().insert_frames(rid, body.start_frame, body.end_frame, fps=body.fps)
    return {"id": rid}
 
 
@app.get("/videos/{video_id}/rallies", response_model=list[RallyOut])
def list_rallies(video_id: uuid.UUID):
    """List all rallies for a video."""
    with db()._cursor(commit=False) as cur:
        cur.execute(
            "SELECT id, video_id, rally_number, start_frame, end_frame, "
            "start_sec, end_sec, clip_path, created_at, metadata "
            "FROM rallies WHERE video_id = %s ORDER BY rally_number",
            (video_id,),
        )
        return [dict(r) for r in cur.fetchall()]
 
 
@app.get("/rallies/{rally_id}", response_model=RallyOut)
def get_rally(rally_id: uuid.UUID):
    """Get a single rally."""
    with db()._cursor(commit=False) as cur:
        cur.execute(
            "SELECT id, video_id, rally_number, start_frame, end_frame, "
            "start_sec, end_sec, clip_path, created_at, metadata "
            "FROM rallies WHERE id = %s",
            (rally_id,),
        )
        row = cur.fetchone()
        if not row:
            raise HTTPException(404, "Rally not found")
        return dict(row)
 
 
@app.get("/rallies/{rally_id}/detail", response_model=RallyDetail)
def get_rally_detail(rally_id: uuid.UUID):
    """
    The big one: returns everything the frontend needs to render a rally
    — rally metadata, ball path, trajectory arcs, and play-by-play.
    """
    # Rally metadata
    with db()._cursor(commit=False) as cur:
        cur.execute(
            "SELECT id, video_id, rally_number, start_frame, end_frame, "
            "start_sec, end_sec, clip_path, created_at, metadata "
            "FROM rallies WHERE id = %s",
            (rally_id,),
        )
        rally_row = cur.fetchone()
        if not rally_row:
            raise HTTPException(404, "Rally not found")
 
    return {
        "rally": dict(rally_row),
        "ball_path": db().get_ball_path(rally_id),
        "trajectories": db().get_trajectories_for_rally(rally_id),
        "play_events": db().get_play_summary(rally_id),
    }
 
 
# ═════════════════════════════════════════════════════════════
# BALL DETECTIONS
# ═════════════════════════════════════════════════════════════
 
@app.post("/rallies/{rally_id}/ball-detections", status_code=201)
def upload_ball_detections(rally_id: uuid.UUID, body: BallDetectionBatch):
    """Batch-upload ball detections for a rally."""
    db().insert_ball_detections(
        rally_id,
        [d.model_dump() for d in body.detections],
        model_run_id=body.model_run_id,
    )
    return {"inserted": len(body.detections)}
 
 
@app.get("/rallies/{rally_id}/ball-path", response_model=list[BallPathPoint])
def get_ball_path(rally_id: uuid.UUID):
    """Ordered ball positions for a rally — ready for trajectory rendering."""
    return db().get_ball_path(rally_id)
 
 
# ═════════════════════════════════════════════════════════════
# PLAYER DETECTIONS
# ═════════════════════════════════════════════════════════════
 
@app.post("/rallies/{rally_id}/player-detections", status_code=201)
def upload_player_detections(rally_id: uuid.UUID, body: PlayerDetectionBatch):
    """Batch-upload player detections for a rally."""
    db().insert_player_detections(
        rally_id,
        [d.model_dump() for d in body.detections],
        model_run_id=body.model_run_id,
    )
    return {"inserted": len(body.detections)}
 
 
@app.get(
    "/rallies/{rally_id}/players-at-frame",
    response_model=list[PlayerPositionOut],
)
def get_players_at_frame(
    rally_id: uuid.UUID,
    frame: int = Query(..., description="Absolute frame number"),
):
    """All player positions at a specific frame — for rendering a court snapshot."""
    return db().get_player_positions_at_frame(rally_id, frame)
 
 
# ═════════════════════════════════════════════════════════════
# TRAJECTORIES
# ═════════════════════════════════════════════════════════════
 
@app.post("/rallies/{rally_id}/trajectories", response_model=dict, status_code=201)
def create_trajectory(rally_id: uuid.UUID, body: TrajectoryCreate):
    """Store a fitted trajectory arc segment."""
    tid = db().insert_trajectory(
        rally_id=rally_id,
        segment_index=body.segment_index,
        start_frame=body.start_frame,
        end_frame=body.end_frame,
        fit_params=body.fit_params,
        peak_height=body.peak_height,
        speed_avg=body.speed_avg,
        model_run_id=body.model_run_id,
    )
    return {"id": tid}
 
 
@app.get("/rallies/{rally_id}/trajectories", response_model=list[TrajectoryOut])
def list_trajectories(rally_id: uuid.UUID):
    """All trajectory segments for a rally."""
    return db().get_trajectories_for_rally(rally_id)
 
 
# ═════════════════════════════════════════════════════════════
# PLAY EVENTS
# ═════════════════════════════════════════════════════════════
 
@app.post("/rallies/{rally_id}/play-events", response_model=dict, status_code=201)
def create_play_event(rally_id: uuid.UUID, body: PlayEventCreate):
    """Record a classified contact event."""
    eid = db().insert_play_event(
        rally_id=rally_id,
        event_order=body.event_order,
        contact=body.contact,
        frame_number=body.frame_number,
        trajectory_id=body.trajectory_id,
        player_label=body.player_label,
        team=body.team,
        ball_x=body.ball_x,
        ball_y=body.ball_y,
        ball_speed=body.ball_speed,
        confidence=body.confidence,
        model_run_id=body.model_run_id,
    )
    return {"id": eid}
 
 
@app.get("/rallies/{rally_id}/play-events", response_model=list[PlayEventOut])
def list_play_events(rally_id: uuid.UUID):
    """Play-by-play for a rally."""
    return db().get_play_summary(rally_id)
 
 
# ═════════════════════════════════════════════════════════════
# MODEL RUNS
# ═════════════════════════════════════════════════════════════
 
@app.post("/videos/{video_id}/model-runs", response_model=dict, status_code=201)
def create_model_run(video_id: uuid.UUID, body: ModelRunCreate):
    """Log the start of an ML model run."""
    rid = db().start_model_run(
        video_id=video_id,
        model_name=body.model_name,
        model_version=body.model_version,
        run_params=body.run_params,
    )
    return {"id": rid}
 
 
@app.patch("/model-runs/{run_id}/finish", response_model=dict)
def finish_model_run(run_id: uuid.UUID, status: str = "done"):
    """Mark a model run as complete."""
    db().finish_model_run(run_id, status=status)
    return {"status": status}
 
 
# ═════════════════════════════════════════════════════════════
# HEALTH CHECK
# ═════════════════════════════════════════════════════════════
 
@app.get("/health")
def health():
    """Quick liveness check."""
    return {"status": "ok"}