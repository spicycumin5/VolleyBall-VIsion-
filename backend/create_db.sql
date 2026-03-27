-- Volleyball Practice Video Analysis — PostgreSQL Schema
-- Stores: source videos, rally clips, per-frame detections
-- (ball + players), ball trajectories, and play classifications.

BEGIN;

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- source videos
CREATE TABLE videos (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    file_path       TEXT        NOT NULL,
    filename        TEXT        NOT NULL,
    duration_sec    REAL,
    fps             REAL        NOT NULL DEFAULT 30.0,
    width           INT,
    height          INT,
    recorded_at     TIMESTAMPTZ,
    ingested_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    metadata        JSONB       DEFAULT '{}'::jsonb
);

COMMENT ON TABLE videos IS 'Raw practice videos before any processing';

-- rallies 
CREATE TABLE rallies (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    video_id        UUID        NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    rally_number    INT         NOT NULL,              -- 1-indexed within the video
    start_frame     INT         NOT NULL,
    end_frame       INT         NOT NULL,
    start_sec       REAL        NOT NULL,
    end_sec         REAL        NOT NULL,
    clip_path       TEXT,                               -- path to exported clip file
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    metadata        JSONB       DEFAULT '{}'::jsonb,    -- e.g. {"score_before": "12-9"}

    UNIQUE (video_id, rally_number),
    CHECK  (end_frame > start_frame)
);

CREATE INDEX idx_rallies_video ON rallies(video_id);

COMMENT ON TABLE rallies IS 'Each rally segmented from the source video';

-- (optional) frames

CREATE TABLE frames (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    rally_id        UUID        NOT NULL REFERENCES rallies(id) ON DELETE CASCADE,
    frame_number    INT         NOT NULL,              -- absolute frame in source video
    rel_frame       INT         NOT NULL,              -- relative frame within rally
    timestamp_sec   REAL        NOT NULL,

    UNIQUE (rally_id, frame_number)
);

CREATE INDEX idx_frames_rally ON frames(rally_id);

-- player detection

CREATE TYPE team_side AS ENUM ('home', 'away', 'unknown');

CREATE TABLE player_detections (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    frame_id        UUID        NOT NULL REFERENCES frames(id) ON DELETE CASCADE,
    rally_id        UUID        NOT NULL REFERENCES rallies(id) ON DELETE CASCADE,
    player_label    TEXT,                               -- tracker-assigned ID, e.g. "P3"
    team            team_side   NOT NULL DEFAULT 'unknown',

    -- bounding box (normalised 0-1 coords)
    bbox_x          REAL NOT NULL,
    bbox_y          REAL NOT NULL,
    bbox_w          REAL NOT NULL,
    bbox_h          REAL NOT NULL,

    -- optional keypoints / pose (store as JSONB array)
    pose_keypoints  JSONB,

    confidence      REAL CHECK (confidence BETWEEN 0 AND 1),
    tracker_id      INT,                                -- re-ID across frames
    metadata        JSONB DEFAULT '{}'::jsonb
);

CREATE INDEX idx_player_det_frame  ON player_detections(frame_id);
CREATE INDEX idx_player_det_rally  ON player_detections(rally_id);
CREATE INDEX idx_player_det_tracker ON player_detections(rally_id, tracker_id);

COMMENT ON TABLE player_detections
    IS 'One row per player per frame. bbox in normalised coords';

-- ball detections

CREATE TABLE ball_detections (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    frame_id        UUID        NOT NULL REFERENCES frames(id) ON DELETE CASCADE,
    rally_id        UUID        NOT NULL REFERENCES rallies(id) ON DELETE CASCADE,

    -- centre of ball (normalised 0-1)
    x               REAL NOT NULL,
    y               REAL NOT NULL,
    radius_px       REAL,                               -- in original pixel space

    confidence      REAL CHECK (confidence BETWEEN 0 AND 1),
    is_interpolated BOOLEAN NOT NULL DEFAULT FALSE,     -- filled by trajectory smoother
    metadata        JSONB DEFAULT '{}'::jsonb
);

CREATE INDEX idx_ball_det_frame ON ball_detections(frame_id);
CREATE INDEX idx_ball_det_rally ON ball_detections(rally_id);

-- trajectories of ball

CREATE TABLE ball_trajectories (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    rally_id        UUID        NOT NULL REFERENCES rallies(id) ON DELETE CASCADE,
    segment_index   INT         NOT NULL,               -- sub-arc within rally (reset on contact)
    start_frame     INT         NOT NULL,
    end_frame       INT         NOT NULL,

    -- polynomial or spline coefficients (JSON)
    fit_params      JSONB       NOT NULL,               -- e.g. {"type":"quadratic","cx":[…],"cy":[…]}
    peak_height     REAL,
    speed_avg       REAL,                               -- px/frame or m/s if calibrated
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),

    UNIQUE (rally_id, segment_index)
);

CREATE INDEX idx_traj_rally ON ball_trajectories(rally_id);

-- play events

CREATE TYPE contact_type AS ENUM (
    'serve', 'pass', 'set', 'attack', 'block',
    'dig', 'free_ball', 'unknown'
);

CREATE TABLE play_events (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    rally_id        UUID          NOT NULL REFERENCES rallies(id) ON DELETE CASCADE,
    frame_id        UUID          REFERENCES frames(id),
    trajectory_id   UUID          REFERENCES ball_trajectories(id),
    event_order     INT           NOT NULL, -- 1-indexed within rally
    contact         contact_type  NOT NULL DEFAULT 'unknown',
    player_label    TEXT, -- which player made the contact
    team            team_side,

    -- ball state at contact
    ball_x          REAL,
    ball_y          REAL,
    ball_speed      REAL,

    confidence      REAL CHECK (confidence BETWEEN 0 AND 1),
    metadata        JSONB DEFAULT '{}'::jsonb,

    UNIQUE (rally_id, event_order)
);

CREATE INDEX idx_events_rally ON play_events(rally_id);

-- model runs

CREATE TABLE model_runs (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    video_id        UUID        REFERENCES videos(id),
    model_name      TEXT        NOT NULL,               -- e.g. "yolov8-volleyball-det"
    model_version   TEXT,
    run_params      JSONB       DEFAULT '{}'::jsonb,
    started_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at     TIMESTAMPTZ,
    status          TEXT        NOT NULL DEFAULT 'running'
);

-- tag every detection / event row back to its model run
ALTER TABLE player_detections ADD COLUMN model_run_id UUID REFERENCES model_runs(id);
ALTER TABLE ball_detections   ADD COLUMN model_run_id UUID REFERENCES model_runs(id);
ALTER TABLE ball_trajectories ADD COLUMN model_run_id UUID REFERENCES model_runs(id);
ALTER TABLE play_events       ADD COLUMN model_run_id UUID REFERENCES model_runs(id);

-- useful views

-- full ball path for a rally (ordered)
CREATE VIEW v_ball_path AS
SELECT
    bd.rally_id,
    f.frame_number,
    f.timestamp_sec,
    bd.x,
    bd.y,
    bd.confidence,
    bd.is_interpolated
FROM ball_detections bd
JOIN frames f ON f.id = bd.frame_id
ORDER BY bd.rally_id, f.frame_number;

-- Play-by-play summary
CREATE VIEW v_play_summary AS
SELECT
    pe.rally_id,
    pe.event_order,
    pe.contact,
    pe.player_label,
    pe.team,
    pe.ball_x,
    pe.ball_y,
    pe.ball_speed,
    f.timestamp_sec
FROM play_events pe
LEFT JOIN frames f ON f.id = pe.frame_id
ORDER BY pe.rally_id, pe.event_order;

COMMIT;