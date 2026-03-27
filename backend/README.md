# Volleyball Video Analysis — PostgreSQL Backend

## Quick start

```bash
# 1. Create the database
createdb volleyball

# 2. Apply the schema
psql -d volleyball -f schema.sql

# 3. Install the Python dependency
pip install psycopg2-binary

# 4. Run the example pipeline (uses stub data)
python example_pipeline.py
```

## Architecture

```
Source Video
    │
    ├── Rally Segmentation ──► rallies
    │
    └── Per-rally processing
         ├── Frame extraction ──► frames
         ├── Ball detection ───► ball_detections
         ├── Player detection ─► player_detections
         ├── Trajectory fitting ► ball_trajectories
         └── Play classification ► play_events
```

## Schema overview

| Table | Purpose |
|---|---|
| `videos` | Raw source video metadata |
| `rallies` | Segmented rally clips (start/end frame, path to clip file) |
| `frames` | One row per frame per rally — the join key for all detections |
| `ball_detections` | Ball x/y (normalised 0–1) per frame, with confidence |
| `player_detections` | Player bounding boxes per frame, with tracker ID and team |
| `ball_trajectories` | Fitted arc segments (polynomial coefficients) between contacts |
| `play_events` | Classified contacts: serve, pass, set, attack, block, dig |
| `model_runs` | Provenance — which model version produced which rows |

## Coordinate system

All spatial values are **normalised to [0, 1]** relative to the video frame.
Origin is top-left: `(0, 0)` = top-left, `(1, 1)` = bottom-right.
This keeps the schema resolution-independent.

## Key queries

```sql
-- Ball trajectory for rally 3
SELECT frame_number, x, y, confidence
FROM v_ball_path
WHERE rally_id = (SELECT id FROM rallies WHERE rally_number = 3 LIMIT 1)
ORDER BY frame_number;

-- Play-by-play for a rally
SELECT * FROM v_play_summary
WHERE rally_id = '...';

-- All player positions at a specific frame
SELECT player_label, team, bbox_x, bbox_y, bbox_w, bbox_h
FROM player_detections pd
JOIN frames f ON f.id = pd.frame_id
WHERE pd.rally_id = '...' AND f.frame_number = 150;

-- Average ball speed per rally
SELECT r.rally_number, avg(bt.speed_avg) AS avg_speed
FROM ball_trajectories bt
JOIN rallies r ON r.id = bt.rally_id
GROUP BY r.rally_number
ORDER BY r.rally_number;
```

## Python API (`db.py`)

```python
from db import VolleyballDB

db = VolleyballDB("dbname=volleyball user=postgres")

# Ingest
vid  = db.insert_video("/data/vid.mp4", fps=30, width=1920, height=1080)
rid  = db.insert_rally(vid, rally_number=1, start_frame=0, end_frame=450)
fids = db.insert_frames(rid, 0, 450)
db.insert_ball_detections(rid, [{"frame_number": 10, "x": 0.5, "y": 0.3, "confidence": 0.9}])

# Query
path = db.get_ball_path(rid)           # ordered (frame, x, y, conf)
plays = db.get_play_summary(rid)       # event_order, contact, player, team
players = db.get_player_positions_at_frame(rid, 150)
arcs = db.get_trajectories_for_rally(rid)
```

## Extending

- **Court calibration**: add a `court_calibrations` table mapping pixel coords to real-world metres via a homography matrix. Store the 3×3 matrix as JSONB.
- **Video thumbnails**: add a `thumbnail_path` column to `rallies`.
- **Team rosters**: add a `players` table with jersey number, name, position, and FK from `player_detections.player_label`.
- **Statistics aggregation**: build materialised views for kill %, pass rating, etc.