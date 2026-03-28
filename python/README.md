I asked Chat to make this.

# Python CLI

This directory contains the Python video-analysis pipeline used to detect players, classify player actions, track the volleyball, and write machine-readable outputs for later use.

## What lives here

- `python/test.py` is the main CLI for the standard detection and tracking pipeline.
- `python/test_sam3_hybrid.py` runs the same pipeline but adds SAM-based player masks and mask-driven action classification.
- `python/test_yolo26x_pose.py` runs the same pipeline but adds pose keypoints for tracked players.
- `python/yolo_tune.py` prepares task-specific YOLO datasets and trains separate player, action, and ball models.
- `python/player_action_db.py` writes optional SQLite output for downstream querying.

## Setup

Run these commands from the project root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r python/requirements.txt
```

You will also need the model weights referenced by your command, such as YOLO checkpoints and the StrongSORT ReID weights.

## Main CLI usage

The standard pipeline is:

```bash
python python/test.py --input path/to/video.mp4 --player_model yolov8m_players.pt --action_model yolov8m_actions.pt --ball_model yolo26x.pt
```

Typical options:

- `--input` required input video.
- `--output` optional annotated MP4 path. If omitted, the CLI writes `<input>_tracked.mp4` next to the input video.
- `--json_output` optional frame-JSON path. If omitted, the CLI writes a JSON file beside the output MP4 or input video.
- `--action_json_output` optional path for a flattened action table.
- `--db_output` optional SQLite database path.
- `--device` runtime target such as `auto`, `cpu`, or `0`.
- `--conf`, `--player_conf`, `--action_conf`, `--ball_conf` confidence thresholds.
- `--imgsz`, `--player_imgsz`, `--action_imgsz`, `--ball_imgsz` inference sizes.
- `--ball_blacklist` blacklist stationary false-ball locations after this many frames; default `5`.
- `--ball_memory_frames` live ball tracker coast window in frames; default `120`.
- `--ball_gap_fill_frames` maximum post-process interpolation gap in frames; default `480`.
- `--memory_frames` player/canonical-ID memory window in frames; default `360`.
- `--cubic` switch post-process ball interpolation from physics mode to cubic mode.
- `--no-mp4` skip video writing and only emit structured outputs.

Example with explicit outputs:

```bash
python python/test.py \
  --input data/example.mp4 \
  --player_model yolov8m_players.pt \
  --action_model yolov8m_actions.pt \
  --ball_model yolo26x.pt \
  --output outputs/example_tracked.mp4 \
  --json_output outputs/example_tracked.json \
  --action_json_output outputs/example_actions.json \
  --db_output outputs/example.sqlite \
  --device 0
```

## Variant CLIs

Use the same project-root invocation pattern for the experimental variants.

### SAM hybrid

Adds mask polygons for tracked players and runs action recognition on SAM-masked player crops.

```bash
python python/test_sam3_hybrid.py \
  --input data/example.mp4 \
  --player_model yolov8m_players.pt \
  --action_model yolov8m_actions.pt \
  --ball_model yolo26x.pt \
  --sam_model sam3.pt
```

Extra options include `--sam_model`, `--sam_imgsz`, `--sam_crop_pad`, `--sam_alpha`, and `--sam_min_area`.

### Pose hybrid

Adds per-player pose keypoints.

```bash
python python/test_yolo26x_pose.py \
  --input data/example.mp4 \
  --player_model yolo26x-pose \
  --action_model yolov8m_actions.pt \
  --ball_model yolo26x.pt
```

Extra options include `--keypoint_conf` and the pose-specific player model.

## JSON output

The main structured output is a standard JSON array. Each element is one frame record written during inference and finalized during the rewrite pass in `python/tracking_shared.py`.

One element looks like this:

```json
[
  {
    "frame": 42,
    "ball": {
      "tid": 1,
      "box": [812, 322, 828, 338],
      "center": [820, 330],
      "conf": 0.91,
      "predicted": false
    },
    "players": [
      {
        "tid": 3,
        "box": [100, 240, 180, 470],
        "conf": 0.88,
        "state": "set",
        "state_conf": 0.74
      }
    ]
  }
]
```

Field notes:

- `frame` is the zero-based frame index.
- `ball` is either `null` or the selected primary ball track for that frame.
- `ball.predicted` is `true` when the post-process rewrite filled the gap using the selected interpolation mode.
- `players` contains one object per tracked player visible in that frame.
- `tid` is the canonical tracked player ID, not just the raw tracker ID for that frame.
- `state` is one of `player`, `block`, `defense`, `serve`, `set`, or `spike`.

Variant-specific JSON fields:

- `python/test_sam3_hybrid.py` may add `mask_polygons` to each player record.
- `python/test_yolo26x_pose.py` may add `keypoints`, where each keypoint has `id`, `x`, `y`, and `conf`.

## Additional output files

The CLI can also emit two secondary outputs.

### Action table JSON

`--action_json_output` writes a normal JSON array containing one row per non-`player` action event:

```json
[
  {
    "player_id": 3,
    "action": "set",
    "frame": 42,
    "action_conf": 0.74
  }
]
```

### SQLite database

`--db_output` creates a SQLite database with:

- `videos` for source-video metadata.
- `players` for player lifetimes inside a processed video.
- `player_actions` for per-frame action rows and boxes.

This is managed by `python/player_action_db.py`.

## How the Python pipeline works

At a high level, the CLI processes a video frame by frame and combines detector outputs with tracking logic.

### 1. Load models and video

- The entry script parses CLI arguments with `argparse`.
- `python/tracking_shared.py` resolves the runtime device and loads YOLO models.
- OpenCV reads the input video for inference first, then the annotated MP4 is rendered in a second pass from the rewritten JSON.

### 2. Run per-frame detection

- The standard CLI uses either one unified YOLO model or separate `player`, `action`, and `ball` models.
- Player detections become the input to StrongSORT tracking.
- In the standard and pose CLIs, action detections are matched back onto tracked player boxes by IoU.
- In the SAM hybrid, SAM builds a player mask first and the action model runs on that masked player crop.
- Ball detections are passed into a custom multi-ball tracker.

### 3. Track identities over time

- Player identity tracking uses BoxMOT StrongSORT plus appearance embeddings.
- `python/tracking_shared.py` promotes raw tracker IDs into longer-lived canonical IDs so a player can keep the same `tid` across temporary occlusions or ID switches.
- Motion checks and embedding galleries reduce unrealistic teleports and help reacquire the same player later.

### 4. Track the volleyball

- `python/ball_tracker.py` maintains a Kalman-filter-based tracker for one or more candidate balls.
- During inference, the tracker follows live detections, keeps short-lived ball memory, and blacklists stationary false-ball locations.
- After inference, `python/tracking_shared.py` rewrites the saved frame JSON, retroactively removes blacklisted false balls, and fills valid gaps with either physics interpolation or cubic interpolation.

### 5. Serialize outputs

- Each processed frame is collected into the main frame JSON array.
- Optional action rows are accumulated into a separate JSON file.
- Optional database rows are inserted into SQLite for later analysis.
- The final MP4 is rendered from the rewritten frame JSON so the video matches the structured output exactly.

### 6. Optional richer player geometry

- The SAM hybrid refines each tracked player box into segmentation polygons.
- The SAM hybrid can use `--sam_imgsz` independently of `--imgsz`; if omitted, it defaults to `--imgsz`.
- The pose hybrid attaches COCO-style pose keypoints to each tracked player.
- Both variants keep the same core detection, tracking, and export flow.

## Model training helper

`python/yolo_tune.py` is the training-oriented CLI in this directory. It prepares separate YOLO datasets for players, actions, and ball detection, then optionally trains each task-specific model.

Example:

```bash
python python/yolo_tune.py --model yolo26x.pt --epochs 50 --batch 8 --device 0
```

Useful flags include `--tasks`, `--parallel`, `--skip-train`, and `--continue-training`.

## Practical notes

- Commands in this README assume you run them from the project root.
- If you omit `--json_output`, the CLI still chooses a default JSON path based on the input or output filename.
- If you omit `--output`, the CLI still writes an annotated MP4 unless you pass `--no-mp4`.
- The default ball detection threshold is `--ball_conf 0.8`.
- By default, post-process ball gap filling uses physics mode; pass `--cubic` to switch to cubic interpolation.
- `--dino` is deprecated and ignored.
