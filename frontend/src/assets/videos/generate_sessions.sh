#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRONTEND_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
VIDEOS_DIR="$FRONTEND_DIR/public/videos"
ANNOTATIONS_DIR="$FRONTEND_DIR/public/annotations"
THUMBNAILS_DIR="$FRONTEND_DIR/public/thumbnails"
OUTPUT_FILE="$SCRIPT_DIR/sessions.js"

if [[ ! -d "$VIDEOS_DIR" ]]; then
  printf 'Videos directory not found: %s\n' "$VIDEOS_DIR" >&2
  exit 1
fi

if [[ ! -d "$ANNOTATIONS_DIR" ]]; then
  printf 'Annotations directory not found: %s\n' "$ANNOTATIONS_DIR" >&2
  exit 1
fi

mapfile -t video_files < <(python3 - "$VIDEOS_DIR" <<'PY'
from pathlib import Path
import sys

video_dir = Path(sys.argv[1])
extensions = {'.mp4', '.mov', '.avi', '.mkv', '.m4v', '.webm'}
for path in sorted(p for p in video_dir.iterdir() if p.is_file() and p.suffix.lower() in extensions):
    print(path.name)
PY
)

mapfile -t thumbnail_files < <(python3 - "$THUMBNAILS_DIR" <<'PY'
from pathlib import Path
import sys

thumb_dir = Path(sys.argv[1])
if not thumb_dir.exists():
    raise SystemExit(0)

extensions = {'.png', '.jpg', '.jpeg', '.webp'}
for path in sorted(p for p in thumb_dir.iterdir() if p.is_file() and p.suffix.lower() in extensions):
    print(path.name)
PY
)

if [[ ${#video_files[@]} -eq 0 ]]; then
  printf 'No video files found in %s\n' "$VIDEOS_DIR" >&2
  exit 1
fi

{
  printf 'const SESSIONS = [\n'
  first_written=1

  for idx in "${!video_files[@]}"; do
    video_file="${video_files[$idx]}"
    stem="${video_file%.*}"

    annotation_file=""
    for candidate in \
      "$ANNOTATIONS_DIR/${stem}_filtered.json" \
      "$ANNOTATIONS_DIR/${stem}.json"
    do
      if [[ -f "$candidate" ]]; then
        annotation_file="$(basename "$candidate")"
        break
      fi
    done

    if [[ -z "$annotation_file" ]]; then
      printf 'Skipping %s: no matching annotation JSON found\n' "$video_file" >&2
      continue
    fi

    action_file="${annotation_file%.json}_actions.json"

    if [[ ${#thumbnail_files[@]} -gt 0 ]]; then
      thumbnail_file="${thumbnail_files[$((idx % ${#thumbnail_files[@]}))]}"
      thumbnail_value="/thumbnails/$thumbnail_file"
    else
      thumbnail_value="/thumbnails/thumbnail0.png"
    fi

    title="${stem//_/ }"
    if [[ "$first_written" -eq 0 ]]; then
      printf ',\n'
    fi
    first_written=0
    printf '  {\n'
    printf '    key: "%s",\n' "$stem"
    printf '    title: "%s",\n' "$title"
    printf '    thumbnail: "%s",\n' "$thumbnail_value"
    printf '    videoSrc: "/videos/%s",\n' "$video_file"
    printf '    annotationUrl: "/annotations/%s",\n' "$annotation_file"
    printf '    actionURL: "/annotations/%s",\n' "$action_file"
    printf '    clips: [],\n'
    printf '  }'
  done

  printf '\n];\n\nexport default SESSIONS;\n'
} > "$OUTPUT_FILE"

printf 'Wrote %s\n' "$OUTPUT_FILE"
