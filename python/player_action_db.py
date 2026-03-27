#!/usr/bin/env python3

import sqlite3
from pathlib import Path


class PlayerActionDatabase:
    def __init__(self, db_path, video_path, fps, total_frames, flush_interval=100):
        self.db_path = Path(db_path)
        self.video_path = str(video_path)
        self.fps = float(fps)
        self.total_frames = int(total_frames)
        self.flush_interval = max(int(flush_interval), 1)
        self.pending_player_updates = {}
        self.pending_action_rows = []

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.db_path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=NORMAL")
        self._create_schema()
        self.video_id = self._create_video()

    def _create_schema(self):
        cursor = self.connection.cursor()
        cursor.executescript(
            """
            CREATE TABLE IF NOT EXISTS videos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                input_path TEXT NOT NULL,
                fps REAL NOT NULL,
                total_frames INTEGER NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS players (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id INTEGER NOT NULL,
                player_tid INTEGER NOT NULL,
                first_frame INTEGER NOT NULL,
                last_frame INTEGER NOT NULL,
                UNIQUE(video_id, player_tid),
                FOREIGN KEY(video_id) REFERENCES videos(id)
            );

            CREATE TABLE IF NOT EXISTS player_actions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id INTEGER NOT NULL,
                player_tid INTEGER NOT NULL,
                frame_idx INTEGER NOT NULL,
                action TEXT NOT NULL,
                action_conf REAL,
                player_conf REAL,
                x1 INTEGER,
                y1 INTEGER,
                x2 INTEGER,
                y2 INTEGER,
                FOREIGN KEY(video_id) REFERENCES videos(id)
            );

            CREATE INDEX IF NOT EXISTS idx_player_actions_video_frame
            ON player_actions(video_id, frame_idx);

            CREATE INDEX IF NOT EXISTS idx_player_actions_video_player
            ON player_actions(video_id, player_tid);
            """
        )
        self.connection.commit()

    def _create_video(self):
        cursor = self.connection.cursor()
        cursor.execute(
            "INSERT INTO videos (input_path, fps, total_frames) VALUES (?, ?, ?)",
            (self.video_path, self.fps, self.total_frames),
        )
        self.connection.commit()
        return int(cursor.lastrowid)

    def add_frame_players(self, frame_idx, players):
        for player in players:
            player_tid = int(player["tid"])
            first_frame, _ = self.pending_player_updates.get(player_tid, (frame_idx, frame_idx))
            self.pending_player_updates[player_tid] = (min(first_frame, frame_idx), frame_idx)

            box = player.get("box") or [None, None, None, None]
            self.pending_action_rows.append(
                (
                    self.video_id,
                    player_tid,
                    int(frame_idx),
                    str(player.get("state", "player")),
                    self._maybe_float(player.get("state_conf")),
                    self._maybe_float(player.get("conf")),
                    self._maybe_int(box[0]),
                    self._maybe_int(box[1]),
                    self._maybe_int(box[2]),
                    self._maybe_int(box[3]),
                )
            )

        if len(self.pending_action_rows) >= self.flush_interval:
            self.flush()

    def flush(self):
        if not self.pending_player_updates and not self.pending_action_rows:
            return

        cursor = self.connection.cursor()
        for player_tid, (first_frame, last_frame) in self.pending_player_updates.items():
            cursor.execute(
                """
                INSERT INTO players (video_id, player_tid, first_frame, last_frame)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(video_id, player_tid) DO UPDATE SET
                    first_frame = MIN(first_frame, excluded.first_frame),
                    last_frame = MAX(last_frame, excluded.last_frame)
                """,
                (self.video_id, player_tid, first_frame, last_frame),
            )

        if self.pending_action_rows:
            cursor.executemany(
                """
                INSERT INTO player_actions (
                    video_id,
                    player_tid,
                    frame_idx,
                    action,
                    action_conf,
                    player_conf,
                    x1,
                    y1,
                    x2,
                    y2
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                self.pending_action_rows,
            )

        self.connection.commit()
        self.pending_player_updates.clear()
        self.pending_action_rows.clear()

    def close(self):
        self.flush()
        self.connection.close()

    @staticmethod
    def _maybe_float(value):
        if value is None:
            return None
        return float(value)

    @staticmethod
    def _maybe_int(value):
        if value is None:
            return None
        return int(value)
