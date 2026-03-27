#!/usr/bin/env python3

from collections import deque

import cv2
import numpy as np

from scipy.optimize import linear_sum_assignment


class KalmanBallTracker:
    def __init__(self, delta_t=1 / 30, gravity=1200.0):
        # State Vector: [x, y, dx, dy] (Position and Velocity)
        # Measurement Vector: [x, y] (What TrackNet actually sees)
        self.kf = cv2.KalmanFilter(4, 2)
        self.gravity = np.array([[np.float32(gravity)]], dtype=np.float32)

        # Measurement Matrix (Translates state to measurement)
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], np.float32)

        self.set_delta_t(delta_t)

        # Measurement Noise (How much we trust TrackNet - lower means we snap to the visual data immediately)
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-4

        self.is_tracking = False

    def set_delta_t(self, delta_t):
        dt = max(float(delta_t), 1e-3)
        self.delta_t = dt

        # State Transition Matrix (Physics Engine: x_new = x + dx * dt, y_new = y + dy * dt)
        self.kf.transitionMatrix = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], np.float32)

        # Constant-acceleration control input for gravity.
        self.kf.controlMatrix = np.array([
            [0.0],
            [0.5 * dt * dt],
            [0.0],
            [dt]
        ], np.float32)

        q_pos = max(0.03 * dt * dt, 1e-4)
        q_vel = max(0.03 * dt, 1e-4)
        self.kf.processNoiseCov = np.diag([q_pos, q_pos, q_vel, q_vel]).astype(np.float32)

    def predict(self, apply_gravity=True):
        """Advances the physics model by one frame and returns the predicted (x, y)."""
        if not self.is_tracking:
            return None

        control = self.gravity if apply_gravity else np.zeros((1, 1), dtype=np.float32)
        prediction = self.kf.predict(control)
        return (int(prediction[0]), int(prediction[1]))

    def set_velocity(self, vx, vy):
        if not self.is_tracking:
            return

        state_pre = self.kf.statePre.copy()
        state_post = self.kf.statePost.copy()
        state_pre[2, 0] = np.float32(vx)
        state_pre[3, 0] = np.float32(vy)
        state_post[2, 0] = np.float32(vx)
        state_post[3, 0] = np.float32(vy)
        self.kf.statePre = state_pre
        self.kf.statePost = state_post

    def get_velocity(self):
        if not self.is_tracking:
            return 0.0, 0.0
        return float(self.kf.statePost[2, 0]), float(self.kf.statePost[3, 0])

    def correct(self, x, y):
        """Updates the physics model with the hard visual data from TrackNet."""
        measurement = np.array([[np.float32(x)], [np.float32(y)]])

        if not self.is_tracking:
            # First time seeing the ball: Hard-set the position and zero the velocity
            self.kf.statePre = np.array([[measurement[0,0]], [measurement[1,0]], [0], [0]], np.float32)
            self.kf.statePost = np.array([[measurement[0,0]], [measurement[1,0]], [0], [0]], np.float32)
            self.is_tracking = True

        self.kf.correct(measurement)

    def reset(self):
        """Clears the tracker if the ball is gone for too long (e.g., play is dead)."""
        self.is_tracking = False


class MultiBallTracker:
    def __init__(self, max_coast_frames=20, distance_thresh=150, fps=30.0, gravity=600.0):
        self.tracks = {}
        self.next_id = 1
        self.max_coast_frames = max_coast_frames
        self.distance_thresh = distance_thresh
        self.delta_t = 1.0 / fps if fps and fps > 0 else 1.0 / 30.0
        self.fps = fps if fps and fps > 0 else 30.0
        self.gravity = gravity
        self.frame_index = 0

    def _estimate_velocity(self, history):
        if len(history) < 2:
            return None

        first_frame, first_x, first_y = history[0]
        last_frame, last_x, last_y = history[-1]
        frame_gap = max(last_frame - first_frame, 1)
        dt = max(frame_gap * self.delta_t, 1e-3)
        vx = (last_x - first_x) / dt
        vy = (last_y - first_y) / dt

        max_speed = 0.75 * max(self.distance_thresh, 1.0) * self.fps
        speed = float(np.hypot(vx, vy))
        if speed > max_speed:
            scale = max_speed / speed
            vx *= scale
            vy *= scale

        return vx, vy

    def _update_track_velocity(self, track):
        velocity = self._estimate_velocity(track['history'])
        if velocity is None:
            track['has_velocity'] = False
            return

        track['kalman'].set_velocity(*velocity)
        track['velocity'] = [float(velocity[0]), float(velocity[1])]
        track['has_velocity'] = True

    def update(self, ball_dets):
        self.frame_index += 1

        # 1. Parse Detections
        det_centers = []
        det_confs = []
        det_boxes = []
        for det in ball_dets:
            x1, y1, x2, y2, conf, cls = det
            det_centers.append([(x1 + x2) / 2, (y1 + y2) / 2])
            det_confs.append(conf)
            det_boxes.append([float(x1), float(y1), float(x2), float(y2)])
        
        det_centers = np.array(det_centers)
        active_ids = list(self.tracks.keys())

        # 2. Predict existing tracks using their Kalmans
        track_preds = []
        for tid in active_ids:
            track = self.tracks[tid]
            track['kalman'].set_delta_t(self.delta_t)
            apply_gravity = track['has_velocity'] and track['missed'] > 0
            pred = track['kalman'].predict(apply_gravity=apply_gravity)
            if pred is None:
                pred = tuple(map(int, track['pos']))
            track_preds.append(pred)
        track_preds = np.array(track_preds)

        # 3. Match Detections to Tracks
        matched_indices = []
        unmatched_dets = set(range(len(det_centers)))
        matched_tracks = set()

        if len(active_ids) > 0 and len(det_centers) > 0:
            cost_matrix = np.linalg.norm(track_preds[:, None] - det_centers[None, :], axis=2)
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r, c] < self.distance_thresh:
                    matched_indices.append((r, c))
                    unmatched_dets.remove(c)
                    matched_tracks.add(active_ids[r])

        # 4. Update Matched Tracks
        for r, c in matched_indices:
            tid = active_ids[r]
            cx, cy = det_centers[c]
            track = self.tracks[tid]
            
            # Add to total distance moved
            last_pos = track['pos']
            dist_moved = np.linalg.norm(np.array([cx, cy]) - np.array(last_pos))
            track['total_dist'] += dist_moved

            track['history'].append((self.frame_index, float(cx), float(cy)))
            self._update_track_velocity(track)
            track['kalman'].correct(cx, cy)
            if track['has_velocity']:
                track['kalman'].set_velocity(*track['velocity'])

            track['pos'] = [float(cx), float(cy)]
            track['box'] = det_boxes[c]
            track['size'] = [
                det_boxes[c][2] - det_boxes[c][0],
                det_boxes[c][3] - det_boxes[c][1],
            ]
            track['missed'] = 0
            track['conf'] = det_confs[c]
            track['is_predicted'] = False

        # 5. Coast Unmatched Tracks (Occlusion)
        for idx, tid in enumerate(active_ids):
            if tid not in matched_tracks:
                track = self.tracks[tid]
                track['missed'] += 1
                track['is_predicted'] = True

                if track['has_velocity']:
                    predicted_pos = track_preds[idx].tolist()
                else:
                    predicted_pos = list(track['pos'])

                track['pos'] = predicted_pos
                width, height = track['size']
                cx, cy = track['pos']
                track['box'] = [
                    cx - (width / 2.0),
                    cy - (height / 2.0),
                    cx + (width / 2.0),
                    cy + (height / 2.0),
                ]

                if not track['has_velocity'] and len(track['history']) > 0:
                    last_frame, last_x, last_y = track['history'][-1]
                    track['history'][-1] = (last_frame, float(last_x), float(last_y))

        # 6. Spawn New Tracks
        for c in unmatched_dets:
            cx, cy = det_centers[c]
            new_kalman = KalmanBallTracker(delta_t=self.delta_t, gravity=self.gravity)
            new_kalman.correct(cx, cy)
            box = det_boxes[c]
            history = deque([(self.frame_index, float(cx), float(cy))], maxlen=3)
            
            self.tracks[self.next_id] = {
                'kalman': new_kalman,
                'total_dist': 0.0,
                'pos': [float(cx), float(cy)],
                'box': box,
                'size': [box[2] - box[0], box[3] - box[1]],
                'missed': 0,
                'conf': det_confs[c],
                'is_predicted': False,
                'history': history,
                'velocity': [0.0, 0.0],
                'has_velocity': False,
            }
            self.next_id += 1

        # 7. Purge Dead Tracks
        dead_tracks = [tid for tid, data in self.tracks.items() if data['missed'] > self.max_coast_frames]
        for tid in dead_tracks:
            del self.tracks[tid]

        # 8. Identify Primary Ball (Most Distance Moved)
        primary_id = None
        max_dist = -1
        
        # Tie-breaker for the very first frame: use highest confidence
        if len(self.tracks) > 0 and all(t['total_dist'] == 0 for t in self.tracks.values()):
            primary_id = max(self.tracks.keys(), key=lambda k: self.tracks[k]['conf'])
        else:
            for tid, data in self.tracks.items():
                if data['total_dist'] > max_dist:
                    max_dist = data['total_dist']
                    primary_id = tid

        return primary_id, self.tracks
