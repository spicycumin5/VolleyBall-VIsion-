#!/usr/bin/env python3

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

    def predict(self):
        """Advances the physics model by one frame and returns the predicted (x, y)."""
        if not self.is_tracking:
            return None

        prediction = self.kf.predict(self.gravity)
        return (int(prediction[0]), int(prediction[1]))

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
    def __init__(self, max_coast_frames=20, distance_thresh=150, fps=30.0, gravity=1200.0):
        self.tracks = {}
        self.next_id = 1
        self.max_coast_frames = max_coast_frames
        self.distance_thresh = distance_thresh
        self.delta_t = 1.0 / fps if fps and fps > 0 else 1.0 / 30.0
        self.gravity = gravity

    def update(self, ball_dets):
        # 1. Parse Detections
        det_centers = []
        det_confs = []
        for det in ball_dets:
            x1, y1, x2, y2, conf, cls = det
            det_centers.append([(x1 + x2) / 2, (y1 + y2) / 2])
            det_confs.append(conf)
        
        det_centers = np.array(det_centers)
        active_ids = list(self.tracks.keys())

        # 2. Predict existing tracks using their Kalmans
        track_preds = []
        for tid in active_ids:
            self.tracks[tid]['kalman'].set_delta_t(self.delta_t)
            pred = self.tracks[tid]['kalman'].predict()
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
            
            # Add to total distance moved
            last_pos = self.tracks[tid]['pos']
            dist_moved = np.linalg.norm(np.array([cx, cy]) - np.array(last_pos))
            self.tracks[tid]['total_dist'] += dist_moved
            
            self.tracks[tid]['kalman'].correct(cx, cy)
            self.tracks[tid]['pos'] = [cx, cy]
            self.tracks[tid]['missed'] = 0
            self.tracks[tid]['conf'] = det_confs[c]
            self.tracks[tid]['is_predicted'] = False

        # 5. Coast Unmatched Tracks (Occlusion)
        for tid in active_ids:
            if tid not in matched_tracks:
                self.tracks[tid]['missed'] += 1
                self.tracks[tid]['is_predicted'] = True
                self.tracks[tid]['pos'] = track_preds[active_ids.index(tid)].tolist()

        # 6. Spawn New Tracks
        for c in unmatched_dets:
            cx, cy = det_centers[c]
            new_kalman = KalmanBallTracker(delta_t=self.delta_t, gravity=self.gravity)
            new_kalman.correct(cx, cy)
            
            self.tracks[self.next_id] = {
                'kalman': new_kalman,
                'total_dist': 0.0,
                'pos': [cx, cy],
                'missed': 0,
                'conf': det_confs[c],
                'is_predicted': False
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
