#!/usr/bin/env python3

import cv2
import numpy as np

class KalmanBallTracker:
    def __init__(self):
        # State Vector: [x, y, dx, dy] (Position and Velocity)
        # Measurement Vector: [x, y] (What TrackNet actually sees)
        self.kf = cv2.KalmanFilter(4, 2)
        
        # Measurement Matrix (Translates state to measurement)
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], np.float32)
        
        # State Transition Matrix (Physics Engine: x_new = x + dx, y_new = y + dy)
        self.kf.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], np.float32)
        
        # Process Noise (How much we trust the physics - lower means smoother but slower to turn)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        
        # Measurement Noise (How much we trust TrackNet - lower means we snap to the visual data immediately)
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-4
        
        self.is_tracking = False

    def predict(self):
        """Advances the physics model by one frame and returns the predicted (x, y)."""
        if not self.is_tracking:
            return None
            
        prediction = self.kf.predict()
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
