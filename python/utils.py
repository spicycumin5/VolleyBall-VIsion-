#!/usr/bin/env python3
import cv2
from collections import defaultdict, deque
import numpy as np


track_history = defaultdict(lambda: deque(maxlen=10))

def get_center(box):
    x1, y1, x2, y2 = box
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2])


def box_area(box):
    x1, y1, x2, y2 = box
    return (x2 - x1) * (y2 - y1)


def motion_distance(prev_center, new_center):
    return np.linalg.norm(prev_center - new_center)


def is_far_player(box, threshold=5000):
    return box_area(box) < threshold


def is_motion_consistent(track_id, new_center, max_jump=150):
    """
    Reject matches where the object teleports unrealistically.
    """
    if len(track_history[track_id]) == 0:
        return True

    prev_center = track_history[track_id][-1]
    dist = motion_distance(prev_center, new_center)

    return dist < max_jump


def is_blurry(img, threshold=50):
    return cv2.Laplacian(img, cv2.CV_64F).var() < threshold


def is_on_player(bx, by, player_boxes):
    for (x1, y1, x2, y2) in player_boxes:
        # Calculate the y-coordinate for the bottom of the "head zone"
        head_bottom = y1 + (y2 - y1) * 0.25

        # Check if the ball's (bx, by) is inside this upper rectangle
        if x1 <= bx <= x2 and y1 <= by <= head_bottom:
            return True

    return False
