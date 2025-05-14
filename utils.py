import cv2
import numpy as np

def compute_speed(p1, p2, time_diff):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    distance = np.sqrt(dx**2 + dy**2)
    return distance / time_diff  # pixels per second

def draw_annotations(frame, track_id, x, y, speed, distance):
    cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)
    text = f"ID:{track_id} Speed:{speed:.2f}px/s Dist:{distance:.2f}px"
    cv2.putText(frame, text, (int(x), int(y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
