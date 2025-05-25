from ultralytics import YOLO
import supervision as sv
import numpy as np
import pandas as pd
import sys 
sys.path.append('../')
from utils import read_stub, save_stub
from core.ball import Ball
import os, pickle

class BallTracker:
    """
    A class that handles basketball detection and tracking using YOLO.

    This class provides methods to detect the ball in video frames, process detections
    in batches, and refine tracking results through filtering and interpolation.
    """
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.ball = Ball()
        class_name = self.__class__.__name__.replace("Tracker", "").lower()
        self.cache_path = f"cache/{class_name}_detections.pkl"

    def detect_frames(self, frames):
        """
        Detect the ball in a sequence of frames using batch processing.

        Args:
            frames (list): List of video frames to process.

        Returns:
            list: YOLO detection results for each frame.
        """
        if os.path.exists(self.cache_path):
            with open(self.cache_path, "rb") as f:
                print(f"Loading cached detections from {self.cache_path}")
                return pickle.load(f)
        batch_size=20 
        detections = []
        for i in range(0,len(frames),batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size],conf=0.5)
            detections += detections_batch

        # Save detections for future use
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        with open(self.cache_path, "wb") as f:
            pickle.dump(detections, f)

        return detections

    def get_object_tracks(self, frames):
        """
        Get ball tracking results for a sequence of frames with optional caching.

        Args:
            frames (list): List of video frames to process.

        Returns:
            list: List of dictionaries containing ball tracking information for each frame.
        """

        detections = self.detect_frames(frames)

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v:k for k,v in cls_names.items()}

            # Covert to supervision Detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            chosen_bbox = None
            max_confidence = 0
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                confidence = frame_detection[2]

                if cls_id == cls_names_inv['Ball']:
                    if max_confidence<confidence:
                        chosen_bbox = bbox
                        max_confidence = confidence

            if chosen_bbox is not None:
                self.ball.add_bbox(frame_num, chosen_bbox)

        self.remove_wrong_detections()
        # Interpolate Ball Tracks
        self.interpolate_ball_positions()
        return self.ball

    def remove_wrong_detections(self):
        maximum_allowed_distance = 25
        bbox_per_frame = self.ball.bbox_per_frame
        valid_bboxes = {}
        last_good_frame_index = None

        frame_indices = sorted(bbox_per_frame.keys())

        for idx in frame_indices:
            current_box = bbox_per_frame[idx]

            if last_good_frame_index is None:
                last_good_frame_index = idx
                valid_bboxes[idx] = current_box
                continue

            last_box = bbox_per_frame[last_good_frame_index]
            frame_gap = idx - last_good_frame_index
            max_distance = maximum_allowed_distance * frame_gap

            dist = np.linalg.norm(np.array(current_box[:2]) - np.array(last_box[:2]))

            if dist <= max_distance:
                valid_bboxes[idx] = current_box
                last_good_frame_index = idx

        self.ball.bbox_per_frame = valid_bboxes

    def interpolate_ball_positions(self):
        bbox_per_frame = self.ball.bbox_per_frame
        max_frame = max(bbox_per_frame.keys())

        df = pd.DataFrame(index=range(max_frame + 1), columns=['x1', 'y1', 'x2', 'y2'])

        for frame_idx, bbox in bbox_per_frame.items():
            df.loc[frame_idx] = bbox

        df = df.astype(float).interpolate().bfill()

        # Update ball object
        interpolated_bboxes = df.to_dict('index')
        self.ball.bbox_per_frame = {idx: list(bbox.values()) for idx, bbox in interpolated_bboxes.items()}
