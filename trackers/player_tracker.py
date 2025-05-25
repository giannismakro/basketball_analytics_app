from ultralytics import YOLO
import supervision as sv
import sys
from core.player import Player
import os
import pickle


sys.path.append('../')
from utils import read_stub, save_stub


class PlayerTracker:
    """
    A class that handles player detection and tracking using YOLO and ByteTrack.

    This class combines YOLO object detection with ByteTrack tracking to maintain consistent
    player identities across frames while processing detections in batches.
    """

    def __init__(self, model_path):
        """
        Initialize the PlayerTracker with YOLO model and ByteTrack tracker.

        Args:
            model_path (str): Path to the YOLO model weights.
        """
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()
        self.read_from_stub = False
        self.stub_path = None

    def detect_frames(self, frames, cache_path="cache/yolo_detections.pkl"):
        """
        Detect players in a sequence of frames using batch processing.

        Args:
            frames (list): List of video frames to process.

        Returns:
            list: YOLO detection results for each frame.
        """
        batch_size = 20
        detections = []
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                print(f"Loading cached detections from {cache_path}")
                return pickle.load(f)


        # Detecting frames in batches is faster
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i + batch_size], conf=0.5)
            detections += detections_batch

        # Save to cache
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(detections, f)
        return detections

    def get_player_objects(self, frames) -> dict:
        """
        Get player tracking results for a sequence of frames with optional caching.
        Returns:
            list: List of dictionaries containing player tracking information for each frame,
                where each dictionary maps player IDs to their player objects.
        """
        tracks = read_stub(self.read_from_stub, self.stub_path)
        if tracks is not None:
            if len(tracks) == len(frames):
                return tracks

        detections = self.detect_frames(frames)

        players = {}  # track_id → Player

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v: k for k, v in cls_names.items()}

            # Covert to supervision Detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            for det in detection_with_tracks:
                bbox = det[0].tolist()
                cls_id = det[3]
                track_id = det[4]

                if cls_id == cls_names_inv['Player']:
                    if track_id not in players:
                        players[track_id] = Player(track_id, track_id)
                    players[track_id].bboxs_per_frame[frame_num] = bbox

        return players