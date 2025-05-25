from ultralytics import YOLO
import supervision as sv
import os, pickle
from utils import read_stub, save_stub

class BaseTracker:
    def __init__(self, model_path, target_class_name):
        self.model = YOLO(model_path)
        self.tracker = getattr(sv, "ByteTrack", None)()  # Only used by PlayerTracker
        self.target_class_name = target_class_name
        self.read_from_stub = False
        class_name = self.__class__.__name__.replace("Tracker", "").lower()
        self.cache_path = f"cache/{class_name}_detections.pkl"

    def detect_frames(self, frames, batch_size=20):
        if os.path.exists(self.cache_path):
            with open(self.cache_path, "rb") as f:
                print(f"Loading cached detections from {self.cache_path}")
                return pickle.load(f)

        detections = []
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i + batch_size], conf=0.5)
            detections += detections_batch

        # Save detections for future use
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        with open(self.cache_path, "wb") as f:
            pickle.dump(detections, f)

        return detections

    def track_objects(self, detections, frames):
        raise NotImplementedError("Must be implemented in subclass")

    def get_tracks(self, frames):
        detections = self.detect_frames(frames)
        tracks = self.track_objects(detections, frames)
        return tracks
