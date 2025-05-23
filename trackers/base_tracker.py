from ultralytics import YOLO
import supervision as sv
from utils import read_stub, save_stub

class BaseTracker:
    def __init__(self, model_path, target_class_name):
        self.model = YOLO(model_path)
        self.tracker = getattr(sv, "ByteTrack", None)()  # Only used by PlayerTracker
        self.target_class_name = target_class_name

    def detect_frames(self, frames, batch_size=20):
        detections = []
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i + batch_size], conf=0.5)
            detections += detections_batch
        return detections

    def track_objects(self, detections, frames):
        raise NotImplementedError("Must be implemented in subclass")

    def get_tracks(self, frames, read_from_stub=False, stub_path=None):
        tracks = read_stub(read_from_stub, stub_path)
        if tracks is not None and len(tracks) == len(frames):
            return tracks

        detections = self.detect_frames(frames)
        tracks = self.track_objects(detections, frames)
        save_stub(stub_path, tracks)
        return tracks
