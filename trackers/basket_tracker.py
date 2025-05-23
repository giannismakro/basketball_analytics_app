from trackers.base_tracker import BaseTracker
from utils import read_stub, save_stub
import supervision as sv
import numpy as np

class HoopTracker(BaseTracker):
    """
    A class that detects basketball hoops (baskets) in video frames using YOLO.
    Assumes the hoop remains stationary and aggregates detection over multiple frames.
    """

    def __init__(self, model_path):
        super().__init__(model_path, "Hoop")  # 'Basket' must be in your model class names

    def track_objects(self, detections, frames):
        """
        Detects hoops across multiple frames and returns a single hoop position,
        averaged from all valid detections.
        """
        basket_boxes = []

        for detection in detections:
            cls_names_inv = {v: k for k, v in detection.names.items()}
            detection_supervision = sv.Detections.from_ultralytics(detection)

            for det in detection_supervision:
                bbox, cls_id = det[0].tolist(), det[3]
                if cls_id == cls_names_inv[self.target_class_name]:
                    basket_boxes.append(bbox)

        if not basket_boxes:
            print("No basket detected.")
            return [{} for _ in frames]

        # Average the basket boxes over all frames
        avg_box = np.mean(np.array(basket_boxes), axis=0).tolist()
        track = {1: {"bbox": avg_box}}

        # Return the same basket for all frames (static hoop)
        return [track for _ in frames]
