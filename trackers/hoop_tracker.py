from trackers.base_tracker import BaseTracker
import supervision as sv

from core.hoop import Hoop

class HoopTracker(BaseTracker):
    def __init__(self, model_path):
        super().__init__(model_path, "Hoop")

    def track_objects(self, detections, frames):
        frame_width = frames[0].shape[1]
        midpoint_x = frame_width // 2

        left_hoop = Hoop(label="left")
        right_hoop = Hoop(label="right")

        for frame_idx, detection in enumerate(detections):
            cls_names_inv = {v: k for k, v in detection.names.items()}
            detection_supervision = sv.Detections.from_ultralytics(detection)

            left_best = None  # (bbox, confidence)
            right_best = None

            for det in detection_supervision:
                bbox = det[0].tolist()
                confidence = det[2]
                cls_id = det[3]

                if cls_id == cls_names_inv[self.target_class_name]:
                    x1, _, x2, _ = bbox
                    x_center = (x1 + x2) / 2

                    if x_center < midpoint_x:
                        if not left_best or confidence > left_best[1]:
                            left_best = (bbox, confidence)
                    else:
                        if not right_best or confidence > right_best[1]:
                            right_best = (bbox, confidence)

            if left_best:
                left_hoop.add_bbox(frame_idx, left_best[0])
            if right_best:
                right_hoop.add_bbox(frame_idx, right_best[0])

        return left_hoop, right_hoop

