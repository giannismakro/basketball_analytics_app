import numpy as np
import cv2
from ultralytics import YOLO

class YOLODetector:
    def __init__(self, model_path='models/yolov8n.pt', conf_thresh=0.3, classes_to_track=[0]):
        self.model = YOLO(model_path)
        self.conf_thresh = conf_thresh
        self.classes_to_track = classes_to_track  # e.g. [0] for person

    def detect(self, frame):
        results = self.model(frame, verbose=False)[0]
        detections = []
        h, w = frame.shape[:2]

        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls = int(box.cls[0])

            # Apply class filter and confidence threshold
            if conf >= self.conf_thresh and cls in self.classes_to_track:
                # Clip box to frame
                x1 = max(0, min(x1, w - 1))
                x2 = max(0, min(x2, w - 1))
                y1 = max(0, min(y1, h - 1))
                y2 = max(0, min(y2, h - 1))

                bbox = [x1, y1, x2, y2]
                detections.append((bbox, conf, cls))

        return detections
