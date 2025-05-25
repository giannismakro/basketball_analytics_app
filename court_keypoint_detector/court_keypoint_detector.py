from ultralytics import YOLO
import supervision as sv
import sys 
sys.path.append('../')
import os, pickle

class CourtKeypointDetector:
    """
    The CourtKeypointDetector class uses a YOLO model to detect court keypoints in image frames. 
    It also provides functionality to draw these detected keypoints on the frames.
    """
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        class_name = self.__class__.__name__.replace("Tracker", "").lower()
        self.cache_path = f"cache/{class_name}_detections.pkl"
    
    def get_court_keypoints(self, frames,read_from_stub=False, stub_path=None):
        """
        Detect court keypoints for a batch of frames using the YOLO model. If requested, 
        attempts to read previously detected keypoints from a stub file before running the model.

        Args:
            frames (list of numpy.ndarray): A list of frames (images) on which to detect keypoints.
            read_from_stub (bool, optional): Indicates whether to read keypoints from a stub file 
                instead of running the detection model. Defaults to False.
            stub_path (str, optional): The file path for the stub file. If None, a default path may be used. 
                Defaults to None.

        Returns:
            list: A list of detected keypoints for each input frame.
        """
        court_keypoints = None
        if os.path.exists(self.cache_path):
            with open(self.cache_path, "rb") as f:
                print(f"Loading cached detections from {self.cache_path}")
                return pickle.load(f)

        
        batch_size=20
        court_keypoints = []
        for i in range(0,len(frames),batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size],conf=0.5)
            for detection in detections_batch:
                court_keypoints.append(detection.keypoints)

        # Save detections for future use
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        with open(self.cache_path, "wb") as f:
            pickle.dump(court_keypoints, f)


        return court_keypoints