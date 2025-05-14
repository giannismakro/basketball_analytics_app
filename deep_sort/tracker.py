from deep_sort_realtime.deepsort_tracker import DeepSort

class DeepSORTTracker:
    def __init__(self):
        self.tracker = DeepSort(max_age=20)

    def update_tracks(self, detections, frame):
        tracks = self.tracker.update_tracks(detections, frame=frame)
        valid_tracks = []
        for track in tracks:
            if not track.is_confirmed():
                continue
            valid_tracks.append(track)
        return valid_tracks
