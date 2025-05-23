class Ball:
    """
    Represents a basketball's movement across frames.
    Stores trajectory and provides utilities for interaction analysis (e.g., scoring).
    """

    def __init__(self, ball_id=1):
        self.ball_id = ball_id
        self.bbox_per_frame = {}  # frame_index -> bbox [x1, y1, x2, y2]
        self.smoothed = False

    def add_bbox(self, frame_idx, bbox):
        self.bbox_per_frame[frame_idx] = bbox

    def get_center(self, frame_idx):
        bbox = self.bbox_per_frame.get(frame_idx)
        if bbox:
            x1, y1, x2, y2 = bbox
            return [(x1 + x2) / 2, (y1 + y2) / 2]
        return None

    def set_last_owner(self, player):
        self.last_owner = player
        player.set_has_ball()