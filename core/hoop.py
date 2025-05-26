class Hoop:
    def __init__(self, label, team_name=None):
        self.label = label
        self.team_name = team_name
        self.bbox_per_frame = {}  # frame_idx: bbox

    def add_bbox(self, frame_idx, bbox):
        self.bbox_per_frame[frame_idx] = bbox

    def get_center(self, frame_idx):
        bbox = self.bbox_per_frame.get(frame_idx)
        if bbox:
            x1, y1, x2, y2 = bbox
            return [(x1 + x2) / 2, (y1 + y2) / 2]
        return None

    def get_bbox(self, frame_idx):
        return self.bbox_per_frame.get(frame_idx, None)

    def is_collision(self, ball, frame_idx):
        hoop_bbox = self.get_bbox(frame_idx)
        ball_center = ball.get_center(frame_idx)
        if hoop_bbox is None or ball_center is None:
            #print("Returning false")
            return False

        hx1, hy1, hx2, hy2 = hoop_bbox
        bx, by = ball_center
        #if hx1 <= bx <= hx2 or hy1 <= by <= hy2:
        #    print("KARAMMMMMMMMMMMReturning true")
        #else:
        #    print(f"!!!!!Karammmm Returning false {(hx1 <= bx <= hx2)} and {(hy1 <= by <= hy2)}")
        return hx1 <= bx <= hx2 or hy1 <= by <= hy2

    def is_collision_with_ellipse(self, ball, frame_idx):
        ball_center = ball.get_center(frame_idx)
        if ball_center is None or frame_idx not in self.bbox_per_frame:
            return False

        # Get the hoop bbox and derive ellipse center and axes
        x1, y1, x2, y2 = self.bbox_per_frame[frame_idx]
        h = (x1 + x2) / 2
        k = y2  # Bottom center of bbox
        a = max(1, (x2 - x1) / 2)  # Horizontal radius
        b = 0.35 * a  # Vertical radius (from your ellipse drawing)

        x, y = ball_center
        ellipse_value = ((x - h) ** 2) / (a ** 2) + ((y - k) ** 2) / (b ** 2)
        #print(f"Ellipse value: {ellipse_value}")
        return ellipse_value <= 100.0