class Basket:
    def __init__(self, team_name, bbox):
        self.team_name = team_name
        self.bbox = bbox  # [x1, y1, x2, y2]

    def get_center(self):
        x1, y1, x2, y2 = self.bbox
        return [(x1 + x2) / 2, (y1 + y2) / 2]

    def is_collision(self, ball_bbox):
        bx, by = Ball.get_center_from_bbox(ball_bbox)
        hx1, hy1, hx2, hy2 = self.bbox
        return hx1 <= bx <= hx2 and hy1 <= by <= hy2
