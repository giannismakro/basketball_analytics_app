from core.player_stats import PlayerStats

class Player:
    def __init__(self, player_id, name, position=None, team=None):
        self.player_id = player_id
        self.name = name
        self.position = position
        self.team = team
        self.track_id = None
        self.bbox = None
        self.bboxs_per_frame = {}
        self.location_history = []  # (frame_id, x, y)
        self.stats = PlayerStats()
        self.has_ball_last = False


    def reset_ball_status(self):
        self.has_ball_last = False

    def set_has_ball(self):
        self.has_ball_last = True

    def update_position(self, x, y, frame_id):
        self.location_history.append((frame_id, x, y))

    def update_stats(self, event):
        self.stats.update_from_event(event)

    def get_scurrent_location(self):
        if self.location_history:
            return self.location_history[-1][1:]
        return None

    def __str__(self):
        return f"Player({self.name}, ID={self.player_id})"
