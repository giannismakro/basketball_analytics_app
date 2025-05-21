from core.player_stats import PlayerStats

class Player:
    def __init__(self, player_id, name, position=None, team=None):
        self.player_id = player_id
        self.name = name
        self.position = position
        self.team = team
        self.track_id = None
        self.bounding_box = None
        self.location_history = []  # (frame_id, x, y)
        self.stats = PlayerStats()

    def update_position(self, x, y, frame_id):
        self.location_history.append((frame_id, x, y))

    def update_bounding_box(self, bbox):
        self.bounding_box = bbox

    def update_stats(self, event):
        self.stats.update_from_event(event)

    def get_current_location(self):
        if self.location_history:
            return self.location_history[-1][1:]
        return None

    def to_dict(self):
        return {
            "player_id": self.player_id,
            "name": self.name,
            "team": self.team,
            "position": self.position,
            "track_id": self.track_id,
            "bounding_box": self.bounding_box,
            "stats": self.stats.to_dict(),
        }

    def __str__(self):
        return f"Player({self.name}, ID={self.player_id})"
