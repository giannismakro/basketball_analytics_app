class PlayerStats:
    def __init__(self):
        self.points = 0
        self.assists = 0
        self.rebounds_offensive = 0
        self.rebounds_defensive = 0
        self.steals = 0
        self.blocks = 0
        self.turnovers = 0
        self.fouls = 0
        self.shots_made = 0
        self.shots_attempted = 0
        self.field_goals_made = 0
        self.field_goals_attempted = 0
        self.three_pointers_made = 0
        self.three_pointers_attempted = 0
        self.free_throws_made = 0
        self.free_throws_attempted = 0

        self.shot_locations = []  # (x, y, result)
        self.pass_count = 0
        self.pass_targets = {}  # {player_id: count}

    def add_points(self, pts):
        self.points += pts

    def record_shot(self, made, is_three=False):
        self.field_goals_attempted += 1
        if made:
            self.field_goals_made += 1
            self.points += 3 if is_three else 2
            if is_three:
                self.three_pointers_made += 1
        if is_three:
            self.three_pointers_attempted += 1

    def shooting_percentage(self):
        if self.shots_attempted == 0:
            return 0
        return self.shots_made / self.shots_attempted

    def record_assist(self):
        self.assists += 1

    def to_dict(self):
        return {
            'points': self.points,
            'assists': self.assists,
            'rebounds_offensive': self.rebounds_offensive,
            'rebounds_defensive': self.rebounds_defensive,
            'steals': self.steals,
            'blocks': self.blocks,
            'turnovers': self.turnovers,
            'fouls': self.fouls,
            'shots_made': self.shots_made,
            'shots_attempted': self.shots_attempted,
            'field_goals_made': self.field_goals_made,
            'field_goals_attempted': self.field_goals_attempted,
            'three_pointers_made': self.three_pointers_made,
            'three_pointers_attempted': self.three_pointers_attempted,
            'free_throws_made': self.free_throws_made,
            'free_throws_attempted': self.free_throws_attempted,
            'shot_locations': self.shot_locations,
            'pass_count': self.pass_count,
            'pass_targets': self.pass_targets
        }


    @classmethod
    def from_dict(cls, data):
        stats = cls()
        stats.points = data.get('points', 0)
        stats.assists = data.get('assists', 0)
        stats.rebounds_offensive = data.get('rebounds_offensive', 0)
        stats.rebounds_defensive = data.get('rebounds_defensive', 0)
        stats.steals = data.get('steals', 0)
        stats.blocks = data.get('blocks', 0)
        stats.turnovers = data.get('turnovers', 0)
        stats.fouls = data.get('fouls', 0)
        stats.shots_made = data.get('shots_made', 0)
        stats.shots_attempted = data.get('shots_attempted', 0)
        stats.field_goals_made = data.get('field_goals_made', 0)
        stats.field_goals_attempted = data.get('field_goals_attempted', 0)
        stats.three_pointers_made = data.get('three_pointers_made', 0)
        stats.three_pointers_attempted = data.get('three_pointers_attempted', 0)
        stats.free_throws_made = data.get('free_throws_made', 0)
        stats.free_throws_attempted = data.get('free_throws_attempted', 0)
        stats.shot_locations = data.get('shot_locations', [])
        stats.pass_count = data.get('pass_count', 0)
        stats.pass_targets = data.get('pass_targets', {})

        return stats
