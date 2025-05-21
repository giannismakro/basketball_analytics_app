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

    def record_assist(self):
        self.assists += 1

    def to_dict(self):
        return self.__dict__
