class Team:
    def __init__(self, name, color):
        self.name = name
        self.color = color
        self.basket = None  # Instance of Basket
        self.players = []     # List[Player]
        self.score = 0

    def add_player(self, player):
        self.players.append(player)
        player.team = self

    def update_score(self, points):
        self.score += points
