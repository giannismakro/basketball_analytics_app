from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch

# TODO a team should have x players each so this function should one be taking a teams argumment
def generate_game_summary_pdf(filename, teams, players):
    """
    Generates a PDF summary of the game stats.

    Args:
        filename (str): Output PDF filename.
        teams (list): List of team objects with `name` and `score`.
        players (dict): Dictionary of player_id -> player object.
    """
    c = canvas.Canvas(filename, pagesize=A4)
    width, height = A4
    margin = 50
    y = height - margin

    def draw_text(text, font_size=12, offset=20, bold=False):
        nonlocal y
        font_name = "Helvetica-Bold" if bold else "Helvetica"
        c.setFont(font_name, font_size)
        c.drawString(margin, y, text)
        y -= offset

    # --- Final Score ---
    draw_text("Final Score", font_size=16, bold=True)
    for team in teams:
        draw_text(f"{team.name}: {team.score}", font_size=14)
    y -= 10

    # --- Team Stats ---
    draw_text("Team Stats", font_size=16, bold=True)
    for team in teams:
        draw_text(f"{team.name}", font_size=14, bold=True)
        if hasattr(team, "stats"):
            for k, v in team.stats.items():
                draw_text(f"  {k.replace('_', ' ').title()}: {v}")
    y -= 10

    # --- Player Stats ---
    draw_text("Player Stats", font_size=16, bold=True)
    for player_id, player in players.items():
        draw_text(f"Player #{player_id}", font_size=14, bold=True)
        if hasattr(player, "name"):
            draw_text(f"  Name: {player.name}")
        if hasattr(player, "team") and player.team is not None:
            draw_text(f"  Team: {player.team.name}")
        if hasattr(player, "stats"):
            stats_dict = player.stats.to_dict() if hasattr(player.stats, "to_dict") else player.stats
            for k, v in stats_dict.items():
                draw_text(f"  {k.replace('_', ' ').title()}: {v}")
        y -= 10

        # Add new page if too low
        if y < 100:
            c.showPage()
            y = height - margin

    c.save()
