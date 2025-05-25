from .utils import draw_ellipse,draw_triangle

class PlayerTracksDrawer:
    """
    A class responsible for drawing player tracks and ball possession indicators on video frames.

    Attributes:
        default_player_team_id (int): Default team ID used when a player's team is not specified.
        team_1_color (list): RGB color used to represent Team 1 players.
        team_2_color (list): RGB color used to represent Team 2 players.
    """
    def __init__(self,team_1_color=[255, 245, 238],team_2_color=[128, 0, 0]):
        """
        Initialize the PlayerTracksDrawer with specified team colors.

        Args:
            team_1_color (list, optional): RGB color for Team 1. Defaults to [255, 245, 238].
            team_2_color (list, optional): RGB color for Team 2. Defaults to [128, 0, 0].
        """
        self.default_player_team_id = 1
        self.team_1_color=team_1_color
        self.team_2_color=team_2_color

    def draw(self, video_frames, players, player_assignment, ball_aquisition):
        """
        Draw player tracks and ball possession indicators on a list of video frames.

        Args:
            video_frames (list): A list of frames (as NumPy arrays or image objects) on which to draw.
            players (dict): Dictionary mapping track_id to Player objects, each containing bboxs_per_frame.
            player_assignment (list): A list of dictionaries indicating team assignments for each player
                                      in the corresponding frame.
            ball_aquisition (list): A list indicating which player has possession of the ball in each frame.

        Returns:
            list: A list of frames with player tracks and ball possession indicators drawn on them.
        """
        output_video_frames = []

        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()
            # Safely get player assignments, defaulting to empty dict if out of range
            player_assignment_for_frame = (player_assignment[frame_num]
                                           if frame_num < len(player_assignment)
                                           else {})

            # Safely get ball acquisition info, defaulting to None if out of range
            player_id_has_ball = (ball_aquisition[frame_num]
                                  if frame_num < len(ball_aquisition)
                                  else None)

            for track_id, player in players.items():
                bbox = player.bboxs_per_frame.get(frame_num)
                if bbox is None:
                    continue  # Skip player if no bbox for this frame

                team_id = player_assignment_for_frame.get(track_id, self.default_player_team_id)
                color = self.team_1_color if team_id == 1 else self.team_2_color

                frame = draw_ellipse(frame, bbox, color, track_id)

                if player_id_has_ball is not None and track_id == player_id_has_ball:
                    frame = draw_triangle(frame, bbox, (0, 0, 255))

            output_video_frames.append(frame)

        return output_video_frames

