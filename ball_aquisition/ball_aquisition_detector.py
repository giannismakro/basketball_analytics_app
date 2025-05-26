import sys 
sys.path.append('../')
from utils.bbox_utils import measure_distance, get_center_of_bbox

class BallAcquisitionDetector:
    """
    Detects ball acquisition by players in a basketball game.

    This class determines which player is most likely in possession of the ball
    by analyzing bounding boxes for both the ball and the players. It combines
    distance measurements between the ball and key points of each player's bounding
    box with containment ratios of the ball within a player's bounding box.
    """

    def __init__(self):
        """
        Initialize the BallAcquisitionDetector with default thresholds.

        Attributes:
            possession_threshold (int): Maximum distance (in pixels) at which
                a player can be considered to have the ball if containment is insufficient.
            min_frames (int): Minimum number of consecutive frames required for a player
                to be considered in possession of the ball.
            containment_threshold (float): Containment ratio above which a player
                is considered to hold the ball without requiring distance checking.
        """
        self.possession_threshold = 50
        self.min_frames = 11
        self.containment_threshold = 0.8
        self.possession_list = []
        
    def get_key_basketball_player_assignment_points(self, player_bbox,ball_center):
        """
        Compute a list of key points around a player's bounding box.

        Key points are used to measure distance to the ball more accurately than
        using just the center of the bounding box.

        Args:
            bbox (tuple or list): A bounding box in the format (x1, y1, x2, y2).

        Returns:
            list of tuple: A list of (x, y) coordinates representing key points
            around the bounding box.
        """
        ball_center_x = ball_center[0]
        ball_center_y = ball_center[1]

        x1, y1, x2, y2 = player_bbox
        width = x2 - x1
        height = y2 - y1

        output_points = []    
        if ball_center_y > y1 and ball_center_y < y2:
            output_points.append((x1, ball_center_y))
            output_points.append((x2, ball_center_y))

        if ball_center_x > x1 and ball_center_x < x2:
            output_points.append((ball_center_x, y1))
            output_points.append((ball_center_x, y2))

        output_points += [
            (x1 + width//2, y1),          # top center
            (x2, y1),                      # top right
            (x1, y1),                      # top left
            (x2, y1 + height//2),          # center right
            (x1, y1 + height//2),          # center left
            (x1 + width//2, y1 + height//2), # center point
            (x2, y2),                      # bottom right
            (x1, y2),                      # bottom left
            (x1 + width//2, y2),          # bottom center
            (x1 + width//2, y1 + height//3), # mid-top center
        ]
        return output_points
    
    def calculate_ball_containment_ratio(self, player_bbox, ball_bbox):
        """
        Calculate how much of the ball is contained within a player's bounding box.

        This is computed as the ratio of the intersection of the bounding boxes
        to the area of the ball's bounding box.

        Args:
            player_bbox (tuple or list): The player's bounding box (x1, y1, x2, y2).
            ball_bbox (tuple or list): The ball's bounding box (x1, y1, x2, y2).

        Returns:
            float: A value between 0.0 and 1.0 indicating what fraction of the
            ball is inside the player's bounding box.
        """
        px1, py1, px2, py2 = player_bbox
        bx1, by1, bx2, by2 = ball_bbox
        
        intersection_x1 = max(px1, bx1)
        intersection_y1 = max(py1, by1)
        intersection_x2 = min(px2, bx2)
        intersection_y2 = min(py2, by2)
        
        if intersection_x2 < intersection_x1 or intersection_y2 < intersection_y1:
            return 0.0
            
        intersection_area = (intersection_x2 - intersection_x1) * (intersection_y2 - intersection_y1)
        ball_area = (bx2 - bx1) * (by2 - by1)
        
        return intersection_area / ball_area
    
    def find_minimum_distance_to_ball(self, ball_center, player_bbox):
        """
        Compute the minimum distance from any key point on a player's bounding box
        to the center of the ball.

        Args:
            ball_center (tuple): (x, y) coordinates of the center of the ball.
            player_bbox (tuple): A bounding box (x1, y1, x2, y2) for the player.

        Returns:
            float: The smallest distance from the ball center to
            any key point on the player's bounding box.
        """
        key_points = self.get_key_basketball_player_assignment_points(player_bbox,ball_center)
        return min(measure_distance(ball_center, point) for point in key_points)
    
    def find_best_candidate_for_possession(self, ball_center, player_tracks_frame, ball_bbox):
        """
        Determine which player in a single frame is most likely to have the ball.

        Players who have a high containment ratio of the ball are prioritized.
        If no player has a high containment ratio, the player with the smallest
        distance to the ball that is below the possession threshold is selected.

        Args:
            ball_center (tuple): (x, y) coordinates of the ball center.
            player_tracks_frame (dict): Mapping from player_id to info about that player,
                including a 'bbox' key with (x1, y1, x2, y2).
            ball_bbox (tuple): Bounding box for the ball (x1, y1, x2, y2).

        Returns:
            int: (best_player_id), or (-1 ) if none found.
        """
        high_containment_players = []
        regular_distance_players = []
        
        for player_id, player_obj in player_tracks_frame.items():
            player_bbox = player_obj.bbox
            if not player_bbox:
                continue
                
            containment = self.calculate_ball_containment_ratio(player_bbox, ball_bbox)
            min_distance = self.find_minimum_distance_to_ball(ball_center, player_bbox)

            if containment > self.containment_threshold:
                high_containment_players.append((player_id, min_distance))
            else:
                regular_distance_players.append((player_id, min_distance))

        # First priority: players with high containment
        if high_containment_players:
            best_candidate = max(high_containment_players, key=lambda x: x[1])
            return best_candidate[0]
            
        # Second priority: players within distance threshold
        if regular_distance_players:
            best_candidate = min(regular_distance_players, key=lambda x: x[1])
            if best_candidate[1] < self.possession_threshold:
                return best_candidate[0]
                
        return -1
    
    def detect_ball_possession(self, players, ball_object):
        """
        Detect which player has the ball in each frame based on bounding box information.

        Args:
            players (dict): Mapping from player_id (track_id) to Player objects.
            ball_object (Ball): Ball object with bbox_per_frame data.

        Returns:
            list: A list with player_id who has possession in each frame, or -1 if none.
        """
        num_frames = max(ball_object.bbox_per_frame.keys()) + 1
        self.possession_list = [-1] * num_frames
        consecutive_possession_count = {}

        for frame_num in range(num_frames):
            if frame_num not in ball_object.bbox_per_frame:
                continue

            ball_bbox = ball_object.bbox_per_frame[frame_num]
            if not ball_bbox:
                continue

            ball_center = get_center_of_bbox(ball_bbox)

            # Build player data for this frame
            players_in_frame = {}
            for player_id, player_obj in players.items():
                if frame_num in player_obj.bboxs_per_frame:
                    players_in_frame[player_id] = player_obj

            best_player_id = self.find_best_candidate_for_possession(
                ball_center,
                players_in_frame,
                ball_bbox
            )

            if best_player_id != -1:
                number_of_consecutive_frames = consecutive_possession_count.get(best_player_id, 0) + 1
                consecutive_possession_count = {best_player_id: number_of_consecutive_frames}

                if number_of_consecutive_frames >= self.min_frames:
                    self.possession_list[frame_num] = best_player_id
            else:
                consecutive_possession_count = {}

        return self.possession_list
