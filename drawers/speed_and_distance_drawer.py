import cv2

class SpeedAndDistanceDrawer():
    def __init__(self):
        pass

    def draw(self, video_frames, players, player_distances_per_frame, player_speed_per_frame):
        output_video_frames = []
        total_distances = {}

        for frame_num, (frame, player_distance, player_speed) in enumerate(
                zip(video_frames, player_distances_per_frame, player_speed_per_frame)
        ):
            output_frame = frame.copy()

            # Update total distance
            for player_id, distance in player_distance.items():
                total_distances[player_id] = total_distances.get(player_id, 0) + distance

            # Loop over all players
            for player_id, player_obj in players.items():
                bbox = player_obj.bboxs_per_frame.get(frame_num)
                if bbox is None:
                    continue  # Player not visible in this frame

                x1, y1, x2, y2 = bbox
                position = [int((x1 + x2) / 2), int(y2) + 40]

                distance = total_distances.get(player_id)
                speed = player_speed.get(player_id)

                if speed is not None:
                    cv2.putText(output_frame, f"{speed:.2f} km/h", position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0),
                                2)
                if distance is not None:
                    cv2.putText(output_frame, f"{distance:.2f} m", (position[0], position[1] + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

            output_video_frames.append(output_frame)

        return output_video_frames
