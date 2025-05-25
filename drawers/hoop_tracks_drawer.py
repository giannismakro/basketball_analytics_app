from .utils import draw_ellipse

class HoopTracksDrawer:
    """
    A class responsible for drawing hoop locations across video frames.

    Attributes:
        left_hoop_color (tuple): RGB color for the left team's hoop.
        right_hoop_color (tuple): RGB color for the right team's hoop.
    """
    def __init__(self, left_hoop_color=(0, 255, 0), right_hoop_color=(255, 0, 0)):
        """
        Initialize the drawer with colors for each hoop.

        Args:
            left_hoop_color (tuple): RGB color for left hoop (default green).
            right_hoop_color (tuple): RGB color for right hoop (default red).
        """
        self.left_hoop_color = left_hoop_color
        self.right_hoop_color = right_hoop_color

    def draw(self, video_frames, left_hoop, right_hoop):
        """
        Draws the two hoops on each frame using their bounding boxes.

        Args:
            video_frames (list): A list of video frames (NumPy arrays).
            left_hoop (Hoop): Hoop object representing the left hoop.
            right_hoop (Hoop): Hoop object representing the right hoop.

        Returns:
            list: List of annotated frames.
        """
        output_frames = []

        for frame_idx, frame in enumerate(video_frames):
            frame = frame.copy()

            # Draw left hoop if available
            left_bbox = left_hoop.get_bbox(frame_idx)
            if left_bbox is not None:
                #print(f"Left hoop bbox detected at frame {frame_idx}: {left_bbox}")
                frame = draw_ellipse(frame, left_bbox, self.left_hoop_color, label="Left Hoop")

            # Draw right hoop if available
            right_bbox = right_hoop.get_bbox(frame_idx)
            if right_bbox is not None:
                #print(f"Right hoop bbox detected at frame {frame_idx}: {right_bbox}")

                frame = draw_ellipse(frame, right_bbox, self.right_hoop_color, label="Right Hoop")

            output_frames.append(frame)

        return output_frames
