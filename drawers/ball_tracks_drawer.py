from .utils import draw_triangle

class BallTracksDrawer:
    """
    A drawer class responsible for drawing ball tracks on video frames.

    Attributes:
        ball_pointer_color (tuple): The color used to draw the ball pointers (in BGR format).
    """

    def __init__(self):
        """
        Initialize the BallTracksDrawer instance with default settings.
        """
        self.ball_pointer_color = (0, 255, 0)

    def draw(self, video_frames, ball_object):
        """
        Draws ball pointers on each video frame based on the Ball object's tracking data.

        Args:
            video_frames (list): A list of video frames (as NumPy arrays or image objects).
            ball_object (Ball): A Ball instance containing bounding boxes per frame.

        Returns:
            list: A list of processed video frames with drawn ball pointers.
        """
        output_video_frames = []

        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            # Skip if the frame doesn't have ball data
            if frame_num not in ball_object.bbox_per_frame:
                output_video_frames.append(frame)
                continue

            bbox = ball_object.bbox_per_frame[frame_num]

            if bbox is not None:
                frame = draw_triangle(frame, bbox, self.ball_pointer_color)

            output_video_frames.append(frame)

        return output_video_frames
