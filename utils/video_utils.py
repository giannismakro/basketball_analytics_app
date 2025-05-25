"""
A module for reading and writing video files.

This module provides utility functions to load video frames into memory and save
processed frames back to video files, with support for common video formats.
"""

import cv2
import os

def read_video(video_path):
    """
    Read all frames from a video file into memory.

    Args:
        video_path (str): Path to the input video file.

    Returns:
        list: List of video frames as numpy arrays.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(video_path)
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    return frames

def save_video(ouput_video_frames,output_video_path):
    """
    Save a sequence of frames as a video file.

    Creates the necessary directories if they don't exist and writes frames using XVID codec.

    Args:
        ouput_video_frames (list): List of frames to save.
        output_video_path (str): Path where the video should be saved.
    """
    if output_video_path is None:
        print("No output video path provided.")

    # If folder doesn't exist, create it
    if not os.path.exists(os.path.dirname(output_video_path)):
        print("Creating folder for output video.")
        os.makedirs(os.path.dirname(output_video_path))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, 24, (ouput_video_frames[0].shape[1], ouput_video_frames[0].shape[0]))
    for frame in ouput_video_frames:
        out.write(frame)
    out.release()


import cv2

def check_for_shots(video_frames, ball_object, baskets):
    output_video_frames = []
    for frame_idx in range(len(video_frames)):
        frame = video_frames[frame_idx]
        frame = check_for_shot(ball_object, baskets, frame_idx, frame)
        output_video_frames.append(frame)
    return  output_video_frames


def check_for_shot(ball, hoops, frame_idx, video_frame):
    shot_display_duration = 30  # frames to show text

    if ball.shot_display_frames_remaining > 0:
        frame_height, frame_width = video_frame.shape[:2]
        text_y = int(frame_height * 0.50)  # 25% down the screen
        cv2.putText(video_frame, ball.last_shot_result, (100, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 5.0,
                    (0, 255, 0) if "Made" in ball.last_shot_result else (0, 0, 255),
                    4)
        ball.shot_display_frames_remaining -= 1
        return video_frame

    for hoop in hoops:
        if not hoop.is_collision(ball, frame_idx):
            continue

        prev_center = ball.get_center(frame_idx - 1)
        curr_center = ball.get_center(frame_idx)
        if not prev_center or not curr_center:
            continue

        if not hoop.is_collision_with_ellipse(ball, frame_idx):
            result = "Shot Missed"
            color = (0, 0, 255)
            if ball.last_owner:
                ball.last_owner.stats['missed'] += 1
        else:
            result = "Shot Made"
            color = (0, 255, 0)
            if ball.last_owner:
                ball.last_owner.stats['made'] += 1
                if hasattr(ball.last_owner, 'team'):
                    ball.last_owner.team.score += 2

        # Store for display
        ball.last_shot_result = result
        ball.shot_display_frames_remaining = shot_display_duration
        print(f"Shot detected at frame {frame_idx}: {result}")
        frame_height, frame_width = video_frame.shape[:2]
        text_y = int(frame_height * 0.50)  # 25% down the screen
        cv2.putText(video_frame, result, (100, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                    5.0, color, 4)
        break

    return video_frame
