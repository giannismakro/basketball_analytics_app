"""
A utility module providing functions for drawing shapes on video frames.

This module includes functions to draw triangles and ellipses on frames, which can be used
to represent various annotations such as player positions or ball locations in sports analysis.
"""

import cv2 
import numpy as np
import sys 
sys.path.append('../')
from utils import get_center_of_bbox, get_bbox_width, get_foot_position


def draw_triangle(frame, bbox, color):
    """
    Draws a filled triangle on the given frame at the specified bounding box location.

    Args:
        frame (numpy.ndarray): The frame on which to draw the triangle.
        bbox (tuple): A tuple representing the bounding box (x, y, width, height).
        color (tuple): The color of the triangle in BGR format.

    Returns:
        numpy.ndarray: The frame with the triangle drawn on it.
    """
    y= int(bbox[1])
    x,_ = get_center_of_bbox(bbox)

    triangle_points = np.array([
        [x,y],
        [x-10,y-20],
        [x+10,y-20],
    ])
    cv2.drawContours(frame, [triangle_points],0,color, cv2.FILLED)
    cv2.drawContours(frame, [triangle_points],0,(0,0,0), 2)

    return frame

import cv2

def draw_ellipse(frame, bbox, color, label=None):
    """
    Draws an ellipse and an optional label on the given frame at the specified bounding box location.

    Args:
        frame (numpy.ndarray): The frame on which to draw the ellipse.
        bbox (tuple): A tuple (x1, y1, x2, y2) representing the bounding box.
        color (tuple): The color of the ellipse in BGR format.
        label (str or int, optional): The label to display inside a rectangle. Defaults to None.

    Returns:
        numpy.ndarray: The frame with the ellipse and optional label drawn on it.
    """
    x1, y1, x2, y2 = map(int, bbox)
    x_center = (x1 + x2) // 2
    y2 = int(y2)
    width = max(1, (x2 - x1) // 2)  # Avoid zero width

    # Draw the ellipse
    cv2.ellipse(
        frame,
        center=(x_center, y2),
        axes=(width, int(0.35 * width)),
        angle=0.0,
        startAngle=-45,
        endAngle=235,
        color=color,
        thickness=2,
        lineType=cv2.LINE_4
    )

    # Draw label if provided
    if label is not None:
        label_str = str(label)
        font_scale = 0.6
        thickness = 2
        (text_width, text_height), _ = cv2.getTextSize(label_str, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

        padding = 5
        rect_w = text_width + 2 * padding
        rect_h = text_height + 2 * padding

        rect_x1 = x_center - rect_w // 2
        rect_y1 = y2 + 15
        rect_x2 = rect_x1 + rect_w
        rect_y2 = rect_y1 + rect_h

        # Draw filled rectangle
        cv2.rectangle(frame, (rect_x1, rect_y1), (rect_x2, rect_y2), color, cv2.FILLED)

        # Draw text
        cv2.putText(
            frame,
            label_str,
            (rect_x1 + padding, rect_y2 - padding),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),  # Black text
            thickness
        )

    return frame
