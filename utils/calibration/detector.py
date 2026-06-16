import cv2
import numpy as np

def find_corners(gray_image, chessboard_size):
    """
    Finds chessboard corners in a grayscale image.
    Tries with findChessboardCornersSB first, falls back to findChessboardCorners.
    """
    try:
        # New detector
        corners = cv2.findChessboardCornersSB(
            gray_image,
            chessboard_size
        )

        if isinstance(corners, tuple):
            found, corners = corners
        else:
            found = corners is not None

    except Exception:
        # Old detector
        found, corners = cv2.findChessboardCorners(
            gray_image,
            chessboard_size
        )

    if found:
        # Refine corner positions
        criteria = (
            cv2.TERM_CRITERIA_EPS +
            cv2.TERM_CRITERIA_MAX_ITER,
            30,
            0.001
        )
        corners_subpix = cv2.cornerSubPix(
            gray_image,
            corners,
            (11, 11),
            (-1, -1),
            criteria
        )
        return True, corners_subpix

    return False, None
