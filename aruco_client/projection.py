import numpy as np
import cv2

def project_marker_to_board(marker, calibration, marker_size):
    """
    Projects a marker's 3D corners from camera-space into the 2D board-grid space.

    This uses the 3D transformation (rotation and translation) calculated during
    calibration to correctly place the marker's contour on the grid, regardless 
    of its height or orientation.

    Returns:
        An array of 4 (x_grid, y_grid) corner points or None.
    """
    # 1. Check if 3D calibration is complete
    if (calibration.cam_to_board_rotation is None or
        calibration.cam_to_board_translation is None or
        calibration.board_width_m == 0 or
        calibration.board_height_m == 0):
        return None

    # 2. Define marker corners in its own local 3D space (centered at origin)
    #    The Y-axis points DOWN in the standard ArUco coordinate system.
    half_size = marker_size / 2.0
    local_corners = np.array([
        [-half_size, -half_size, 0],  # Top-left
        [ half_size, -half_size, 0],  # Top-right
        [ half_size,  half_size, 0],  # Bottom-right
        [-half_size,  half_size, 0]   # Bottom-left
    ], dtype=np.float32)

    # 3. Transform local marker corners to camera space
    # Get rotation matrix from the marker's rotation vector
    r_matrix, _ = cv2.Rodrigues(marker.rvec)
    
    # Transform each local corner to camera space: P_cam = R_marker * P_local + T_marker
    cam_space_corners = (r_matrix @ local_corners.T).T + marker.tvec

    # 4. Transform 3D camera-space corners to 3D board-space
    # P_board = R_cam_to_board * P_cam + T_cam_to_board
    board_space_corners = (calibration.cam_to_board_rotation @ cam_space_corners.T).T + \
                          calibration.cam_to_board_translation
    
    # 5. Convert metric board coordinates to grid (0-8) coordinates
    projected_grid_points = []
    for point in board_space_corners:
        x_board_m, y_board_m, _ = point
        
        x_grid = (x_board_m / calibration.board_width_m) * 8.0
        # Invert Y-axis: board's 3D Y is "up", grid's 2D Y is "down"
        y_grid = 8.0 - ((y_board_m / calibration.board_height_m) * 8.0)
        projected_grid_points.append([x_grid, y_grid])

    return np.array(projected_grid_points, dtype=np.float32)