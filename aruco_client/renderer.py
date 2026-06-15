import cv2
import numpy as np
from .projection import project_marker_to_board

class Renderer:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.frame = np.full((height, width, 3), 255, dtype=np.uint8)
        self.board_to_image_matrix = None

    def clear_frame(self):
        """Resets the frame to a blank white state."""
        self.frame[:] = 255

    def draw_table_grid(self, table_zone, pixels_per_meter):
        """Draws the calibrated 8x8 grid and stores the transformation matrix."""
        if not table_zone or pixels_per_meter == 0:
            return

        inset_m = 0.005  # 5mm
        inset_px = inset_m * pixels_per_meter

        dst_pts = np.float32(table_zone)
        centroid = np.mean(dst_pts, axis=0)
        
        inset_dst_pts = []
        for pt in dst_pts:
            vector = centroid - pt
            normalized_vector = vector / np.linalg.norm(vector)
            inset_dst_pts.append(pt + normalized_vector * inset_px)
        
        inset_dst_pts = np.array(inset_dst_pts, dtype=np.float32)

        grid_src_pts = np.float32([[0, 0], [8, 0], [8, 8], [0, 8]])
        
        # Calculate and store the board-to-image transformation matrix
        self.board_to_image_matrix = cv2.getPerspectiveTransform(grid_src_pts, inset_dst_pts)

        grid_color = (0, 255, 0)
        for i in range(1, 8):
            # Vertical lines
            line_src_v = np.float32([[[i, 0]], [[i, 8]]])
            line_dst_v = cv2.perspectiveTransform(line_src_v, self.board_to_image_matrix)
            pt1_v, pt2_v = tuple(line_dst_v[0][0].astype(int)), tuple(line_dst_v[1][0].astype(int))
            cv2.line(self.frame, pt1_v, pt2_v, grid_color, 1)

            # Horizontal lines
            line_src_h = np.float32([[[0, i]], [[8, i]]])
            line_dst_h = cv2.perspectiveTransform(line_src_h, self.board_to_image_matrix)
            pt1_h, pt2_h = tuple(line_dst_h[0][0].astype(int)), tuple(line_dst_h[1][0].astype(int))
            cv2.line(self.frame, pt1_h, pt2_h, grid_color, 1)

    def draw_boundary(self, boundary_markers, calibrated_zone):
        """Draws the table boundary, either live or the calibrated one."""
        if calibrated_zone:
            pts = np.array(calibrated_zone, dtype=np.int32)
            cv2.polylines(self.frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
        elif len(boundary_markers) == 4:
            sorted_ids = sorted(list(boundary_markers.keys()))
            p0, p1, p2, p3 = (boundary_markers[id].get_center() for id in sorted_ids)
            cv2.line(self.frame, tuple(p0), tuple(p1), (0, 255, 0), 2)
            cv2.line(self.frame, tuple(p1), tuple(p2), (0, 255, 0), 2)
            cv2.line(self.frame, tuple(p2), tuple(p3), (0, 255, 0), 2)
            cv2.line(self.frame, tuple(p3), tuple(p0), (0, 255, 0), 2)

    def draw_markers(self, markers, no_text=False):
        """Draws all detected markers and their information."""
        for marker_id, marker in markers.items():
            cv2.polylines(self.frame, [marker.corners], isClosed=True, color=(255, 0, 0), thickness=2)
            
            if not no_text:
                text_id = f"id={marker_id}"
                text_coords = f"({marker.tx*1000:.0f},{marker.ty*1000:.0f},{marker.tz*1000:.0f})"
                text_pos = marker.corners[0]
                cv2.putText(self.frame, text_id, (text_pos[0], text_pos[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 0), 1)
                cv2.putText(self.frame, text_coords, (text_pos[0], text_pos[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 0), 1)

    def draw_control_marker_distance(self, control_marker, distance):
        """Displays the proximity distance for the control marker."""
        if control_marker and distance != float('inf'):
            display_text = f"dist={distance*1000:.0f}mm"
            text_pos = control_marker.get_center()
            cv2.putText(self.frame, display_text, (text_pos[0] + 20, text_pos[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

    def draw_ui_info(self, camera_threshold_mm):
        """Draws static UI text on the frame."""
        cv2.putText(self.frame, "0 - calibration", (10, self.height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        if camera_threshold_mm > 0:
             cv2.putText(self.frame, f"Cam Thresh: {camera_threshold_mm:.0f}mm", (10, self.height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    def show_frame(self):
        """Displays the OpenCV frame."""
        cv2.imshow("Client", self.frame)
        return cv2.waitKey(1) & 0xFF
    
    def draw_projected_markers(self, markers, calibration, marker_size):
        # Ensure the grid and transformation matrix are ready
        if self.board_to_image_matrix is None:
            return

        for marker in markers.values():
            # Get the marker's projected corners in board space (0-8)
            board_coords = project_marker_to_board(marker, calibration, marker_size)
            if board_coords is None:
                continue

            # Transform the board corner points to image (pixel) space
            image_points = cv2.perspectiveTransform(
                np.array([board_coords]), 
                self.board_to_image_matrix
            )[0]
            
            # Draw the red projected contour
            cv2.polylines(
                self.frame, 
                [np.int32(image_points)], 
                isClosed=True, 
                color=(0, 0, 255), 
                thickness=2
            )
            
            # Optionally, add text near the first corner
            text_pos = image_points[0]
            cv2.putText(
                self.frame,
                f"id={marker.marker_id}",
                (int(text_pos[0]) + 5, int(text_pos[1]) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 0, 255),
                1
            )
