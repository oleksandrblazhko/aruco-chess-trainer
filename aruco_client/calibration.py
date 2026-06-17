import time
import numpy as np

class Calibration:
    def __init__(self, duration=3.0):
        self.duration = duration
        self.is_calibrating = False
        self.data = {}
        
        # --- Results ---
        # 2D screen projection
        self.table_zone = []
        self.pixels_per_meter = 0
        self.avg_boundary_z = 0
        
        # 3D transformation to board space
        self.cam_to_board_rotation = None
        self.cam_to_board_translation = None
        self.board_width_m = 0
        self.board_height_m = 0
        self.boundary_marker_tvecs = {}

    def start(self):
        """Starts the calibration process."""
        print("Starting calibration...")
        self.is_calibrating = True
        self.start_time = time.time()
        self.data = {}
        self.table_zone = []
        self.pixels_per_meter = 0
        self.avg_boundary_z = 0
        self.cam_to_board_rotation = None
        self.cam_to_board_translation = None
        self.board_width_m = 0
        self.board_height_m = 0
        self.boundary_marker_tvecs = {}


    def is_running(self):
        return self.is_calibrating

    def update(self, markers, boundary_ids):
        """Collects data for boundary markers during calibration."""
        if not self.is_calibrating:
            return

        elapsed = time.time() - self.start_time
        if elapsed > self.duration:
            # This calls the lambda in main.py, which only takes one argument
            self.finish(markers) 
            return

        for marker_id in boundary_ids:
            if marker_id in markers:
                marker = markers[marker_id]
                if marker_id not in self.data:
                    self.data[marker_id] = {'centers': [], 'widths': [], 'tvecs': []}
                self.data[marker_id]['centers'].append(marker.get_center())
                self.data[marker_id]['widths'].append(marker.get_pixel_width())
                self.data[marker_id]['tvecs'].append(marker.get_pos_3d())

    def finish(self, markers, marker_size, boundary_ids):
        """Calculates the final calibration results, including 3D transform."""
        self.is_calibrating = False
        print("Calibration finished. Calculating average positions...")
        
        # Calculate average positions for all visible boundary markers first
        avg_marker_data = {}
        for marker_id in boundary_ids:
            if marker_id in self.data and len(self.data[marker_id]['centers']) > 0:
                avg_marker_data[marker_id] = {
                    'center': np.mean(self.data[marker_id]['centers'], axis=0),
                    'tvec': np.mean(self.data[marker_id]['tvecs'], axis=0),
                    'widths': self.data[marker_id]['widths']
                }

        if len(avg_marker_data) != 4:
            print("Error: Could not define table zone. Not all 4 boundary markers were visible.")
            self.data = {}
            return

        # Sort marker IDs based on the spatial position of their average center point
        marker_centers = {mid: data['center'] for mid, data in avg_marker_data.items()}
        
        points = np.array(list(marker_centers.values()))
        centroid = np.mean(points, axis=0)
        
        # Create a list of tuples (angle, marker_id) for sorting
        id_angle_list = []
        for marker_id, center in marker_centers.items():
            angle = np.arctan2(center[1] - centroid[1], center[0] - centroid[0])
            id_angle_list.append((angle, marker_id))
            
        # Sort by angle
        id_angle_list.sort()
        sorted_ids = [marker_id for angle, marker_id in id_angle_list]

        # Now, build the results using the correctly sorted IDs
        temp_zone = []
        visible_marker_widths = []
        avg_tvecs = {}
        for marker_id in sorted_ids:
            data = avg_marker_data[marker_id]
            temp_zone.append(data['center'].astype(int))
            avg_tvecs[marker_id] = data['tvec']
            visible_marker_widths.extend(data['widths'])
        
        if len(temp_zone) == 4:
            self.table_zone = temp_zone
            self.boundary_marker_tvecs = avg_tvecs
            
            # --- Calculate Average Boundary Z-depth ---
            all_tvecs = np.array(list(avg_tvecs.values()))
            self.avg_boundary_z = np.mean(all_tvecs[:, 2])
            print(f"Average boundary plane depth (tz): {self.avg_boundary_z:.3f}m")

            print("Calibration successful: Table zone and 3D positions defined.")
            
            # --- Calculate 3D Transformation ---
            self._calculate_3d_transform(sorted_ids, avg_tvecs)

            # --- Calculate 2D metrics (can be removed later) ---
            if visible_marker_widths and marker_size > 0:
                avg_pixel_width = np.mean(visible_marker_widths)
                self.pixels_per_meter = avg_pixel_width / marker_size
                print(f"Pixels/meter ratio: {self.pixels_per_meter:.2f}")
        else:
            print("Error: Could not define table zone. Not all boundary markers were visible.")
        
        self.data = {}  # Clear data for next time

    def _calculate_3d_transform(self, sorted_ids, avg_tvecs):
        """Calculates the rotation and translation from camera space to board space."""
        try:
            # We assume a standard order: TL, TR, BR, BL
            tl_id, tr_id, br_id, bl_id = sorted_ids
            
            # Define board coordinate system based on bottom-left, bottom-right, top-left
            origin_tvec = avg_tvecs[bl_id] # Bottom-Left is origin (0,0,0)
            x_end_tvec = avg_tvecs[br_id]  # Bottom-Right defines X axis
            y_end_tvec = avg_tvecs[tl_id]  # Top-Left defines Y axis

            # Raw axes from markers
            x_axis_raw = x_end_tvec - origin_tvec
            y_axis_raw = y_end_tvec - origin_tvec

            # Store physical board dimensions
            self.board_width_m = np.linalg.norm(x_axis_raw)
            self.board_height_m = np.linalg.norm(y_axis_raw)
            
            # Create an orthonormal basis
            x_axis = x_axis_raw / self.board_width_m
            # The z-axis is orthogonal to the board plane
            z_axis = np.cross(x_axis, y_axis_raw)
            z_axis = z_axis / np.linalg.norm(z_axis)
            # The y-axis must be orthogonal to both X and Z
            y_axis = np.cross(z_axis, x_axis)

            # This is the rotation from board space to camera space
            board_to_cam_rotation = np.array([x_axis, y_axis, z_axis]).T
            # The inverse rotation (cam->board) is the transpose
            self.cam_to_board_rotation = board_to_cam_rotation.T
            
            # The translation is the negated rotation of the origin's tvec
            self.cam_to_board_translation = -self.cam_to_board_rotation @ origin_tvec
            
            print(f"3D transform calculated. Board size: {self.board_width_m*100:.1f}cm x {self.board_height_m*100:.1f}cm")

        except (KeyError, IndexError) as e:
            print(f"Error calculating 3D transform: Not all required boundary markers found. {e}")
