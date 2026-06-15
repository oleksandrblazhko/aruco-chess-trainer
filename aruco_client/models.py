import time
import numpy as np

class Marker:
    def __init__(self, marker_id, tx, ty, tz, roll, pitch, yaw, rvec, corners):
        self.marker_id = marker_id
        self.update(tx, ty, tz, roll, pitch, yaw, rvec, corners)

    def update(self, tx, ty, tz, roll, pitch, yaw, rvec, corners):
        # Keep for backward compatibility and display
        self.tx = tx
        self.ty = ty
        self.tz = tz
        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw
        
        # Store raw vectors for 3D calculations
        self.rvec = np.array(rvec, dtype=np.float32)
        self.tvec = np.array([tx, ty, tz], dtype=np.float32)

        self.corners = np.array(corners, dtype=np.int32).reshape((4, 2))
        self.timestamp = time.time()

    def get_center(self):
        return np.mean(self.corners, axis=0).astype(int)

    def get_pos_3d(self):
        """Returns the translation vector (tvec)."""
        return self.tvec
    
    def get_pixel_width(self):
        """Calculates the average width of the marker in pixels."""
        d1 = np.linalg.norm(self.corners[0] - self.corners[1])
        d2 = np.linalg.norm(self.corners[2] - self.corners[3])
        return (d1 + d2) / 2
