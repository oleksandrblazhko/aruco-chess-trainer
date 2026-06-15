import time
import numpy as np

class Calibration:
    def __init__(self, duration=3.0):
        self.duration = duration
        self.is_calibrating = False
        self.data = {}
        
        # Results
        self.table_zone = []
        self.pixels_per_meter = 0
        self.avg_boundary_z = 0

    def start(self):
        """Starts the calibration process."""
        print("Starting calibration...")
        self.is_calibrating = True
        self.start_time = time.time()
        self.data = {}
        self.table_zone = []
        self.pixels_per_meter = 0
        self.avg_boundary_z = 0

    def is_running(self):
        return self.is_calibrating

    def update(self, markers, boundary_ids):
        """Collects data for boundary markers during calibration."""
        if not self.is_calibrating:
            return

        elapsed = time.time() - self.start_time
        if elapsed > self.duration:
            self.finish(markers) # The lambda in main.py will pass the other args
            return

        for marker_id in boundary_ids:
            if marker_id in markers:
                if marker_id not in self.data:
                    self.data[marker_id] = {'centers': [], 'widths': [], 'zs': []}
                self.data[marker_id]['centers'].append(markers[marker_id].get_center())
                self.data[marker_id]['widths'].append(markers[marker_id].get_pixel_width())
                self.data[marker_id]['zs'].append(markers[marker_id].tz)

    def finish(self, markers, marker_size, boundary_ids):
        """Calculates the final calibration results."""
        self.is_calibrating = False
        print("Calibration finished. Calculating average positions...")
        
        temp_zone = []
        visible_marker_widths = []
        visible_marker_zs = []
        sorted_ids = sorted(list(boundary_ids))

        for marker_id in sorted_ids:
            if marker_id in self.data and len(self.data[marker_id]['centers']) > 0:
                avg_point = np.mean(self.data[marker_id]['centers'], axis=0).astype(int)
                temp_zone.append(avg_point)
                visible_marker_widths.extend(self.data[marker_id]['widths'])
                visible_marker_zs.extend(self.data[marker_id]['zs'])
            else:
                print(f"Warning: No data collected for boundary marker {marker_id}.")
        
        if len(temp_zone) == 4:
            self.table_zone = temp_zone
            print("Calibration successful: Table zone defined.")

            if visible_marker_zs:
                self.avg_boundary_z = np.mean(visible_marker_zs)
                print(f"Calibrated Avg Boundary Z: {self.avg_boundary_z:.4f}m")
            else:
                print("Warning: Could not calculate calibrated average boundary Z.")

            if visible_marker_widths and marker_size > 0:
                avg_pixel_width = np.mean(visible_marker_widths)
                self.pixels_per_meter = avg_pixel_width / marker_size
                print(f"Pixels/meter ratio: {self.pixels_per_meter:.2f}")
            else:
                 print("Warning: Could not calculate pixels/meter ratio.")
        else:
            print("Error: Could not define table zone. Not all boundary markers were visible.")
        
        self.data = {} # Clear data for next time
