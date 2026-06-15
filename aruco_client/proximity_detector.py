import time
import numpy as np
from .config import AUDIO_GRACE_PERIOD

class ProximityDetector:
    def __init__(self, audio_manager):
        self.audio_manager = audio_manager
        
        # Camera proximity state
        self.camera_prox_last_seen = {}
        
        # Control marker proximity state
        self.locked_object_state = None
        self.last_proximity_time = 0

    def check_camera_proximity(self, markers, objects_data, camera_distance_threshold):
        """
        Manages audio feedback for markers that are too close to the camera.
        """
        if camera_distance_threshold <= 0:
            return

        current_time = time.time()
        
        # 1. Start sounds for newly close markers
        for marker_id, marker in markers.items():
            if marker_id in objects_data and objects_data[marker_id].get("obj_type") != "control":
                if marker.tz < camera_distance_threshold:
                    self.camera_prox_last_seen[marker_id] = current_time
                    obj_def = objects_data[marker_id]
                    if obj_def.get("audio_name"):
                        # This method is smart and won't restart an already playing sound
                        self.audio_manager.play_camera_prox_sound(marker_id, obj_def["audio_name"])

        # 2. Stop sounds for markers that are no longer close (after a grace period)
        for marker_id in list(self.audio_manager.camera_prox_channels.keys()):
            time_since_last_close = current_time - self.camera_prox_last_seen.get(marker_id, 0)
            if time_since_last_close > AUDIO_GRACE_PERIOD:
                self.audio_manager.stop_camera_prox_sound(marker_id)
                if marker_id in self.camera_prox_last_seen:
                    del self.camera_prox_last_seen[marker_id]

    def check_control_marker_proximity(self, control_marker, markers, objects_data, proximity_threshold):
        """
        Manages the "lock-on" audio logic for the control marker.
        Returns the closest distance for display purposes.
        """
        current_time = time.time()
        closest_dist_for_display = float('inf')

        if not control_marker:
            # If control marker is not visible, check if we need to stop audio
            if self.locked_object_state and (current_time - self.last_proximity_time > AUDIO_GRACE_PERIOD):
                self.audio_manager.stop_looping_sound()
                self.locked_object_state = None
            return closest_dist_for_display

        # --- Step 1: Check and maintain the existing lock ---
        if self.locked_object_state:
            raw_distance = np.linalg.norm(control_marker.get_pos_3d() - self.locked_object_state["saved_pos"])
            closest_dist_for_display = raw_distance

            # Smoothing
            distances = self.locked_object_state["recent_distances"]
            distances.append(raw_distance)
            if len(distances) > 5: distances.pop(0)
            smoothed_distance = np.mean(distances)

            break_threshold = proximity_threshold * 1.5  # Hysteresis

            if smoothed_distance >= break_threshold:
                self.locked_object_state = None
                # Don't stop audio immediately, wait for grace period
            else:
                self.audio_manager.play_looping_sound(self.locked_object_state["audio_name"])
                self.last_proximity_time = current_time
        
        # --- Step 2: If no lock, search for a new one ---
        else:
            for obj_def in objects_data.values():
                obj_marker_id = obj_def["marker_id"]
                if obj_def.get("obj_type") != "control" and obj_marker_id in markers:
                    obj_marker = markers[obj_marker_id]
                    distance = np.linalg.norm(control_marker.get_pos_3d() - obj_marker.get_pos_3d())

                    if distance < proximity_threshold:
                        self.locked_object_state = {
                            "marker_id": obj_marker_id,
                            "saved_pos": obj_marker.get_pos_3d(),
                            "audio_name": obj_def["audio_name"],
                            "recent_distances": [distance]
                        }
                        self.audio_manager.play_looping_sound(self.locked_object_state["audio_name"])
                        self.last_proximity_time = current_time
                        closest_dist_for_display = distance
                        break
        
        # --- Step 3: Handle audio stop grace period ---
        if not self.locked_object_state and (current_time - self.last_proximity_time > AUDIO_GRACE_PERIOD):
            self.audio_manager.stop_looping_sound()
            
        # --- Step 4: Update distance for visual display even if not locked
        if not self.locked_object_state:
            for obj_def in objects_data.values():
                if obj_def.get("obj_type") != "control" and obj_def["marker_id"] in markers:
                    obj_marker = markers[obj_def["marker_id"]]
                    distance = np.linalg.norm(control_marker.get_pos_3d() - obj_marker.get_pos_3d())
                    if distance < (1.5 * proximity_threshold):
                        closest_dist_for_display = min(distance, closest_dist_for_display)

        return closest_dist_for_display
