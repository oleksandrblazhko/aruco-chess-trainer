import asyncio
import time
import cv2

from .config import parse_arguments, load_objects_config, MARKER_TIMEOUT
from .models import Marker
from .audio_manager import AudioManager
from .osc_server import OSCServer
from .calibration import Calibration
from .proximity_detector import ProximityDetector
from .renderer import Renderer

async def main_loop(args, objects_data, control_marker_id, audio_base_dir, boundary_ids):
    # --- Initialization ---
    renderer = Renderer(args.width, args.height)
    audio_manager = AudioManager(audio_base_dir)
    calibration = Calibration()
    proximity_detector = ProximityDetector(audio_manager)

    markers = {}  # Main dictionary to store marker objects
    
    scale_x = args.width / args.source_width
    scale_y = args.height / args.source_height
    osc_server = OSCServer(args.ip, args.port, markers, scale_x, scale_y)
    await osc_server.start()

    proximity_threshold = args.marker_size + args.proximity_gap
    camera_distance_threshold = 0.0

    # --- Main Loop ---
    try:
        while True:
            current_time = time.time()
            renderer.clear_frame()

            # --- Prune old markers ---
            for marker_id, marker in list(markers.items()):
                if current_time - marker.timestamp > MARKER_TIMEOUT:
                    del markers[marker_id]

            # --- Calibration Logic ---
            if calibration.is_running():
                calibration.update(markers, boundary_ids)
                # The finish logic is now inside the class, check if it finished
                if not calibration.is_running():
                     # Update thresholds and ratios once calibration is done
                     if calibration.avg_boundary_z > 0:
                        camera_distance_threshold = calibration.avg_boundary_z / 2.0
            
            # --- Proximity Detection ---
            if not args.no_proximity_check:
                # 1. Camera Proximity
                proximity_detector.check_camera_proximity(markers, objects_data, camera_distance_threshold)
                
                # 2. Control Marker Proximity
                control_marker = markers.get(control_marker_id)
                closest_dist = proximity_detector.check_control_marker_proximity(
                    control_marker, markers, objects_data, proximity_threshold
                )

            # --- Visualization ---
            boundary_markers_viz = {mid: markers[mid] for mid in boundary_ids if mid in markers}
            renderer.draw_boundary(boundary_markers_viz, calibration.table_zone)
            if calibration.table_zone:
                renderer.draw_table_grid(calibration.table_zone, calibration.pixels_per_meter)
            
            # Use the original stable rendering method
            renderer.draw_markers(markers, args.no_text)

            if not args.no_proximity_check:
                renderer.draw_control_marker_distance(markers.get(control_marker_id), closest_dist)
            
            renderer.draw_ui_info(camera_distance_threshold * 1000)

            key = renderer.show_frame()

            if key == ord('q'):
                break
            elif key == ord('0'):
                if not calibration.is_running():
                    calibration.start()
                    # Pass the marker size to the finish method when it's called internally
                    calibration.finish = lambda m: Calibration.finish(calibration, m, args.marker_size, boundary_ids)
                    asyncio.create_task(audio_manager.play_calibration_beeps())
            
            await asyncio.sleep(1/60)

    finally:
        # --- Cleanup ---
        osc_server.close()
        audio_manager.stop_all_sounds()
        audio_manager.quit()
        cv2.destroyAllWindows()
        print("Client shut down gracefully.")


async def init_main():
    args = parse_arguments()
    objects_data, control_marker_id, audio_base_dir, boundary_ids = load_objects_config()

    if not objects_data:
        print("Exiting due to configuration errors.")
        return

    await main_loop(args, objects_data, control_marker_id, audio_base_dir, boundary_ids)

if __name__ == '__main__':
    try:
        asyncio.run(init_main())
    except KeyboardInterrupt:
        pass
