import argparse
import json
import os

# --- Global Constants ---
MARKER_TIMEOUT = 0.2  # seconds
AUDIO_GRACE_PERIOD = 0.5 # seconds

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", default="127.0.0.1", help="The ip to listen on")
    parser.add_argument("--port", type=int, default=9000, help="The port to listen on")
    parser.add_argument("--width", type=int, default=1280, help="Client window width")
    parser.add_argument("--height", type=int, default=720, help="Client window height")
    parser.add_argument("--source-width", type=int, default=640, help="Source (server) camera width")
    parser.add_argument("--source-height", type=int, default=480, help="Source (server) camera height")
    parser.add_argument("--marker-size", type=float, default=0.010, help="Physical size of the markers in meters")
    parser.add_argument("--proximity-gap", type=float, default=0.015, help="Desired visual gap between markers for proximity alert (in meters)")
    parser.add_argument("--no-text", action="store_true", help="Disable drawing text for each marker to improve performance")
    parser.add_argument("--no-proximity-check", action="store_true", help="Disable proximity check to improve performance")
    return parser.parse_args()

def load_objects_config(file_path="objects.json"):
    try:
        script_dir = os.path.dirname(__file__)
        objects_json_path = os.path.join(script_dir, file_path) # Look in the same directory
        with open(objects_json_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        audio_base_dir = config.get("audio_directory", "audio")
        objects_data = {obj["marker_id"]: obj for obj in config.get("objects", [])}
        
        control_marker_id = None
        boundary_ids = set()
        for obj in config.get("objects", []):
            if obj.get("obj_type") == "control":
                control_marker_id = obj["marker_id"]
            if obj.get("obj_type") == "border":
                boundary_ids.add(obj["marker_id"])

        print(f"Loaded {len(objects_data)} objects from {file_path}. Control marker ID: {control_marker_id}. Audio directory: {audio_base_dir}")
        print(f"Found {len(boundary_ids)} boundary markers: {boundary_ids}")
        return objects_data, control_marker_id, audio_base_dir, boundary_ids

    except FileNotFoundError:
        print(f"Error: {file_path} not found.")
        return {}, None, "audio", set()
    except json.JSONDecodeError:
        print(f"Error: Could not decode {file_path}. Check for valid JSON format.")
        return {}, None, "audio", set()
