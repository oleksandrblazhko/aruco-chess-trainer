# Обробка параметрів запуску.

import argparse
import json

def load_valid_marker_ids(filepath):
    """Loads the valid marker IDs from the client's objects.json file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        valid_ids = {obj['marker_id'] for obj in data.get('objects', [])}
        print(f"Loaded {len(valid_ids)} valid marker IDs.")
        return valid_ids
    except FileNotFoundError:
        print(f"Warning: Objects file not found at '{filepath}'. No ID whitelist will be used.")
        return None
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Warning: Could not parse '{filepath}'. Invalid format: {e}. No ID whitelist will be used.")
        return None

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--cam", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)

    parser.add_argument(
        "--objects",
        type=str,
        default="../aruco_client/objects.json",
        help="Path to the objects.json file to create a marker ID whitelist."
    )

    parser.add_argument(
        "--pattern",
        type=int,
        default=1
    )

    parser.add_argument(
        "--size",
        type=float,
        default=0.010
    )

    parser.add_argument(
        "--flip",
        action="store_true",
        default=True
    )

    parser.add_argument(
        "--win",
        action="store_true",
        default=True
    )

    return parser.parse_args()
