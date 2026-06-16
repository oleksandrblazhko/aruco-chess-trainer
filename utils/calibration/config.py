import argparse

# ---------- Configuration ----------
CHESSBOARD_SIZE = (9, 6)
SQUARE_SIZE = 0.025
SAMPLES_NEEDED = 20
CALIBRATION_FILE = "camera_ext.json"

CAPTURE_DELAY = 2.0
MIN_CENTER_SHIFT = 40.0  # pixels

def get_args():
    parser = argparse.ArgumentParser(
        description="Camera Calibration using Chessboard"
    )

    parser.add_argument(
        "--cam",
        type=int,
        default=0,
        help="Camera index"
    )

    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Capture width"
    )

    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Capture height"
    )

    parser.add_argument(
        "--file",
        type=str,
        default=CALIBRATION_FILE,
        help="Output calibration file"
    )

    return parser.parse_args()
