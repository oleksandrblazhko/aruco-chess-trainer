import cv2
import numpy as np
from . import config
from .camera import CameraManager
from .detector import find_corners
from .sample_collector import SampleCollector
from .calibrator import (
    perform_calibration,
    calculate_reprojection_error,
    save_calibration_data
)
from .ui import (
    draw_text,
    draw_corners,
    show_frame,
    destroy_windows,
    wait_key,
)
from .quality_analyzer import QualityAnalyzer

def run_validation(args, mtx, dist):
    print("\n=== VALIDATION PHASE ===")
    
    try:
        cam_manager = CameraManager(args.cam, args.width, args.height)
    except IOError as e:
        print(e)
        return

    locations = {
        "Center": "Place the board in the center of the view and press any key.",
        "Top-Left": "Place the board in the top-left corner and press any key.",
        "Top-Right": "Place the board in the top-right corner and press any key.",
        "Bottom-Left": "Place the board in the bottom-left corner and press any key.",
        "Bottom-Right": "Place the board in the bottom-right corner and press any key.",
    }
    
    validation_errors = {}

    objp = np.zeros(
        (config.CHESSBOARD_SIZE[0] * config.CHESSBOARD_SIZE[1], 3),
        np.float32
    )
    objp[:, :2] = np.mgrid[
        0:config.CHESSBOARD_SIZE[0],
        0:config.CHESSBOARD_SIZE[1]
    ].T.reshape(-1, 2)
    objp *= config.SQUARE_SIZE

    for name, instruction in locations.items():
        print(f"\n{instruction}")
        while True:
            frame = cam_manager.get_frame()
            if frame is None:
                print("Frame capture error")
                cam_manager.release()
                destroy_windows()
                return

            draw_text(frame, instruction, (20, 40), color=(0, 255, 255))
            show_frame(frame, window_name="Validation")

            key = wait_key(100)
            if key != -1:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                found, corners = find_corners(gray, config.CHESSBOARD_SIZE)
                if found:
                    _, rvecs, tvecs, _ = cv2.solvePnPRansac(objp, corners, mtx, dist)
                    
                    projected, _ = cv2.projectPoints(
                        objp,
                        rvecs,
                        tvecs,
                        mtx,
                        dist
                    )
                    error = cv2.norm(
                        corners,
                        projected,
                        cv2.NORM_L2
                    ) / len(projected)
                    validation_errors[name] = error
                    print(f"{name} error: {error:.4f} px")
                    break
                else:
                    print("Could not find chessboard. Please try again.")

    cam_manager.release()
    destroy_windows()

    print("\n=== VALIDATION REPORT ===")
    for name, error in validation_errors.items():
        print(f"  {name:<15}: {error:.4f} px")


def main():
    args = config.get_args()
    
    try:
        cam_manager = CameraManager(args.cam, args.width, args.height)
    except IOError as e:
        print(e)
        return

    collector = SampleCollector(config.CHESSBOARD_SIZE, config.SQUARE_SIZE)
    analyzer = QualityAnalyzer(args.width, args.height)

    print("\n=== Camera Calibration ===")
    print(f"Chessboard: {config.CHESSBOARD_SIZE[0]}x{config.CHESSBOARD_SIZE[1]}")
    print(f"Need {config.SAMPLES_NEEDED} samples")
    print("Move chessboard around the image")
    print("Press q to quit\n")

    image_size = None
    status_text = "No chessboard"

    while collector.get_sample_count() < config.SAMPLES_NEEDED:
        frame = cam_manager.get_frame()
        if frame is None:
            print("Frame capture error")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if image_size is None:
            image_size = gray.shape[::-1]

        found, corners = find_corners(gray, config.CHESSBOARD_SIZE)

        if found:
            draw_corners(frame, config.CHESSBOARD_SIZE, corners, found)
            captured, status_text = collector.add_sample(
                corners, config.MIN_CENTER_SHIFT, config.CAPTURE_DELAY
            )
            if captured:
                analyzer.add_sample(corners)
        else:
            status_text = "No chessboard"
        
        draw_text(
            frame,
            f"Samples: {collector.get_sample_count()}/{config.SAMPLES_NEEDED}",
            (20, 40)
        )
        draw_text(
            frame,
            status_text,
            (20, 80),
            color=(0, 0, 255),
            font_scale=0.8
        )
        draw_text(
            frame,
            f"Grid Visited: {len(analyzer.grid_visits)}/{analyzer.grid_size[0] * analyzer.grid_size[1]}",
            (20, 120),
            color=(255, 0, 0),
            font_scale=0.8
        )
        show_frame(frame)

        key = wait_key()
        if key & 0xFF == ord("q"):
            print("Cancelled")
            cam_manager.release()
            destroy_windows()
            return
    
    cam_manager.release()
    destroy_windows()

    if collector.get_sample_count() < 10:
        print("Too few samples")
        return

    objpoints, imgpoints = collector.get_points()

    print("\nPerforming calibration...")
    rms, mtx, dist, rvecs, tvecs = perform_calibration(
        objpoints, imgpoints, image_size
    )

    if mtx is None:
        return

    reprojection_error = calculate_reprojection_error(
        objpoints, imgpoints, rvecs, tvecs, mtx, dist
    )

    analyzer.generate_report(reprojection_error)

    save_calibration_data(args.file, mtx, dist)
    print(f"\nSaved: {args.file}")
    
    run_validation(args, mtx, dist)


if __name__ == "__main__":
    main()
