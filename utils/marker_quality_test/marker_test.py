import cv2
import time
import numpy as np
import os

CAM_INDEX = 0
MARKER_SIZE = 0.035
TEST_TIME = 15

if os.path.exists('camera_ext.json'):
    with open('camera_ext.json', 'r') as json_file:
        camera_data = json.load(json_file)
    camera_matrix = np.array(camera_data["mtx"])
    dist_coeffs = np.array(camera_data["dist"])
    print("INFO: Camera calibration loaded from camera_ext.json")



def run_test(dictionary, name):

    print(f"\n====================")
    print(f"TEST: {name}")
    print(f"====================")

    aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary)
    params = cv2.aruco.DetectorParameters()

    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    tvec_history = []
    rvec_history = []
    times = []
    lost = 0

    start = time.time()

    # Use globally defined camera_matrix and dist_coeffs

    # If camera_ext.json was not found, initialize with default values
    if 'camera_matrix' not in globals() or camera_matrix is None:
        fx, fy = 640, 480
        cx, cy = fx / 2, fy / 2
        camera_matrix = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float32)
        dist_coeffs = np.zeros((5, 1))
        print("WARNING: camera_ext.json not found or loaded, using default camera parameters.")

    while time.time() - start < TEST_TIME:

        t0 = time.time()
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        corners, ids, _ = cv2.aruco.detectMarkers(
            gray,
            aruco_dict,
            parameters=params
        )

        if ids is not None:

            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners,
                MARKER_SIZE,
                camera_matrix,
                dist
            )

            # draw axes
            for i in range(len(rvecs)):
                cv2.drawFrameAxes(
                    frame,
                    camera_matrix,
                    dist,
                    rvecs[i],
                    tvecs[i],
                    0.03
                )

            tvec_history.append(tvecs[0][0])
            rvec_history.append(rvecs[0][0])

        else:
            lost += 1

        # FPS (реальний)
        fps = 1.0 / (time.time() - t0 + 1e-6)

        cv2.putText(frame, f"TEST: {name}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.putText(frame, f"FPS: {int(fps)}", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.putText(frame, f"Lost: {lost}", (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("ArUco Experiment", frame)

        times.append(time.time() - t0)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Stopped by user")
            cap.release()
            cv2.destroyAllWindows()
            return None

    cap.release()
    cv2.destroyAllWindows()

    # -------- STATS --------

    def stats(data):
        if len(data) == 0:
            return np.zeros(3)
        data = np.array(data)
        return np.std(data, axis=0)

    print("\nRESULTS")
    print("FPS:", len(times) / TEST_TIME)
    print("Lost frames:", lost)
    print("Tvec std:", stats(tvec_history))
    print("Rvec std:", stats(rvec_history))

    return {
        "fps": len(times) / TEST_TIME,
        "tstd": stats(tvec_history),
        "rstd": stats(rvec_history),
        "lost": lost
    }


if __name__ == "__main__":

    res_4x4 = run_test(cv2.aruco.DICT_4X4_50, "DICT_4X4_50")

    print("\n====================")
    print("FIRST TEST FINISHED")
    print("====================")
    input("Fix setup if needed, then press ENTER for next test...")

    res_5x5 = run_test(cv2.aruco.DICT_ARUCO_ORIGINAL, "DICT_ARUCO_ORIGINAL")

    print("\n====================")
    print("FINAL COMPARISON")
    print("====================")

    print("\n4x4 FPS:", res_4x4["fps"])
    print("5x5 FPS:", res_5x5["fps"])

    print("\nTvec stability")
    print("4x4:", res_4x4["tstd"])
    print("5x5:", res_5x5["tstd"])

    print("\nRvec stability")
    print("4x4:", res_4x4["rstd"])
    print("5x5:", res_5x5["rstd"])