import cv2

def draw_text(frame, text, position, color=(0, 255, 0), font_scale=1, thickness=2):
    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        thickness
    )

def draw_corners(frame, chessboard_size, corners, found):
    cv2.drawChessboardCorners(
        frame,
        chessboard_size,
        corners,
        found
    )

def show_frame(frame, window_name="Calibration"):
    cv2.imshow(window_name, frame)

def destroy_windows():
    cv2.destroyAllWindows()

def wait_key(delay=1):
    return cv2.waitKey(delay)
