# Створення детектора маркерів.

import cv2

ARUCO_DICTS = {
    0: cv2.aruco.DICT_ARUCO_ORIGINAL,
    1: cv2.aruco.DICT_4X4_250,
    2: cv2.aruco.DICT_APRILTAG_36h11,
}

def create_detector(pattern):

    dictionary = cv2.aruco.getPredefinedDictionary(
        ARUCO_DICTS.get(
            pattern,
            cv2.aruco.DICT_ARUCO_ORIGINAL
        )
    )

    params = cv2.aruco.DetectorParameters()

    # Налаштування параметрів для зменшення хибних спрацьовувань.
    # Documentation: https://docs.opencv.org/4.x/d1/dcd/structcv_1_1aruco_1_1DetectorParameters.html

    # Збільшити точність полігональної апроксимації контурів, що вимагає,
    # щоб виявлені контури були більш квадратними.
    params.polygonalApproxAccuracyRate = 0.05

    # Обмежити мінімальний та максимальний розмір периметра виявлених контурів.
    # Це допомагає відфільтрувати дуже малі шуми та дуже великі об'єкти (наприклад, клітинки дошки).
    params.minMarkerPerimeterRate = 0.04  # Злегка збільшено, щоб відфільтрувати дрібніші шуми

    # Налаштування параметрів адаптивної порогової обробки для кращого розділення маркерів від фону.
    params.adaptiveThreshWinSizeStep = 5  # Менший крок для більш детальної бінаризації
    params.adaptiveThreshConstant = 7     # Константа, що віднімається від середнього значення

    # Збільшити рівень корекції помилок для більш надійного розпізнавання ID маркерів.
    params.errorCorrectionRate = 0.8

    return cv2.aruco.ArucoDetector(
        dictionary,
        params
    )

def detect_markers(detector, frame):

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    return detector.detectMarkers(gray)
