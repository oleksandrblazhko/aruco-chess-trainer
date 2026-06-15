# Створення MarkerData.

import cv2

from pose import rmat_to_euler
from models.marker_data import MarkerData

def build_marker_data(
    marker_id,
    corner,
    rvec,
    tvec
):

    R, _ = cv2.Rodrigues(rvec)

    roll, pitch, yaw = rmat_to_euler(R)

    tx, ty, tz = tvec[0]

    return MarkerData(
        marker_id=int(marker_id),

        # Keep for backward compatibility
        tx=float(tx),
        ty=float(ty),
        tz=float(tz),
        roll=float(roll),
        pitch=float(pitch),
        yaw=float(yaw),
        
        # Add raw vectors for new clients
        rvec=rvec,
        tvec=tvec,

        corners=corner.tolist()
    )
