# Замість передачі сирих масивів створюється окремий клас.

from dataclasses import dataclass
import numpy as np

@dataclass
class MarkerData:

    marker_id: int

    # Euler angles and individual components for backward compatibility
    tx: float
    ty: float
    tz: float
    roll: float
    pitch: float
    yaw: float
    
    # Raw vectors for new, precise calculations
    rvec: np.ndarray
    tvec: np.ndarray

    corners: list
