from dataclasses import dataclass
import numpy as np


@dataclass
class ResultadoSegmentacao:
    roi: np.ndarray
    mascara: np.ndarray
    segmentada: np.ndarray
    bbox_pixel: tuple[int, int, int, int]