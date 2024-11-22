import math
import cv2 as cv
import numpy as np
from typing import Tuple
from numpy.typing import ArrayLike


def make_size_divisible_by(width: int, height: int, factor: int) -> Tuple[int, int]:
    rows = height // factor + (1 if height % factor else 0)
    cols = width // factor + (1 if width % factor else 0)   
    return cols * factor, rows * factor


def make_array_divisible_by(arr: ArrayLike, factor: int) -> np.ndarray:
    arr = np.asarray(arr)
    if not arr.ndim in [3, 4]:
        raise ValueError("Expected a 3D or 4D array as input")
    
    h, w = arr.shape[-3: -1]
    w_new, h_new = make_size_divisible_by(w, h, factor)
    pads = [(0, h_new - h), (0, w_new - w), (0, 0)]
    if arr.ndim == 4:
        pads.insert(0, (0, 0))

    return np.pad(arr, pads, mode='constant', constant_values=0)
