import numpy as np
from numpy.typing import ArrayLike


def make_divisible_by(val: ArrayLike, div_factor: int) -> np.ndarray:
    val = np.asarray(val)
    if not val.ndim in [3, 4]:
        raise ValueError("Expected a 3D or 4D array as input")

    height, width = val.shape[-3: -1]
    rows = height // div_factor + (1 if height % div_factor else 0)
    cols = width // div_factor + (1 if width % div_factor else 0)

    padding = [(0, rows * div_factor - height), (0, cols * div_factor - width), (0, 0)]
    if val.ndim == 4:
        padding.insert(0, (0, 0))
    val = np.pad(val, padding, mode='constant', constant_values=0)

    return val
