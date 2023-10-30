import math

import numpy as np


def normalize(x):
    xmin = np.min(x)
    xmax = np.max(x)
    return (x - xmin) / (xmax - xmin)


def compute_rectangle_dims_from_area(n: int, ratio: int):
    """Compute integer-valued rectangle side dimensions from area.

    x and y approximately satisfy x * y == n
    x and y satisfy x * y <= n
    x and y satisfy x == ratio * y
    """
    y = math.isqrt(n // ratio)
    x = ratio * y
    return x, y
