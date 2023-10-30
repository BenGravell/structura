import numpy as np

import math_utils


def test_normalize():
    x = np.array([0, 1, 2, 3, 4])
    xn = math_utils.normalize(x)
    xn_expected = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    np.testing.assert_allclose(xn, xn_expected)


def test_compute_rectangle_dims_from_area():
    n = 100000
    ratio = 3
    x, y = math_utils.compute_rectangle_dims_from_area(n, ratio)
    xy = x * y
    assert x == ratio * y
    assert xy <= n  # strict upper bound
    assert xy > 0.99 * n  # should hold for large n and ratio not too far from 1
