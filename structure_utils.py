import numpy as np


def compute_element_stiffness_vector(poisson_ratio: float):
    return np.array(
        [
            1 / 2 - poisson_ratio / 6,
            1 / 8 + poisson_ratio / 8,
            -1 / 4 - poisson_ratio / 12,
            -1 / 8 + 3 * poisson_ratio / 8,
            -1 / 4 + poisson_ratio / 12,
            -1 / 8 - poisson_ratio / 8,
            poisson_ratio / 6,
            1 / 8 - 3 * poisson_ratio / 8,
        ]
    )


def compute_element_stiffness_matrix(poisson_ratio: float):
    scale = 1 / (1 - poisson_ratio * poisson_ratio)
    k = compute_element_stiffness_vector(poisson_ratio)
    M = np.array(
        [
            [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
            [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
            [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
            [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
            [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
            [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
            [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
            [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]],
        ]
    )
    return scale * M
