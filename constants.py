import structure_utils


# Minimum time interval between image draw updates.
# Needs to be fairly high to work reliably in the deployed app, but can be zero when running locally.
DRAW_UPDATE_SEC = 0.5


# TODO expose via app options
YOUNGS_MODULUS_MIN = 1e-9
YOUNGS_MODULUS_MAX = 1.0
POISSON_RATIO = 0.3
ELEMENT_STIFFNESS_MATRIX = structure_utils.compute_element_stiffness_matrix(POISSON_RATIO)


COLORMAP_OPTIONS = [
    "turbo",
    "viridis",
    "inferno",
    "cividis",
    "Greys",
    "Blues",
]
