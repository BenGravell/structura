import constants


def test_youngs_modulus():
    """Sanity checks for Youngs modulus."""
    assert constants.YOUNGS_MODULUS_MIN > 0
    assert constants.YOUNGS_MODULUS_MAX > 0
    assert constants.YOUNGS_MODULUS_MAX > constants.YOUNGS_MODULUS_MIN
