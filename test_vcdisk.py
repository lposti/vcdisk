import numpy as np
import pytest

from vcdisk import vcdisk

rad = np.logspace(-1, 1.5, 100)
md, rd = 1e10, 1.0
sb  = md / (2*np.pi*rd**2) * np.exp(-rad/rd)

def test_inputs():
    # input rad, sb
    with pytest.raises(TypeError):
        vcdisk(0, 0)
    with pytest.raises(TypeError):
        vcdisk(rad, 0)
    with pytest.raises(TypeError):
        vcdisk(0, sb)
    with pytest.raises(ValueError):
        vcdisk(rad, np.ones(len(rad)+1))

    # input z0
    with pytest.raises(TypeError):
        vcdisk(rad, sb, z0=[])

    # input zsamp
    with pytest.raises(TypeError):
        vcdisk(rad, sb, zsamp='no')

    # input rsamp
    with pytest.raises(TypeError):
        vcdisk(rad, sb, rsamp='no')
    with pytest.raises(TypeError):
        vcdisk(rad, sb, rsamp=rad)

def test_output_type():
    assert type(vcdisk(rad, sb)) is np.ndarray
    assert type(vcdisk(rad, sb, z0=1)) is np.ndarray
    assert type(vcdisk(rad, sb, z0=1.0)) is np.ndarray
    assert type(vcdisk(rad, sb, zsamp='log')) is np.ndarray
    assert type(vcdisk(rad, sb, zsamp='lin')) is np.ndarray
    assert type(vcdisk(rad, sb, zsamp=[0.1, 1.0, 10.0])) is np.ndarray
    assert type(vcdisk(rad, sb, zsamp=rad)) is np.ndarray
    assert type(vcdisk(rad, sb, rsamp='log')) is np.ndarray
    assert type(vcdisk(rad, sb, rsamp='lin')) is np.ndarray
    assert type(vcdisk(rad, sb, rsamp='nat')) is np.ndarray
