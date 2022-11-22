import numpy as np
import pytest

from vcdisk import vcdisk, vc_thin_expdisk

rad = np.logspace(-1, 1.5, 100)
md, rd = 1e10, 1.0
sb  = md / (2*np.pi*rd**2) * np.exp(-rad/rd)
rhoz_simple = lambda x: x**-2
rhoz_compl  = lambda x,t: t*x**-2

def test_inputs():
    # input rad, sb
    with pytest.raises(TypeError):
        vcdisk(0, 0)
    with pytest.raises(TypeError):
        vcdisk(rad, 0)
    with pytest.raises(TypeError):
        vcdisk(0, sb)
    with pytest.raises(ValueError):
        vcdisk(rad, [])
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

    # input rhoz
    with pytest.raises(TypeError):
        vcdisk(rad, sb, rhoz='no')

    # input rhoz_args
    with pytest.raises(TypeError):
        vcdisk(rad, sb, rhoz=rhoz_simple, rhoz_args=0)


    #-----------------------------
    # input R
    with pytest.raises(TypeError):
        vc_thin_expdisk('o',md,rd)
    with pytest.raises(ValueError):
        vc_thin_expdisk([],md,rd)

    # input Md and Rd
    with pytest.raises(TypeError):
        vc_thin_expdisk(rad,'o',rd)
    with pytest.raises(TypeError):
        vc_thin_expdisk(rad,md,'o')


def test_output_type():
    assert type(vc_thin_expdisk(rad, md, rd)) is np.ndarray
    assert type(vc_thin_expdisk(1, md, rd)) is np.float64
    assert type(vc_thin_expdisk(list(rad), md, rd)) is np.ndarray

    #-----------------------------
    assert type(vcdisk(rad, sb)) is np.ndarray
    assert type(vcdisk(list(rad), list(sb))) is np.ndarray
    assert type(vcdisk(rad, sb, z0=1)) is np.ndarray
    assert type(vcdisk(rad, sb, z0=1.0)) is np.ndarray
    assert type(vcdisk(rad, sb, zsamp='log')) is np.ndarray
    assert type(vcdisk(rad, sb, zsamp='lin')) is np.ndarray
    assert type(vcdisk(rad, sb, zsamp=[0.1, 1.0, 10.0])) is np.ndarray
    assert type(vcdisk(rad, sb, zsamp=rad)) is np.ndarray
    assert type(vcdisk(rad, sb, rsamp='log')) is np.ndarray
    assert type(vcdisk(rad, sb, rsamp='lin')) is np.ndarray
    assert type(vcdisk(rad, sb, rsamp='nat')) is np.ndarray
    assert type(vcdisk(rad, sb, rhoz='cosh')) is np.ndarray
    assert type(vcdisk(rad, sb, rhoz='exp')) is np.ndarray
    assert type(vcdisk(rad, sb, rhoz=rhoz_simple)) is np.ndarray
    assert type(vcdisk(rad, sb, rhoz=rhoz_compl, rhoz_args={"t":1.0})) is np.ndarray
