import numpy as np
import pytest

from vcdisk import vcdisk, vcdisk_thinexp, vcbulge_sph, vcbulge_ellip, vcbulge_sersic, sersic

rad = np.logspace(-1, 1.5, 100)
rad_nan = rad.copy()
rad_nan[0] = np.nan
md, rd = 1e10, 1.0
n = 1.0
sb  = md / (2*np.pi*rd**2) * np.exp(-rad/rd)
sb_nan = sb.copy()
sb_nan[0] = np.nan
rhoz_simple = lambda x: np.exp(-x**2)
rhoz_compl  = lambda x,t: t*np.exp(-x**2)
rhoz_flare1 = lambda x,y: np.exp(-x/(1.0+y*1.0))
rhoz_flare2 = lambda x,y,t: np.exp(-x/(1.0+y*t))
sers = sersic(md, rd, n)

def test_inputs():
    #-- vcdisk
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
    with pytest.raises(ValueError):
        vcdisk(rad_nan, sb)
    with pytest.raises(ValueError):
        vcdisk(rad, sb_nan)

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


    #-- vcdisk_thinexp
    # input R
    with pytest.raises(TypeError):
        vcdisk_thinexp('o',md,rd)
    with pytest.raises(ValueError):
        vcdisk_thinexp([],md,rd)
    with pytest.raises(ValueError):
        vcdisk_thinexp(rad_nan,md,rd)

    # input Md and Rd
    with pytest.raises(TypeError):
        vcdisk_thinexp(rad,'o',rd)
    with pytest.raises(TypeError):
        vcdisk_thinexp(rad,md,'o')


    #-- vcbulge_ellip
    # input q and inc
    with pytest.raises(ValueError):
        vcbulge_ellip(rad, sb, q=-1.0)
    with pytest.raises(ValueError):
        vcbulge_ellip(rad, sb, q=2.0)
    with pytest.raises(ValueError):
        vcbulge_ellip(rad, sb, inc=-1.0)
    with pytest.raises(ValueError):
        vcbulge_ellip(rad, sb, inc=180.0)

    #-- vcbulge_sersic
    # input rad
    with pytest.raises(TypeError):
        vcbulge_sersic('rad', md, rd, n)
    with pytest.raises(ValueError):
        vcbulge_sersic([], md, rd, n)
    with pytest.raises(ValueError):
        vcbulge_sersic(rad_nan, md, rd, n)
    with pytest.raises(ValueError):
        vcbulge_sersic(rad, md, rd, -1.0)
    with pytest.raises(ValueError):
        vcbulge_sersic(rad, md, rd, 10.0)


def test_output_type():
    #-- vcdisk_thinexp
    assert type(vcdisk_thinexp(rad, md, rd)) is np.ndarray
    # assert type(vcdisk_thinexp(1, md, rd)) is np.float64
    assert type(vcdisk_thinexp(list(rad), md, rd)) is np.ndarray


    #-- vcdisk
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
    assert type(vcdisk(rad, sb, rhoz=rhoz_flare1, flaring=True)) is np.ndarray
    assert type(vcdisk(rad, sb, rhoz=rhoz_flare2, rhoz_args={"t":1.0}, flaring=True)) is np.ndarray


    #-- vcbulge_sph
    assert type(vcbulge_sph(rad, sb)) is np.ndarray


    #-- vcbulge_ellip
    assert type(vcbulge_ellip(rad, sb)) is np.ndarray
    assert type(vcbulge_ellip(rad, sb, q=0.9)) is np.ndarray
    assert type(vcbulge_ellip(rad, sb, q=1.0)) is np.ndarray
    assert type(vcbulge_ellip(rad, sb, inc=60.0)) is np.ndarray
    assert type(vcbulge_ellip(rad, sb, q=0.9, inc=60.0)) is np.ndarray


    #-- vcbulge_sersic
    assert type(vcbulge_sersic(rad, md, rd, n)) is np.ndarray
    assert type(vcbulge_sersic(list(rad), md, rd, n)) is np.ndarray


    #-- sersic
    assert type(sers(rad)) is np.ndarray
