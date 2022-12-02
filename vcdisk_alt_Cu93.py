# vcdisk_alt_Cu93
#
# author: Lorenzo Posti <lorenzo.posti@gmail.com>
#
#
# This is an alternative implementation of the double integral to get the potential
# Phi(R,z) of an axisymmetric disk. This follows eq. (27)-(28) of Cuddeford (1993)
# https://articles.adsabs.harvard.edu/pdf/1993MNRAS.262.1076C
# The derivation of the formula with K instead of Q_-1/2 is due to a transformation
# of eq. (27) following Byrd & Friedman (1971), page 248 eq 560.01, which I found
# in https://gitlab.com/iogiul/galpynamics/-/wikis/Potential-of-discs
#

import numpy as np
from scipy.integrate import simpson, quad
from scipy.special import ellipk

# constants
G_GRAV = 4.301e-6 # kpc km^2 s^-2 M_sun^-1


def vcdisk_alt_Cu93(rad, sb, z0=0.3, rhoz='cosh', rhoz_args=None, zsamp='log', rsamp='log', flaring=False):
    ################
    # input checks #
    ################
    #
    # checks on rad, sb
    if type(rad) is list: rad = np.asarray(rad)
    if type(sb)  is list: sb  = np.asarray(sb)
    if type(rad) is np.ndarray and type(sb) is np.ndarray:
        pass
    else:
        raise TypeError("rad and sb must be lists or np.arrays")
    if len(rad)<1 or len(sb)<1:
        raise ValueError("rad and sb must be arrays of size >1")
    if len(rad) != len(sb):
        raise ValueError("rad and sb must be arrays of the same size")

    # checks on z0
    if type(z0) is float:
        pass
    else:
        try:
            z0 = float(z0)
        except:
            raise TypeError("z0 must be a float")

    # vertical sampling
    l = None
    nz, z0_fac = 200, 15.0
    if type(zsamp) is str and zsamp=='log': l = np.logspace(np.log10(z0/z0_fac), np.log10(z0*z0_fac), nz)
    if type(zsamp) is str and zsamp=='lin': l = np.linspace(z0/z0_fac, z0*z0_fac, nz)
    if type(zsamp) is list and type(zsamp[0]) is float: l = np.asarray(zsamp)
    if type(zsamp) is np.ndarray: l = zsamp
    if l is None:
        raise TypeError("zsamp must be either 'log', 'lin' or an np.array")

    # radial sampling
    u, rs_fac = None, 2
    if type(rsamp) is str and rsamp=='log': u = np.logspace(np.log10(rad.min()), np.log10(rad.max()), rs_fac*len(rad))
    if type(rsamp) is str and rsamp=='lin': u = np.linspace(rad.min(), rad.max(), rs_fac*len(rad))
    if type(rsamp) is str and rsamp=='nat': u = rad
    if u is None:
        raise TypeError("rsamp must be either 'log', 'lin' or 'nat'")


    # linear interpolation of the input surface density
    smdisk = np.interp(u, rad, sb)

    phi = -4*G_GRAV * np.array([1/np.sqrt(R) * simpson(simpson(intfunc_Cu93(u, l, R, 0,
                                                                            smdisk,
                                                                            z0=z0,
                                                                            rhoz=rhoz,
                                                                            rhoz_args=rhoz_args,
                                                                            flaring=flaring
                                                                           ), u), l) for R in rad])
    v_circ = np.sqrt(rad*np.gradient(phi, rad, axis=0))

    return v_circ


def intfunc_Cu93(u, lp, R, z, smdisk, z0=0.3, rhoz='cosh', rhoz_args=None, flaring=False):

    l = lp[:,None]
    x = (R**2 + u**2 + (l-z)**2) / (2*R*u)
    y = 2/(1+x)

    # these are different choices for the vertical density profile
    rho_uz = None
    if rhoz=='cosh':
        rho_uz = smdisk / (2*z0) * np.cosh(np.abs(l)/z0)**-2
    if rhoz=='exp':
        rho_uz = smdisk / (2*z0) * np.exp(-np.abs(l)/z0)
    if not flaring and callable(rhoz) and rhoz_args is None and type(rhoz(z)) is np.ndarray:
        norm = quad(rhoz, 0, np.inf)[0]
        rho_uz = smdisk / (2*norm) * rhoz(np.abs(l))
    if flaring:
        if callable(rhoz) and rhoz_args is None and type(rhoz(z,u)) is np.ndarray:
            # assuming rhoz is normalised
            rho_uz = smdisk * rhoz(np.abs(l), u)
        elif callable(rhoz) and type(rhoz(z, u, **rhoz_args)) is np.ndarray:
            # assuming rhoz is normalised
            rho_uz = smdisk * rhoz(np.abs(l), u, **rhoz_args)
    if not flaring and rhoz_args is not None:
        try:
            if callable(rhoz) and type(rhoz(z, **rhoz_args)) is np.ndarray:
                # note that rhoz_args must be given in the correct positional order!
                norm = quad(rhoz, 0, np.inf, args=tuple(rhoz_args.values()))[0]
                rho_uz = smdisk / (2*norm) * rhoz(np.abs(l), **rhoz_args)
        except:
            raise TypeError("rhoz_args is a dictionary of additional arguments of "+
                             "the rhoz callable function")

    if rho_uz is None:
        raise TypeError("rhoz must be 'cosh', 'exp' or a callable function")

    if rho_uz is None:
        raise TypeError("rhoz must be 'cosh', 'exp' or a callable function")

    return np.sqrt(u*y) * ellipk(np.sqrt(y)) * rho_uz
