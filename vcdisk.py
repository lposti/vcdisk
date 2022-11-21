# vcdisk: Rotation curves of thick truncated galaxy disks
#
# author: Lorenzo Posti <lorenzo.posti@gmail.com>
# license: BSD-2
#
#
'''
Rotation curves of thick truncated galaxy disks (:mod:`vcdisk`)
========================================
This is a minimal python package to solve Poisson's equation and to compute the
circular velocity curve of a truncated disk with non-zero thickness and arbitrary
radial density profile. This implements the algorithm of
[Casertano (1983)](https://ui.adsabs.harvard.edu/abs/1983MNRAS.203..735C)
The package can be installed using pip::
    pip install vcdisk
Reference/API
-------------
.. currentmodule:: vcdisk
.. autosummary::
   :toctree: api
   :nosignatures:
   vcdisk
   _integrand
   vc_razorthin

'''

__all__ = [
    'vcdisk',
    'vc_thin_expdisk',
    '_integrand',
]


import numpy as np
from scipy.integrate import simpson
from scipy.special import ellipk, ellipe, i0, i1, k0, k1

# constants
G_GRAV = 4.301e-6 # kpc km^2 s^-2 M_sun^-1

# circular velocity of a razor-thin exponential disk
def vc_thin_expdisk(R, Md, Rd):
    y = R/2./Rd
    return np.nan_to_num(np.sqrt(2*G_GRAV*Md/Rd*y**2*(i0(y)*k0(y)-i1(y)*k1(y))))

def _integrand(u, r, smdisk, xi, z0=0.3, rhoz='cosh', rhoz_args=None):
    z = xi[:,None]
    x = (r**2+u**2+z**2)/(2*r*u)
    p = x-np.sqrt(x**2-1)

    # these are different choices for the vertical density profile
    rho_uz = None
    if rhoz=='cosh':
        rho_uz = smdisk / (2*z0) * np.cosh(z/z0)**-2
    if rhoz=='exp':
        rho_uz = smdisk / (2*z0) * np.exp(-z/z0)
    if callable(rhoz) and rhoz_args is None and type(rhoz(z)) is np.ndarray:
        rho_uz = smdisk / (2*z0) * rhoz(z)
    if rhoz_args is not None:
        try:
            if callable(rhoz) and type(rhoz(z, **rhoz_args)) is np.ndarray:
                    rho_uz = smdisk / (2*z0) * rhoz(z, **rhoz_args)
        except:
            raise TypeError("rhoz_args is a dictionary of additional arguments of "+
                             "the rhoz callable function")
    if rho_uz is None:
        raise TypeError("rhoz must be 'cosh', 'exp' or a callable function")

    # derivative of rho(u,z) w.r.t. u
    # term in square brackets in Eq. 4 of Casertano (1983)
    drho_du = np.gradient(rho_uz, u, axis=1)

    return 2/np.pi*np.sqrt(u/r*p) * (ellipk(p) - ellipe(p)) * drho_du

def get_vstar_disk(rad, sb, z0=0.3, rhoz='cosh', rhoz_args=None):
    """
    Calculate the circular velocity of a thick disk of arbitrary
    surface density using the method of Casertano (1983, MNRAS, 203, 735).

    :param rad: array of radii in kpc
    :type rad: list or np.array
    :param sb: array of surface densities in M_sun / kpc^2
    :type sb: list or np.array
    :param z0: disk scaleheight in kpc. Default: 0.3 kpc
    :type z0: optional|float
    :raise lumache.InvalidKindError: If the kind is invalid.
    :return: array of V_star velocities in km / s
    :rtype: np.array

    """

    # radial and vertical sampling
    rr = np.logspace(np.log10(rad.min()), np.log10(rad.max()), 100)
    xi = np.logspace(np.log10(z0/10.0),np.log10(z0*10.0),100)
#     xi = np.linspace(z0/10.0, z0*10.0,8)


    # linear interpolation of the input surface density
    smdisk = np.interp(rr, rad, sb)

    # calculation of the double integral (Eq 4 in Casertano 1983)
    rad_force = 4 * np.pi * G_GRAV *\
                np.array([simpson(simpson(_integrand(rr, r, smdisk, xi,
                                                     z0=z0,
                                                     rhoz=rhoz,
                                                     rhoz_args=rhoz_args
                                                    ).T, xi), rr) for r in rad])

    # getting the circular velocity
    # Eq (6) of Casertano (1983) is modified to account for the sign
    # of the radial force: the velocity is negative if there is a net force
    # away from from the galaxy centre (same as in gipsy)
    v_circ = -np.sign(rad_force) * np.sqrt(rad * np.abs(rad_force))

    return v_circ
