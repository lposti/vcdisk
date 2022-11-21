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

def get_vstar_disk(rad, sb, z0=0.3, rhoz='cosh', rhoz_args=None,
                   zsamp='log', rsamp='log'):
    """
    Calculate the circular velocity of a thick disk of arbitrary
    surface density using the method of Casertano (1983, MNRAS, 203, 735).

    :param rad: array of radii in kpc.
    :type rad: list or np.array.
    :param sb: array of surface densities in M_sun / kpc^2.
    :type sb: list or np.array.
    :param z0: disk scaleheight in kpc. Default: 0.3 kpc.
    :type z0: optional|float.
    :param rhoz: vertical density function. Can either be one of two
        hardcoded options, 'cosh' (default) or 'exp', or it can be
        any user-defined function with input and output np.arrays.
        The function should define the vertical surface density in
        M_sun / kpc^2 and it should be normalized such that rhoz(0)=1.
        It can have additional arguments handled by rhoz_args.
    :type rhoz: optional|str or callable.
    :param rhoz_args: dictionary of arguments of the user-defined
        function rhoz.
    :param zsamp: sampling in the z-direction. Can be either
        'log' (default) for logarithmically spaced values,
        'lin' for linearly spaced values, or a user-defined np.array.
        If 'log' or 'lin' an np.array is created in the range
        [z0/10, z0*10] with size 100.
    :type zsamp: optional|str or np.array.
    :param rsamp: sampling in the R-direction. Can be either
        'log' (default) for logarithmically spaced values,
        'lin' for linearly spaced values, or 'nat' for the natural
        spacing of the data in input rad.
    :type rsamp: optional|str
    :type rhoz_args: optional|dict.
    :raise lumache.InvalidKindError: If the kind is invalid.
    :return: array of V_star velocities in km / s.
    :rtype: np.array.

    """

    # vertical sampling
    xi = None
    nz, z0_fac = 100, 10.0
    if zsamp=='log': xi = np.logspace(np.log10(z0/z0_fac), np.log10(z0*z0_fac), nz)
    if zsamp=='lin': xi = np.linspace(z0/z0_fac, z0*z0_fac, nz)
    if type(zsamp) is np.ndarray: xi = zsamp
    if xi is None:
        raise TypeError("zsamp must be either 'log', 'lin' or an np.array")

    # radial sampling
    rr, rs_fac = None, 2
    if rsamp=='log': rr = np.logspace(np.log10(rad.min()), np.log10(rad.max()), rs_fac*len(rad))
    if rsamp=='lin': rr = np.linspace(rad.min(), rad.max(), rs_fac*len(rad))
    if rsamp=='nat': rr = rad
    if rr is None:
        raise TypeError("rsamp must be either 'log', 'lin' or 'nat'")


    # linear interpolation of the input surface density
    smdisk = np.interp(rr, rad, sb)

    # calculation of the double integral (Eq 4 in Casertano 1983)
    rad_force = 4 * np.pi * G_GRAV *\
                np.array([simpson(simpson(_integrand(rr, xi, r, smdisk,
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


def _integrand(u, xi, r, smdisk, z0=0.3, rhoz='cosh', rhoz_args=None):
    """
    Integrand in Eq 4 of Casertano (1983, MNRAS, 203, 735).
    Note that here I actually use the notation of Eq A17, which is
    equivalent to Eq 4.
    This function is called by the main function get_vstar_disk.

    :param u: radial variable of integration.
    :type u: list or np.array.
    :param xi: vertical variable of integration.
    :type xi: list or np.array.
    :param r: radius at which the radial force is evaluated
    :type r: float.
    :param smdisk: array of surface densities in M_sun / kpc^2.
    :type smdisk: list or np.array.
    :param z0: disk scaleheight in kpc. Default: 0.3 kpc.
    :type z0: optional|float.
    :param rhoz: vertical density function. Can either be one of two
        hardcoded options, 'cosh' (default) or 'exp', or it can be
        any user-defined function with input and output np.arrays.
        The function should define the vertical surface density in
        M_sun / kpc^2 and it should be normalized such that rhoz(0)=1.
        It can have additional arguments handled by rhoz_args.
    :type rhoz: optional|str or callable.
    :param rhoz_args: dictionary of arguments of the user-defined
        function rhoz.
    :return: 2-D array of the radial force integrand, with shape
        (len(xi), len(u)).
    :rtype: np.array.

    """
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
