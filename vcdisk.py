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
    '_integrand',
    'vc_thin_expdisk',
]


import numpy as np
from scipy.integrate import simpson
from scipy.special import ellipk, ellipe, i0, i1, k0, k1

# constants
G_GRAV = 4.301e-6 # kpc km^2 s^-2 M_sun^-1

def vcdisk(rad, sb, z0=0.3, rhoz='cosh', rhoz_args=None, zsamp='log', rsamp='log'):
    """
    Calculate the circular velocity of a thick disk of arbitrary
    surface density using the method of Casertano (1983, [1]_).

    Parameters
    ----------
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

    Returns
    -------
    :return: array of V_star velocities in km / s.
    :rtype: np.array.

    Example
    -------

    >>> import numpy as np.
    >>> from vcdisk import vcdisk
    >>> md, rd = 1e10, 2.0
    >>> r = np.linspace(0.1, 30.0, 50)
    >>> sb = md / (2*np.pi*rd**2) * np.exp(-r/rd)
    >>> vc = vcdisk(r, sb)

    Notes
    -----
    The circular velocity on the mid-plane of a disk can be written as
    :math:`V_{\rm disk}(r) = \sqrt{-r\,F_{r,\rm disk}(r)` where :math:`F_{r,\rm disk}(r)`
    is the radial force on the plane of the disk. The radial force can be
    calculated as

    .. math::

        F_{r,\rm disk}(r) = 4\pi G \int_0^\infty \mathrm{d}u \int_0^\infty \mathrm{d}z \,\,2\sqrt{\frac{u}{rp}} \,\frac{\mathcal{K}(p)-\mathcal{E}(p)}{\pi}\, \frac{\partial \rho(u,z)}{\partial u},

    where :math:`p = x-\sqrt{x^2-a}` and :math:`x = (r^2+u^2+z^2)/(2ru)` (see Eqs. 4-5-6
    in [1]_).

    References
    ----------
    .. [1] Casertano, 1983, MNRAS, 203, 735. Rotation curve of the edge-on spiral
    galaxy NGC 5907 : disc and halo masses. doi:10.1093/mnras/203.3.735


    """


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
    xi = None
    nz, z0_fac = 100, 10.0
    if type(zsamp) is str and zsamp=='log': xi = np.logspace(np.log10(z0/z0_fac), np.log10(z0*z0_fac), nz)
    if type(zsamp) is str and zsamp=='lin': xi = np.linspace(z0/z0_fac, z0*z0_fac, nz)
    if type(zsamp) is list and type(zsamp[0]) is float: xi = np.asarray(zsamp)
    if type(zsamp) is np.ndarray: xi = zsamp
    if xi is None:
        raise TypeError("zsamp must be either 'log', 'lin' or an np.array")

    # radial sampling
    rr, rs_fac = None, 2
    if type(rsamp) is str and rsamp=='log': rr = np.logspace(np.log10(rad.min()), np.log10(rad.max()), rs_fac*len(rad))
    if type(rsamp) is str and rsamp=='lin': rr = np.linspace(rad.min(), rad.max(), rs_fac*len(rad))
    if type(rsamp) is str and rsamp=='nat': rr = rad
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
    Integrand in Eq. (4) of Casertano (1983, MNRAS, 203, 735).
    Note that here I actually use the notation of Eq. (A17), which is
    equivalent to Eq. (4).
    This function is called by the main function :py:func:`vcdisk`.

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
    x = (r**2+u**2+z**2)/(2*r*u) # Eq. 5
    p = x-np.sqrt(x**2-1)        # Eq. 5

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


def vc_thin_expdisk(R, Md, Rd):
    """
    Circular velocity of an infinitely thin exponential disk.
    This result was first derived by Freeman (1970), ApJ, 160, 811,
    see their Eq. (10).

    :param R: radii in kpc where to calculate V_circ.
    :type R: float or list or np.array.
    :param Md: disk mass in M_sun.
    :type Md: float.
    :param Rd: exponential scale-length of the disk in kpc
    :type Rd: float.

    :return: 1-D array (same shape as R) of the circular velocities
        in km / s.
    :rtype: float or np.array.

    """

    # checks on input
    if type(R) is list: R = np.asarray(R)
    if type(R) is np.ndarray:
        pass
    else:
        try:
            assert type(float(R)) is float
        except:
            raise TypeError("R must be a float scalar, a list of floats or a np.array")

    if type(R) is np.ndarray and len(R)<1:
        raise ValueError("the size of the R array must be >0")

    try:
        assert type(float(Md)) is float
        assert type(float(Rd)) is float
    except:
        raise TypeError("Md and Rd must be float scalars")

    y = R/2./Rd
    return np.nan_to_num(np.sqrt(2*G_GRAV*Md/Rd*y**2*(i0(y)*k0(y)-i1(y)*k1(y))))
