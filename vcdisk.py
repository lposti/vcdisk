# vcdisk: Rotation curves of thick truncated galaxy disks
#
# author: Lorenzo Posti <lorenzo.posti@gmail.com>
# license: BSD-2
#
#

__all__ = [
    'vcdisk',
    'integrand',
    'vc_thin_expdisk',
]


import numpy as np
from scipy.integrate import simpson, quad
from scipy.special import ellipk, ellipe, i0, i1, k0, k1

# constants
G_GRAV = 4.301e-6 # kpc km^2 s^-2 M_sun^-1

def vcdisk(rad, sb, z0=0.3, rhoz='cosh', rhoz_args=None, flaring=False, zsamp='log', rsamp='log'):
    r"""
    Circular velocity of a thick disk of arbitrary surface density.
    This function uses the method of [Casertano83]_ to calculate
    the radial force on the disk plane and then the circular velocity.

    :param rad: array of radii in :math:`\rm kpc`.
    :type rad: list or numpy.ndarray
    :param sb: array of surface densities in :math:`\rm M_\odot / kpc^2`.
    :type sb: list or numpy.ndarray
    :param z0: disk scaleheight in kpc. Default: 0.3 :math:`\rm kpc`.
    :type z0: float, optional
    :param rhoz: vertical density function. Can either be one of two
        hardcoded options, ``'cosh'`` (default) or ``'exp'``, or it can be
        any user-defined function with input and output numpy.ndarray.
        The function should define the vertical surface density in
        :math:`\rm M_\odot / kpc^2` and it should be normalized such that
        ``rhoz(0)=1``.
        It can have additional arguments handled by ``rhoz_args``. See
        :ref:`rhoz-label` for details.
    :type rhoz: str or callable, optional
    :param rhoz_args: dictionary of arguments of the user-defined
        function ``rhoz``.
    :type rhoz_args: dict, optional
    :param flaring: whether the vertical density function ``rhoz`` depends
        also on radius, i.e. if :math:`\rho_z=\rho_z(z,R)`. If ``True``, then
        ``rhoz`` must be a callable function with signature ``rhoz(z, R)`` or
        ``rhoz(z, R, **rhoz_args)``. This option assumes that the vertical
        density provided is normalized, i.e.
        :math:`\int {\rm d}z\, \rho_z(z,R)=1\,\,\,\forall R`.
    :type flaring: bool, optional
    :param zsamp: sampling in the z-direction. Can be either
        ``'log'`` (default) for logarithmically spaced values,
        ``'lin'`` for linearly spaced values, or a user-defined np.array.
        If ``'log'`` or ``'lin'`` an numpy.ndarray is created in the range
        [z0/10, z0*10] with size 100.
    :type zsamp: str or numpy.ndarray, optional
    :param rsamp: sampling in the :math:`R`-direction. Can be either
        ``'log'`` (default) for logarithmically spaced values,
        ``'lin'`` for linearly spaced values, or ``'nat'`` for the natural
        spacing of the data in input ``rad``.
    :type rsamp: str, optional
    :return: array of :math:`V_{\rm disk}` velocities in :math:`\rm km/s`.
    :rtype: numpy.ndarray

    .. seealso::

        `gipsy.rotmod <https://www.astro.rug.nl/~gipsy/tsk/rotmod.dc1>`_, `galpynamics <https://gitlab.com/iogiul/galpynamics>`_

    .. _notes-label:

    Notes
    =====

    .. _background-label:

    Background
    ----------

    The circular velocity on the mid-plane of a disk galaxy can be written as

    .. math::
        V_{\rm disk}(r) = \sqrt{-r\,F_{r,\rm disk}(r)},

    where :math:`F_{r,\rm disk}(r)` is the radial force on the plane of the disk,
    which is calculated as

    .. math::

        F_{r,\rm disk}(r) = 4\pi G \int_0^\infty \mathrm{d}u \int_0^\infty \mathrm{d}z \,\,2\sqrt{\frac{u}{rp}} \,\frac{\mathcal{K}(p)-\mathcal{E}(p)}{\pi}\, \frac{\partial \rho(u,z)}{\partial u}.

    Here :math:`p = x-\sqrt{x^2-a}`, :math:`x = (r^2+u^2+z^2)/(2ru)`, and
    :math:`\mathcal{K}` and :math:`\mathcal{E}` are the complete elliptic integrals
    respectively of the first and second kinds (see Eqs. 4-5-6 and Appendix A of
    [Casertano83]_ for full details).

    This program follows the notation of `gipsy <https://www.astro.rug.nl/~gipsy/>`_
    in defining a *signed* circular velocity which has the opposite sign of the
    radial force, i.e.

    .. math::
        V_{\rm disk}(r) = -{\rm sign}\left[F_{r,\rm disk}(r)\right]\sqrt{r\,|F_{r,\rm disk}(r)|}.

    This convention is such that :math:`V_{\rm disk}` is negative if there is a net
    force away from the galaxy center. This can happen, for instance, in cases where
    the surface density has a depression close to the center, as it is often observed
    in the neutral hydrogen disks of nearby spiral galaxies.

    .. _rhoz-label:

    Vertical density
    ----------------

    The radial force on the disk plane depends both on the radial surface density
    as well as on the vertical density distribution of the disk.
    While the radial surface density of can be typically obtained from observations
    of the surface brightness, in rotation curve studies the vertical density
    distribution is often unknown.

    Here we provide the user with several choices for the vertical density profile of
    the disk.

    Vertical density: constant scaleheight
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    The simplest and the most common assumption is that the axisymmetric disk has
    a constant scaleheight, which implies that the density is separable as
    :math:`\rho(R,z)=\rho_R(R)\rho_z(z)`. Thus, defining the surface density on
    the disk plane as :math:`\Sigma(R)=\int {\rm d}z\, \rho(R,z)`, we can write
    :math:`\rho(R,z)=\Sigma(R)\rho_z(z)/N`, where :math:`N` is the
    normalization of the vertical density :math:`N=\int {\rm d}z\, \rho_z(z)`.

    We provide full implementations of two popular choices for :math:`\rho_z(z)`
    that are often used to describe disk galaxies, and we also provide an interface
    for the user to supply their own vertical density profile.

    * ``'cosh'`` (**default**): this corresponds to :math:`\rho_z(z)=\cosh^{-2}(z/z_0)`, with :math:`N=2z_0`, i.e. the so-called [vdKruitSearle81]_ disk,
    * ``'exp'``: this corresponds to :math:`\rho_z(z)=\exp(-z/z_0)`, with :math:`N=2z_0`.
    * the user may also choose to specify a custom vertical density profile by passing a callable function as the ``rhoz`` argument.

    .. note::
        If the user specifies a callable ``rhoz``, :py:func:`vcdisk.vcdisk` will try
        to normalize it for you using :func:`scipy.integrate.quad` to solve the integral
        :math:`\int {\rm d}z\, \rho_z(z)`. Do look out for potential issues related
        to this normalization.

    .. warning::
        If the user-defined function ``rhoz`` has signature ``rhoz(z, **rhoz_args)``,
        then make sure that the ``rhoz_args`` dictionary supplied to :py:func:`vcdisk.vcdisk`
        is in the correct positional order, since keys get lost when calculating the
        normalization integral.


    Vertical density: flaring disks
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    Another possibility is that the vertical density :math:`\rho_z` depends also on
    radius, so that :math:`\rho(R,z)` is not separable. Often this happens since
    galactic disks flare in the outer regions, where the gravitational support is
    lower, thus one may wish to compute the circular velocity in the case where
    the scaleheight depends on radius, :math:`z_0=z_0(R)`.

    By activating the option ``flaring=True``, :py:func:`vcdisk.vcdisk` computes the
    potential with a :math:`\rho_z(z,R)` specified via the ``rhoz`` parameter.
    As an example, this could be something like
    :math:`\rho_z(z,R) = e^{-z/z_0(R)}/2z_0(R)`.

    .. note::
        If ``flaring=True``, :py:func:`vcdisk.vcdisk` expects the callable ``rhoz``
        function to be *already normalized*, such that
        :math:`\int {\rm d}z\, \rho_z(z,R)=1\,\,\,\forall R`.

    Implementation details
    ----------------------

    :py:func:`vcdisk.vcdisk` uses Simpson's method, implemented in
    :func:`scipy.integrate.simpson`, to compute the double integral quickly and
    efficiently. The ``scipy`` library is also used for the implementation of the
    elliptic integrals :func:`scipy.special.ellipk` and :func:`scipy.special.ellipe`.

    The arbitrary sampling in the :math:`z`-direction unfortunately seems to somewhat
    have a (small) impact on the derived :math:`V_{\rm disk}`. This is probably
    because of numerical noise in the double integral when Simpson's method is used
    on very steep functions such as 3-D densities of galaxy disks. After extensive
    tests where I compared the results with analytic formulae for a thin disk and
    with `gipsy <https://www.astro.rug.nl/~gipsy/>`_'s implementation, I have come
    to the conclusion that a logarithmic sampling in the :math:`z`-direction
    in the range :math:`z\in[z_0/10, 10\,z_0]` with 100 samples is a good compromise
    between speed and accuracy.

    References
    ----------

    .. [Casertano83] Casertano, S. 1983, MNRAS, 203, 735. Rotation curve of the edge-on spiral galaxy NGC 5907: disc and halo masses. `doi:10.1093/mnras/203.3.735 <https://doi.org/10.1093/mnras/203.3.735>`_
    .. [vdKruitSearle81] van der Kruit, P. C. & Searle, L. 1981, A&A, 95, 105. Surface photometry of edge-on spiral galaxies. I - A model for the three-dimensional distribution of light in galactic disks.

    Example
    =======

    >>> import numpy as np
    >>> from vcdisk import vcdisk
    >>> md, rd = 1e10, 2.0                        # mass, scalelength of the disk
    >>> r = np.linspace(0.1, 30.0, 50)            # radii samples
    >>> sb = md / (2*np.pi*rd**2) * np.exp(-r/rd) # exponential disk surface density
    >>> vcdisk(r, sb)
    array([ 6.36888305, 38.63854589, ..., 38.26020293, 37.82648552])

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
                np.array([simpson(simpson(integrand(rr, xi, r, smdisk,
                                                    z0=z0,
                                                    rhoz=rhoz,
                                                    rhoz_args=rhoz_args,
                                                    flaring=flaring
                                                    ).T, xi), rr) for r in rad])

    # getting the circular velocity
    # Eq (6) of Casertano (1983) is modified to account for the sign
    # of the radial force: the velocity is negative if there is a net force
    # away from from the galaxy centre (same as in gipsy)
    v_circ = -np.sign(rad_force) * np.sqrt(rad * np.abs(rad_force))

    return v_circ


def integrand(u, xi, r, smdisk, z0=0.3, rhoz='cosh', rhoz_args=None, flaring=False):
    r"""
    Integrand function for the radial force integral.
    This is the integrand Eq. (4) of [Casertano83]_.

    :param u: radial variable of integration in :math:`\rm kpc`.
    :type u: list or numpy.ndarray
    :param xi: vertical variable of integration in :math:`\rm kpc`.
    :type xi: list or numpy.ndarray
    :param r: radius in :math:`\rm kpc` at which the radial force
        is evaluated
    :type r: float
    :param smdisk: array of surface densities in :math:`\rm M_\odot / kpc^2`.
    :type smdisk: list or numpy.ndarray
    :param z0: disk scaleheight in :math:`\rm kpc`. Default: :math:`0.3 \rm kpc`.
    :type z0: float, optional
    :param rhoz: vertical density function. Can either be one of two
        hardcoded options, ``'cosh'`` (default) or ``'exp'``, or it can be
        any user-defined function with input and output numpy.ndarray.
        The function should define the vertical surface density in
        :math:`\rm M_\odot / kpc^2` and it should be normalized such that
        ``rhoz(0)=1``.
        It can have additional arguments handled by ``rhoz_args``. See
        :ref:`rhoz-label` for details.
    :type rhoz: str or callable, optional
    :param rhoz_args: dictionary of arguments of the user-defined
        function ``rhoz``.
    :type rhoz_args: dict, optional
    :param flaring: whether the vertical density function ``rhoz`` depends
        also on radius, i.e. if :math:`\rho_z=\rho_z(z,R)`. If ``True``, then
        ``rhoz`` must be a callable function with signature ``rhoz(z, R)`` or
        ``rhoz(z, R, **rhoz_args)``. This option assumes that the vertical
        density provided is normalized, i.e.
        :math:`\int {\rm d}z\, \rho_z(z,R)=1\,\,\,\forall R`.
    :type flaring: bool, optional
    :return: 2-D array of the radial force integrand, with shape
        ``(len(xi), len(u))``.
    :rtype: numpy.ndarray

    .. seealso::

        :py:func:`vcdisk.vcdisk`

    Notes
    =====

    Note that here I actually use the notation of Eq. (A17), which is
    equivalent to Eq. (4).
    This function is called by the main function :py:func:`vcdisk.vcdisk`.

    References
    ----------

    .. [Casertano83] Casertano, 1983, MNRAS, 203, 735. Rotation curve of the edge-on spiral galaxy NGC 5907: disc and halo masses. `doi:10.1093/mnras/203.3.735 <https://doi.org/10.1093/mnras/203.3.735>`_


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
    if not flaring and callable(rhoz) and rhoz_args is None and type(rhoz(z)) is np.ndarray:
        norm = quad(rhoz, 0, np.inf)[0]
        rho_uz = smdisk / (2*norm) * rhoz(z)
    if flaring:
        if callable(rhoz) and rhoz_args is None and type(rhoz(z,u)) is np.ndarray:
            # assuming rhoz is normalised
            rho_uz = smdisk * rhoz(z, u)
        elif callable(rhoz) and type(rhoz(z, u, **rhoz_args)) is np.ndarray:
            # assuming rhoz is normalised
            rho_uz = smdisk * rhoz(z, u, **rhoz_args)
    if not flaring and rhoz_args is not None:
        try:
            if callable(rhoz) and type(rhoz(z, **rhoz_args)) is np.ndarray:
                ###
                # take extra care that rhoz_args is given in the correct positional order!!!
                ###
                norm = quad(rhoz, 0, np.inf, args=tuple(rhoz_args.values()))[0]
                rho_uz = smdisk / (2*norm) * rhoz(z, **rhoz_args)
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
    r"""
    Circular velocity of an infinitely thin exponential disk.
    See [Freeman70]_, Eq. (10).

    :param R: radii in :math:`\rm kpc`.
    :type R: float or list or numpy.ndarray
    :param Md: disk mass in :math:`\rm M_\odot`.
    :type Md: float
    :param Rd: exponential scale-length of the disk in :math:`\rm kpc`
    :type Rd: float
    :return: 1-D array (same shape as R) of the circular velocities
        in :math:`\rm km/s`.
    :rtype: float or numpy.ndarray

    References
    ----------

    .. [Freeman70] Freeman (1970), ApJ, 160, 811. On the Disks of Spiral and S0 Galaxies. `https://ui.adsabs.harvard.edu/abs/1970ApJ...160..811F/ <https://ui.adsabs.harvard.edu/abs/1970ApJ...160..811F/>`_

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
