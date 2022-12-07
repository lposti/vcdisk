# vcdisk: Rotation curves of thick galaxy disks and flattened spheroids
#
# author: Lorenzo Posti <lorenzo.posti@gmail.com>
# license: BSD-2
#
#

__all__ = [
    'vcdisk',
    'vcbulge_sph',
    'vcbulge',
    'vcbulge_sersic',
    'sersic',
    'integrand',
    'vcdisk_thinexp',
]


import numpy as np
from scipy.integrate import simpson, quad, IntegrationWarning
from scipy.special import ellipk, ellipe, i0, i1, k0, k1, gamma

import warnings
warnings.filterwarnings("ignore", category=IntegrationWarning)

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
    rad, sb = check_rad_sb(rad, sb)

    # checks on z0
    z0 = check_float(z0, 'z0')

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
    Integrand function for the radial force integral in :py:func:`vcdisk.vcdisk`.

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
                # note that rhoz_args must be given in the correct positional order!
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


def vcdisk_thinexp(r, md, rd):
    r"""
    Circular velocity of an infinitely thin exponential disk.
    See [Freeman70]_, Eq. (10).

    :param r: radii in :math:`\rm kpc`.
    :type r: float or list or numpy.ndarray
    :param md: disk mass in :math:`\rm M_\odot`.
    :type md: float
    :param rd: exponential scale-length of the disk in :math:`\rm kpc`
    :type rd: float
    :return: 1-D array (same shape as ``r``) of the circular velocities
        in :math:`\rm km/s`.
    :rtype: float or numpy.ndarray

    References
    ----------

    .. [Freeman70] Freeman, 1970, ApJ, 160, 811. On the Disks of Spiral and S0 Galaxies. `https://ui.adsabs.harvard.edu/abs/1970ApJ...160..811F/ <https://ui.adsabs.harvard.edu/abs/1970ApJ...160..811F/>`_

    """

    # check rad
    r = check_rad(r, 'r')

    # check md, rd
    md = check_float(md, 'md')
    rd = check_float(rd, 'rd')

    y = r/2./rd
    return np.nan_to_num(np.sqrt(2*G_GRAV*md/rd*y**2*(i0(y)*k0(y)-i1(y)*k1(y))))


def vcdisk_offplane(rad, zs, rho_rz):
    r"""
    Circular velocity off the mid plane of a thick disk.

    This uses Eq. (27)-(28) of [Cuddeford93]_ to compute the gravitational potential,
    and the circular velocity, at any position :math:`(R,z)` of thick disk of
    arbitrary density.

    :param rad: array of radii in :math:`\rm kpc`.
    :type rad: list or numpy.ndarray
    :param zs: array of height above the plane in :math:`\rm kpc`.
    :type zs: list or numpy.ndarray
    :param rho_rz: array of surface densities in :math:`\rm M_\odot / kpc^2`.
        Its shape must be ``rho_rz.shape == (len(zs), len(rad))``. If the
        array is sampled from a function ``rho(R,z)``, then this can be obtained
        with ``rho_rz = rho(rad, z[:, None])``.
    :type rho_rz: list or numpy.ndarray
    :return: array of :math:`V_{\rm disk}` velocities in :math:`\rm km/s`.
    :rtype: numpy.ndarray

    .. seealso::

        :py:func:`vcdisk.vcdisk`, `galpynamics <https://gitlab.com/iogiul/galpynamics>`_

    .. warning::

        This function is not as accurate as :py:func:`vcdisk.vcdisk` on the disk
        plane and it is currently under development.

    Notes
    =====

    This routine solves the integral in Eq. (27) of [Cuddeford93]_ to evaluate the
    gravitational potential of a thick disk galaxy

    .. math::

        \Phi(R,z) = -\frac{2G}{\sqrt{R}} \int_{-\infty}^\infty {\rm d}l \int_0^\infty {\rm d}u\,\sqrt{u}\rho(u,l)\,Q_{-1/2}(x),

    where :math:`Q_\lambda` is the Legendre function and

    .. math::

        x = \frac{R^2+u^2+(z-l)^2}{2Ru}.

    From Eq. (560.01) in [ByrdFriedman71]_ we know that :math:`Q_{-1/2}(x) = y\,\mathcal{K}(y)`,
    where :math:`\mathcal{K}` is the complete elliptic integral of the first kind and

    .. math::
        y^2 = \frac{2}{1+x} = \frac{4Ru}{R^2+u^2+2Ru+(z-l)^2}.

    Thus we can write the potential as

    .. math::

        \Phi(R,z) = -\frac{2G}{\sqrt{R}} \int_{-\infty}^\infty {\rm d}l \int_0^\infty {\rm d}u\,\sqrt{u}\rho(u,l)\,y\,\mathcal{K}(y).

    :py:func:`vcdisk.vcdisk_offplane` solves this integral to get a 2D array for the
    gravitational potential in :math:`(R,z)` and then computes the circular velocity
    of the disk as

    .. math::

        V_{\rm disk}(R,z) = \sqrt{R\left.\frac{\partial\Phi(R,z)}{\partial R}\right|_{z}}.

    The double integral is computed with Simpson's method implemented in
    :func:`scipy.integrate.simpson`.


    References
    ----------

    .. [ByrdFriedman71] Byrd & Friedman, 1971, Springer-Verlag, Berlin. Handbook of Elliptic Integrals for Engineers and Scientists, 2nd Edition. `http://dx.doi.org/10.1007/978-3-642-65138-0 <http://dx.doi.org/10.1007/978-3-642-65138-0>`_
    .. [Cuddeford93] Cuddeford, 1993, MNRAS, 262, 1076. On the potentials of galactic discs. `https://doi.org/10.1093/mnras/262.4.1076 <https://doi.org/10.1093/mnras/262.4.1076>`_

    Example
    =======

    >>> import numpy as np
    >>> from vcdisk import vcdisk_offplane
    >>> md, rd, z0 = 1e10, 1.0, 0.3                                               # mass, rd, z0
    >>> rad = np.linspace(0.1, 40.0, 20)                                          # radii samples
    >>> zs  = np.linspace(0.0, 3.0,  20)                                          # height samples
    >>> rho = lambda R,z: md / (4*np.pi*rd**2*z0) * np.exp(-R/rd) * np.exp(-z/z0) # density in (R,z)
    >>> rho_rz = rho(rad, zs[:,None])
    >>> vcdisk_offplane(rad, zs, rho_rz)
    array([[ 69.9369551 , 280.68843945, ..., 31.73661284,  31.67214993],
            ...
            [  4.19006384,  36.38315685, ..., 31.6072344 ,  31.55062705]])

    """

    # input checks
    rad = check_rad(rad, 'rad')
    zs  = check_rad(zs,  'zs')

    if type(rho_rz) is list: rho_rz = np.asarray(rho_rz)
    if type(rho_rz) is np.ndarray:
        pass
    else:
        raise TypeError("rho_rz must be a list or np.array")

    if rho_rz.shape != (len(rad), len(zs)):
        raise TypeError("rho_rz have shape: rho_rz.shape == (len(rad), len(zs))")

    phi = -4*G_GRAV * np.array([[1/np.sqrt(R) * simpson(simpson(integrand_offplane(rad, zs[:,None], rho_rz, R, z), rad), zs)
                                 for R in rad] for z in zs])
    v_circ = np.sqrt(rad*np.gradient(phi, rad, axis=1))

    return v_circ


def integrand_offplane(u, l, rho_ul, R, z):
    r"""
    Integrand function for the integral in :py:func:`vcdisk.vcdisk_offplane`.

    :param u: radial variable of integration in :math:`\rm kpc`. Its shape
        must be ``u.shape == (len(u),)``.
    :type u: list or numpy.ndarray
    :param l: vertical variable of integration in :math:`\rm kpc`. Its shape
        must be ``l.shape == (len(l),1)``. This can be obtained from a standard
        array as ``l[:,None]``.
    :type l: list or numpy.ndarray
    :param rho_rz: array of surface densities in :math:`\rm M_\odot / kpc^2`.
        Its shape must be ``rho_rz.shape == (len(l), len(u))``. If the
        array is sampled from a function ``rho(R,z)``, then this can be obtained
        with ``rho_rz = rho(u, l[:, None])``.
    :type rho_rz: list or numpy.ndarray
    :param R: radius in :math:`\rm kpc` at which the potential is evaluated
    :type R: float
    :param z: height in :math:`\rm kpc` at which the potential is evaluated
    :type z: float
    :return: 2-D array of the potential integrand with shape ``(len(l), len(u))``.
    :rtype: numpy.ndarray

    .. seealso::

        :py:func:`vcdisk.vcdisk_offplane`


    """

    ysq = 4*R*u / (R**2 + u**2 + 2*R*u + np.clip(z-l, 1e-6, None)**2)

    return np.sqrt(u*ysq) * ellipk(np.sqrt(ysq)) * rho_ul



def vcbulge_sph(rad, sb):
    r"""
    Circular velocity of a spherical bulge of arbitrary surface density.

    This function inverts Abel's integral equation to calculate the 3D density
    of a spherically symmetric bulge from its projected surface density
    (see Eq. B.72 in [BT2008]_).

    :param rad: array of radii in :math:`\rm kpc`.
    :type rad: list or numpy.ndarray
    :param sb: array of surface densities in :math:`\rm M_\odot / kpc^2`.
    :type sb: list or numpy.ndarray
    :return: array of :math:`V_{\rm bulge}` velocities in :math:`\rm km/s`.
    :rtype: numpy.ndarray

    .. seealso::

        :py:func:`vcdisk.vcbulge`

    Notes
    =====

    :py:func:`vcdisk.vcbulge_sph` inverts Abel's integral to get the 3D density

    .. math::

        \rho(r) = -\frac{1}{\pi} \int_r^\infty \frac{{\rm d}I}{{\rm d}R} \frac{{\rm d}R}{\sqrt{R^2-r^2}},

    where :math:`I(R)` is the projected surface density. Since this integral has
    a singularity at the lower bound of integration, where :math:`R=r`, it is better
    to change the integration variable to :math:`u={\rm arccosh}{(R/r)}` so that the integral
    becomes

    .. math::

        \rho(r) = -\frac{1}{\pi} \int_0^\infty I'(r\cosh{u}) {\rm d}u,

    where :math:`I'` is the first derivative of :math:`I`. Then, the mass profile
    can be written as :math:`M(r)=4\pi\int_0^r x^2\rho(x) {\rm d}x` and finally
    :math:`V_{\rm bulge}(r)=\sqrt{GM(r)/r}.`

    The :math:`\rho(r)` integral is computed with :func:`scipy.integrate.quad`, the
    derivative of the input surface density profile :math:`I'` is discretized with
    :func:`numpy.gradient`, and the mass integral is computed with
    :func:`scipy.integrate.simpson`.

    References
    ----------

    .. [BT2008] Binney & Tremaine, 2008, Princeton University Press, NJ USA. Galactic Dynamics: Second Edition.

    """

    # input checks
    rad, sb = check_rad_sb(rad, sb)

    # 3d density rho(r) from surface density I(R)
    # by inverting Abel's integral equation
    with np.errstate(over='ignore'):
        rhom = -1/np.pi * np.array([quad(lambda u: np.interp(r*np.cosh(u), rad, np.gradient(sb,rad)), 0, np.inf)[0]
                                    for r in rad])

    # mass profile
    mass = 4*np.pi * np.array([simpson((rad**2 * rhom)[rad<=R], rad[rad<=R]) for R in rad])

    v_circ = np.sqrt(G_GRAV * mass / rad)

    return v_circ



def vcbulge(rad, sb, q=0.99, inc=0.):
    r"""
    Circular velocity of a spheroidal oblate bulge of arbitrary surface density.

    This function calculates :math:`V_{\rm bulge}` for a flattened bulge, whose
    isodensity surfaces :math:`\rho=\rho(m)` are stratified on similar spheroids
    with :math:`m^2=R^2+(z^2/q^2)`. This is done with an Abel inversion to get
    the 3D density :math:`\rho(m)` from the observed projected density and
    then by integration of the 3D density to get the gravitational potential.
    See e.g. Eq. (2.132) in [BT2008]_.

    :param rad: array of radii in :math:`\rm kpc`.
    :type rad: list or numpy.ndarray
    :param sb: array of surface densities in :math:`\rm M_\odot / kpc^2`.
    :type sb: list or numpy.ndarray
    :param q: intrinsic axis ratio of the spheroid. This is related to the ellipticity
        of the observed isophotal contours :math:`\epsilon` and the inclination angle
        :math:`i` (i.e. the ``inc`` parameter) by
        :math:`(1-\epsilon)^2 = q^2+(1-q^2)\cos^2 i`. This parameter is :math:`0<q<1`
        for *oblate* bulges. The spherical case :math:`q=1` is singular in this
        formulation and will fallback to :py:func:`vcdisk.vcbulge_sph`.
    :type q: float, optional
    :param inc: inclination in degrees of the line-of-sight with respect to the
        symmetry axis of the spheroid. ``inc=0`` is edge-on, ``inc=90`` is face-on.
    :type inc: float, optional
    :return: array of :math:`V_{\rm bulge}` velocities in :math:`\rm km/s`.
    :rtype: numpy.ndarray

    .. seealso::

        :py:func:`vcdisk.vcbulge_sph`, :py:func:`vcdisk.vcbulge_sersic`

    Notes
    =====

    This function calculates :math:`V_{\rm bulge}` for a spheroidal bulge, whose
    isodensity surfaces are stratified on similar spheroids:

    .. math::

        \rho=\rho(m), \qquad {\rm with}\,\,\, m^2=R^2+\frac{z^2}{q^2},

    where :math:`q` is the intrinsic axis ratio. The calculation is done following
    the formalism of [Noordermeer08]_, using their Eq. (10).

    In practice this is done in two steps. First the 3D density :math:`\rho(m)`
    is computed with an Abel inversion:

    .. math::

        \rho(m) = -\frac{1}{\pi} \int_m^\infty \frac{{\rm d}I}{{\rm d}R} \frac{{\rm d}R}{\sqrt{R^2-m^2}},

    where :math:`I(R)` is the observed surface density. Since this integral has
    a singularity at the lower bound of integration, where :math:`R=m`, it is better
    to change the integration variable to :math:`u={\rm arccosh}{(R/m)}` so that the integral
    becomes

    .. math::

        \rho(m) = -\frac{1}{\pi} \int_0^\infty I'(m\cosh{u}) {\rm d}u,

    where :math:`I'` is the first derivative of :math:`I`. With the 3D density :math:`\rho(m)`
    at hand, the circular velocity can then be computed as

    .. math::

        V^2_{\rm bulge}(r) = -4\pi\,Gq\sqrt{\sin^2i+\frac{1}{q}\cos^2i} \int_0^r \frac{m^2\rho(m){\rm d}m}{\sqrt{r^2-e^2m^2}},

    where :math:`e=\sqrt{1-q^2}` is the intrinsic ellipticity. However, also this integral
    has a singularity at :math:`m=r/e`, thus it is convenient to change variables to
    :math:`u=\arcsin{me/r}` so that

    .. math::

        V^2_{\rm bulge}(r) = -4\pi\,Gq\frac{r^2}{e^3}\sqrt{\sin^2i+\frac{1}{q}\cos^2i} \int_0^{\arcsin{e}} \rho\left(\frac{r}{e}\sin{u}\right) \sin^2u \,{\rm d}u.

    Both the :math:`\rho(m)` and the :math:`V_{\rm bulge}` integrals are computed
    with :func:`scipy.integrate.quad` and the derivative of the input surface density
    profile :math:`I'` is discretized with :func:`numpy.gradient`.

    .. warning::

        With the current implementation using :func:`scipy.integrate.quad` this function
        is considerably slower than :py:func:`vcdisk.vcdisk`, which instead computes
        discretized integrals with :func:`scipy.integrate.simpson`.

    .. warning::

        The integral consistently results in a numerical overflow of ``cosh`` and
        an IntegrationWarning on :func:`scipy.integrate.quad`. Thus these two warnings
        have been suppressed.


    References
    ----------

    .. [Noordermeer08] Noordermeer, 2008, MNRAS, 385, 1359. The rotation curves of flattened Sérsic bulges. `https://ui.adsabs.harvard.edu/abs/2008MNRAS.385.1359N/ <https://ui.adsabs.harvard.edu/abs/2008MNRAS.385.1359N/>`_

    """

    # input checks
    rad, sb = check_rad_sb(rad, sb)

    # checks on q and inc
    if q==1.0:
        print ("the spherical case q=1 is handled with vcbulge_sph")
        return vcbulge_sph(rad, sb)
    q, inc = check_q_inc(q, inc)

    # intrinsic ellipticity
    e=np.sqrt(1-q**2)

    # geometric factor
    # see Sec. 2.1 in Noordermeer (2008)
    geom_factor = np.sqrt(np.sin(np.radians(inc))**2 + np.cos(np.radians(inc))**2/q**2)

    # first get 3d density with Abel integral
    with np.errstate(over='ignore'):
        rhom = np.array([-1/np.pi * geom_factor *
                         quad(lambda u: np.interp(m*np.cosh(u), rad, np.gradient(sb,rad)), 0, np.inf)[0]
                         for m in rad])


    # then get circular velocity
    v_circ = np.array([np.sqrt(4*np.pi*G_GRAV * q *
                               quad(lambda u: np.interp(r/e*np.sin(u), rad, rhom)*r**2/e**3*np.sin(u)**2,
                                    0, np.arcsin(e))[0]) for r in rad])

    return v_circ

def vcbulge_sersic(rad, mtot, re, n, q=0.99, inc=0.):
    r"""
    Circular velocity of a flattened Sersic bulge.

    This is the same as :py:func:`vcdisk.vcbulge`, but for an analytic
    [Sersic68]_ surface density profile. The implementation follows Eq.s
    (10)-(14) in [Noordermeer08]_.

    :param rad: array of radii in :math:`\rm kpc`.
    :type rad: list or numpy.ndarray
    :param mtot: total mass in :math:`\rm M_\odot` of the Sersic bulge.
    :type mtot: float
    :param re: effective radius in :math:`\rm kpc` of the Sersic bulge.
    :type re: float
    :param n: Sersic index, :math:`\rm 0 < n \leq 8`.
    :type n: float
    :param q: intrinsic axis ratio of the spheroid. This is related to the ellipticity
        of the observed isophotal contours :math:`\epsilon` and the inclination angle
        :math:`i` (i.e. the ``inc`` parameter) by
        :math:`(1-\epsilon)^2 = q^2+(1-q^2)\cos^2 i`. This parameter is :math:`0<q<1`
        for *oblate* bulges. The spherical case :math:`q=1` is singular in this
        formulation and will fallback to :py:func:`vcdisk.vcbulge_sph`.
    :type q: float, optional
    :param inc: inclination in degrees of the line-of-sight with respect to the
        symmetry axis of the spheroid. ``inc=0`` is edge-on, ``inc=90`` is face-on.
    :type inc: float, optional
    :return: array of :math:`V_{\rm bulge}` velocities in :math:`\rm km/s`.
    :rtype: numpy.ndarray

    .. seealso::

        :py:func:`vcdisk.vcbulge`, :py:func:`vcdisk.vcbulge_sph`, :py:class:`vcdisk.sersic`

    Notes
    =====

    This function calculates :math:`V_{\rm bulge}` for a spheroidal bulge, whose
    observed surface density can be approximated with a Sersic profile

    .. math::

        I(R) = I_e \exp\left\{ -b_n\left[\left(\frac{R}{R_e}\right)^\frac{1}{n}-1\right] \right\},

    where :math:`R_e` is the effective radius, i.e. the radius containing half the
    total mass of the spheroid, :math:`I_e` is the surface density at the effective
    radius, and :math:`n` is the Sersic index, which determines the concentration
    of the density profile. The derivative of :math:`I(R)` is also analytic:

    .. math::

        \frac{{\rm d}I(R)}{{\rm d}R} = -\frac{I_e\,b_n}{n\,R_e} \exp\left\{ -b_n\left[\left(\frac{R}{R_e}\right)^\frac{1}{n}-1\right] \right\} \left(\frac{R}{R_e}\right)^{\frac{1}{n}-1}.

    With this expression, recalling that :math:`e=\sqrt{1-q^2}` is the intrinsic ellipticity,
    the circular velocity profile of the bulge becomes (e.g. Eq. 2.132 in [BT2008]_):

    .. math::

        V^2_{\rm bulge}(r) = \mathcal{C} \int_0^r \left[ \int_m^\infty \frac{ \exp\left\{ -b_n\left[\left(R/R_e\right)^{1/n}-1\right] \right\} \left(R/R_e\right)^{1/n-1} }{\sqrt{R^2-m^2}} {\rm d}R \right]\frac{m^2}{\sqrt{r^2-e^2m^2}}{\rm d}m,

    .. math::

        \mathcal{C} = \frac{4\,G q I_e\,b_n}{nR_e} \sqrt{\sin^2i+\frac{1}{q}\cos^2i}.

    As in :py:func:`vcdisk.vcbulge`, it is convenient to change integration
    variables since both integrals present singularities: :math:`u={\rm arccosh}{(R/m)}`
    is used for the inner integral in :math:`{\rm d}R`, while :math:`t=\arcsin{me/r}`
    is used for the outer integral in :math:`{\rm d}m`.

    References
    ----------

    .. [Sersic68] Sersic, 1968, Argentina: Observatorio Astronomico. Atlas de Galaxias Australes. `https://ui.adsabs.harvard.edu/abs/1968adga.book.....S/ <https://ui.adsabs.harvard.edu/abs/1968adga.book.....S/>`_


    """

    # check rad
    if type(rad) is list: rad = np.asarray(rad)
    if type(rad) is np.ndarray:
        pass
    else:
        raise TypeError("rad must be a list or np.array")
    if len(rad)<1:
        raise ValueError("rad must be an array of size >1")
    if np.isnan(np.sum(rad)):
        raise ValueError("there are NaNs in rad. Maybe try with np.nan_to_num(rad)")

    # checks on m, re, n
    mtot = check_float(mtot, 'mtot')
    re   = check_float(re, 're')
    n    = check_float(n,  'n')

    if n<=0 or n>8:
        raise ValueError("n must be 0 < n <= 8")

    # checks on q and inc
    q, inc = check_q_inc(q, inc)

    # ellipticity
    e=np.sqrt(1-q**2)

    # geometric factor
    # see Sec. 2.1 in Noordermeer (2008)
    geom_factor = np.sqrt(np.sin(np.radians(inc))**2 + np.cos(np.radians(inc))**2/q**2)

    # Sersic profile
    sers = sersic(mtot, re, n)

    with np.errstate(over='ignore'):
        rhom = np.array([-1/np.pi * geom_factor *
                         quad(lambda u: sers.deriv(m*np.cosh(u)), 0, np.inf)[0] for m in rad])
        # too large value of u may cause numerical overflow on cosh
        if np.isnan(np.sum(rhom)):
            rhom = np.array([-1/np.pi * geom_factor *
                         quad(lambda u: sers.deriv(m*np.cosh(u)), 0, 50.0)[0] for m in rad])

    v_circ = np.array([np.sqrt(4*np.pi*G_GRAV * q *
                               quad(lambda u: np.interp(R/e*np.sin(u), rad, rhom)*R**2/e**3*np.sin(u)**2,
                                    0, np.arcsin(e))[0]) for R in rad])

    return v_circ


class sersic():
    r"""
    Class for Sersic profiles.

    This class creates a [Sersic68]_ profile from the total mass, the effective
    radius (i.e. the radius containing 50% of the total mass), and the Sersic
    index. It has two implemented methods:

    * ``__call__`` returns the value

    .. math::

        I(R) = I_e \exp\left\{ -b_n\left[\left(\frac{R}{R_e}\right)^\frac{1}{n}-1\right] \right\},

    * ``deriv`` returns the first derivative

    .. math::
        \frac{{\rm d}I(R)}{{\rm d}R} = -\frac{I_e\,b_n}{n\,R_e} \exp\left\{ -b_n\left[\left(\frac{R}{R_e}\right)^\frac{1}{n}-1\right] \right\} \left(\frac{R}{R_e}\right)^{\frac{1}{n}-1},

    where :math:`b_n = 2n -1/3 + (4/405)n^{-1} + o(n^{-2})` (see [CiottiBertin1999]_) and
    :math:`I_e` is the surface density at the effective radius :math:`R_e` and it is
    related to the total mass as

    .. math::

        I_e = \frac{M}{2\pi n R_e^2} \frac{b_n^{2n}}{e^{b_n}\Gamma(2n)},

    where :math:`\Gamma` is the complete gamma function (see [GrahamDriver05]_).

    :param mtot: total mass in :math:`\rm M_\odot`.
    :type mtot: float
    :param re: effective radius in :math:`\rm kpc`.
    :type re: float
    :param n: Sersic index :math:`0 < n \leq 8`.
    :type n: float

    References
    ----------

    .. [CiottiBertin1999] Ciotti & Bertin, 1999, A&A, 352, 447. Analytical properties of the :math:`R^{1/m}` law. `https://ui.adsabs.harvard.edu/abs/1999A%26A...352..447C/ <https://ui.adsabs.harvard.edu/abs/1999A%26A...352..447C/>`_
    .. [GrahamDriver05] Graham & Driver, 2005, PASA, 22, 118. A Concise Reference to (Projected) Sérsic :math:`R^{1/n}` Quantities, Including Concentration, Profile Slopes, Petrosian Indices, and Kron Magnitudes `https://doi.org/10.1071/AS05001 <https://doi.org/10.1071/AS05001>`_

    """
    def __init__(self, mtot, re, n):
        self.mtot, self.re, self.n = mtot, re, n
        self.bn = 2*self.n -1./3. +4./405./self.n
        self.Ie = self.mtot / (2*np.pi*self.n*self.re**2) * self.bn**(2*self.n) / np.exp(self.bn) / gamma(2*self.n)

    def __call__(self, R):
        """
        Returns the value :math:`I(R)`
        """
        return self.Ie * np.exp(-self.bn * ((R/self.re)**(1/self.n)-1))

    def deriv(self, R):
        """
        Returns the value :math:`I'(R)`
        """
        return -self.Ie*self.bn/(self.n*self.re) * np.exp(-self.bn*((R/self.re)**(1/self.n)-1)) * (R/self.re)**(1/self.n-1.0)

def check_rad(r, name='r'):
    """
    Type checks on the common input ``r``.
    """
    if type(r) is list: r = np.asarray(r)
    if type(r) is np.ndarray:
        pass
    else:
        raise TypeError(name+" must be a list or np.array")
    if len(r)<1:
        raise ValueError(name+" must be an array of size >1")
    if np.isnan(np.sum(r)):
        raise ValueError("there are NaNs in "+name+". Maybe try with np.nan_to_num("+name+")")

    return r

def check_rad_sb(rad, sb):
    """
    Type checks on the common inputs ``rad`` and ``sb``.
    """
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

    # nan check
    if np.isnan(np.sum(rad)):
        raise ValueError("there are NaNs in rad. Maybe try with np.nan_to_num(rad)")
    if np.isnan(np.sum(sb)):
        raise ValueError("there are NaNs in sb. Maybe try with np.nan_to_num(sb)")

    return rad, sb

def check_float(x, name):
    """
    Type check on float
    """
    if type(x) is float:
        pass
    else:
        try:
            x = float(x)
        except:
            raise TypeError(name+" must be a float")

    return x

def check_q_inc(q, inc):
    """
    Type checks on the common inputs ``q`` and ``inc``.
    """
    q   = check_float(q, 'q')
    inc = check_float(inc, 'inc')

    if q<=0.0:
        raise ValueError("q must be positive")
    if q>1.0:
        raise ValueError("q must be <1, can't do prolate bulges")

    if inc<0.0 or inc>90.0:
        raise ValueError("the inclination in degrees must be 0 <= inc <= 90")

    return q, inc
