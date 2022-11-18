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
    'vc_razorthin',
    '_integrand',
]


import numpy as np
from scipy.integrate import simps
from scipy.special import ellipk, ellipe, i0, i1, k0, k1

# constants
G_GRAV = 4.301e-6 # kpc km^2 s^-2 M_sun^-1

# exponential disc circular velocity
def vc_razorthin(R, Md, Rd):
    y = R/2./Rd
    return np.nan_to_num(np.sqrt(2*G_GRAV*Md/Rd*y**2*(i0(y)*k0(y)-i1(y)*k1(y))))


def _integrand(rad, r, smdisc, xxii, z0=0.3, verbose=False):
    u, xi = np.copy(rad), xxii[:,None]
    X = (r**2+u**2+xi**2)/(2*r*u)
    p = X-np.sqrt(X**2-1)

    # these are different choices for the vertical density profile
    # rho_uxi = smdisc / (2*z0) * (np.cosh(xi/z0))**-2
    rho_uxi = smdisc * (np.cosh(xi/z0))**-2 / z0 / 2.2
    drho_du = np.gradient(rho_uxi, u, axis=1)

    if (r==rad[0]) & (verbose):
        print ("%.3e" % (2*np.pi*simps(simps(rho_uxi*u, u), xxii)))
    return 2/np.pi*np.sqrt(u/r*p) * (ellipk(p) - ellipe(p)) * drho_du

def get_vstar_disc(RR, LL, z0=0.3):
    _rr, xxii = np.linspace(RR.min(), RR.max(), 100), np.logspace(-2,3,1000)
    smdisc = np.interp(_rr, RR, LL)
    return np.array([np.sqrt(-4*np.pi*G_GRAV*r*simps(simps(_integrand(_rr, r, smdisc, xxii, z0=z0).T, xxii), _rr))
                     for r in RR])
