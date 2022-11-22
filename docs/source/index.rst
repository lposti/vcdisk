Rotation curves of thick truncated galaxy disks
========================================
This is a minimal python package to solve Poisson's equation and to compute the
circular velocity curve of a truncated disk with non-zero thickness and arbitrary
radial density profile. This implements the algorithm of
[Casertano (1983)](https://ui.adsabs.harvard.edu/abs/1983MNRAS.203..735C)
The package can be installed using pip::
    pip install vcdisk

Contents
--------

.. toctree::

   usage
   api

Reference/API
-------------
.. currentmodule:: vcdisk
.. autosummary::
   :toctree: api
   :nosignatures:
   vcdisk
   _integrand
   vc_razorthin
