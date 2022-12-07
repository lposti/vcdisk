``vcdisk``: Rotation curves of disk galaxies
############################################

A minimal python module to solve Poisson's equation in galactic disks.
This is useful to compute i) the circular velocity of a **thick galaxy disk**
with arbitrary surface density and vertical density profile, and ii) the circular
velocity of a **flattened bulge** with arbitrary surface density.

The ``vcdisk`` module is a compact toolbox indispensable for modelling
galaxy rotation curves, since it allows to determine the gravitational field
of the observed baryonic matter (stars, gas, dust etc.) just from photometric
observations, with minimal assumptions.

:py:func:`~vcdisk.vcdisk` calculates the radial force on the disk plane of a
thick disk component, with an arbitrarily sampled surface density profile and a
user-defined vertical density profile. :py:func:`~vcdisk.vcbulge` computes
the gravitational field on the mid-plane of an axisymmetric spheroidal oblate (or
spherical) bulge component, with an arbitrarily sampled surface density profile.

.. image:: vcdisk_banner.png

Install
=======

The package can be installed using

.. code-block:: bash

   pip install vcdisk

Getting started
===============

.. toctree::
  :maxdepth: 1

  notebooks/usage.ipynb

Reference/API
=============

.. toctree::
  :maxdepth: 2

  api
  changelog
