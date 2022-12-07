# `vcdisk`: Rotation curves of disk galaxies
![](coverage.svg)
[![DOI](https://zenodo.org/badge/566744612.svg)](https://zenodo.org/badge/latestdoi/566744612)
[![Documentation Status](https://readthedocs.org/projects/vcdisk/badge/?version=latest)](https://vcdisk.readthedocs.io/en/latest/?badge=latest)

A minimal python module to solve Poisson's equation in galactic disks.
This is useful to compute i) the circular velocity of a **thick galaxy disk**
with arbitrary surface density and vertical profiles, and ii) the circular
velocity of a **flattened bulge** with arbitrary surface density.

The `vcdisk` module is a compact toolbox indispensable for modelling
galaxy rotation curves, since it allows to determine the gravitational field
of the observed baryonic matter (stars, gas, dust etc.) just from photometric
observations, with minimal assumptions.

`vcdisk.vcdisk` calculates the radial force on the disk plane of a
thick disk component, with an arbitrarily sampled surface density profile and a
user-defined vertical density profile. `vcdisk.vcbulge` computes
the gravitational field on the mid-plane of an axisymmetric spheroidal oblate (or
spherical) bulge component, with an arbitrarily sampled surface density profile.

### Read the full documentation at: [https://vcdisk.readthedocs.io/en/latest/](https://vcdisk.readthedocs.io/en/latest/)
