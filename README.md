# `vcdisk`: Rotation curves of thick galaxy disks
![](coverage.svg)
[![DOI](https://zenodo.org/badge/566744612.svg)](https://zenodo.org/badge/latestdoi/566744612)

A minimal python module to solve Poisson's equation in a thick galactic disk.
This is useful to compute the circular velocity on the disk plane of a galaxy
with observed surface density profile and arbitrary, user-defined vertical profile.

`vcdisk` implements the algorithm of
[Casertano (1983)](https://ui.adsabs.harvard.edu/abs/1983MNRAS.203..735C), which
calculates the radial force on the disk plane as a 2-D integral (their Eq. 4) and
then derives the velocity of a circular orbit on the disk plane.

### Read the full documentation at: [https://vcdisk.readthedocs.io/en/latest/](https://vcdisk.readthedocs.io/en/latest/)
