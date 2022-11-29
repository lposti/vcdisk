=========
Changelog
=========

version: 0.2.0
--------------

* Added the option of a flaring disk in vcdisk
* Fixed bugs on the normalization of the vertical profiles
* Added automatic calculation of the normalization of ``rhoz`` with :func:`scipy.integrate.quad`
* Added vcdisk_alt_Cu93.py, which implements the formula of `Cuddeford (1993) <https://ui.adsabs.harvard.edu/abs/1993MNRAS.262.1076C/>`_ for Phi(R,z), i.e. above the plane
* Added full docs for flaring option, including an example in the notebook usage.ipynb

version: 0.1.2
--------------

* Added zenodo DOI.

version: 0.1.1
--------------

* Working `RTD documentation <https://vcdisk.readthedocs.io/en/latest/>`_

version: 0.1.0
--------------

* First released on November 25, 2022. Copyright, Lorenzo Posti. BSD-2 License.
* Python project specified with pyproject.toml file
* Added unit testing: coverage 100%
* Added full Sphinx documentation
* Added "Getting started" jupyter notebook tutorial
* Uploaded the package to PyPI
