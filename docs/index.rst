.. elle documentation master file, created by
   sphinx-quickstart on Thu Feb  4 16:25:15 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

elle
====

elle is a toolkit for modelling the line profile distortions and surface
velocities of a star due to a transiting planet. The software is a Python
implementation of the `reloaded <https://arxiv.org/abs/1602.00322>`_ 
Rossiter-McLaughlin method to measure `spin-orbit angles
<https://arxiv.org/abs/1709.06376>`_ in exoplanet systems. The method is
currently only applicable to the cross-correlation functions (CCF) of stellar
spectra observed with stabilized high-resolution spectrographs. Some of the
features of *elle* include:

* various convenience functions for e.g. resampling, normalizing, and fitting CCFs 
  to recover the planet-occulted light.
* brightness-weighted stellar surface velocities due to rotation and
  centre-to-limb convective variation.
* publication-ready plots of the planet *trace* and stellar surface
  velocities.

*elle* is available on `Github <https://github.com/vedad/elle>`_. If you run
into any trouble please open an issue.


.. toctree::
   :maxdepth: 2
   :caption: Using elle

   user/install
   user/api

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   tutorials/quickstart

.. Indices and tables
.. ==================

..
  * :ref:`genindex`

..
  * :ref:`modindex`

..
  * :ref:`search`
