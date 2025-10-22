Installation
============

KLASSEZ can be installed from PyPI through:

::

   pip install klassez

Another option is to clone the `GitHub repository`_ and install the package with ``pip`` directly from the source folder:


.. _GitHub repository: https://github.com/MetallerTM/klassez

::

   git clone git@github.com:Metallertm/klassez.git

   cd klassez
   pip install .

The required dependencies are sorted out automatically in either case.


Import instructions
===================

Initialize the package by writing, at the top of your file:

::

   from klassez import *

This line executes the following code:

::

        import os
        import sys
        import numpy as np
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        from copy import deepcopy
        from pprint import pprint

        from . import fit, misc, sim, figures, processing, anal

        from .Spectra import Spectrum_1D, pSpectrum_1D, Spectrum_2D, pSpectrum_2D, Pseudo_2D

        from .config import CM, CM_2D, COLORS, cron



This means these can be not imported in your code, as **KLASSEZ**
already does it for you.

An alternative, safer version to prevent overwriting of custom functions
is:

::

   import klassez as kz

In this case, additional packages for the main script must be declared
explicitely.



Initializing ``KLASSEZ`` also grants access to ``CM`` and ``COLORS``.

``CM`` is a dictionary of colormaps taken from ``seaborn`` and saved in
a dictionary whose keys are their names, so that also ``matplotlib`` can
use them. You can inspect the keys through:

::

   print(CM.keys())

There is a restricted list of colormaps, ``CM_2D``, that should be used
for visualizing 2D spectra.

``COLORS`` is:

::

   colors = [ 'tab:blue', 'tab:red', 'tab:green', 'tab:orange', 'tab:cyan', 'tab:purple', 'tab:pink', 'tab:gray', 'tab:brown', 'tab:olive', 'salmon', 'indigo', 'm', 'c', 'g', 'r', 'b', 'k', ]

repeated cyclically ten times and stored as tuple.

Other two "quality of life" variables are ``figures.figsize_small`` and
``figures.figsize_large``, which correspond to figure panel sizes of
:math:`3.59 \times 2.56` inches and :math:`15 \times 8` inches,
respectively. The former suits well for saving figures of spectra with
font sizes of about 10 pt, whereas the latter are best for GUIs and
withstand font sizes of about 14 pt.

For NMR: the variable ``sim.gamma`` is a dictionary containing the
gyromagnetic ratio, in MHz/T, of all the magnetically-active nuclei. For
instance:

::

   print(sim.gamma['13C'])
   >>> 10.70611

A decorator function called ``cron`` is defined in the top-level script
``config``, and imported by ``__init__``, so that you can use it after
writing:

::

   from klassez import cron

This decorator allows to measure the runtime of a function, and print it
on standard output once it ended.

