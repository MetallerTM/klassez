.. _intro_klassez:

Introduction
============

**KLASSEZ** is a python package written to handle 1D and 2D NMR data.
The aim of the project is to provide a toolkit, consisting of
"black-box" functions organized in modules, that could be used to read,
process and analyze such data in a flexible manner, so to adapt to the
needs of the individual users. However, the open-source nature of the
package grants the user the chance to open the lid of these black-boxes
and understand the gears that stand behind the function call.

The development of the toolkit started with ``python 3.8`` and therefore
it is compatible with that version. Nevertheless, the use of
``python 3.10`` is advised.

The key objects provided by **KLASSEZ** are the classes ``Spectrum_1D``
and ``Spectrum_2D``, that are able to fulfil the aims of the package
with a few lines of code. The classes are able to read both simulated
(i.e. generated with a custom-made input file) and experimental
datasets. The latter feature was tested with Bruker data after the
removal of the digital filter (run command ``convdta`` in TopSpin), but
should be compatible with other kind of spectrometers, thanks to the
remarkable work made by J. J. Helmus and coworkers with their
**nmrglue** package [1]_. Either the FID or the spectrum processed with
external solver can be read from **KLASSEZ** by using the classes
``Spectrum_nD`` or ``pSpectrum_nD``, respectively.

The ``processing`` module, besides the classical functions used for the
processing of NMR data (window functions, Fourier transform, etc.),
includes denoising algorithms based on Multivariate Curve
Resolution [2]_ and on Cadzow method [3]_. Details are illustrated in
the description of the functions.

Functions to show and analyze data in real time are provided, with
dedicated GUIs. However, it is better to rely on the standalone
functions, enclosed in the single modules, to save the figures. In fact,
the ``figures`` module offers a wide plethora of functions (all based on
``matplotlib``) to plot the data with a high degree of customization for
the appearance.

The fitting functions use ``lmfit`` to build the initial guess and to
minimize the difference between the experimental data and the model,
generated with a Voigt profile in the time domain and then
Fourier-transformed, in the least-square sense (employing the
Levenberg-Marquardt algorithm implemented in ``scipy``). For this
purpose, the class ``Voigt_fit`` of the ``fit`` module includes
attribute functions to construct an initial guess interactively, fit the
data, and save the parameters in dedicated files.

Regarding the development of the package, I would like to acknowledge
Letizia Fiorucci for her contribution in the design and the
implementation of several functions, and for the alpha-testing.

.. [1]
   https://www.nmrglue.com/

.. [2]
   `Multivariate Curve Resolution: 50 years addressing the mixture
   analysis problem - A
   review <https://www.sciencedirect.com/science/article/pii/S0003267020310771>`__

.. [3]
   `Denoising NMR time-domain signal by singular-value decomposition
   accelerated by graphics processing
   units <https://www.sciencedirect.com/science/article/pii/S0926204014000356?via%3Dihub>`__
