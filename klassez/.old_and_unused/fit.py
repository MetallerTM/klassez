#! /usr/bin/env python3

import numpy as np
from . import misc, fit


def build_baseline(ppm_scale, C, L=None):
    """
    Builds the baseline calculating the polynomion with the given coefficients, and summing up to the right position.

    .. error::

        Old function!! Legacy


    Parameters
    ----------
    ppm_scale : 1darray
        ppm scale of the spectrum
    C : list
        Parameters coefficients. No baseline corresponds to False.
    L : list
        List of window regions. If it is None, the baseline is built on the whole ``ppm_scale``

    Returns
    ----------
    baseline : 1darray
        Self-explanatory.
    """

    # Just in case you have a unique polynomion
    if isinstance(C, list) is False:
        C = [np.array(C)]
    if L is None:
        L = ppm_scale[0], ppm_scale[-1]
    if isinstance(L, list) is False:
        L = [L]

    poly = []
    baseline = np.zeros_like(ppm_scale)
    for k, coeff in enumerate(C):
        if coeff is False:      # No baseline
            continue
        lims = misc.ppmfind(
                ppm_scale, L[k][0])[0], misc.ppmfind(ppm_scale, L[k][1])[0]     # Find the indexes on ppm_scale
        lims = min(lims), max(lims)                 # Sort them to avoid stupid mistakes
        size_x = int(np.abs(lims[0] - lims[1]))     # Size of the polynomion scale
        x = np.linspace(0, 1, size_x)[::-1]         # Build the polynomion scale as the one of the fit
        poly.append(misc.polyn(x, coeff))           # Computes the polynomion
        baseline[lims[0]:lims[1]] += poly[-1]       # Sum it to the baseline in the correct position
    return baseline


def join_par(filenames, ppm_scale, joined_name=None):
    """
    Load a series of parameters fit files. Join them together, returning a unique array of signal parameters, a list of coefficients for the baseline, and a list of tuples for the regions.
    Also, uses the coefficients and the regions to directly build the baseline according to the ppm windows.

    .. error::

        Old function!! Legacy


    Parameters
    ----------
    filenames : list
        List of directories of the input files.
    ppm_scale : 1darray
        ppm scale of the spectrum. Used to build the baseline
    joined_name : str or None
        If it is not None, concatenates the files in the list ``filenames`` and saves them in a single file named ``joined_name``.

    Returns
    ----------
    V : 2darray
        Array of joined signal parameters
    C : list
        Parameters coefficients. No baseline corresponds to False.
    L : list
        List of window regions.
    baseline : 1darray
        Baseline built from ``C`` and ``L``.
    """
    if isinstance(joined_name, str):
        f = open(joined_name, 'w')

    V = []      # Values
    C = []      # Polynomion
    L = []      # Window limits

    for k in range(len(filenames)):
        # Read file
        tmp = fit.read_par(filenames[k])
        V.append(tmp[0])
        C.append(tmp[1])
        L.append(tmp[2])

    if isinstance(joined_name, str):
        for k in range(len(filenames)):
            fit.write_par(V[k], C[k], L[k], f)
            if k < len(filenames) - 1:  # Add separator
                f.write('\n*{}*\n\n'.format('-'*64))
            else:                       # Add closing statement
                f.write('\n***{:^60}***'.format('END OF FILE'))
                f.close()
        print('Joined parameters are saved in {}'.format(joined_name))

    # Check if the regions superimpose
    sup = np.zeros_like(ppm_scale)
    for k in range(len(L)):
        lims = misc.ppmfind(ppm_scale, L[k][0])[0], misc.ppmfind(ppm_scale, L[k][1])[0]  # Find the indexes on ppm_scale
        lims = min(lims), max(lims)                 # Sort them to avoid stupid mistakes
        sup[lims[0]:lims[1]] += 1                   # Add 1 in the highlighted region

    # If there are superimposed regions, do not compute the baseline
    if np.any(sup > 1):
        print('Warning: Superimposed regions detected! Baseline not computed.')
        baseline = np.zeros_like(ppm_scale)
    else:
        baseline = fit.build_baseline(ppm_scale, C, L)

    # Stack the signal parameters to make a unique matrix
    V = np.concatenate(V, axis=0)

    return V, C, L, baseline


def calc_fit_lines(ppm_scale, limits, t_AQ, SFO1, o1p, N, V, C=False):
    """
    Given the values extracted from a fit input/output file, calculates the signals, the total fit function, and the baseline.

    .. error::

        Old function!! Legacy


    Parameters
    ----------
    ppm_scale : 1darray
        PPM scale of the spectrum
    limits : tuple
        (left, right) in ppm
    t_AQ : 1darray
        Acquisition timescale
    SFO1 : float
        Larmor frequency of the nucleus /ppm
    o1p : float
        Pulse carrier frequency /ppm
    N : int
        Size of the final spectrum.
    V : 2darray
        Matrix containing the values to build the signals.
    C : 1darray
        Baseline polynomion coefficients. False to not use the baseline

    Returns
    ----------
    sgn : list
        Voigt signals built using ``V``
    Total : 1darray
        sum of all the ``sgn``
    baseline : 1darray
        Polynomion built using ``C``. False if ``C`` is False.
    """
    lim1 = misc.ppmfind(ppm_scale, limits[0])[0]
    lim2 = misc.ppmfind(ppm_scale, limits[1])[0]
    lim1, lim2 = min(lim1, lim2), max(lim1, lim2)

    x = np.linspace(0, 1, ppm_scale[lim1:lim2].shape[-1])[::-1]
    # Make the polynomion only if C contains its coefficients
    if C is False:
        baseline = np.zeros_like(x)
    else:
        baseline = misc.polyn(x, C)

    # Make the signals
    sgn = []
    Total = np.zeros_like(x) + 1j*np.zeros_like(x)
    for i in range(V.shape[0]):
        sgn.append(fit.make_signal(t_AQ, *V[i], SFO1=SFO1, o1p=o1p, N=N))
        Total += sgn[i][lim1:lim2]

    return sgn, Total, baseline
