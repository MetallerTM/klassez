#! /usr/bin/env python3

import numpy as np
from numpy import linalg
import scipy
from scipy import linalg as slinalg
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.widgets import Button, Cursor, LassoSelector
from matplotlib.path import Path as MplPath
import lmfit
from datetime import datetime
import warnings
from pathlib import Path
from copy import deepcopy

from . import fit, misc, sim, processing, anal, gui
from .config import cprint
from .Spectra import Spectrum_1D

print = cprint

"""
Contains a series of processing functions for different purposes
"""

# CPMG processing


def sum_echo_train(datao, n, n_echoes, i_p=0):
    """
    Sum up a CPMG echo-train FID into echoes so to be enchance the SNR.
    This function calls ``processing.split_echo_train`` with the same parameters.

    Parameters
    ----------
    datao : ndarray
        FID with an echo train on its last dimension
    n : int
        number of points that separate one echo from the next
    n_echoes : int
        number of echoes to sum
    i_p : int
        Number of offset points

    Returns
    -------
    data_p : ndarray
        Summed echoes

    .. seealso::

        :func:`klassez.gui.interactive_echo_param`
    """
    # Separate the echoes
    data = processing.split_echo_train(datao, n, n_echoes, i_p)
    # Sum on the first dimension
    data_p = np.sum(data, axis=0)

    return data_p


def split_echo_train(datao, n, n_echoes, i_p=0):
    """
    Separate a CPMG echo-train FID into echoes so to be processed separately.
    The first decay, i.e. the native FID, is extracted, and corresponds to echo number 0.
    Then, for each echo, the left side (reversed) is summed up to its right part.

    Parameters
    ----------
    datao : ndarray
        FID with an echo train on its last dimension
    n : int
        number of points that separate one echo from the next
    n_echoes : int
        number of echoes to extract. If it is 0, extracts only the first decay
    i_p : int
        Number of offset points

    Returns
    -------
    data_p : (n+1)darray
        Separated echoes

    .. seealso::

        :func:`klassez.gui.interactive_echo_param`
    """
    # Take account of the offset points
    data = datao[..., i_p:]
    # nm = middle point. +1 if n is odd
    if np.mod(n, 2) == 0:
        nm = n // 2
    else:
        nm = n // 2 + 1

    # Where to save the echoes
    datap = []
    datap.append(datao[..., :nm])   # Add first decay

    for i in range(n_echoes):
        c = (i+1)*n                         # Echo centre
        A = slice(c-nm+1, c+1)              # Left part to echo centre
        B = slice(c, c+nm)                  # Right part from echo centre

        datal = data[..., A][..., ::-1]      # Left part, reversed
        datar = data[..., B]                # Right part

        # Reversing in time means to change sign to the imaginary part
        if np.iscomplexobj(data):
            datal = np.conj(datal)
        datap.append(datal + datar)  # Sum up
    # Create the output data by stacking the echoes. This adds a dimension
    data_p = np.stack(datap)

    return data_p

# -----------------------------------------------------------------------


def quad(fid):
    """
    Subtracts from the FID the arithmetic mean of its last quarter. The real and imaginary channels are treated separately.

    Parameters
    ----------
    fid : ndarray
        Self-explanatory.

    Returns
    -------
    fid : ndarray
        Processed FID.
    """
    size = fid.shape[-1]
    qsize = size//4
    avg_re = np.average(fid[..., -qsize:].real)
    avg_im = np.average(fid[..., -qsize:].imag)
    fid.real = fid.real - avg_re
    fid.imag = fid.imag - avg_im
    return fid


def qpol(fid):
    """
    Fits the FID with a 4-th degree polynomion, then subtracts it from the original FID. The real and imaginary channels are treated separately.

    Parameters
    ----------
    fid : ndarray
        Self-explanatory.

    Returns
    -------
    fid_corr : ndarray
        Processed FID
    """
    # Fits the FID with a 4th degree polinomion
    size = fid.shape[-1]
    x = np.linspace(0, size, size)

    coeff_re = fit.lsp(fid.real, x, n=5)
    coeff_im = fit.lsp(fid.imag, x, n=5)
    c = coeff_re + 1j*coeff_im

    fid_corr = fid - misc.polyn(x, c)
    return fid_corr

# -------------------------------------------------------------------------------------------------------------------------------------------------------
# window functions


def qsin(data, ssb):
    r"""
    Sine-squared apodization.

    .. math::

        a(x) = \sin^2 \biggl[\frac{\pi}{SSB} + \pi \biggl(1 - \frac{1}{SSB}\biggr) x \biggr]

    Parameters
    ----------
    data: ndarray
        FID to be processed
    ssb: int
        Sine bell shift.

    Returns
    -------
    datap: ndarray
        Apodized data

    .. seealso::

        :func:`klassez.processing.apodf`
    """

    if ssb == 0 or ssb == 1:
        off = 0
    else:
        off = 1/ssb
    end = 1
    size = data.shape[-1]
    apod = np.power(np.sin(np.pi * off + np.pi * (end - off) * np.arange(size) / (size)).astype(data.dtype), 2).astype(data.dtype)
    return apod * data


def sin(data, ssb):
    r"""
    Sine apodization.

    .. math::

        a(x) = \sin \biggl[\frac{\pi}{SSB} + \pi \biggl(1 - \frac{1}{SSB}\biggr) x \biggr]

    Parameters
    ----------
    data: ndarray
        FID to be processed
    ssb: int
        Sine bell shift.

    Returns
    -------
    datap: ndarray
        Apodized data

    .. seealso::

        :func:`klassez.processing.apodf`

    """
    if ssb == 0 or ssb == 1:
        off = 0
    else:
        off = 1/ssb
    end = 1
    size = data.shape[-1]
    apod = np.sin(np.pi * off + np.pi * (end - off) * np.arange(size) / (size)).astype(data.dtype)
    return apod * data


def em(data, lb, sw):
    r"""
    Exponential apodization

    .. math::

        a(x) = \exp\biggl[-\pi \frac{LB}{2\,SW} x \biggr]

    Parameters
    ----------
    data: ndarray
        Input data
    lb: float
        Lorentzian broadening. It should be positive.
    sw: float
        Spectral width /Hz

    Returns
    -------
    datap: ndarray
        Apodized data

    .. seealso::

        :func:`klassez.processing.apodf`
    """
    lb = lb / (2*sw)
    apod = np.exp(- np.pi * np.arange(data.shape[-1]) * lb).astype(data.dtype)
    return apod * data


def gm(data, lb, gb, gc, sw):
    r"""
    Gaussian apodization.
    The parameter ``lb`` controls the sharpening factor of a rising exponential, and behaves exactly as in ``processing.em``.
    In contrast, ``gb`` controls the gaussian decay factor.
    ``gc`` moves the central point of the gaussian filter.

    .. math::

        a(x) = \exp\biggl\{ -\pi \frac{LB}{SW} x - \biggl[ \pi \frac{GB}{SW} \biggl(GC (N-1) -x \biggr) \biggr]^2 \biggr\}

    Apply this function VERY CAREFULLY. Choose the right values through the interactive processing.

    Parameters
    ----------
    data : ndarray
        Input data
    lb : float
        Lorentzian sharpening /Hz. It should be negative.
    gb : float
        Gaussian broadening. It should be positive.
    gc : float
        Gaussian center, relatively to the FID length: :math:`0 \leq g_c \leq 1`
    sw : float
        Spectral width /Hz

    Returns
    -------
    pdata : ndarray
        Processed data

    .. seealso::

        :func:`klassez.processing.apodf`
    """
    size = data.shape[-1]
    x = np.arange(size)
    A = - np.pi * lb / sw * x
    B = np.pi * gb / sw * (gc * (size - 1) - x)
    apod = np.exp(A - B**2)
    return apod * data


def gmb(data, lb, gb, sw):
    r"""
    Bruker-style Gaussian apodization.

    .. math::

        a(x) = \exp\biggl[ -N \frac{\pi}{SW} \biggl( LB\,x - \frac{LB}{2\,GB}x^2 \biggr) \biggr]

    Apply this function VERY CAREFULLY. Choose the right values through the interactive processing.

    Parameters
    ----------
    data : ndarray
        Input data
    lb : float
        Lorentzian sharpening /Hz. It should be negative.
    gb : float
        Gaussian broadening. It should be positive.
    sw : float
        Spectral width /Hz

    Returns
    -------
    pdata : ndarray
        Processed data

    .. seealso::

        :func:`klassez.processing.apodf`
    """
    size = data.shape[-1]
    x = np.arange(size)/size
    apod = np.exp(- np.pi / sw * size * (lb * x - lb / (2 * gb) * x**2))
    return apod * data

# zero-filling


def zf(data, size):
    """
    Zero-filling of ``data`` up to ``size`` in its last dimension.

    Parameters
    ----------
    data : ndarray
        Array to be zero-filled
    size : int
        Number of points of the last dimension after zero-filling

    Returns
    -------
    datazf : ndarray
        Zero-filled data

    .. seealso::

        :func:`klassez.processing.apodf`
    """
    def zf_pad(data, pad):
        size = list(data.shape)
        size[-1] = int(pad)
        z = np.zeros(size, dtype=data.dtype)
        return np.concatenate((data, z), axis=-1)
    zpad = size - data.shape[-1]
    if zpad <= 0:
        zpad = 0
    datazf = zf_pad(data, pad=zpad)
    return datazf

# Fourier transform


def ft(data0, alt=False, fcor=0.5):
    """
    Fourier transform in NMR sense.
    This means it returns the reversed spectrum.

    Parameters
    ----------
    data0 : ndarray
        Array to Fourier-transform
    alt : bool
        negates the sign of the odd points, then take their complex conjugate. Required for States-TPPI processing.
    fcor : float
        weighting factor for FID 1st point. Default value (0.5) prevents baseline offset

    Returns
    -------
    dataft : ndarray
        Transformed data
    """
    data = np.copy(data0)
    data[..., 0] = data[..., 0] * fcor
    if alt:
        data[..., 1::2] = data[..., 1::2] * -1
        data.imag = data.imag * -1
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        dataft = np.fft.fftshift(np.fft.fft(data, axis=-1).astype(data.dtype), -1)[..., ::-1]
    return dataft


def ift(data0, alt=False, fcor=0.5):
    """
    Inverse Fourier transform in NMR sense.
    This means that the input dataset is reversed before to do iFT.

    Parameters
    ----------
    data0 : ndarray
        Array to Fourier-transform
    alt : bool
        negates the sign of the odd points, then take their complex conjugate. Required for States-TPPI processing.
    fcor : float
        weighting factor for FID 1st point. Default value (0.5) prevents baseline offset

    Returns
    -------
    dataft : ndarray
        Transformed data
    """
    data = np.copy(data0)[..., ::-1]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        dataft = np.fft.ifft(np.fft.ifftshift(data, -1), axis=-1).astype(data.dtype)
    if alt:
        dataft[..., 1::2] = dataft[..., 1::2] * -1
        dataft.imag = dataft.imag * -1
    dataft[..., 0] = dataft[..., 0] / fcor
    return dataft


def rev(data):
    """
    Reverse data over its last dimension
    """
    datarev = data[..., ::-1]
    return datarev

    # phase correction


def ps(data, ppmscale=None, p0=None, p1=None, pivot=None, interactive=False, reference=None):
    """
    Applies phase correction on the last dimension of data.
    The pivot is set at the center of the spectrum by default.
    Missing parameters will be inserted interactively.

    Parameters
    ----------
    data : ndarray
        Input data
    ppmscale : 1darray or None
        PPM scale of the spectrum. Required for pivot and interactive phase correction
    p0 : float
        Zero-order phase correction angle /degrees
    p1 : float
        First-order phase correction angle /degrees
    pivot : float or None.
        First-order phase correction pivot /ppm. If None, it is the center of the spectrum.
    interactive : bool
        If True, all the parameters will be ignored and the interactive phase correction panel will be opened.
    reference : list of 1darray or Spectrum_1D object
        Reference spectrum to be used for phasing. Can be also given as ``[ppm, spectrum]``

    Returns
    -------
    datap : ndarray
        Phased data
    final_values : tuple
        Employed values of the phase correction. ``(p0, p1, pivot)``

    .. seealso::

        :func:`klassez.gui.interactive_phase_1D`
    """
    if p0 is None and p1 is None:
        interactive = True
    elif p0 is None and p1 is not None:
        p0 = 0
    elif p1 is None and p0 is not None:
        p1 = 0

    if not np.iscomplexobj(data):
        raise ValueError('Data is not complex! Impossible to phase')

    if ppmscale is None and interactive is True and pivot is not None:
        raise ValueError('PPM scale not supplied. Aborting...')

    if interactive is True and len(data.shape) < 2:
        datap, final_values = gui.interactive_phase_1D(ppmscale, data, reference)
    else:
        p0 = p0 * np.pi / 180
        p1 = p1 * np.pi / 180
        size = data.shape[-1]
        pvscale = np.arange(size) / size
        if pivot is None:
            pv = 0.5
        else:
            pv = (misc.ppmfind(ppmscale, pivot)[0] / size)
        apod = np.exp(1j * (p0 + p1 * (pvscale - pv))).astype(data.dtype)
        datap = data * apod
        final_values = p0*180/np.pi, p1*180/np.pi, pivot
    return datap, final_values


def eae(data):
    """
    Shuffles data if the spectrum is acquired with ``FnMODE = 'Echo-Antiecho'``.
    NOTE: introduces -90° phase shift in F1, to be corrected after the processing

    .. code-block:: python

        pdata = np.zeros_like(data)
        pdata[::2] = (data[::2].real - data[1::2].real) + 1j*(data[::2].imag - data[1::2].imag)
        pdata[1::2] = -(data[::2].imag + data[1::2].imag) + 1j*(data[::2].real + data[1::2].real)

    Parameters
    ----------
    data : 2darray
        FID in echo-antiecho format

    Returns
    -------
    pdata : 2darray
        FID in States-TPPI format
    """
    pdata = np.zeros_like(data)
    pdata[::2] = (data[::2].real - data[1::2].real) + 1j*(data[::2].imag - data[1::2].imag)
    pdata[1::2] = -(data[::2].imag + data[1::2].imag) + 1j*(data[::2].real + data[1::2].real)
    return pdata


def tp_hyper(data):
    """
    Computes the hypercomplex transpose of data.
    Needed for the processing of data acquired in a phase-sensitive manner in the indirect dimension.

    Parameters
    ----------
    data : 2darray
        Hypercomplex data to be transposed

    Returns
    -------
    datap : 2darray
        Transposed data
    """
    def ri2c(data):
        s = list(data.shape)
        s[-1] = s[-1]*2
        n = np.empty(s, data.real.dtype)
        n[..., ::2] = data.real
        n[..., 1::2] = data.imag
        return n

    def c2ri(data):
        temp = np.array(data.flat[0] + 1j*data.flat[1])
        s = list(data.shape)
        s[-1] = s[-1] // 2
        n = np.empty(s, temp.dtype)
        del temp
        n.real = data.real[..., ::2]
        n.imag = data.real[..., 1::2]
        return n
    datatp = np.array(c2ri(ri2c(data).T), dtype='complex64')
    return datatp


def unpack_2D(data):
    """
    Separates hypercomplex data into 4 distinct components

    Parameters
    ----------
    data : 2darray
        Hypercomplex matrix

    Returns
    -------
    rr : 2darray
        Real F2, Real F1
    ir : 2darray
        Imaginary F2, Real F1
    ri : 2darray
        Real F2, Imaginary F1
    ii : 2darray
        Imaginary F2, Imaginary F1
    """
    rr = data.real[::2]
    ir = data.imag[::2]
    ri = data.real[1::2]
    ii = data.imag[1::2]
    return rr, ir, ri, ii


def repack_2D(rr, ir, ri, ii):
    """
    Renconstruct hypercomplex 2D NMR data given the 4 components

    Parameters
    ----------
    rr : 2darray
        Real F2, Real F1
    ir : 2darray
        Imaginary F2, Real F1
    ri : 2darray
        Real F2, Imaginary F1
    ii : 2darray
        Imaginary F2, Imaginary F1

    Returns
    -------
    data : 2darray
        Hypecomplex matrix
    """
    data = np.empty((2*rr.shape[0], rr.shape[1]), dtype='complex64')
    data.real[::2] = rr
    data.imag[::2] = ir
    data.real[1::2] = ri
    data.imag[1::2] = ii
    return data


def td_eff(data, tdeff):
    """
    Uses only the first ``tdeff`` points of data. Equivalent to box-like apodization.
    ``tdeff`` must be a list as long as the dimensions:

    .. code-block:: python

        tdeff = [F1, F2, ..., Fn]

    Parameters
    ----------
    data : ndarray
        Data to be trimmed
    tdeff : list of int
        Number of points to be used in each dimension

    Returns
    -------
    datain : ndarray
        Trimmed data
    """
    datain = np.copy(data)

    def trim(datain, n):
        return datain[..., :n]

    ndim = len(datain.shape)
    # if tdeff is a number, make it list
    if isinstance(tdeff, int):
        L = tdeff
        tdeff = []
        tdeff.append(L)
        del L

    tdeff = tdeff[::-1]     # to obtain correct final shape

    if len(tdeff) != ndim:       # Check
        raise ValueError('Shape mismatch between datain and tdeff')

    X = tuple(np.roll(np.arange(ndim), 1))  # Roll the dimensions to the right

    for k in range(ndim):
        if tdeff[k]:
            datain = trim(datain, tdeff[k])
        datain = np.transpose(datain, X)

    return datain


def fp(data, wf=None, zf=None, fcor=0.5, tdeff=0):
    """
    Performs the full processing of a 1D NMR FID (``data``).
    Also works for pseudo-2D: only applies the processing on the last dimension.

    Parameters
    ----------
    data : ndarray
        Input data
    wf : dict
        ``{'mode': function to be used, 'parameters': different from each function}``
    zf : int
        final size of spectrum
    fcor : float
        weighting factor for the FID first point
    tdeff : int
        number of points of the FID to be used for the processing.

    Returns
    -------
    datap : ndarray
        Processed data

    .. seealso::

        :func:`klassez.processing.td_eff`

        :func:`klassez.processing.apodf`

        :func:`klassez.processing.zf`

        :func:`klassez.processing.ft`
    """
    # Correct the shape of tdeff for pseudo_2D spectra, xf2, and stuff
    if len(data.shape) > 1 and not isinstance(tdeff, (list, tuple, np.ndarray)):
        tdeff = [0 for w in range(len(data.shape)-1)] + [tdeff]
    # Window function
    # Rectangle
    datap = processing.td_eff(data, tdeff)
    # The real one
    datap *= processing.apodf(datap.shape, wf)

    # Zero-filling
    if zf is not None:
        datap = processing.zf(datap, zf)
    else:
        datap = processing.zf(datap, 0)
    if datap.shape[-1] < 2 * data.shape[-1]:
        warnings.warn('Not enough zerofilling for Hilbert Transform', stacklevel=2)

    # FT
    datap = processing.ft(datap, fcor=fcor)
    return datap


def apodf(size, wf):
    """
    Generates a function to be used as apodization function on the basis of the ``wf`` dictionary.
    The behavior is controlled by the ``mode`` key. Each ``mode`` reads the attributes of the corresponding positions, e.g.:

    .. code-block:: python

        mode:   em, qsin, qsin
        lb:      5,    0,    0
        ssb:     0,    2,    3

    will compute the function:

    .. code-block:: python

        processing.em(lb=5) * processing.qsin(ssb=2) * processing.qsin(ssb=3)

    Parameters
    ----------
    size : tuple
        Dimension of data to be windowed
    wf : dict
        Dictionary of window functions modes and parameters

    Returns
    -------
    apod_func : np.ndarray
        Custom apodization function of dimension ``size``

    .. seealso::

        :func:`klassez.processing.em`

        :func:`klassez.processing.sin`

        :func:`klassez.processing.qsin`

        :func:`klassez.processing.gm`

        :func:`klassez.processing.gmb`
    """

    # First input of the window functions, so that I get the function and not the processed data
    ones = np.ones(size)

    # Remove the bool to avoid it to raise errors
    if wf['mode'] is None:
        wf['mode'] = 'no'

    # For compatibility if one wants to use only one function:
    if isinstance(wf['mode'], str):
        for key, item in wf.items():    # loop in the dictionary
            if 'sw' not in key:     # SW is always the only one
                # Put the element in a list
                wf[key] = [item]
                # Remove extra lists that might appear
                wf[key] = misc.listsqueeze(wf[key])

    # Placeholder: we start from all ones
    apod_func = deepcopy(ones)
    # Loop on the modes
    for k, mode in enumerate(wf['mode']):
        # Case switch on the mode, creates tmp as apod. func. on the fly
        if mode == 'qsin':
            tmp = processing.qsin(ones, ssb=wf['ssb'][k])
        elif mode == 'sin':
            tmp = processing.sin(ones, ssb=wf['ssb'][k])
        elif mode == 'em':
            tmp = processing.em(ones, lb=wf['lb'][k], sw=wf['sw'])
        elif mode == 'gm':
            tmp = processing.gm(ones, lb=wf['lb_gm'][k], gb=wf['gb_gm'][k], sw=wf['sw'], gc=wf['gc'][k])
        elif mode == 'gmb':
            tmp = processing.gmb(ones, lb=wf['lb_gm'][k], gb=wf['gb'][k], sw=wf['sw'])
        else:   # It means "no"
            continue
        # Since it is after continue, only works if mode != no
        apod_func *= tmp

    return apod_func


def inv_fp(data, wf=None, size=None, fcor=0.5):
    """
    Performs the full inverse processing of a 1D NMR spectrum (``data``).

    Parameters
    ----------
    data : 1darray
        Spectrum
    wf : dict
        ``{'mode': function to be used, 'parameters': different from each function}``
    size : int
        initial size of the FID
    fcor : float
        weighting factor for the FID first point

    Returns
    -------
    pdata : 1darray
        FID
    """
    # IFT
    pdata = processing.ift(data, fcor=fcor)
    # Reverse zero-filling
    if size is not None:
        pdata = processing.td_eff(pdata, size)
    # Reverse window function
    if wf is not None:
        if wf['mode'] is None:
            apod = np.ones_like(pdata)
        if wf['mode'] == 'qsin':
            apod = processing.qsin(pdata, ssb=wf['ssb'])/pdata
        if wf['mode'] == 'sin':
            apod = processing.sin(pdata, ssb=wf['ssb'])/pdata
        if wf['mode'] == 'em':
            apod = processing.em(pdata, lb=wf['lb'], sw=wf['sw'])/pdata
        if wf['mode'] == 'gm':
            apod = processing.gm(pdata, lb=wf['lb'], gb=wf['gb'], sw=wf['sw'])/pdata
        pdata = pdata / apod
    return pdata


def xfb(data, wf=[None, None], zf=[None, None], fcor=[0.5, 0.5], tdeff=[0, 0], u=True, FnMODE='States-TPPI'):
    """
    Performs the full processing of a 2D NMR FID (``data``).
    The returned values depend on ``u``: if it is True, returns a sequence of 2darrays depending on FnMODE, otherwise just the complex/hypercomplex data after FT in both dimensions.

    Parameters
    ----------
    data : 2darray
        Input data
    wf : sequence of dict
        (F1, F2); ``{'mode': function to be used, 'parameters': different from each function}``
    zf : sequence of int
        final size of spectrum, (F1, F2)
    fcor : sequence of float
        weighting factor for the FID first point, (F1, F2)
    tdeff : sequence of int
        number of points of the FID to be used for the processing, (F1, F2)
    u : bool
        choose if to unpack the hypercomplex spectrum into separate arrays or not
    FnMODE : str
        Acquisition mode in F1

    Returns
    -------
    datap : 2darray or tuple of 2darray
        Processed data or tuple of 2darray

    .. seealso::

        :func:`klassez.processing.apodf`

        :func:`klassez.processing.zf`

        :func:`klassez.processing.ft`

        :func:`klassez.processing.unpack_2D`
    """

    # First of all, cut the data
    data = processing.td_eff(data, tdeff)

    original_size = data.shape

    # Processing the direct dimension
    # Window function
    data *= processing.apodf(data.shape, wf[1])
    # Zero-filling
    if zf[1] is not None:
        data = processing.zf(data, zf[1])
    else:
        data = processing.zf(data, 0)
    if data.shape[-1] < 2 * original_size[-1]:
        warnings.warn('Not enough zerofilling for Hilbert Transform in F2', stacklevel=2)
    # FT
    data = processing.ft(data, fcor=fcor[1])

    if FnMODE == 'QF-nofreq' or FnMODE == 'No':
        pass
    else:    # Processing the indirect dimension
        # If FnMODE is 'QF', do normal transpose instead of hyper
        if FnMODE == 'QF':
            data = data.T
        else:
            data = processing.tp_hyper(data)

        # Window function
        data *= processing.apodf(data.shape, wf[0])

        # Zero-filling
        if zf[0] is not None:
            data = processing.zf(data, zf[0])
        else:
            data = processing.zf(data, 0)
        if data.shape[-1] < 2 * original_size[0]:
            warnings.warn('Not enough zerofilling for Hilbert Transform in F1', stacklevel=2)

        # FT
        # Discriminate between F1 acquisition modes
        if FnMODE == 'States-TPPI':
            data = processing.ft(data, alt=True, fcor=fcor[0])
        elif FnMODE == 'Echo-Antiecho' or FnMODE == 'QF':
            data = processing.ft(data, fcor=fcor[0])
        else:
            raise NotImplementedError('Unknown acquisition mode in F1. Aborting...')
        if FnMODE == 'States-TPPI' or FnMODE == 'QF':
            data = processing.rev(data)                     # reverse data
        # Transpose back
        if FnMODE == 'QF':
            data = data.T
        else:
            data = processing.tp_hyper(data)
        # Unpack and/or return processed data
    if u:                                           # unpack or not
        if FnMODE in ['QF', 'QF-nofreq', 'No']:
            return data.real, data.imag
        else:
            return processing.unpack_2D(data)           # rr, ir, ri, ii
    else:
        return data


def inv_xfb(data, wf=[None, None], size=(None, None), fcor=[0.5, 0.5], FnMODE='States-TPPI'):
    """
    Reverts the full processing of a 2D NMR FID (data).

    Parameters
    ----------
    data : 2darray
        Input data, complex or hypercomplex
    wf : list of dict
        list of two entries [F1, F2]. Each entry is a dictionary of window functions
    size : list of int
        Initial size of FID
    fcor : list of float
        first fid point weighting factor [F1, F2]
    FnMODE : str
        Acquisition mode in F1

    Returns
    -------
    data : 2darray
        Processed data

    .. seealso::

        :func:`klassez.processing.repack_2D`

        :func:`klassez.processing.apodf`

        :func:`klassez.processing.td_eff`

    """

    # Processing the indirect dimension
    # If FnMODE is 'QF', do normal transpose instead of hyper
    if FnMODE == 'QF':
        data = data.T
    else:
        data = processing.tp_hyper(data)

    if FnMODE == 'States-TPPI' or FnMODE == 'QF':
        data = processing.rev(data)                     # reverse data

    # IFT on F1
    # Discriminate between F1 acquisition modes
    if FnMODE == 'States-TPPI':
        data = processing.ift(data, alt=True, fcor=fcor[0])
    elif FnMODE == 'Echo-Antiecho' or FnMODE == 'QF':
        data = processing.ift(data, fcor=fcor[0])
    else:
        raise NotImplementedError('Unknown acquisition mode in F1. Aborting...')

    # Revert zero-filling
    if size[0] is not None:
        data = processing.td_eff(data, (0, size[0]))

    # Reverse window function
    if wf[0] is not None:
        if wf[0]['mode'] is None:
            apod = np.ones_like(data)
        if wf[0]['mode'] == 'qsin':
            apod = processing.qsin(data, ssb=wf[0]['ssb'])/data
        if wf[0]['mode'] == 'sin':
            apod = processing.sin(data, ssb=wf[0]['ssb'])/data
        if wf[0]['mode'] == 'em':
            apod = processing.em(data, lb=wf[0]['lb'], sw=wf[0]['sw'])/data
        if wf[0]['mode'] == 'gm':
            apod = processing.gm(data, lb=wf[0]['lb'], gb=wf[0]['gb'], sw=wf[0]['sw'])/data
        data = data / apod

    # Transpose back
    if FnMODE == 'QF':
        data = data.T
    else:
        data = processing.tp_hyper(data)

    # IFT on F2
    data = processing.ift(data, fcor=fcor[1])

    # Revert zero-filling
    if size[1] is not None:
        data = processing.td_eff(data, (0, size[1]))

    # Reverse window function
    if wf[1] is not None:
        if wf[1]['mode'] is None:
            apod = np.ones_like(data)
        if wf[1]['mode'] == 'qsin':
            apod = processing.qsin(data, ssb=wf[1]['ssb'])/data
        if wf[1]['mode'] == 'sin':
            apod = processing.sin(data, ssb=wf[1]['ssb'])/data
        if wf[1]['mode'] == 'em':
            apod = processing.em(data, lb=wf[1]['lb'], sw=wf[1]['sw'])/data
        if wf[1]['mode'] == 'gm':
            apod = processing.gm(data, lb=wf[1]['lb'], gb=wf[1]['gb'], sw=wf[1]['sw'])/data
        data = data / apod

    return data


def make_scale(size, dw, rev=True):
    """
    Computes the frequency scale of the NMR spectrum, given the number of points and the employed dwell time (the REAL one, not the TopSpin one!).
    ``rev=True`` is required for the correct frequency arrangement in the NMR sense.

    Parameters
    ----------
    size: int
        Number of points of the frequency scale
    dw : float
        Time spacing in the time dimension
    rev: bool
        Reverses the scale

    Returns
    -------
    fqscale: 1darray
        The computed frequency scale.
    """
    fqscale = np.fft.fftshift(np.fft.fftfreq(size, d=dw))
    if rev:
        fqscale = fqscale[::-1]
    return fqscale

# ------------------------------------------------------------------------------------


def tabula_rasa(data, lvl=0.05, cmap=cm.Blues_r):
    """
    This function is to be used in SIFT algorithm.
    Allows interactive selection using a Lasso widget of the region of the spectrum
    which contain signal. Returns a masking matrix, of the same shape as data, whose entries
    are 1 inside the selection and 0 outside.
    """
    # Make the figure
    fig = plt.figure('Tabula Rasa')
    fig.set_size_inches(12, 8)
    plt.subplots_adjust(left=0.15, bottom=0.15)
    ax = fig.add_subplot()
    # Define grid
    xscale = np.arange(data.shape[1])
    yscale = np.arange(data.shape[0])

    thesignal = []          # List of the selected regions
    sgn_reg = 0             # Temporary storage of selected region
    mask = np.zeros_like(data)      # Mask matrix of zeros

    # Define 'save' button
    box = plt.axes([0.8, 0.025, 0.10, 0.07])
    button = Button(box, 'SAVE')

    def onselect(verts):
        # Function connected to the lasso
        nonlocal sgn_reg
        # raw selection of data
        path = MplPath(verts, closed=True)
        selected = []
        for i in yscale:
            for j in xscale:
                if path.contains_point((float(j), float(i))):
                    selected.append([j, i])

        # Create convex hull around the raw lasso
        CH = ConvexHull(np.array(selected))
        # Create delimiting wall
        xhull = list(CH.points[CH.vertices, 0])
        xhull.append(CH.points[CH.vertices[0], 0])
        xhull = np.array(xhull)
        yhull = list(CH.points[CH.vertices, 1])
        yhull.append(CH.points[CH.vertices[0], 1])
        yhull = np.array(yhull)

        # Update the plot
        hull.set(visible=True)
        hull.set_data(xhull, yhull)

        # Fine selection of points on the basis of the hull
        path = MplPath(CH.points[CH.vertices], closed=True)
        selected = []
        for i in yscale:
            for j in xscale:
                if path.contains_point((float(j), float(i))):
                    selected.append([j, i])
        # Store the selected points in a non-local variable
        sgn_reg = np.array(selected)

        plt.draw()

    def save(event):
        # Function connected to the button

        thesignal.append(sgn_reg)       # Save the region

        CH = ConvexHull(sgn_reg)        # Compute convex hull
        # Create the walls
        xhull = list(CH.points[CH.vertices, 0])
        xhull.append(CH.points[CH.vertices[0], 0])
        xhull = np.array(xhull)
        yhull = list(CH.points[CH.vertices, 1])
        yhull.append(CH.points[CH.vertices[0], 1])
        yhull = np.array(yhull)

        ax.plot(xhull, yhull, 'g')      # Plot the region walls on the figure forever
        hull.set(visible=False)         # Turn off the lasso

    # Parameters for contour
    norm = np.max(data)
    contour_start = norm * lvl
    contour_num = 16
    contour_factor = 1.40
    cl = contour_start * contour_factor**np.arange(contour_num)

    ax.contour(xscale, yscale, data, cl, cmap=cmap, linewidths=0.5)     # plot the contours

    hull, = ax.plot(0, 0, visible=False)             # Create variable for the lasso selection on screen

    # Set limits
    ax.set_ylim(data.shape[0], 0)

    # Widgets
    LassoSelector(ax, onselect)
    button.on_clicked(save)

    plt.show()
    plt.close()

    # Fill the masking matrix on the basis of the selected region
    #   If you selected something, set as '1' the highlighted points
    if len(thesignal) > 0:
        thesignal = np.concatenate(thesignal)
        for k in range(thesignal.shape[0]):
            mask[thesignal[k, 1], thesignal[k, 0]] = 1
    #   If you did not select anything, the masking matrix does not alter the spectrum
    else:
        mask = np.ones_like(data)
    return mask

# Phase correction


def integral(fx, x=None, dx=None, lims=None, use_bas=False):
    """
    Calculates the primitive of ``fx``. If ``fx`` is a multidimensional array, the integrals are computed along the last dimension.

    Parameters
    ----------
    fx : ndarray
        Function (array) to integrate
    x : 1darray or None
        Independent variable. Determines the integration step. If None, it is the point scale
    dx : float or None
        Integration step. If ``None``, computes it from the resolution of ``x``
    lims : tuple or None
        Integration range in the ``x`` scale. If None, the whole function is integrated.
    use_bas : bool
        Subtracts the straight line that connects the limit window before the integration (``True``) or not (``False``)

    Returns
    -------
    Fx : ndarray
        Integrated function.

    .. seealso::

        :func:`klassez.misc.trim_data`
    """
    # Copy variables for check
    fx_in = np.copy(fx)
    if x is None:   # Make the point scale
        x_in = np.arange(fx.shape[-1])
    else:
        x_in = np.copy(x)
    # Integration step
    if dx is None:
        dx = misc.calcres(x_in)

    if lims is None:    # whole range
        x_tr, fx_tr = np.copy(x_in), np.copy(fx_in)
    else:
        # Trim data according to lims
        x_tr, fx_tr = misc.trim_data(x_in, fx_in, lims)

    if use_bas:
        bas = processing.sl_bas_onidx(fx_tr, (0, len(x_tr)-1))
    else:
        bas = np.zeros_like(x_tr)

    # Integrate
    Fx = np.cumsum(fx_tr - bas, axis=-1) * dx
    return Fx


def integrate(fx, x=None, dx=None, lims=None, use_bas=False):
    """
    Calculates the definite integral of ``fx`` as ``I = F[-1] - F[0]`` (basically it applies the fundamental theorem of calculus).
    If ``fx`` is a multidimensional array, the integrals are computed along the last dimension.

    Parameters
    ----------
    fx : ndarray
        Function (array) to integrate
    x : 1darray or None
        Independent variable. Can determine the integration step. If ``None``, it is the point scale.
    dx : float or None
        Integration step. If ``None``, computes it from the resolution of ``x``
    lims : tuple or None
        Integration range according to ``x``. If ``None``, the whole function is integrated.
    use_bas : bool
        Subtracts the straight line that connects the limit window before the integration (``True``) or not (``False``)

    Returns
    -------
    integ : float
        Integrated function.

    .. seealso::

        :func:`klassez.processing.integral`
    """
    Fx = processing.integral(fx, x, dx, lims, use_bas)
    # Calculus fundamental theorem
    integ = Fx[..., -1] - Fx[..., 0]
    return integ


def pknl(data, grpdly=0, onfid=False):
    """
    Compensate for the Bruker group delay at the beginning of FID through a first-order phase correction of ``p1 = - 360 * GRPDLY`` degrees.
    If applied on the FID (``onfid=True``), it is equivalent to a left circular shift of ``GRPDLY`` points.
    However, in order to accomodate for also non-integer ``GRPDLY``, it is computed by doing the Fourier transform on the fly.

    Parameters
    ----------
    data : ndarray
        Input data. Be sure it is complex!
    grpdly : int
        Number of points that make the group delay.
    onfid : bool
        If it is True, performs FT before to apply the phase correction, and IFT after.

    Returns
    -------
    datap : ndarray
        Corrected data
    """
    # Safety check
    assert np.iscomplexobj(data), print('Input data is not complex')

    if onfid is True:   # FT, ps, IFT
        data_ft = processing.ft(data)
        datap_ft, *_ = processing.ps(data_ft, p1=-360*grpdly)
        datap = processing.ift(datap_ft)
        return datap
    else:   # Just ps
        datap, *_ = processing.ps(data, p1=-360*grpdly)
        return datap


def convdta(data, grpdly=0, scaling=1):
    """
    Removes the digital filtering to obtain a spectrum similar to the command CONVDTA performed by TopSpin.
    However, they will differ a little bit because of the digitization.
    These differences are not invisible to human's eye.

    .. note::

        Since TopSpin version 4.0 the algorithm changed somehow, and the result is different.

    Parameters
    ----------
    data : ndarray
        FID with digital filter
    grpdly : int
        Number of points that the digital filter consists of. Key $GRPDLY in acqus file
    scaling : float
        Scaling factor of the resulting FID. Needed to match TopSpin's intensities.

    Returns
    -------
    data_in : ndarray
        FID without the digital filter. It will have ``grpdly`` points less than ``data``.
    """
    # Safety copy
    data_in = np.copy(data)

    # Circular shift to put the digital filter at the end of FID
    data_in = np.roll(data_in, -grpdly, axis=-1)
    # Digital filter, reversed to make it look like a FID.
    dig_filt = data_in[..., -grpdly:][::-1]

    # Subtract the digital filter, reversed, from the start of the FID
    data_in[..., :grpdly] -= dig_filt
    # Trim the digital filter at the end of FID
    data_in = data_in[..., :-grpdly]
    # Correct the intensities
    data_in *= scaling
    return data_in


def calibration(ppmscale, S, ref=None):
    """
    Scroll the ppm scale of spectrum to make calibration.
    The interface offers two guidelines: the red one, labelled 'reference signal' remains fixed, whereas the green one ('calibration value') moves with the ppm scale.
    The ideal calibration procedure consists in placing the red line on the signal you want to use as reference,
    and the green line on the ppm value that the reference signal must assume in the calibrated spectrum.
    Then, scroll with the mouse until the two lines are superimposed.
    You can use another spectrum as reference.

    Parameters
    ----------
    ppmscale : 1darray
        The ppm scale to be calibrated
    S : 1darray
        The spectrum to calibrate
    ref : list of 1darray or Spectrum_1D object
        Reference spectrum to be used for calibration. If list, ``[ppm scale, spectrum]``

    Returns
    -------
    offset : float
        Difference between original scale and new scale. This must be summed up to the original ppm scale to calibrate the spectrum.
    """

    # initialize values
    if ppmscale[0] < ppmscale[-1]:
        S = S[::-1]
        ppmscale = ppmscale[::-1]
    ppmscale0 = np.copy(ppmscale)       # save original scale for reset

    offset = 0                          # initialize returned value
    calstep = 0.25                      # calibration step

    # Initialize guidelines positions
    #   Fixed one
    g_idx = len(ppmscale)//2
    g_pos = ppmscale[g_idx]
    #   Mobile one
    d_idx = len(ppmscale)//2
    d_pos = ppmscale[g_idx]

    # Make the figure
    fig = plt.figure('Calibration')
    fig.set_size_inches(15, 8)
    plt.subplots_adjust(left=0.1, bottom=0.125, right=0.875, top=0.85)
    ax = fig.add_subplot()

    # Boxes and widgets
    #   Buttons
    box_save = plt.axes([0.905, 0.125, 0.09, 0.08])
    button = Button(box_save, 'SAVE\nAND\nEXIT', hovercolor='0.975')
    box_reset = plt.axes([0.905, 0.825-0.05, 0.09, 0.07])
    reset_button = Button(box_reset, 'RESET', hovercolor='0.975')
    box_up = plt.axes([0.955, 0.545-0.05, 0.04, 0.06])
    up_button = Button(box_up, r'$\uparrow$', hovercolor='0.975')
    box_down = plt.axes([0.905, 0.545-0.05, 0.04, 0.06])
    down_button = Button(box_down, r'$\downarrow$', hovercolor='0.975')
    box_overlay = plt.axes([0.905, 0.345-0.05, 0.09, 0.07])
    overlay_button = Button(box_overlay, 'OVERLAY', hovercolor='0.975')

    def increase_step(event):
        # up
        nonlocal calstep
        calstep *= 2
        stext.set_text(f'{"Step":^9s}\n{calstep:^9.3f}')
        fig.canvas.draw()

    def decrease_step(event):
        # down
        nonlocal calstep
        calstep /= 2
        stext.set_text(f'{"Step":^9s}\n{calstep:^9.3f}')
        fig.canvas.draw()

    def save(event):
        # Calculate the calibration offset and close figure
        nonlocal offset
        offset = ppmscale[0] - ppmscale0[0]
        plt.close()

    def reset(event):
        nonlocal calstep, ppmscale
        calstep = 0.25
        ppmscale = np.copy(ppmscale0)
        on_scroll(event)
        fig.canvas.draw()

    def mouse_click(event):
        if event.inaxes != ax:
            return
        if event.button == 1:
            move_fixed(event)
        elif event.button == 3:
            move_mobile(event)

    def move_fixed(event):
        # set position of the red bar
        x = event.xdata
        if (event.dblclick and event.button == 1) or event.button == 2:
            nonlocal g_pos, g_idx
            g_pos = x
            g_idx = misc.ppmfind(ppmscale, g_pos)[0]
            guide.set_xdata((x,))
        gtext.set_text(f'{"Mobile":^9s}\n{g_pos:^9.3f}')
        fig.canvas.draw()

    def move_mobile(event):
        # set position of the green bar
        x = event.xdata
        if x is not None:
            if (event.dblclick and event.button == 3):
                nonlocal d_pos, d_idx
                d_pos = x
                d_idx = misc.ppmfind(ppmscale, d_pos)[0]
                dguide.set_xdata((x,))
            dtext.set_text(f'{"Fixed":^9s}\n{d_pos:^9.3f}')
        fig.canvas.draw()

    def on_scroll(event):
        # move the scale
        nonlocal ppmscale
        if event.button == 'up':
            ppmscale += calstep
        if event.button == 'down':
            ppmscale -= calstep
        spect.set_xdata(ppmscale)
        guide.set_xdata((ppmscale[g_idx],))
        dguide.set_xdata((d_pos,))
        gtext.set_text(f'{"Mobile":^9s}\n{ppmscale[g_idx]:^9.3f}')
        dtext.set_text(f'{"Fixed":^9s}\n{d_pos:^9.3f}')
        fig.canvas.draw()

    def overlay(event):
        # Bring red over green in one go
        nonlocal ppmscale
        step2move = d_pos - ppmscale[g_idx]
        ppmscale += step2move
        spect.set_xdata(ppmscale)
        guide.set_xdata((ppmscale[g_idx],))
        dguide.set_xdata((d_pos,))
        gtext.set_text(f'{"Mobile":^9s}\n{ppmscale[g_idx]:^9.3f}')
        dtext.set_text(f'{"Fixed":^9s}\n{d_pos:^9.3f}')
        fig.canvas.draw()

    if ref is not None:
        if isinstance(ref, Spectrum_1D):
            ax.plot(ref.ppm, ref.S.real/max(ref.S.real)*max(S.real), c='k', lw=0.8, label='Reference')
        else:
            ax.plot(ref[0], ref[1].real/max(ref[1].real)*max(S.real), c='k', lw=0.8, label='Reference')
    spect, = ax.plot(ppmscale, S.real, c='tab:blue', lw=0.8, label='Spectrum')    # plot spectrum

    # Plot the guidelines
    guide = ax.axvline(x=g_pos, lw=0.7, c='tab:red')        # static
    dguide = ax.axvline(x=d_pos, lw=0.7, c='tab:green')     # dynamic
    #   green and red lines position
    gtext = plt.text(0.950, 0.750, f'{"Mobile":^9s}\n{g_pos:^9.3f}', ha='center', va='top', fontsize=16, transform=fig.transFigure, c='tab:red')
    dtext = plt.text(0.950, 0.690, f'{"Fixed":^9s}\n{d_pos:^9.3f}', ha='center', va='top', fontsize=16, transform=fig.transFigure, c='tab:green')
    stext = plt.text(0.950, 0.630, f'{"Step":^9s}\n{calstep:^9.3f}', ha='center', va='top', fontsize=16, transform=fig.transFigure, c='k')

    # Make cool figure
    T = np.max(np.array(S).real)
    B = np.min(np.array(S).real)
    ax.set_ylim(B - 0.01*T, T + 0.01*T)

    ax.set_xlabel(r'$\delta\,$ /ppm')

    instruction_text = '''Set the red bar with left double click on the reference signal of your spectrum.\n
                       Set the green bar with right double click on where you want the red bar to be after the calibration.\n
                       Scroll with the mouse to move the spectrum. Press "overlay" to align the red bar with the green bar in a single go.'''

    ax.legend()
    misc.mathformat(ax)
    misc.pretty_scale(ax, (max(ppmscale), min(ppmscale)), 'x')
    misc.pretty_scale(ax, ax.get_ylim(), 'y')
    misc.set_fontsizes(ax, 20)
    ax.set_title(instruction_text, fontsize=16)

    # Connect widgets to functions
    button.on_clicked(save)
    overlay_button.on_clicked(overlay)
    reset_button.on_clicked(reset)
    up_button.on_clicked(increase_step)
    down_button.on_clicked(decrease_step)
    cursor = Cursor(ax, useblit=True, horizOn=False, color='k', linewidth=0.4)
    cursor.vertOn = True
    fig.canvas.mpl_connect('button_press_event', mouse_click)
    fig.canvas.mpl_connect('scroll_event', on_scroll)

    plt.show()
    plt.close(1)

    print('Offset: {: .3f} /ppm'.format(offset), c='violet')

    return offset

# -----------------------------------------------------------------------------------------
# MCR and related


def mcr_stack(input_data, P='H'):
    """
    Performs matrix augmentation by assembling input_data according to the positioning matrix ``P``.
    ``P`` has two default modes: ``'H'`` = horizontal stacking; ``'V'`` = vertical stacking. Otherwise, a custom ``P`` matrix can be given as follows.
    The entries of the ``P`` matrix are the indices of the data in input_data. The shape of the matrix determines the final arrangement. See example.
    If each dataset in ``input_data`` has dimensions ``(m, n)`` and ``P`` has dimensions ``(u,v)``, then the returned data matrix will have dimensions ``(m*u, n*v)``.

    Parameters
    ----------
    input_data : 3darray or list of 2darray
        Contains the spectra to be stacked together. The index that runs on the datasets must be the first one.
    P : str or 2darray
        ``'H'`` for horizontal stacking, ``'V'`` for vertical stacking, or custom matrix as explained in the description

    Returns
    -------
    data : 2darray
        Augmented data matrix.

    Examples
    --------
    If ``input_data = [a, b, c, d, e, f]``, and one wants to obtain ``[[a, b], [d,c], [f, e]]``, the correspondant ``P`` matrix is:

    .. code-block:: python

        P = [
            [0, 1],
            [3, 2],
            [5, 4]
            ]

    .. seealso::

        :func:`klassez.processing.mcr_unpack`
    """
    # Get the number of datasets
    if isinstance(input_data, list):
        nds = len(input_data)
        Q = input_data
    else:   # if it is not a list, make it be manually
        nds = input_data.shape[0]
        Q = [input_data[w] for w in range(nds)]

    # Compute the P matrix
    if isinstance(P, str):  # default options
        if P == 'H':    # Horizontal
            P = np.arange(nds).reshape(1, -1)
        elif P == 'V':  # Vertical
            P = np.arange(nds).reshape(-1, 1)
        else:   # Unknown
            raise ValueError('Unrecognized P type')
    elif isinstance(P, np.ndarray):  # Check if the dimensions are compatible
        assert np.prod(P.shape) == nds, 'Wrong P shape'

    # Assemble the data
    for k in range(nds):
        # Compute mask matrix
        Mk = np.zeros_like(P)
        # Find the position of the k-th spectrum in P
        i, j = np.where(P == k)
        # Set that position as a 1 in Mk, all the rest is 0
        Mk[i[-1], j[-1]] = 1     # np.where returns lists of 1 number each
        if k == 0:  # Make the variable
            data = np.kron(Mk, Q[k])
        else:       # Add it
            data += np.kron(Mk, Q[k])
    return data


def mcr_unpack(C, S, nds, P='H'):
    """
    Reverts matrix augmentation of :func:`klassez.processing.mcr_stack`.
    The denoised spectra can be calculated by matrix multiplication:

    .. code-block:: python

        for k in range(nds):
            D[k] = C_f[k] @ S_f[k]

    Parameters
    ----------
    C : 2darray
        MCR C matrix
    S : 2darray
        MCR S matrix
    nds : int
        number of experiments
    P : str or 2darray
        ``'H'`` for horizontal stacking, ``'V'`` for vertical stacking, or custom matrix as explained in the description of ``mcr_stack``

    Returns
    -------
    C_f : list of 2darray
        Disassembled MCR C matrix
    S_f : list of 2darray
        Disassembled MCR C matrix

    .. seealso::

        :func:`klassez.processing.mcr_stack`
    """
    # Compute the P matrix
    if isinstance(P, str):  # default options
        if P == 'H':    # Horizontal
            P = np.arange(nds).reshape(1, -1)
        elif P == 'V':  # Vertical
            P = np.arange(nds).reshape(-1, 1)
        else:   # Unknown
            raise ValueError('Unrecognized P type')
    elif isinstance(P, np.ndarray):  # Check if the dimensions are compatible
        assert np.prod(P.shape) == nds, 'Wrong P shape'

    # Compute the dimension of each original dataset
    m = C.shape[0] // P.shape[0]    # num. rows of C / num. exp. per column
    n = S.shape[-1] // P.shape[-1]  # num. columns of S / num. exp. per row

    # Initialize variables for storing the final C and S matrices
    C_f, S_f = [], []

    for k in range(nds):    # Loop on the datasets
        i, j = np.where(P == k)     # find whe position of the k-th dataset in P
        # Compute slices for delimiting the k-th spectrum
        #   in C: all the columns, rows according to P
        rslice = slice(i[0]*m, i[0]*m + m)
        C_f.append(C[rslice, ...])
        #   in S: all the rows, columns according to P
        cslice = slice(j[0]*n, j[0]*n + n)
        S_f.append(S[..., cslice])

    return np.array(C_f), np.array(S_f)


def calc_nc(data, s_n):
    """
    Calculates the optimal number of components to be used for either MCR or SVD, given the standard deviation of the noise.
    The threshold value is calculated as stated in Theorem 1 of `this article`_ .

    .. _this article: https://arxiv.org/abs/1710.09787v2

    Parameters
    ----------
    data : 2darray
        Input data
    s_n : float
        Noise standard deviation

    Returns
    -------
    n_c : int
        Number of components

    .. seealso::

        :func:`klassez.processing.lrd`

        :func:`klassez.processing.mcr`
    """
    M, N = data.shape

    S = slinalg.svdvals(data)

    b = M/N
    c = (1/2**0.5) * (1 + b + (1 + 14*b + b**2)**0.5)**0.5
    threshold = s_n * ((c + 1/c) * (c + b/c))**0.5

    threshold *= S[0]
    for k in range(len(S)):
        if S[k] < threshold:
            n_c = k+1
            break

    return n_c


def simplisma(D, nc, f=10, oncols=True):
    """
    Finds the first ``nc`` purest components of matrix ``D`` using the *simplisma* algorithm, proposed by `Windig and Guilment`_ .
    If ``oncols=True``, this function estimates ``S`` with simplisma, then calculates :math:`C = D S^+`.
    If ``oncols=False``, this function estimates ``C`` with *simplisma*, then calculates :math:`S = C^+ D`. ``f`` defines the percentage of allowed noise.

    .. _Windig and Guilment: https://pubs.acs.org/doi/10.1021/ac00014a016

    Parameters
    ----------
    D : 2darray
        Input data, of dimensions ``(m, n)``
    nc : int
        Number of components to be found. This determines the final size of the ``C`` and ``S`` matrices.
    f : float
        Percentage of allowed noise.
    oncols : bool
        If True, simplisma estimates the ``S`` matrix, otherwise estimates ``C``.

    Returns
    -------
    C : 2darray
        Estimation of the ``C`` matrix, of dimensions ``(m, nc)``.
    S : 2darray
        Estimation of the ``S`` matrix, of dimensions ``(nc, n)``.

    .. seealso::

        :func:`klassez.processing.mcr`

        :func:`klassez.processing.mcr_als`

    """

    rows = D.shape[0]       # number of rows of D
    cols = D.shape[1]       # number of columns of D

    if oncols:
        # on columns
        m = np.zeros(rows).astype(D.dtype)
        s = np.zeros(rows).astype(D.dtype)

        for i in range(rows):
            m[i] = np.mean(D[i, :])      # mean of the i-th row
            s[i] = np.std(D[i, :])       # STD of the i-th row

        # Correction factor for the noise 'alpha'
        a = 0.01 * f * max(m)

        print('Computing 1° purest variable...', end='\r')
        # Rescaling of data for lambda: makes determinant of COO
        # proportional only to the independance between variables
        L = (s**2 + (m + a)**2)**0.5  # lambda corrected for alpha
        Dl = np.zeros_like(D)
        for i in range(rows):
            Dl[i, :] = D[i, :] / L[i]

        Q = (1/cols) * Dl @ Dl.T      # Correlation-around-origin matrix

        # Calculation of the weighting factors:
        # express the independency between the variables
        w = np.zeros((rows, nc)).astype(D.dtype)       # Weights
        p_s = np.zeros((rows, nc)).astype(D.dtype)     # Pure components spectra
        s_s = np.zeros((rows, nc)).astype(D.dtype)     # STD spectra

        # First weight
        w[:, 0] = (s**2 + m**2) / (s**2 + (m + a)**2)
        p0 = s / (m + a)    # First purity spectrum
        pv, ipv = [], []    # Purest variables and correspondant index

        p_s[:, 0] = w[:, 0] * p0
        s_s[:, 0] = w[:, 0] * s

        # 1st purest variable
        pv.append(max(p_s[:, 0]))
        ipv.append(np.argmax(p_s[:, 0]))

        # Matrix for computing the determinants
        #   It has the following structure, where Q denotes the COO matrix
        #   and p# the index of the # purest component:
        """
            Q[i,i]          Q[i,p1]         Q[i,p2]         ... Q[i,p(i-1)]
            Q[p1,i]         Q[p1,p1]        Q[p1,p2]        ... Q[p1,p(i-1)]
            Q[p2,i]         Q[p2,p1]        Q[p2,p2]        ... Q[p2,p(i-1)]
            ...             ...             ...             ... ...
            Q[p(i-1),i]     Q[p(i-1),p1]    Q[p(i-1),p2]    ... Q[p(i-1),p(i-1)]
        """
        for c in range(1, nc):      # 'c' cycles on number of components
            print('Computing '+str(c+1)+'° purest variable...', end='\r')
            for i in range(rows):   # i cycles on the number of rows
                W = np.zeros((c+1, c+1)).astype(D.dtype)
                W[0, 0] = Q[i, i]
                for k in range(1, c+1):                 # cycles inside W
                    W[0, k] = Q[i, ipv[k-1]]              # first row \{0,0}
                    W[k, 0] = Q[ipv[k-1], i]              # first column \{0,0}
                    for q in range(1, c+1):
                        W[k, q] = Q[ipv[k-1], ipv[q-1]]   # all the rest, going row per row
                w[i, c] = linalg.det(W)

            p_s[:, c] = p0 * w[:, c]              # Create pure spectrum of c-th component
            s_s[:, c] = s_s[:, 0] * w[:, c]        # Create STD spectrum of c-th component
            pv.append(max(p_s[:, c]))            # Update pure component
            ipv.append(np.argmax(p_s[:, c]))     # Update pure variable

        print('Purest variables succesfully found.\n', c='violet')
        for c in range(nc):
            print('{}° purest variable:\t\t{}'.format(c+1, ipv[c]))

        # MCR "S" matrix (D = CS + E)
        S = np.zeros((nc, cols)).astype(D.dtype)
        for c in range(nc):
            S[c, :] = D[ipv[c], :]
        C = D @ linalg.pinv(S)

    else:
        # on rows
        m = np.zeros((cols)).astype(D.dtype)
        s = np.zeros((cols)).astype(D.dtype)

        for j in range(cols):
            m[j] = np.mean(D[:, j])      # mean of the i-th row
            s[j] = np.std(D[:, j])       # STD of the i-th row

        # Correction factor for the noise 'alpha'
        a = 0.01 * f * max(m)

        print('Computing 1° purest variable...', end='\r')
        # First purity spectrum
        p1 = s / (m + a)        # First purity spectrum
        pv, ipv = [], []        # Purest variables and correspondant index

        # 1st purest variable
        pv.append(max(p1))
        ipv.append(np.argmax(p1))

        # Rescaling of data for lambda: makes determinant of COO
        # proportional only to the independance between variables
        L = (s**2 + (m + a)**2)**0.5  # lambda corrected for alpha
        Dl = np.zeros_like(D)
        for j in range(cols):
            Dl[:, j] = D[:, j] / L[j]

        Q = (1/rows) * Dl.T @ Dl      # Correlation-around-origin matrix

        # Calculation of the weighting factors:
        # express the independency between the variables

        w = np.zeros((cols, nc)).astype(D.dtype)       # Weights
        p_s = np.zeros((cols, nc)).astype(D.dtype)     # Pure components spectra
        s_s = np.zeros((cols, nc)).astype(D.dtype)     # STD spectra

        # First weight
        w[:, 0] = (s**2 + m**2) / (s**2 + (m + a)**2)
        p_s[:, 0] = w[:, 0] * p1
        s_s[:, 0] = w[:, 0] * s

        # Matrix for computing the determinants
        # It has the following structure, where Q denotes the COO matrix
        # and p# the index of the # purest component:
        """
            Q[j,j]          Q[j,p1]         Q[j,p2]         ... Q[j,p(j-1)]
            Q[p1,j]         Q[p1,p1]        Q[p1,p2]        ... Q[p1,p(j-1)]
            Q[p2,j]         Q[p2,p1]        Q[p2,p2]        ... Q[p2,p(j-1)]
            ...             ...             ...             ... ...
            Q[p(j-1),j]     Q[p(j-1),p1]    Q[p(j-1),p2]    ... Q[p(j-1),p(j-1)]
        """
        for c in range(1, nc):      # 'c' cycles on number of components
            print('Computing '+str(c+1)+'° purest variable...', end='\r')
            for j in range(cols):   # j cycles on the number of colums
                W = np.zeros((c+1, c+1)).astype(D.dtype)
                W[0, 0] = Q[j, j]
                for k in range(1, c+1):  # cycles inside W
                    W[0, k] = Q[j, ipv[k-1]]        # first row \{0,0}
                    W[k, 0] = Q[ipv[k-1], j]        # first column \{0,0}
                    for q in range(1, c+1):
                        W[k, q] = Q[ipv[k-1], ipv[q-1]]  # all the rest, going row per row
                w[j, c] = linalg.det(W)

            p_s[:, c] = p_s[:, 0] * w[:, c]      # Create pure spectrum of c-th component
            s_s[:, c] = s_s[:, 0] * w[:, c]      # Create STD spectrum of c-th component
            pv.append(max(p_s[:, c]))          # Update pure component
            ipv.append(np.argmax(p_s[:, c]))   # Update pure variable

        print('Purest variables succesfully found.\n', c='violet')
        for c in range(nc):
            print('{}° purest variable:\t\t{}'.format(c+1, ipv[c]))

        # MCR "C" matrix (D = CS + E)
        C = np.zeros((rows, nc)).astype(D.dtype)
        for c in range(nc):
            C[:, c] = D[:, ipv[c]]
        S = linalg.pinv(C) @ D

    return C, S


def mcr_als(D, C, S, itermax=10000, tol=1e-5):
    r"""
    Performs alternating least squares to get the final ``C`` and ``S`` matrices. Being the fundamental MCR equation:

    .. math::

        D = CS + E

    At the k-th step of the iterative cycle:

    1. :math:`C_{(k)} = D S^+_{(k-1)}`
    2. :math:`S_{(k)} = C^+_{(k)} D`
    3. :math:`E_{(k)} = D - C_{(k)} S_{(k)}`

    Defined ``rC`` and ``rS`` as the Frobenius norm of the difference of ``C`` and ``S`` matrices between two subsequent steps:

    .. math::

        rC = || C_{(k)} - C_{(k-1)} || \qquad
        rS = || S_{(k)} - S_{(k-1)} ||

    The convergence is reached when both ``C`` and ``S`` change less than ``tol`` times the first iteration:

    .. math::

        \frac{ rC_{(k)} }{rC_{1}} \leq tol \quad \text{and} \quad \frac{ rS_{(k)} }{rS_{1}} \leq tol

    Parameters
    ----------
    D : 2darray
        Input data, of dimensions ``(m, n)``
    C : 2darray
        Estimation of the ``C`` matrix, of dimensions ``(m, nc)``.
    S : 2darray
        Estimation of the ``S`` matrix, of dimensions ``(nc, n)``.
    itermax : int
        Maximum number of iterations
    tol : float
        Threshold for the arrest criterion.

    Returns
    -------
    C : 2darray
        Optimized C matrix, of dimensions ``(m, nc)``.
    S : 2darray
        Optimized S matrix, of dimensions ``(nc, n)``.

    .. seealso::

        :func:`klassez.processing.mcr`

        :func:`klassez.processing.simplisma`
    """

    itermax = int(itermax)

    start_time = datetime.now()
    print('\n-----------------------------------------------------\n')
    print('             MCR optimization running...             \n', c='violet')

    convergence_flag = 0
    print(f'{"#":>5s}\t{"C convergence":>12s}\t{"S convergence":>12s}')

    # First round is basically null, hence it happens outside the loop
    C = D @ linalg.pinv(S)
    S = linalg.pinv(C) @ D
    for kk in range(itermax):
        # Copy from previous cycle
        C0 = np.copy(C)
        S0 = np.copy(S)

        # Compute new C, S and E
        C = D @ linalg.pinv(S)
        S = linalg.pinv(C) @ D

        # Compute the Frobenius norm of the difference matrices
        # between two subsequent cycles
        if kk == 0:
            rC0 = linalg.norm(C - C0)
            rS0 = linalg.norm(S - S0)
        rC = linalg.norm(C - C0) / rC0
        rS = linalg.norm(S - S0) / rS0

        # Ongoing print of the residues
        print(f'{kk+1:5.0f}\t{rC:12.6e}\t{rS:12.6e}', end='\r')

        # Arrest criterion
        if (rC < tol) and (rS < tol):
            end_time = datetime.now()
            print('\n\n\tMCR converges in '+str(kk+1)+' steps.', c='violet')
            convergence_flag = 1    # Set to 1 if the arrest criterion is reached
            break

    if not convergence_flag:
        print('\n\n\tMCR does not converge.', c='violet')
    end_time = datetime.now()
    print('\tTotal runtime: {}'.format(end_time - start_time))

    return C, S


def mcr(input_data, nc, f=10, tol=1e-3, itermax=1e4, P='H', oncols=True):
    """
    This is an implementation of Multivariate Curve Resolution for the denoising of 2D NMR data.
    Let us consider a matrix `D`, of dimensions ``(m, n)``, where the starting data are stored. The final purpose of MCR is to decompose the `D` matrix as follows:

    .. math::

        D = CS + E

    where `C` and `S` are matrices of dimension ``(m, nc)`` and ``(nc, n)``, respectively, and `E` contains the part of the data that are not reproduced by the factorization.
    Being `D` the FID of a NMR spectrum, `C` will contain time evolutions of the indirect dimension, and `S` will contain transients in the direct dimension.

    The total MCR workflow can be separated in two parts:
    a first algorithm that produces an initial guess for the three matrices `C`, `S` and `E` (``simplisma``),
    and an optimization step that aims at the removal of the unwanted features of the data by iteratively filling the E matrix (``mcr_als``).
    This function returns the denoised datasets, `CS`, and the single `C` and `S` matrices.

    Parameters
    ----------
    input_data : 2darray or 3darray
        a 3D array containing the set of 2D NMR datasets to be coprocessed stacked along the first dimension. A single 2D array can be passed, if the denoising of a single dataset is desired.
    nc : int
        number of purest components to be looked for;
    f : float
        percentage of allowed noise;
    tol : float
        tolerance for the arrest criterion;
    itermax : int
        maximum number of allowed iterations
    P : str or 2darray
        ``'H'`` for horizontal stacking, ``'V'`` for vertical stacking, or custom matrix as explained in the description of ``mcr_stack``
    oncols : bool
        True to estimate ``S`` with ``processing.simplisma``, False to estimate ``C``.

    Returns
    -------
    CS_f : 2darray or 3darray
        Final denoised data matrix
    C_f : 2darray or 3darray
        Final C matrix
    S_f : 2darray or 3darray
        Final S matrix

    .. seealso::

        :func:`klassez.processing.mcr_stack`

        :func:`klassez.processing.mcr_unpack`

        :func:`klassez.processing.simplisma`

        :func:`klassez.processing.mcr_als`

    """

    # Get number of datasets (nds) from the shape of the input tensor
    if isinstance(input_data, list):
        nds = len(input_data)
    else:
        if len(input_data.shape) == 3:
            nds = input_data.shape[0]
        elif len(input_data.shape) == 2:
            nds = 1
            input_data = np.reshape(input_data, (1, input_data.shape[0], input_data.shape[1]))
        else:
            raise ValueError('Input data is not a matrix!')

    print('\n*****************************************************')
    print('*                                                   *')
    print('*           Multivariate Curve Resolution           *')
    print('*                                                   *')
    print('*****************************************************\n')

    D = processing.mcr_stack(input_data, P=P)           # Matrix augmentation

    # Get initial estimation of C, S and E
    C0, S0 = processing.simplisma(D, nc, f, oncols=oncols)

    # Optimize C and S matrix through Alternating Least Squares
    C, S = processing.mcr_als(D, C0, S0, itermax=itermax, tol=tol)

    # Revert matrix augmentation
    C_f, S_f = processing.mcr_unpack(C, S, nds, P)

    # Obtain the denoised data of the same shape as the input
    CS_f = [C_f[j] @ S_f[j] for j in range(nds)]

    # Reshape if no matrix augmentation is performed
    if nds == 1:
        CS_f = CS_f[0]
        C_f = C_f[0]
        S_f = S_f[0]

    print('\n*****************************************************\n')

    return CS_f, C_f, S_f

# ---------------------------------------------------------------------------------------- #


def lrd(data, nc):
    """
    Denoising method based on Low-Rank Decomposition.
    The algorithm performs a singular value decomposition on data, then keeps only the first ``nc`` singular values while setting all the others to 0.
    Finally, rebuilds the data matrix using the modified singular values.

    Parameters
    ----------
    data : 2darray
        Data to be denoised
    nc : int
        Number of components, i.e. number of singular values to keep

    Returns
    -------
    data_out : 2darray
        Denoised data
    """
    # Safety check on data dimension
    if len(data.shape) != 2:
        raise ValueError('Input data is not 2D. Aborting...')

    print('\n*****************************************************')
    print('*                                                   *')
    print('*                 Low Rank Denoising                *')
    print('*                                                   *')
    print('*****************************************************\n')

    # Make SVD
    print('Performing SVD. This might take a while...', c='violet')
    U, svals, V = linalg.svd(data)
    print('Done.\n', c='violet')
    # Apply hard-thresholding
    svals_p = np.zeros_like(svals)
    svals_p[:nc] = svals[:nc]
    # Reconstruct the denoised data
    data_out = U @ slinalg.diagsvd(svals_p, U.shape[1], V.shape[0]) @ V
    print('Low-Rank Denosing completed.', c='violet')
    print('\n*****************************************************\n')
    return data_out


def cadzow(data, n, nc, print_head=True):
    """
    Performs Cadzow denoising on data, which is a 1D array of ``N`` points.
    The algorithm works as follows:

    1. Transform data in a Hankel matrix ``H`` of dimensions ``(N-n, n)``
    2. Make SVD on :math:`H = U S V`
    3. Keep only the first ``nc`` singular values, and put all the rest to 0 `(S -> S')`
    4. Rebuild :math:`H' = U S' V`
    5. Average the antidiagonals to rebuild the Hankel-type structure, then make 1D array

    Set ``print_head=True`` to display the fancy heading.

    Parameters
    ----------
    data : 1darray
        Input data
    n : int
        Number of columns of the Hankel matrix.
    nc : int
        Number of singular values to keep.
    print_head : bool
        Set it to True to display the fancy heading.

    Returns
    -------
    datap : 1darray
        Denoised data

    .. seealso::

        :func:`klassez.processing.hankel`

        :func:`klassez.processing.lrd`

        :func:`klassez.processing.iterCadzow`
    """
    if print_head is True:
        print('\n*****************************************************')
        print('*                                                   *')
        print('*                   Cadzow denoising                *')
        print('*                                                   *')
        print('*****************************************************\n')

    N = data.shape[-1]

    # Builds a Hankel-type matrix containing in the first row "data" up to index "n-1"
    # and as last column "data" from index "n" to the end
    H = slinalg.hankel(data[:n], data[n-1:]).T

    U, s, V = linalg.svd(H)    # Make SVD
    sp = np.zeros_like(s)      # Create empty array for singular values
    sp[:nc] = s[:nc]           # Keep only the first nc singular values

    Hp = U @ slinalg.diagsvd(sp, H.shape[0], H.shape[1]) @ V                               # Rebuild the new data matrix
    datap = np.array([np.mean(np.diag(Hp[:, ::-1], w)) for w in range(-N+n, n)])[::-1]      # Mean on the antidiagonals

    return datap


def iterCadzow(data, n, nc, itermax=100, f=0.005, print_head=True, print_time=True):
    r"""
    Performs Cadzow denoising on data, which is a 1D array of ``N`` points, in an iterative manner.
    The algorithm works as follows:

    1. Transform data in a Hankel matrix `H` of dimensions ``(N-n, n)``
    2. Make SVD on :math:`H = U S V`
    3. Keep only the first ``nc`` singular values, and put all the rest to 0 `(S -> S')`
    4. Rebuild :math:`H' = U S' V`
    5. Average the antidiagonals to rebuild the Hankel-type structure, then make 1D array
    6. Check arrest criterion: if it is not reached, go to 1, else exit.

    The arrest criterion is, at the k-th step:

    .. math::

        \frac{S_{(k-1)}[nc-1] }{ S_{(k-1)}[0] } - \frac{S_{(k)}[nc-1] }{ S_{(k)}[0] } < f \frac{S_{(0)}[nc-1] }{ S_{(0)}[0] }

    Parameters
    ----------
    data : 1darray
        Data to be processed
    n : int
        Number of columns of the Hankel matrix
    nc : int
        Number of singular values to preserve
    itermax : int
        max number of iterations allowed
    f : float
        factor that appears in the arrest criterion
    print_time : bool
        set it to True to show the time it took
    print_head : bool
        set it to True to display the fancy heading.

    Returns
    -------
    datap : 1darray
        Denoised data

    .. seealso::

        :func:`klassez.processing.cadzow`
    """

    if print_head is True:
        print('\n*****************************************************')
        print('*                                                   *')
        print('*                   Cadzow denoising                *')
        print('*                                                   *')
        print('*****************************************************\n')

    def check_arrcrit(s_0, s_1, nc, tol):
        """
        Arrest criterion:
        check if the difference of the ratio [max(s) / min(s)] between two subsequent iterations is below tol
        """
        r_0 = s_0[0] / s_1[0]
        r_c = s_0[nc-1] / s_1[nc-1]
        R = np.abs(r_0 - r_c)

        if R < tol:
            return R, True
        else:
            return R, False

    def calc_tol(s, nc, f=0.01):
        tol = (s[nc] / s[0]) * f
        return tol

    start_time = datetime.now()

    N = data.shape[-1]

    data0 = data
    # Builds a Hankel-type matrix containing in the first row "data" up to index "n-1"
    # and as last column "data" from index "n" to the end
    H0 = slinalg.hankel(data[:n], data[n-1:]).T

    s0 = slinalg.svdvals(H0)     # Calculate the singular values of H0
    sp = np.zeros_like(s0)      # Create empty array to store the singular values to be kept

    tol = calc_tol(s0, nc, f=f)

    print(f'{"#":>6s} | {"Control":>12s} | {"Target":>12s}')
    for k in range(itermax):
        H0 = slinalg.hankel(data0[:n], data0[n-1:]).T    # Make Hankel
        U, s, V = linalg.svd(H0)                        # Make SVD
        sp[:nc] = s[:nc]                                # Keep only the first nc singular values

        Hp = U @ slinalg.diagsvd(sp, H0.shape[0], H0.shape[1]) @ V                               # Rebuild the new data matrix
        datap = np.array([np.mean(np.diag(Hp[:, ::-1], w)) for w in range(-N+n, n)])[::-1]      # Mean on the antidiagonals

        # Check convergence
        R, Cond = check_arrcrit(s0, s, nc, tol)
        # Print status
        print(f'{k+1:>6.0f} | {R:12.5e} | {tol:12.5e}', end='\r')
        if Cond and k:
            print(f'\nCadzow converges in {k+1} steps.', c='violet')
            break
        else:
            s0 = s
            data0 = datap

    end_time = datetime.now()
    if k+1 == itermax:
        print('\tCadzow does not converge.', c='violet')
    if print_time is True:
        print('Total runtime: {}'.format(end_time - start_time))
    # Add empty line for aesthetic purposes
    print()

    return datap


def cadzow_2D(data, n, nc, i=True, f=0.005, itermax=100, print_time=True):
    """
    Performs the Cadzow denoising method on a 2D spectrum, one transient at the time.
    This function calls either Cadzow or iterCadzow, depending on the parameter ``i``:
    True for ``iterCadzow``, False for normal ``cadzow``.

    Parameters
    ----------
    data : 2darray
        Input data
    n : int
        Number of columns of the Hankel matrix.
    nc : int
        Number of singular values to keep.
    i : bool
        Calls ``processing.cadzow`` if ``i=False``, or ``processing.iterCadzow`` if ``i=True``.
    itermax : int
        Maximum number of iterations allowed.
    f : float
        Factor for the arrest criterion.
    print_time : bool
        Set it to True to display the time spent.

    Returns
    -------
    datap : 2darray
        Denoised data
    """
    start_time = datetime.now()
    print('\n*****************************************************')
    print('*                                                   *')
    print('*                   Cadzow denoising                *')
    print('*                                                   *')
    print('*****************************************************\n')

    datap = np.zeros_like(data)
    for k in range(data.shape[0]):
        print('Processing of transient '+str(k+1)+' of '+str(data.shape[0]), c='violet')
        if i:
            datap[k] = processing.iterCadzow(data[k], n=n, nc=nc, f=f, itermax=itermax, print_head=False, print_time=False)
        else:
            datap[k] = processing.cadzow(data[k], n=n, nc=nc, print_head=False)
    print('Processing has ended!', c='violet')
    end_time = datetime.now()
    if print_time is True:
        print('Total runtime: {}'.format(end_time - start_time))

    return datap

# -------------------------------------------------------------------------------------------------------------------

# BASELINE


def qfil(ppm, data, u, s, SFO1):
    """
    Suppress signals in the spectrum using a gaussian filter.

    Parameters
    ----------
    ppm : 1darray
        ppm scale of the spectrum
    data : ndarray
        Data to be processed. The filter is applied on the last dimension
    u : float
        Position of the filter /ppm
    s : float
        Width of the filter (FWHM) /Hz
    SFO1 : float
        Spectrometer larmor frequency

    Returns
    -------
    pdata : ndarray
        Filtered data
    """
    # Convert fwhm to stdev
    cnv = 2 * (2 * np.log(2))**0.5
    s /= cnv
    # Convert Hz to ppm
    sppm = misc.freq2ppm(s, SFO1)
    # Make the filter
    G = sim.gaussian_filter(ppm, u, sppm)
    # Apply it
    datap = np.zeros_like(data)
    datap[..., :] = data[..., :] * G
    print(f'Applied qfil at {u:.3f} ppm with FWHM = {s * cnv:.0f} Hz.\n', c='violet')
    return datap


def acme(data, m=1, a=5e-5):
    r"""
    Automated phase Correction based on Minimization of Entropy.
    This algorithm allows for automatic phase correction by minimizing the entropy of the m-th derivative of the spectrum,
    as explained in detail by `L. Chen et. al.`_.

    Defined the entropy of `h` as:

    .. math::

        S = - \sum_j h[j] \ln( h[j] )

    and

    .. math::

        h = \frac { | R[j]^{(m)} | }{ \sum_j | R[j]^{(m)} | }

    where

    .. math::

        R = \Re\{ d \, e^{-i \phi} \}

    and :math:`R^{(m)}` is the m-th derivative of `R`, the objective function to minimize is:

    .. math::

        S + P(R)

    where `P(R)` is a penalty function for negative values of the spectrum.

    The phase correction is applied using :func:`klassez.processing.ps`. The values ``p0`` and ``p1`` are fitted using Nelder-Mead algorithm.


    .. _L. Chen et. al.: https://www.sciencedirect.com/science/article/pii/S1090780702000691


    Parameters
    ----------
    data : 1darray
        Spectrum to be phased, complex
    m : int
        Order of the derivative to be computed
    a : float
        Weighting factor for the penalty function

    Returns
    -------
    p0f : float
        Fitted zero-order phase correction, in degrees
    p1f : float
        Fitted first-order phase correction, in degrees
    """

    def entropy(data):
        """
        Compute entropy of data.

        Parameters
        ----------
        data : ndarray
            Input data

        Returns
        -------
        S : float
            Entropy of data
        """
        data_in = np.copy(data)
        if not data_in.all():
            zero_ind = np.flatnonzero(data_in == 0)
            for i in zero_ind:
                data_in[i] = 1e-15

        return - np.sum(data_in * np.log(data_in))

    def mth_derivative(data, m):
        """
        Computes the m-th derivative of data by applying np.gradient m times.

        Parameters
        ----------
        data : 1darray
            Input data
        m : int
            Order of the derivative to be computed

        Returns
        -------
        pdata : 1darray
            m-th derivative of data
        """
        pdata = np.copy(data)
        for k in range(m):
            pdata = np.gradient(pdata)
        return pdata

    def penalty_function(data, a=5e-5):
        """
        F(y) is a function that is 0 for positive y and 1 otherwise.
        The returned value is
            a * sum_j F(y_j) y_j^2

        Parameters
        ----------
        data : 1darray
            Input data
        a : float
            Weighting factor

        Returns
        -------
        p_fun : float
            a * sum_j F(y_j) y_j^2
        """
        signs = - np.sign(data)     # 1 for negative entries, -1 for positive entries
        p_arr = np.array([0 if j < 1 else 1 for j in signs])  # replace all !=1 values in signs with 0
        p_fun = a * np.sum(p_arr * data**2)     # Make the sum
        return p_fun

    def f2min(param, data, m, a):
        """ Cost function for the fit. Applies the algorithm. """
        par = param.valuesdict()
        p0 = par['p0']
        p1 = par['p1']

        # Phase data and take real part
        Rp, *_ = processing.ps(data, p0=p0, p1=p1)
        R = Rp.real

        # Compute the derivative and the h function
        Rm = np.abs(mth_derivative(R, m))
        H = np.sum(Rm)  # Normalization factor
        h = Rm / H

        # Calculate the penalty factor
        P = penalty_function(R, a)

        # Compute the residual
        res = entropy(h) + P
        return res

    if not np.iscomplexobj(data):
        raise ValueError('Input data is not complex.')

    # Define the parameters of the fit
    param = lmfit.Parameters()
    param.add('p0', value=0, min=-180, max=180)
    param.add('p1', value=0, min=-720, max=720)

    # Minimize using simplex method because the residue is a scalar
    minner = lmfit.Minimizer(f2min, param, fcn_args=(np.copy(data), m, a))
    result = minner.minimize(method='nelder', tol=1e-15)
    popt = result.params.valuesdict()

    return popt['p0'], popt['p1']


def whittaker_smoother(data, n=2, s_f=1, w=None):
    """
    Adapted from `P.H.C. Eilers, Anal. Chem 2003, 75, 3631-3636`_.
    Implementation of the smoothing algorithm proposed by Whittaker in 1923.

    .. _P.H.C. Eilers, Anal. Chem 2003, 75, 3631-3636: https://pubs.acs.org/doi/10.1021/ac034173t

    Parameters
    ----------
    data : 1darray
        Data to be smoothed
    n : int
        Order of the difference to be computed
    s_f : float
        Smoothing factor
    w : 1darray or None
        Array of weights. If None, no weighting is applied.

    Returns
    -------
    z : 1darray
        Smoothed data
    """
    # Import things to handle sparse matrices
    import scipy.sparse as sps
    from scipy.sparse.linalg import spsolve

    y = np.copy(data)
    m = data.shape[-1]      # Data dimension

    if w is None:           # Use a vector of ones for the weights
        w = np.ones(m)

    # Compute the derivative matrix directly as sparse
    signs = np.array([(-1)**(n+k) for k in range(n+1)])
    entries = misc.binomial_triangle(n+1)
    D = sps.lil_matrix((m-n, m))    # Empty
    for k in range(n+1):            # Fill only the interesting diagonals
        D.setdiag(signs[k]*entries[k], k)
    D.tocsr()       # Conversion to csr

    W = sps.lil_matrix((m, m))   # Sparse weights matrix
    W.setdiag(w)
    W.tocsr()       # Conversion to csr

    A = sps.csr_matrix(W + s_f * D.T @ D)   # Sparse criterion
    z = spsolve(A, w*y)     # Find solutions using LU factorization

    return z


def rpbc(data, split_imag=False, n=5, basl_method='huber', basl_thresh=0.2, basl_itermax=2000, **phase_kws):
    """
    Reversed Phase and Baseline Correction.
    Allows for the automatic phase correction and baseline subtraction of NMR spectra.
    It is called "reversed" because the baseline is actually computed and subtracted before to perform the phase correction.

    The baseline is computed using a low-order polynomion, built on a scale that goes from -1 to 1, whose coefficients are obtained minimizing a non-quadratic cost function.
    It is recommended to use either ``"tq"`` (truncated quadratic, much faster) or ``"huber"`` (Huber function, slower but sometimes more accurate).
    The user is requested to choose between separating the real and imaginary channel in this step.
    The order of the polynomion and the threshold value are the key parameters for obtaining a good baseline. The used function is :func:`klassez.processing.polyn_basl`

    The phase correction is computed on the baseline-subtracted complex data as described in the SINC algorithm.
    The default parameters are generally fine, but in case of data with poor SNR (approximately SNR < 10) better results can be obtained by increasing the value of the ``e1`` parameter.
    The employed function is :func:`klassez.fit.SINC_phase`

    .. note::
        Not excellent results. Computation might be slow

    Parameters
    ----------
    data : 1darray
        Data to be processed, complex-valued
    split_imag : bool
        If True, computes the baseline on the real and imaginary part separately; else, the set of polynomion coefficients are forced to be the same for both
    n : int
        Number of coefficients of the polynomion, i.e. it will be of degree n-1
    basl_method : str
        Cost function to be minimized for the baseline computation. Look for ``fit.CostFunc``, ``"method"`` attribute
    basl_thresh : float
        Relative threshold value for the non-quadratic behaviour of the cost function. Look for ``fit.CostFunc``, ``"s"`` attribute
    basl_itermax : int
        Maximun number of iterations allowed during the baseline fitting procedure
    phase_kws : keyworded arguments
        Optional arguments for the phase correction. Look for ``fit.SINC_phase`` keyworded arguments for details.

    Returns
    -------
    y : 1darray
        Processed data
    p0 : float
        Zero-order phase correction angle, in degrees
    p1 : float
        First-order phase correction angle, in degrees
    c : 1darray
        Set of coefficients to be used for the baseline computation, starting from the 0-order coefficient

    .. seealso::

        :func:`klassez.fit.SINC_phase`

        :func:`klassez.processing.polyn_basl`

        :class:`klassez.fit.CostFunc`
    """

    # Check if the data is actually complex
    if np.iscomplexobj(data):
        y = np.copy(data)
    else:
        raise ValueError('Input data is not complex. Aborting...')

    # BASELINE COMPUTATION AND SUBTRACTION
    if not n:  # Do not correct the baseline
        c = [0+0j]
        basl = np.zeros_like(y)
    else:
        if split_imag:
            # Compute baseline for real and imaginary parts separately
            basl_r, c_r = fit.polyn_basl(y.real, n=n, method=basl_method, s=basl_thresh, itermax=int(basl_itermax))
            basl_i, c_i = fit.polyn_basl(y.imag, n=n, method=basl_method, s=basl_thresh, itermax=int(basl_itermax))
            # Put them together afterwards
            c = np.array(c_r) + 1j*np.array(c_i)
            basl = basl_r + 1j*basl_i
        else:
            # Compute the baseline on the complex spectrum
            basl, c = fit.polyn_basl(y, n=n, method=basl_method, s=basl_thresh, itermax=int(basl_itermax))

    # Transform c into array for easier handling
    c = np.array(c)
    # Subtract the baseline to the data
    y -= basl

    # PHASE CORRECTION
    # Compute the phase angles
    p0, p1 = fit.sinc_phase(y, **phase_kws)
    # Apply it
    y, *_ = processing.ps(y, p0=p0, p1=p1)

    return y, p0, p1, c


def align(ppm_scale, data, lims, u_off=0.5, ref_idx=0):
    """
    Performs the calibration of a pseudo-2D experiment by circular-shifting the spectra of an appropriate amount.
    The target function aims to minimize the superimposition between a reference spectrum and the others using a brute-force method.

    Parameters
    ----------
    ppm_scale : 1darray
        ppm scale of the spectrum to calibrate
    data : 2darray
        Complex-valued spectrum
    lims : tuple
        (ppm sx, ppm dx) of the calibration region
    u_off : float
        Maximum offset for the circular shift, in ppm
    ref_idx : int
        Index of the spectrum to be used as reference

    Returns
    -------
    data_roll : 2darray
        Calibrated data
    u_cal : list
        Number of point of which the spectra have been circular-shifted
    u_cal_ppm : list
        Correction for the ppm scale of each experiment
    """

    def f2min(param, s_ref, s, span_region):
        """
        Cost function for the fit
        """
        # Unpack the parameters
        par = param.valuesdict()

        # Circular-shift the spectrum
        roll_s = np.roll(s, int(par['u']))

        # Normalize the spectra to their maximum in the calibration region
        s_ref_norm = s_ref.real / max(s_ref.real[span_region])
        roll_s_norm = roll_s.real / max(roll_s.real[span_region])
        # Compute the residuals
        res = s_ref_norm - roll_s_norm
        return res[span_region]

    # Shallow copy
    data_in = np.copy(data)

    # Convert the offset in points
    npoints = int((u_off / misc.calcres(ppm_scale)))

    # Convert the ppm limits into points indeces
    sx = misc.ppmfind(ppm_scale, lims[0])[0]
    dx = misc.ppmfind(ppm_scale, lims[1])[0]
    # Calibration region
    cal_reg = slice(min(sx, dx), max(sx, dx), 1)

    # Get the reference spectrum
    s_ref = data_in[ref_idx]

    # Initialize the output variables
    u_cal = np.empty(data_in.shape[0])  # Shifts in points
    u_cal[ref_idx] = 0      # The reference spectrum does not move!

    u_cal_ppm = np.empty(data_in.shape[0])  # Shifts in ppm
    u_cal_ppm[ref_idx] = 0  # Same here!

    for i, s_i in enumerate(data_in):   # Loop over the experiments
        if i != ref_idx:                # The reference spectrum does not move!
            # Make the parameters of the fit
            param = lmfit.Parameters()
            param.add('u', value=0, max=npoints, min=-npoints)
            param['u'].set(brute_step=1)      # Discrete step of one point

            # Fit
            minner = lmfit.Minimizer(f2min, param, fcn_args=(s_ref, s_i, cal_reg))
            result = minner.minimize(method='brute', max_nfev=1000)

            # Unpack the parameters and store them in the output variables
            popt = result.params.valuesdict()
            u = popt['u']

            u_cal[i] = int(u)
            u_cal_ppm[i] = u * misc.calcres(ppm_scale)

    # Apply the correction
    data_roll = []      # Initialize output variable
    for i, experiment in enumerate(data_in):        # Loop over the experiments
        # Roll the spectra of the appropriate amount and append them to the list
        data_roll.append(np.roll(experiment, int(u_cal[i])))
    # Transform into array
    data_roll = np.array(data_roll)

    return data_roll, u_cal, u_cal_ppm


def lp(data, pred=1, order=8, mode='b'):
    """
    Apply linear prediction on the dataset.
    This method solves the linear system

    .. math::

        D a = d

    where `a` is the array of linear prediction coefficients.

    Parameters
    ----------
    data : 1darray
        FID to be linear-predicted
    pred : int
        Number of points to predict
    order : int
        Number of coefficients to use for the prediction
    mode : str
        ``'f'`` for forward linear prediction, ``'b'`` for backward linear prediction

    Returns
    -------
    newdata : 1darray
        FID with linear prediction applied.
    """
    def make_Dd(x, order, mode):
        L = len(x) - order      # Dimension of the coefficient array
        if mode == 'f':         # Forward lp
            # Characteristic matrix
            D = misc.hankel(x[:-1], order)
            # Coefficient vector
            d = x[order:].reshape(L, 1)
        elif mode == 'b':       # Backward lp
            # Characteristic matrix
            D = misc.hankel(x[1:], order)
            # Coefficient vector
            d = x[:L].reshape(L, 1)
        return D, d

    def find_lpc(D, d):
        """
        Solve the linear system
        """
        # Compute pseudoinverse matrix
        Dpinv = np.linalg.pinv(D)
        # Solve the system
        a = Dpinv @ d
        return a

    def extrapolate(trace, a, pred, mode):
        """
        Apply the correction
        """
        m = len(a)          # Number of coefficients
        M = len(trace)      # Number of points
        # Make placeholder for the new FID
        ntrace = np.empty((M+pred), dtype=trace.dtype)
        if mode == 'f':     # forward lp
            ntrace[:M] = trace  # Copy the "old" fid at the beginning
            for i in range(pred):   # Fill the rest point-by-point
                ntrace[M+i] = np.sum(np.multiply(ntrace[M-m+i: M+i], a.flat))
        if mode == 'b':     # backward lp
            ntrace[-M:] = trace  # Copy the "old" fid at the end
            for i in range(pred):   # Fill the rest point-by-point, going backwards
                ntrace[pred-i-1] = np.sum(np.multiply(ntrace[pred-i: pred+m-i], a.flat))
        return ntrace
    if len(data.shape) != 1:
        raise NotImplementedError('LP available for only 1D datasets')

    N = data.shape[-1]      # Length of the dataset
    # Cut the data to ease the computational workload
    t = min(N, max(10*pred, 2048))  # 500 points or 10x the number of points
    if mode == 'f':     # Take the last part of the fid
        x = data[-t:]
    elif mode == 'b':   # Take the first part of the fid
        x = data[:t]
    # Compute the system
    D, d = make_Dd(x, order, mode)
    # Solve the system
    a = find_lpc(D, d)
    # Apply the correction
    newdata = extrapolate(data, a, pred, mode)
    return newdata


def blp(data, pred=1, order=8):
    """
    Applies backward linear prediction by calling :func:`klassez.processing.lp` with ``mode='b'``.

    Parameters
    ----------
    data : 1darray
        FID to be linear-predicted
    pred : int
        Number of points to predict
    order : int
        Number of coefficients to use for the prediction

    Returns
    -------
    lpdata : 1darray
        FID with linear prediction applied.

    .. seealso::

        :func:`klassez.processing.lp`
    """
    lpdata = lp(data, pred, order, mode='b')
    return lpdata


def stack_fids(*fids, filename=None):
    """
    Stacks together FIDs in order to create a pseudo-2D experiment.
    This function can handle either arrays or Spectrum_1D objects.

    Parameters
    ----------
    fids : sequence of 1darrays or Spectrum_1D objects
        Input data.
    filename : str
        Location for a .npy file to be saved. If None, no file is created.

    Returns
    -------
    p2d : 2darray
        Stacked FIDs.
    """
    p2d_fid = []    # Placeholder
    # Append the FIDs to this list
    for k, fid in enumerate(fids):
        # If 1darray, append it as is
        if isinstance(fid, np.ndarray) and len(fid.shape) == 1:
            p2d_fid.append(fid)
        # If Spectrum_1D, append the "fid" attribute
        elif isinstance(fid, Spectrum_1D):
            p2d_fid.append(fid.fid)
        else:   # Raise an error
            raise ValueError(f'There was a problem in reading the {k+1}° fid.')

    # Pile up the FIDs
    p2d = np.stack(p2d_fid, axis=0)

    # Save the .npy file
    if isinstance(filename, (str, Path)):
        np.save(filename, p2d)

    return p2d


def hilbert(f, axis=-1):
    """
    Computes the Hilbert transform of real vector ``f`` in order to retrieve its imaginary part.
    The algorithm computes the convolution by means of FT, as follows:
    #. make IFT of ``f``: ``a``
    #. compute ``h = [1j for x in range(N) if x < N/2 else -1j]``
    #. Compute ``b = h * a``
    #. Build ``d = a + 1j*b``
    #. make FT of ``d``: ``F``
    #. replace the real part of ``F`` with ``f``

    .. important::
        Make sure that the original spectrum was zero-filled to at least twice the original size of the FID.

    Parameters
    ----------
    f : ndarray
        Array of which you want to compute the imaginary part
    axis : int
        Axis along which to compute the Hilbert transform. Default is -1 (last axis).

    Returns
    -------
    f_cplx : ndarray
        Complex version of ``f``

    .. seealso::
        :func:`klassez.processing.hilbert2`
    """
    # Get the number of dimensions
    ndim = f.ndim
    # Normalize axis to positive index
    if axis < 0:
        axis += ndim
    # Get the number of points along the specified axis
    N = f.shape[axis]
    # Create axes list and move the specified axis to the end
    axes = list(range(ndim))
    axes.append(axes.pop(axis))
    # Transpose f to make the specified axis the last one
    f_trans = np.transpose(f, axes)
    # Suppress warnings for ft of real data
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        # Apply Hilbert transform on transposed data
        # Make ift of the data
        a = np.fft.ifft(f_trans.real)
        # Compute vector h: 1j for the first half, -1j for the other half
        h = 1j * np.ones_like(a)
        h[..., N//2:] *= -1
        # Retrieve imaginary part
        b = a * h
        # Make the full, complex signal
        d = a + 1j * b
        # Do ft
        f_i = np.fft.fft(d)
    # Replace real part of the data with the original one
    f_cplx_trans = f_trans.real + 1j * f_i.imag
    # Create inverse axes to move the last axis back to original position
    inv_axes = list(range(ndim))
    inv_axes.insert(axis, inv_axes.pop(-1))
    # Transpose back to original order
    f_cplx = np.transpose(f_cplx_trans, inv_axes)
    return f_cplx


def hilbert2(data):
    """
    Retrieve the imaginary parts of a hypercomplex dataset, when you only have the `rr` part.

    Given ``Ht = klassez.processing.hilbert``:

    .. code-block:: python

        rr = Ht(rr).real
        ir = Ht(rr).imag
        ri = - Ht(rr.T).T.imag
        ii = Ht(ri).imag

    Parameters
    ----------
    data : 2darray
        rr part

    Returns
    -------
    rr : 2darray
        Real part in f2, real part in f1
    ir : 2darray
        Imaginary part in f2, real part in f1
    ri : 2darray
        Real part in f2, imaginary part in f1
    ii : 2darray
        Imaginary part in f2, imaginary part in f1

    .. seealso::

        :func:`klassez.processing.hilbert`

        :func:`klassez.processing.repack_2D`
    """
    # ir: Ht on F2
    S_rr_ir = processing.hilbert(data, axis=1)
    rr = S_rr_ir.real
    ir = S_rr_ir.imag
    # ri: Ht on F1. The - is because of NMR conventions
    ri = - processing.hilbert(data, axis=0).imag
    # ii: Ht on both F2 and F1
    ii = processing.hilbert(ri).imag

    return rr, ir, ri, ii


def convolve(in1, in2):
    """
    Perform the convolution of the two array by multiplying their inverse Fourier transform.
    The two arrays must have the same dimension.

    Parameters
    ----------
    in1 : ndarray
        First array
    in2 : ndarray
        Second array

    Returns
    -------
    cnv : ndarray
        Convolved array
    """
    assert in1.shape[-1] == in2.shape[-1], 'The two arrays have different dimensions!'
    size = in1.shape[-1]
    in1t = np.fft.ifft(np.fft.ifftshift(in1))
    in2t = np.fft.ifft(np.fft.ifftshift(in2))
    cnvt = in1t * in2t
    # factor size is needed to correct the intensity
    cnv = size * np.fft.fftshift(np.fft.fft(cnvt))
    return cnv


def inv_convolve(in1, in2):
    """
    Perform the inverse-convolution of the two array by dividing their inverse Fourier transform.
    The two arrays must have the same dimension.

    .. important::

        This operation involves a division!!! Might give unexpected and unpleasant results!

    Parameters
    ----------
    in1 : ndarray
        First array
    in2 : ndarray
        Second array

    Returns
    -------
    cnv : ndarray
        Deconvolved array
    """
    assert in1.shape[-1] == in2.shape[-1], 'The two arrays have different dimensions!'
    size = in1.shape[-1]
    in1t = np.fft.ifft(np.fft.ifftshift(in1))
    in2t = np.fft.ifft(np.fft.ifftshift(in2))
    cnvt = in1t * np.linalg.pinv(in2t.reshape(-1, 1)).reshape(-1)
    # factor size is needed to correct the intensity
    cnv = size * np.fft.fftshift(np.fft.fft(cnvt))
    return cnv


def splitcomb(data, taq, J=53.8):
    """
    Applies the processing required for the IPAP virtual decoupling scheme in the direct dimension.
    The data structure must be with the IP in the first half of the direct dimension, and with AP in the second half.
    The default J is 53.8 Hz, correspondant to the CO-Ca coupling.

    Parameters
    ----------
    data : 2darray
        FID of the spectrum to process
    taq : 1darray
        Acquisition timescale
    J : float
        Scalar coupling constant of the coupling to suppress, in Hz

    Returns
    -------
    datap : 2darray
        Decoupled data. The direct dimension is halved with respect to the original FID
    """

    # Split IP and AP FIDs
    N = data.shape[-1]
    # Left one: IP
    data_ip = data[..., :N//2]
    # Right one: AP
    data_ap = data[..., N//2:]

    # Get single feature of the doublet
    data_sum = data_ip + data_ap
    data_dif = data_ip - data_ap

    # Make sure the acquisition timescale matches the dimensions
    taq = processing.extend_taq(taq, N//2)

    # Make Dirac delta functions to shift the data
    delta_sx = np.exp(+1j * np.pi * J * taq)
    delta_dx = np.exp(-1j * np.pi * J * taq)

    # The sum spectrum must go right
    data_sum *= delta_sx
    # The diff spectrum must go left
    data_dif *= delta_dx

    # Now sum the two to double sensitivity
    datap = data_sum + data_dif

    return datap


def apk(ppm, data, SFO1, alpha=3, winsize=50, ap1=True, seethrough=False):
    r"""
    Performs automatic phase correction.

    The algorithm starts with the computation of a mask to separate signal from baseline-only regions.
    This is done via an iterative thresholding, i.e. a point is "signal" if the first derivative of the spectrum in that point is higher than its standard deviation by ``alpha`` times:

    .. math::

        d'[k] > \alpha \, std(d') \implies k \in \text{signal region}

    The selection is further refined by repeating the same procedure on the original data.
    Then, the regions separated by less than winsize are joined together, and the presence of actual peaks in the region is checked with a pick-picker.

    At this point, each region is phased independently with only phase 0. The phase angle is tested in a brute-force manner.
    The cost function minimizes the area below the straight line that connects the borders of the window.
    The first-order phase correction is calculated with a weighted linear regression, where the weights are the integrals of the magnitude of each region.

    .. tip::

        The choice of ``alpha`` and ``winsize`` can be important for the outcome.
        Higher ``alpha`` values make the detection of peaks more stringent -> for spectra with high SNR and sharp peaks a suitable value is 4-6.
        Decreasing ``winsize`` makes the algorithm to estimate more regions.

    Parameters
    ----------
    ppm : 1darray
        ppm scale of the spectrum
    data : 1darray
        Spectrum
    SFO1 : float
        Nucleus' Larmor frequency /MHz
    alpha : float
        Factor that multiplies the std of the spectrum to set the threshold
    winsize : float
        Minimum size of the window that can contain peaks /Hz
    ap1 : bool
        True to adjust both zero and first order, False for only phase zero
    seethrough : bool
        If True, draws a series of diagnostic figures to see what the algorithm is doing

    Returns
    -------
    datap : 1darray
        Phased data
    values : tuple
        Found values ``(p0, p1)``

    .. seealso::

        :func:`klassez.processing.mask_sgn_basl`

        :func:`klassez.processing.ps`

        :func:`klassez.fit.lr`
    """
    def phase0(data, slices):
        """
        Phase the regions independently with only phase 0
        """
        def f2min(p0, y, x):
            """
            Cost function for the fit
            """
            N = max(x)
            # Try to phase
            y_ph = processing.ps(y, p0=p0)[0].real
            # Straight line that passes for the borders of the window
            basl = (y_ph[-1] - y_ph[0]) / N * x + y_ph[0]
            # Where the spectrum is below the baseline
            neg_cut = np.where(y_ph < basl)
            # Minus the area below the baseline so that it becomes positive
            neg_area = - np.trapezoid(y_ph[neg_cut]-basl[neg_cut])
            return neg_area

        def phase_loop(y, x, p0list):
            """
            Get p0 for each region
            """
            # Initialize
            p0_opt = 0
            neg_area = np.inf
            # try all the phases in p0list
            for p0 in p0_list:
                # Find the minimum value through brute force
                neg_area_ph = f2min(p0, y, x)
                if neg_area_ph < neg_area:
                    neg_area = neg_area_ph
                    p0_opt = p0

            return p0_opt

        # Placeholder
        p0s = []
        # Loop on the windows
        for sl in slices:
            # Trim data and make a scale
            y = data[sl]
            x = np.arange(y.shape[-1])

            # Initial list: from -180 to 180 in 20 steps
            p0_start, p0_stop, p0_step = -180, 180, 19
            for n in range(3):
                # Extend the list
                p0_list = np.linspace(p0_start, p0_stop, p0_step)
                # Find the best compromise
                p0_opt = phase_loop(y, x, p0_list)
                # Shrink around the best value
                p0_start = p0_opt - p0_step / 2
                p0_stop = p0_opt + p0_step / 2
                # Repeat three times so that you have about 1 degree inaccuracy
            # Update the optimal value
            p0s.append(p0_opt)
        return np.array(p0s)

    # -----------------------------------------------------------------------------------

    # Get the peak-only regions
    slices, _ = processing.mask_sgn_basl(ppm, data, SFO1, alpha, winsize)

    # Debug figure to see the detected regions
    if seethrough:
        x = np.arange(data.shape[-1])
        fig = plt.figure('Detected regions to be phased')
        fig.set_size_inches(15, 8)
        ax = fig.add_subplot()
        for sl in slices:
            ax.plot(x[sl], data[sl].real)
        misc.pretty_scale(ax, ax.get_xlim(), 'x')
        misc.pretty_scale(ax, ax.get_ylim(), 'y')
        misc.mathformat(ax)
        misc.set_fontsizes(ax, 20)
        plt.show()

    # Smooth the data with a 5 Hz gaussian filter
    gbdata = processing.smooth_g(data, 1/misc.hz2pt(misc.ppm2freq(ppm, SFO1), 5))
    # Phase each region independently with only p0
    p0s = phase0(gbdata, slices)

    if len(p0s) > 1:    # there is more than one region detected
        # Compute the relative positions of the regions on the phase scale (that goes from -0.5 to 0.5)
        rel_pos = np.array([np.mean([sl.start, sl.stop])/data.shape[-1] for sl in slices]) - 0.5
        # The weights are the integrals of the magnitude of the detected regions
        weights = np.array([np.trapezoid(np.abs(data[sl])) for sl in slices])
        # ...normalized
        weights /= np.sum(weights)

        if ap1:  # Compute p1
            # Make a weighted linear regression
            phase_reg, (p1, p0) = fit.lr(p0s, rel_pos, w=weights)

            if seethrough:
                fig = plt.figure('Linear regression for phase 1')
                fig.set_size_inches(15, 8)
                ax = fig.add_subplot()
                ax.errorbar(rel_pos, p0s, yerr=1/weights, c='k', fmt='x')
                ax.plot(rel_pos, phase_reg, ':', c='tab:red')
                ax.set_xlabel('Relative position')
                ax.set_ylabel(r'$\phi_0$ /°')
                misc.pretty_scale(ax, ax.get_xlim(), 'x')
                misc.pretty_scale(ax, ax.get_ylim(), 'y')
                misc.set_fontsizes(ax, 20)
                plt.show()
        else:   # do not compute p1
            p1 = 0

        if np.abs(p1) < 5:  # then it is either not computed or negligible
            # p0 is the weighted average of the p0s
            p0 = np.average(p0s, weights=weights)
            # p1 is 0 by default
            p1 = 0
    else:   # Only one region was detected
        # p0 must be unpacked as the single found value
        p0 = p0s[-1]
        # p1 cannot be estimated
        p1 = 0

    # Apply the correction
    datap, *_ = processing.ps(data, p0=p0, p1=p1)

    print('APK: p0: {:.3f}, p1: {:.3f}\n'.format(p0, p1), c='violet')

    return datap, (p0, p1)


def abc(ppm, data, n=5, lims=None, alpha=2.75, qfil=False, qfilp={'u': 4.7, 's': 10}):
    """
    Automatic computation of a baseline for a spectrum using a thresholding-based method
    for the detection of the baseline-only region, followed by a weighted linear least squares
    optimization with a polynomion of degree n-1.
    The weights are computed on the absolute value of the first derivative of the spectrum.
    Set ``qfil=True`` if there is a very intense solvent peak that would hamper
    the computation of the threshold.

    Parameters
    ----------
    ppm : 1darray
        PPM scale of the spectrum
    data : 1darray
        The spectrum to baseline-correct
    n : int
        Number of coefficients of the polynomial baseline
    lims : tuple or None
        Limits for the region on which to compute the baseline, in ppm
    alpha : float
        The threshold will be set as ``thr = alpha * np.std(np.gradient(data))``
    qfil : bool
        Choose whether to apply a filter on the solvent region (True) or not (False)
    qfilp : dict
        Parameters to be used to compute the filter if ``qfil=True``. Keys:
        ``'u'`` = center of the filter in ppm
        ``'s'`` = width of the filter in Hz

    Returns
    -------
    baseline: 1darray
        Computed baseline

    .. seealso::

        :func:`klassez.processing.qfil`

        :func:`klassez.fit.lsp`

        :func:`klassez.processing.abc_v2`
    """

    def compute_weights(ppm, data, qfil=False, alpha=2.75):
        """ Computes the weights to be used for the weighted least squares optimization """
        if qfil:    # Apply according to qfilp
            d = processing.qfil(ppm, data, qfilp['u'], qfilp['s'])
        else:       # Do nothing
            d = deepcopy(data)
        # Second derivative of the data
        y = np.abs(np.gradient(data))
        # Second derivative of the qfilled data, if the case
        d2 = np.abs(np.gradient(d))
        # Use d2 to compute the threshold
        thr = alpha * anal.noise_std(d2)

        # Find peaks and widths on the second derivative
        peaks, *_ = scipy.signal.find_peaks(y, height=thr, threshold=None, distance=None, prominence=None, width=None, wlen=None, rel_height=0.5, plateau_size=None)
        widths, *_ = scipy.signal.peak_widths(y, peaks, rel_height=0.7, prominence_data=None, wlen=None)

        # Initialize w all equal to 1
        w = np.ones_like(x, dtype=int)

        # Convert peaks positions and widths in x coordinated
        x_peaks = x[peaks]
        widths *= misc.calcres(x)
        # Regions will be (lefts, rights)
        lefts = x_peaks - widths / 2
        rights = x_peaks + widths / 2

        # Inside the intervals, weights are zero
        for left, right in zip(lefts, rights):
            # Condition: l < x < r
            mask = (left <= x) & (x <= right)
            w[mask] = 0
        return w

    # Scale goes from 0 to 1
    x = np.arange(data.shape[-1]) / data.shape[-1]
    # Compute the weights
    w = compute_weights(ppm, data, qfil, alpha)

    # Correct the weights if limits are given
    if lims is not None:
        # Left part: 0 when more left than the lower limit
        wleft = (ppm > min(lims)).astype(int)
        # Right part: 0 when more right than the upper limit
        wright = (ppm < max(lims)).astype(int)
        # Correct the weights
        w *= wleft * wright

    # Compute the coefficients of the polynomion
    c = fit.lsp(data, x, n=n, w=w)
    # Compute the baseline
    baseline = misc.polyn(x, c)
    return baseline


def abc_v2(ppm, data, SFO1, n=5, lims=None, alpha=5, winsize=2, qfil=False, qfilp={'u': 4.7, 's': 10}):
    """
    Automatic computation of a baseline for a spectrum using a thresholding-based method for the detection of the baseline-only region,
    followed by a weighted linear least squares optimization with a polynomion of degree n-1.
    Employs the same method for the detection of signal-free regions of :func:`klassez.processing.apk`.
    Set ``qfil=True`` if there is a very intense solvent peak that would hamper the computation of the threshold.

    Parameters
    ----------
    ppm : 1darray
        PPM scale of the spectrum
    data : 1darray
        The spectrum to baseline-correct
    SFO1 : float
        Nucleus Larmor frequency /MHz
    n : int
        Number of coefficients of the polynomial baseline
    lims : tuple or None
        Limits for the region on which to compute the baseline, in ppm
    alpha : float
        The threshold will be set as thr = alpha * np.std(np.gradient(data))
    winsize : float
        Minimum allowed window containing signals /Hz
    qfil : bool
        Choose whether to apply a filter on the solvent region (True) or not (False)
    qfilp : dict
        Parameters to be used to compute the filter if ``qfil=True``. Keys:
        ``'u'`` = center of the filter in ppm
        ``'s'`` = width of the filter in Hz

    Returns
    -------
    baseline : 1darray
        Computed baseline

    .. seealso::

        :func:`klassez.processing.apk`

        :func:`klassez.fit.lsp`

        :func:`klassez.processing.abc`
    """

    def compute_weights(ppm, data, SFO1, qfil=False, alpha=5, winsize=2):
        """ Computes the weights to be used for the weighted least squares optimization """
        if qfil:    # Apply according to qfilp
            d = processing.qfil(ppm, data, qfilp['u'], qfilp['s'], SFO1)
        else:       # Do nothing
            d = deepcopy(data)

        # Initialize w all equal to 1
        w = np.zeros_like(x, dtype=float)

        _, basl_slices = mask_sgn_basl(ppm, d, SFO1, alpha=alpha, winsize=winsize)

        for sl in basl_slices:
            w[sl] = 1

        if np.all(w == 0):
            w += 0.05

        return w

    # Scale goes from 0 to 1
    x = np.arange(data.shape[-1]) / data.shape[-1]
    # Compute the weights
    w = compute_weights(ppm, data, SFO1, qfil, alpha, winsize)

    # Correct the weights if limits are given
    if lims is not None:
        # Left part: 0 when more left than the lower limit
        wleft = (ppm > min(lims)).astype(int)
        # Right part: 0 when more right than the upper limit
        wright = (ppm < max(lims)).astype(int)
        # Correct the weights
        w *= wleft * wright

    # Compute the coefficients of the polynomion
    c = fit.lsp(data, x, n=n, w=w)
    # Compute the baseline
    baseline = misc.polyn(x, c)
    return baseline


def abs(ppm, data, n=5, lims=None, alpha=2.75, qfil=False, qfilp={'u': 4.7, 's': 10}):
    """
    Computes the baseline correction on data using :func:`klassez.processing.abc`, and gives back the subtracted spectrum.
    The imaginary part of the spectrum is reconstructed using :func:`klassez.processing.hilbert`.

    Parameters
    ----------
    ppm : 1darray
        PPM scale of the spectrum
    data : 1darray
        The spectrum to baseline-correct
    n : int
        Number of coefficients of the polynomial baseline
    lims : tuple or None
        Limits for the region on which to compute the baseline, in ppm
    alpha : float
        The threshold will be set as ``thr = alpha * np.std(np.gradient(data))``
    qfil : bool
        Choose whether to apply a filter on the solvent region (True) or not (False)
    qfilp : dict
        Parameters to be used to compute the filter if ``qfil=True``. Keys:
        ``'u'`` = center of the filter in ppm
        ``'s'`` = width of the filter in Hz

    Returns
    -------
    S : 1darray
        Baseline-subtracted spectrum

    .. seealso::

        :func:`klassez.processing.abc`

        :func:`klassez.processing.hilbert`
    """
    # Compute the baseline
    b = processing.abc(ppm, data.real, n=n, alpha=alpha, lims=lims, qfil=qfil, qfilp=qfilp)
    # Subtract it
    datab = data.real - b
    # Compute the missing imaginary part
    S = processing.hilbert(datab)
    return S


def abs_v2(ppm, data, SFO1, n=5, lims=None, alpha=5, winsize=2, qfil=False, qfilp={'u': 4.7, 's': 10}):
    """
    Computes the baseline correction on data using :func:`klassez.processing.abc_v2`, and gives back the subtracted spectrum.
    The imaginary part of the spectrum is reconstructed using :func:`klassez.processing.hilbert`.

    Parameters
    ----------
    ppm : 1darray
        PPM scale of the spectrum
    data : 1darray
        The spectrum to baseline-correct
    n : int
        Number of coefficients of the polynomial baseline
    lims : tuple or None
        Limits for the region on which to compute the baseline, in ppm
    alpha : float
        The threshold will be set as ``thr = alpha * np.std(np.gradient(data))``
    winsize : float
        Minimum allowed window containing signals /Hz
    qfil : bool
        Choose whether to apply a filter on the solvent region (True) or not (False)
    qfilp : dict
        Parameters to be used to compute the filter if ``qfil=True``. Keys:
        ``'u'`` = center of the filter in ppm
        ``'s'`` = width of the filter in Hz

    Returns
    -------
    S : 1darray
        Baseline-subtracted spectrum

    .. seealso::

        :func:`klassez.processing.abc_v2`

        :func:`klassez.processing.hilbert`
    """
    # Compute the baseline
    b = processing.abc_v2(ppm, data.real, SFO1, n=n, alpha=alpha, winsize=winsize, lims=lims, qfil=qfil, qfilp=qfilp)
    # Subtract it
    datab = data.real - b
    # Compute the missing imaginary part
    S = processing.hilbert(datab)
    return S


def abs2_v2(ppm_f2, data, SFO1, n=5, lims=None, alpha=5, winsize=2, qfil=False, qfilp={'u': 4.7, 's': 10}, FnMODE='States-TPPI'):
    """
    Baseline correction for 2D datasets, alternative version.
    Computes the baseline correction on ``data`` using :func:`klassez.processing.abc_v2` for each row, and gives back the subtracted spectrum.
    The imaginary part of the spectrum is reconstructed using either :func:`klassez.processing.hilbert` or :func:`klassez.processing.hilbert2` depending on ``FnMODE``.

    .. todo::

        Work in progress!

    Parameters
    ----------
    ppm : 1darray
        PPM scale of the spectrum
    data : 1darray
        The spectrum to baseline-correct
    n : int
        Number of coefficients of the polynomial baseline
    lims : tuple or None
        Limits for the region on which to compute the baseline, in ppm
    alpha : float
        The threshold will be set as ``thr = alpha * np.std(np.gradient(data))``
    qfil : bool
        Choose whether to apply a filter on the solvent region (True) or not (False)
    qfilp : dict
        Parameters to be used to compute the filter if ``qfil=True``. Keys:
        ``'u'`` = center of the filter in ppm
        ``'s'`` = width of the filter in Hz

    Returns
    -------
    S : 2darray
        Baseline-subtracted spectrum, either complex or hypercomplex

    .. seealso::

        :func:`klassez.processing.abc_v2`

        :func:`klassez.processing.hilbert`

        :func:`klassez.processing.hilbert2`
    """
    # Compute the baseline
    D = deepcopy(data)
    for k, trace in enumerate(D):
        b = processing.abc_v2(ppm_f2, trace, SFO1, n=n, lims=lims, winsize=winsize, qfil=qfil, qfilp=qfilp)
        # Subtract it
        D[k] = trace - b
    # Compute the missing imaginary part
    if FnMODE in ['States-TPPI', 'Echo-Antiecho']:
        rr, ir, ri, ii = processing.hilbert2(D)
        S = processing.repack_2D(rr, ir, ri, ii)
    else:
        S = processing.hilbert(D)
    return S


def abs2(ppm_f2, data, n=5, lims=None, alpha=2.75, qfil=False, qfilp={'u': 4.7, 's': 10}, FnMODE='States-TPPI'):
    """
    Baseline correction for 2D datasets.
    Computes the baseline correction on ``data`` using :func:`klassez.processing.abc` for each row, and gives back the subtracted spectrum.
    The imaginary part of the spectrum is reconstructed using either :func:`klassez.processing.hilbert` or :func:`klassez.processing.hilbert2` depending on ``FnMODE``.

    Parameters
    ----------
    ppm : 1darray
        PPM scale of the spectrum
    data : 1darray
        The spectrum to baseline-correct
    n : int
        Number of coefficients of the polynomial baseline
    lims : tuple or None
        Limits for the region on which to compute the baseline, in ppm
    alpha : float
        The threshold will be set as ``thr = alpha * np.std(np.gradient(data))``
    qfil : bool
        Choose whether to apply a filter on the solvent region (True) or not (False)
    qfilp : dict
        Parameters to be used to compute the filter if ``qfil=True``. Keys:
        ``'u'`` = center of the filter in ppm
        ``'s'`` = width of the filter in Hz

    Returns
    -------
    S : 2darray
        Baseline-subtracted spectrum, either complex or hypercomplex

    .. seealso::

        :func:`klassez.processing.abc`

        :func:`klassez.processing.hilbert`

        :func:`klassez.processing.hilbert2`
    """
    # Compute the baseline
    D = deepcopy(data)
    for k, trace in enumerate(D):
        b = processing.abc(ppm_f2, trace, n=n, lims=lims, qfil=qfil, qfilp=qfilp)
        # Subtract it
        D[k] = trace - b
    # Compute the missing imaginary part
    if FnMODE in ['States-TPPI', 'Echo-Antiecho']:
        rr, ir, ri, ii = processing.hilbert2(D)
        S = processing.repack_2D(rr, ir, ri, ii)
    else:
        S = processing.hilbert(D)
    return S


def rndc(data):
    """
    Robust Noise Derivative Calculation, `reference`_ . Employed coefficients: ``(42, 48, 27, 8, 1)/512``
    Used to compute the first derivative of a function.

    .. _reference: http://www.holoborodko.com/pavel/numerical-methods/numerical-derivative/smooth-low-noise-differentiators/

    Parameters
    ----------
    data : 1darray
        Input data

    Returns
    -------
    dy : 1darray
        First derivative of data. First and last 5 points are set to zero.
    """

    # Shallow copy
    d = deepcopy(data)

    # Convolution coefficients
    coeff = np.array([42, 48, 27, 8, 1]) / 512

    # We need to compute (d[k+j] - d[k-j]) for j = 1,...,5 and for all k eligible
    # This is the most efficient way I could think of
    couples = [[np.roll(d, -j), np.roll(d, j)] for j in range(1, 6)]
    diffs = np.array([coeff[k] * (couple[0] - couple[1]) for k, couple in enumerate(couples)])

    # 5e-2 is a correction factor to make it the same as np.gradient. IDK where it comes from
    dy = np.sum(diffs, axis=0)
    # Set first and last 5 points to 0 because they are meaningless
    dy[..., :5] = 0
    dy[..., -5:] = 0
    return dy


def smooth_g(d, m):
    """
    Apply a smoothing with a gaussian filter by convolution. The width of the filter is ``1/m``.

    Parameters
    ----------
    d : 1darray
        Data to be smoothed
    m : float
        Inverse width of the filter /pt

    Returns
    -------
    yc : 1darray
        Smoothed data
    """
    # Shallow copy
    data = deepcopy(d)
    # Scale to compute the gaussian
    x = np.arange(data.shape[-1])
    # Make the filter
    Gf = sim.f_gaussian(x, np.mean(x), 1/m)
    # Smooth the data
    yc = processing.convolve(data, Gf)
    return yc


def mask_sgn_basl(ppm, data, SFO1, alpha=3, winsize=50):
    """
    Given an NMR spectrum, this function estimates the signal and baseline regions, and return a list of slices to cut the spectrum accordingly.

    Parameters
    ----------
    ppm : 1darray
        ppm scale of the spectrum
    data : 1darray
        Spectrum
    SFO1 : float
        Nucleus' Larmor frequency /MHz
    alpha : float
        Factor that multiplies the std of the spectrum to set the threshold
    winsize : float
        Minimum size of the window that can contain peaks /Hz

    Returns
    -------
    peak_slices : list of slices
        Slices that trim the data in the signal-only regions
    basl_slices : list of slices
        Slices that trim the data in the baseline-only regions
    """
    def calc_mask(data, alpha=3):
        """
        Compute the mask that divides "signals" from "baseline" regions.
        There is "signal" if the spectrum is higher than alpha * std of the spectrum.

        Parameters
        ----------
        data : 1darray
            Spectrum
        alpha : float
            Factor that multiplies the std of the spectrum to set the threshold

        Returns
        -------
        full_mask : 1darray
            1 if there is signal, 0 is there is not
        """
        # First derivative of data
        d = processing.rndc(data)
        # Make a shallow copy
        d_in = deepcopy(d)

        # Initialize the mask
        full_mask = np.zeros(d.shape[-1])
        for j in range(1000):   # instead of while
            # Compute standard deviation of the data
            std = np.std(np.abs(d_in))
            # where the spectrum is bigger than alpha times the std
            mask = (np.abs(d_in) > alpha*std).astype(int)
            # Add these regions to the mask
            full_mask += mask
            # killmask kills the signal
            killmask = 1 - mask

            if np.all(killmask == 1):   # No more peaks detected
                break
            else:                       # Kill all the signals and start again
                d_in *= killmask

        # Now copy the original dataset
        d_in = deepcopy(data)
        # Apply rhe mask
        d_in *= (1 - full_mask)
        # Do the same thing but on the original dataset to see if we missed anything
        # in the "baseline" regions
        for j in range(1000):   # instead of while
            std = np.std(d_in)
            mask = (np.abs(d_in) > alpha*std).astype(int)
            full_mask += mask
            killmask = 1 - mask
            if np.all(killmask == 1):
                break
            else:
                d_in *= killmask

        return full_mask

    def noise_correlation(data, mask):
        """
        Compute the window at which the noise is correlated to estimate the smoothing factor.

        """

        # Get the borders of the regions
        starts, ends = misc.detect_jumps(mask)

        # Make slices for the signals regions
        slices = [slice(start, end) for start, end in zip(starts, ends)]

        m = 1     # placeholder
        for k, sl in enumerate(slices):  # For each region
            # Trim the data
            y = data[sl]
            # Make a scale
            x = np.arange(y.shape[-1])
            # Useless to consider regions less than 5 points
            if len(x) < 5:
                continue
            # This is basically a correlation function
            corr_func = np.array([
                np.sum(y * np.roll(y, -j))
                for j in range(y.shape[-1])
                                 ])
            # The noise is not correlated anymore if it is less than 60% the maximum correlation
            tol = corr_func[0] * 0.6
            okay = np.where(corr_func < tol)[0]
            if okay.size > 0:   # If this is true, take only that part of the correlation function
                corr_func = corr_func[:okay[0]]
                x = x[:okay[0]]

            # Useless to consider regions less than 5 points
            if len(x) < 5:
                continue

            # Fit with an exponential -> linear regression on the logarithm
            logcf = np.log(corr_func)
            lr, (mk, q) = fit.lr(logcf, x)
            # mk is the slope of the linear regression -> the time constant
            if mk < m:
                # update
                m = mk
        return m

    def mask_to_regions(ppm, d, mask, winsize=50):

        # The minimum allowed window is winsize Hz converted to points
        window = misc.hz2pt(misc.ppm2freq(ppm, SFO1), winsize)

        # Same here, see where there are signals
        starts, ends = misc.detect_jumps(mask)

        # Merge the regions that are not further away than window points
        merged = []     # placeholders
        current_start = starts[0]
        current_end = ends[0]

        for s, e in zip(starts[1:], ends[1:]):   # for each window
            # length of the block
            gap = s - current_end
            if gap < window:
                # Extend the current block
                current_end = e
            else:
                # Add the block to the merged list
                merged.append((current_start, current_end))
                # Update the borders
                current_start = s
                current_end = e
            # Add to the merged list
            merged.append((current_start, current_end))
        # This is the true thing
        true_merged = []
        for w in merged:    # exclude regions less than 5 points
            if max(w) - min(w) > 5:
                true_merged.append(w)

        # Make a new mask
        new_mask = np.zeros_like(mask)
        # This is 1 in the merged regions
        for s, e in true_merged:
            new_mask[s:e] = 1

        return new_mask

    def anypeak(data, mask):
        """
        Correct the slices to get only the ones that contain peaks
        """
        # Make the slices
        slices = [slice(start, end) for start, end in zip(*misc.detect_jumps(mask))]

        true_slices = []    # placeholder
        for sl in slices:   # loop on the slices
            # Trim data
            y = data[sl]
            # Compute magnitude
            magy = np.abs(y)
            # Standard deviation of the magnitude
            std = np.std(magy)

            # Use scipy peak-picker to see if there are peaks taller than 3 times std
            peaks, *_ = scipy.signal.find_peaks(magy-min(magy), height=3*std)

            if len(peaks):  # there are!
                # Extend the window slightly
                start = sl.start - 5 if sl.start - 5 > 0 else 0
                end = sl.stop + 5 if sl.stop + 5 < data.shape[-1] else data.shape[-1]
                # Add the new window to the list
                true_slices.append(slice(start, end))
        return true_slices

    # Compute the mask to separate signal- from baseline-regions
    mask = calc_mask(data, alpha)

    # Apply correction to the mask:
    #   compute correlation on the noise
    corr = noise_correlation(data.real, 1 - mask)
    #   smooth the data with a gaussian filter
    fdata = processing.smooth_g(data, corr)
    #   correct the mask with the filter data
    mask = mask_to_regions(ppm, fdata, mask, winsize)

    # Detect peaks
    peak_slices = anypeak(fdata, mask)

    # Compute complementary slices
    basl_slices = []    # placeholder

    last_end = 0    # we start from zero
    for sl in peak_slices:
        # start at the start and end at the end
        start = sl.start or 0
        end = sl.stop
        if start > last_end:    # avoid loop break
            # New slice is from end to start in peak_slices
            basl_slices.append(slice(last_end, start))
        # Update the end and go again
        last_end = max(last_end, end)
    # Append last part
    if last_end < data.shape[-1]:
        basl_slices.append(slice(last_end, data.shape[-1]))

    return peak_slices, basl_slices


def extend_taq(old_taq, newsize=None):
    """
    Extend the acquisition timescale to a longer size, using the same dwell time

    Parameters
    ----------
    old_taq : 1darray
        Old timescale
    newsize : int
        New size of acqusition timescale, in points

    Returns
    -------
    new_taq : 1darray
        Extended timescale
    """
    # Safety check
    if newsize is None:
        new_taq = np.copy(old_taq)
    elif newsize <= len(old_taq):  # Extend only if needed
        new_taq = np.copy(old_taq)
    else:
        dw = misc.calcres(old_taq)      # Get the dwell time
        new_taq = np.arange(0, dw * newsize, dw)    # Compute new scale
    return new_taq


def sl_bas_onidx(y, x_idx):
    """
    Computes the straight line that connects ``y[min(x_idx)]`` with ``y[max(x_idx)]``.
    If ``y`` is a 2darray, a 2darray of lines is returned, one correspondant to each row of ``y``.
    The lines will be ``max(x_idx) - min(x_idx)`` points long.

    Parameters
    ----------
    y : 1darray or 2darray
        Array for which to compute the connecting lines
    x_idx : tuple
        Points indices of the endpoints of the connecting lines

    Returns
    -------
    bas : 1darray or 2darray
        Computed connecting lines. Of note, this is squeezed before returning!

    .. seealso::

        :func:`numpy.squeeze`
        :func:`klassez.fit.lr`
    """
    # Let's be sure the indices are in the correct order
    x_idx = sorted(x_idx)
    # Array of shape (y.shape[0], 2) of the endpoints
    bas_points = np.array([y[..., q] for q in x_idx]).T
    if len(bas_points.shape) == 1:  # add a dimension to use the for after
        bas_points = [bas_points]

    bas = []    # placeholder
    # Loop on the rows of the endpoints
    for k, tr in enumerate(bas_points):
        # tr contains two points -> only one line exists
        _, (bas_m, bas_q) = fit.lr(tr, x=np.asarray(x_idx))
        # Compute the scale for rendering the connecting line correctly
        xb = np.arange(min(x_idx), max(x_idx) + 1)
        # Add to placeholder
        bas.append(xb * bas_m + bas_q)
    # Remove the extra dimension added before, if needed
    return np.squeeze(np.array(bas))


def sl_bas(x, y, lims=None):
    """
    Computes the straight line that connects ``y`` between ``lims`` on the ``x`` scale.
    If ``y`` is a 2darray, a 2darray of lines is returned, one correspondant to each row of ``y``.

    Parameters
    ----------
    y : 1darray or 2darray
        Array for which to compute the connecting lines
    x : 1darray
        Scale for the referencing of ``lims``
    lims : tuple
        ``(left, right)`` for selecting the endpoints

    Returns
    -------
    bas_overlay : 1darray or 2darray
        Computed connecting lines, with the same shape of ``y``. Of note, this is squeezed before returning!

    .. seealso::

        :func:`numpy.squeeze`
        :func:`klassez.processing.sl_bas_onidx`
    """
    # Get the correct arrays to work with
    if lims is None:
        # Use the whole data
        x_tr = deepcopy(x)
        y_tr = deepcopy(y)
        # Limits are therefore first and last point of the scale
        lims = x[0], x[-1]
    else:
        # Cut the data according to lims
        if len(y.shape) > 1:
            # Use a dummy yscale to cut only on the last dimension
            x_tr, _, y_tr = misc.trim_data_2D(x, np.arange(y.shape[0]), y, xlim=lims)
        else:
            # Just cut
            x_tr, y_tr = misc.trim_data(x, y, lims=lims)

    # Get the indices of the endpoints ON x NOT xtrim, otherwise the overlay does not work
    x_idx = [misc.ppmfind(x, v)[0] for v in lims]
    # Compute the correcting lines
    bas = sl_bas_onidx(y, x_idx)
    # Add a dimension to still use the for
    if len(bas.shape) == 1:
        bas = [bas]

    # Match the shape of y. Use always the left limit as anchor point
    overlay_bas = [misc.sum_overlay(np.zeros_like(x), y_b, x[min(x_idx)], x) for y_b in bas]
    return np.squeeze(np.array(overlay_bas))
