#! /usr/bin/env python3

import os
import sys
import numpy as np
from numpy import linalg
from scipy import linalg as slinalg
from scipy import stats
from scipy.spatial import ConvexHull
import random
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.widgets import Slider, Button, RadioButtons, TextBox, CheckButtons, Cursor, LassoSelector
from matplotlib.path import Path
import seaborn as sns
import nmrglue as ng
import lmfit as l
from datetime import datetime
import warnings

from . import fit, misc, sim, figures, processing
from .config import CM, COLORS, cron
from .Spectra import Spectrum_1D

""" 
Contains a series of processing functions for different purposes
"""



# CPMG processing

def interactive_echo_param(data0):
    """
    Interactive plot that allows to select the parameters needed to process a CPMG-like FID.
    Use the TextBox or the arrow keys to adjust the values.
    You can call processing.sum_echo_train or processing.split_echo_train by starring the return statement of this function, i.e.:
        processing.sum_echo_train(data0, *interactive_echo_train(data0))
    as they are in the correct order to be used in this way.
    -------
    Parameters:
    - data0: ndarray
        CPMG FID
    -------
    Returns:
    - n: int
        Distance between one echo and the next one
    - n_echoes: int
        Number of echoes to sum/split
    - i_p: int
        Offset points from the start of the FID
    """

    # Check for data dimension and safety copy
    if len(data0.shape) == 1:
        data = np.copy(data0)
    elif len(data0.shape) == 2:
        data = np.copy(data0[0,:])
    else:
        raise ValueError('Data shape not supported')

    # Make the figure
    fig = plt.figure('Echo Splitter')
    fig.set_size_inches(figures.figsize_large)
    plt.subplots_adjust(left=0.25, right=0.95, top=0.90, bottom=0.15)
    ax = fig.add_subplot(2,3,(1,5)) # Square plot
    axs = fig.add_subplot(2,3,3)    # Right top
    axt = fig.add_subplot(2,3,6)    # Right bottom

    # Initialize the three values in a dictionary
    param = {
            'n' : 20,
            'n_echoes' : 2,
            'i_p' : 0,
            }

    # ---------------------------------------------------------
    def update_axs():
        """ Redraw the figure """
        # Compute new data
        newdata = processing.sum_echo_train(data, **param)
        # Draw it in the top-right subplot
        sum_sp.set_data(np.arange(len(newdata)), newdata)
        # Make FT and draw it in bottom-right subplot
        new_ft = processing.ft(newdata)
        sum_ft.set_data(np.arange(len(new_ft)), new_ft)
        # Make pretty scales
        misc.pretty_scale(axs, (0, len(newdata)), 'x')
        misc.set_ylim(axs, newdata)
        misc.set_ylim(axt, new_ft)
        misc.pretty_scale(axs, (0, len(newdata)-1), 'x')
        misc.pretty_scale(axt, (0, len(new_ft)-1), 'x')
        misc.pretty_scale(axs, axs.get_ylim(), 'y')
        misc.pretty_scale(axt, axt.get_ylim(), 'y')
        # Write the current values
        for label in radio.labels:
            T = label.get_text()
            val_text[f'{T}'].set_text(f'{param[T]}')
        plt.draw()

    def read_tb(text):
        """ Eval() the input in the textbox, clear it """
        val = eval(text)
        input_tb.text_disp.set_text('')
        return int(val)

    def change_param(text):
        """ Change parameters according to the TextBox """
        nonlocal param
        try:    # Avoid error due to the clear text
            param[f'{radio.value_selected}'] = read_tb(text)
        except SyntaxError:
            pass
        # Draw the red bars and set them visible
        [X.set_xdata((k*param['n']+param['i_p'],)) for k, X in enumerate(sampling)]
        change_nechoes()
        # Redraw the plots
        update_axs()

    def change_nechoes():
        """ Set a certain number of red bars as visible """
        for k, X in enumerate(sampling):
            if k < param['n_echoes']:
                X.set_visible(True)
            else:
                X.set_visible(False)

    def key_press(event):
        """ Edit the param dictionary with uparrow and downarrow """
        nonlocal param
        if event.key == 'up':
            param[f'{radio.value_selected}'] += 1
        elif event.key == 'down':
            param[f'{radio.value_selected}'] -= 1
        else:
            return
        # Redraw the red bars and set them visible
        [X.set_xdata((k*param['n']+param['i_p']),) for k, X in enumerate(sampling)]
        change_nechoes()
        # Redraw the subplots
        update_axs()

    # ---------------------------------------------------------

    # Make the widgets with their boxes
    radio_box = plt.axes([0.025, 0.40, 0.15, 0.35])
    input_box = plt.axes([0.025, 0.20, 0.15, 0.08])
    input_box.set_title('Insert value here')
    input_tb = TextBox(input_box, '')
    radio = RadioButtons(radio_box, list(param.keys()), activecolor='tab:blue')

    # Write the current values to be updated
    val_text = {}
    for k, label in enumerate(radio.labels):
        val_text[f'{label.get_text()}'] = radio_box.text(0.95, label.get_position()[1]-0.025, 
                f'{param[label.get_text()]:.0f}',
                ha='right', va='bottom')

    # Set a scale
    x = np.arange(data.shape[-1])

    ax.plot(x, data, lw=0.5)    # FID
    # Top right plot
    sum_sp, = axs.plot(np.arange(param['n']//2), processing.sum_echo_train(data, **param))
    # Bottom right plot
    sum_ft,  = axt.plot(np.arange(param['n']//2), processing.ft(processing.sum_echo_train(data, **param)))

    # Red bars
    sampling = [ax.axvline(k*param['n'], c='r', lw=0.5) for k in range(data.shape[-1]//param['n'])]
    change_nechoes()    # Draw them

    # Titles
    ax.set_title('FID')
    axs.set_title('Sum FID')
    axt.set_title('Sum Spectrum')

    # Scales
    misc.pretty_scale(ax, (x[0], x[-1]), 'x')
    misc.pretty_scale(ax, ax.get_ylim(), 'y')
    misc.pretty_scale(axs, (0, param['n']//2-1), 'x')
    misc.pretty_scale(axt, (0, param['n']//2-1), 'x')
    misc.pretty_scale(axs, axs.get_ylim(), 'y')
    misc.pretty_scale(axt, axt.get_ylim(), 'y')

    # Connect the widgets to the functions
    input_tb.on_submit(change_param)        # Text box
    fig.canvas.mpl_connect('key_press_event', key_press)    # Keys

    plt.show()
    plt.close()

    return tuple([param[f'{label.get_text()}'] for label in radio.labels])


def sum_echo_train(datao, n, n_echoes, i_p=0):
    """
    Sum up a CPMG echo-train FID into echoes so to be enchance the SNR.
    This function calls processing.split_echo_train with the same parameters.
    -------
    Parameters:
    - datao: ndarray
        FID with an echo train on its last dimension
    - n: int
        number of points that separate one echo from the next
    - n_echoes: int
        number of echoes to sum
    - i_p: int
        Number of offset points
    ------
    Returns:
    - data_p: ndarray
        Summed echoes
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
    -------
    Parameters:
    - datao: ndarray
        FID with an echo train on its last dimension
    - n: int
        number of points that separate one echo from the next
    - n_echoes: int
        number of echoes to extract. If it is 0, extracts only the first decay
    - i_p: int
        Number of offset points
    ------
    Returns:
    - data_p: (n+1)darray
        Separated echoes
    """
    # Take account of the offset points 
    data = datao[..., i_p:]
    # nm = middle point. +1 if n is odd
    if np.mod(n,2) == 0:
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

        datal = data[..., A][...,::-1]      # Left part, reversed
        datar = data[..., B]                # Right part
        
        # Reversing in time means to change sign to the imaginary part
        if np.iscomplexobj(data):
            datal = np.conj(datal)
        datap.append(datal + datar) # Sum up
    # Create the output data by stacking the echoes. This adds a dimension
    data_p = np.stack(datap)

    return data_p

# -----------------------------------------------------------------------

#   fid adjustment
def quad(fid):
    """
    Subtracts from the FID the arithmetic mean of its last quarter. The real and imaginary channels are treated separately.
    -------
    Parameters:
    - fid : ndarray
        Self-explanatory.
    -------
    Returns:
    - fid : ndarray
        Processed FID.
    """
    size = fid.shape[-1]
    qsize = size//4
    avg_re = np.average(fid[...,-qsize:].real)
    avg_im = np.average(fid[...,-qsize:].imag)
    fid.real = fid.real - avg_re
    fid.imag = fid.imag - avg_im
    return fid

def qpol(fid):
    """
    Fits the FID with a 4-th degree polynomion, then subtracts it from the original FID. The real and imaginary channels are treated separately.
    -------
    Parameters:
    - fid : ndarray
        Self-explanatory.
    -------
    Returns:
    - fid_corr : ndarray
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
    """
    Sine-squared apodization.
    --------
    Parameters:
    - ssb: int
        Sine bell shift. 
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
    """
    Sine apodization.
    --------
    Parameters:
    - ssb: int
        Sine bell shift. 
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
    """
    Exponential apodization
    ---------
    Parameters:
    - data: ndarray
        Input data
    - lb: float
        Lorentzian broadening. It should be positive.
    - sw: float
        Spectral width /Hz
    """
    lb = lb / sw
    apod = np.exp(-np.pi * np.arange(data.shape[-1]) * lb).astype(data.dtype)
    return apod * data

def gm(data, lb, gb, gc, sw):
    """
    Gaussian apodization.
    The parameter 'lb' controls the sharpening factor of a rising exponential, and behaves exactly as in processing.em.
    In contrast, 'gb' controls the gaussian decay factor.
    Apply this function VERY CAREFULLY. Choose the right values through the interactive processing.
    -------
    Parameters:
    - data: ndarray
        Input data
    - lb: float
        Lorentzian sharpening /Hz. It should be negative.
    - gb: float
        Gaussian broadening. It should be positive.
    - gc: float
        Gaussian center, relatively to the FID length: 0 <= gc <= 1
    - sw: float
        Spectral width /Hz
    -------
    Returns:
    - pdata: ndarray
        Processed data
    """
    size = data.shape[-1]
    x = np.arange(size)
    A = - np.pi * lb / sw * x
    B = np.pi * gb / sw * (gc * (size - 1) - x)
    apod = np.exp(A - B**2)
    return apod * data

def gmb(data, lb, gb, sw):
    """
    Bruker-style Gaussian apodization.
    Apply this function VERY CAREFULLY. Choose the right values through the interactive processing.
    --------
    Parameters:
    - data: ndarray
        Input data
    - lb: float
        Lorentzian sharpening /Hz. It should be negative.
    - gb: float
        Gaussian broadening. It should be positive.
    - sw: float
        Spectral width /Hz
    -------
    Returns:
    - pdata: ndarray
        Processed data
    """
    size = data.shape[-1]
    x = np.arange(size)/size
    apod = np.exp(- np.pi / sw * size * (lb * x - lb / (2 * gb) * x**2))
    return apod * data

# zero-filling
def zf(data, size):
    """
    Zero-filling of data up to size in its last dimension.
    -------
    Parameters:
    - data: ndarray
        Array to be zero-filled
    - size: int
        Number of points of the last dimension after zero-filling
    -------
    Returns:
    - datazf: ndarray
        Zero-filled data
    """
    def zf_pad(data, pad):
        size = list(data.shape)
        size[-1] = int(pad)
        z = np.zeros(size, dtype=data.dtype)
        return np.concatenate((data, z), axis=-1)
    zpad = size - data.shape[-1]
    if zpad <= 0 :
        zpad = 0
    datazf = zf_pad(data, pad=zpad)
    return datazf

# Fourier transform
def ft(data0, alt=False, fcor=0.5):
    """ 
    Fourier transform in NMR sense.
    This means it returns the reversed spectrum.
    ------------
    Parameters:
    - data0: ndarray
        Array to Fourier-transform
    - alt: bool
        negates the sign of the odd points, then take their complex conjugate. Required for States-TPPI processing.
    - fcor: float
        weighting factor for FID 1st point. Default value (0.5) prevents baseline offset
    ---------
    Returns:
    - dataft: ndarray
        Transformed data
    """
    data = np.copy(data0)
    if not np.iscomplexobj(data):
        # Suppress "casting complex to real" warning
        warnings.simplefilter("ignore")
    size = data.shape[-1]
    data[...,0] = data[...,0] * fcor
    if alt:
        data[...,1::2] = data[...,1::2] * -1
        data.imag = data.imag * -1
    dataft = np.fft.fftshift(np.fft.fft(data, axis=-1).astype(data.dtype), -1)[...,::-1]
    # Restore the normal warnings behavior
    warnings.simplefilter("default")
    return dataft

def ift(data0, alt=False, fcor=0.5):
    """ 
    Inverse Fourier transform in NMR sense.
    This means that the input dataset is reversed before to do iFT.
    ------------
    Parameters:
    - data0: ndarray
        Array to Fourier-transform
    - alt: bool
        negates the sign of the odd points, then take their complex conjugate. Required for States-TPPI processing.
    - fcor: float
        weighting factor for FID 1st point. Default value (0.5) prevents baseline offset
    -----------
    Returns:
    - dataft: ndarray
        Transformed data
    """
    data = np.copy(data0)[...,::-1]
    if not np.iscomplexobj(data):
        # Suppress "casting complex to real" warning
        warnings.simplefilter("ignore")
    size = data.shape[-1]
    dataft = np.fft.ifft(np.fft.ifftshift(data, -1), axis=-1).astype(data.dtype)
    if alt:
        dataft[...,1::2] = dataft[...,1::2] * -1
        dataft.imag = dataft.imag * -1
    dataft[...,0] = dataft[...,0] / fcor
    # Restore the normal warnings behavior
    warnings.simplefilter("default")
    return dataft
    
def rev(data):
    """
    Reverse data over its last dimension
    """
    datarev = data[...,::-1]
    return datarev
    
    # phase correction
def ps(data, ppmscale=None, p0=None, p1=None, pivot=None, interactive=False):
    """
    Applies phase correction on the last dimension of data.
    The pivot is set at the center of the spectrum by default.
    Missing parameters will be inserted interactively.
    -------
    Parameters:
    - data: ndarray
        Input data
    - ppmscale: 1darray or None
        PPM scale of the spectrum. Required for pivot and interactive phase correction
    - p0: float
        Zero-order phase correction angle /degrees
    - p1: float
        First-order phase correction angle /degrees
    - pivot: float or None.
        First-order phase correction pivot /ppm. If None, it is the center of the spectrum.
    - interactive: bool
        If True, all the parameters will be ignored and the interactive phase correction panel will be opened.
    --------
    Returns:
    - datap: ndarray
        Phased data
    - final_values: tuple
        Employed values of the phase correction. (p0, p1, pivot)
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
        datap, final_values = processing.interactive_phase_1D(ppmscale, data)
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
    Shuffles data if the spectrum is acquired with FnMODE = Echo-Antiecho.
    NOTE: introduces -90° phase shift in F1, to be corrected after the processing

    pdata = np.zeros_like(data)
    pdata[::2] = (data[::2].real - data[1::2].real) + 1j*(data[::2].imag - data[1::2].imag)
    pdata[1::2] = -(data[::2].imag + data[1::2].imag) + 1j*(data[::2].real + data[1::2].real)

    ---------
    Parameters:
    - data: 2darray
        FID in echo-antiecho format
    ---------
    Returns:
    - pdata: 2darray
        FID in States-TPPI format
    """
    pdata = np.zeros_like(data)
    pdata[::2] = (data[::2].real - data[1::2].real) + 1j*(data[::2].imag - data[1::2].imag)
    pdata[1::2] = -(data[::2].imag + data[1::2].imag) + 1j*(data[::2].real + data[1::2].real)
    return pdata
    
    
def tp_hyper(data):
    """
    Computes the hypercomplex transpose of data.
    Needed for the processing of data acquired in a phase_sensitive manner
    in the indirect dimension.
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
    Separates hypercomplex data into 4 distinct ser files
    --------
    Parameters:
    - data: 2darray
        Hypercomplex matrix
    --------
    Returns:
    - rr: 2darray
        Real F2, Real F1
    - ir: 2darray
        Imaginary F2, Real F1
    - ri: 2darray
        Real F2, Imaginary F1
    - ii: 2darray
        Imaginary F2, Imaginary F1
    """
    rr = data.real[::2]
    ir = data.imag[::2]
    ri = data.real[1::2]
    ii = data.imag[1::2]
    return rr, ir, ri, ii
    
def repack_2D(rr, ir, ri, ii):
    """
    Renconstruct hypercomplex 2D NMR data given the 4 ser files
    -------
    Parameters:
    - rr: 2darray
        Real F2, Real F1
    - ir: 2darray
        Imaginary F2, Real F1
    - ri: 2darray
        Real F2, Imaginary F1
    - ii: 2darray
        Imaginary F2, Imaginary F1
    -------
    Returns:
    - data: 2darray
        Hypecomplex matrix
    """
    data = np.empty((2*rr.shape[0],rr.shape[1]), dtype='complex64')
    data.real[::2] = rr
    data.imag[::2] = ir
    data.real[1::2] = ri
    data.imag[1::2] = ii
    return data
    
def td_eff(data, tdeff):
    """
    Uses only the first tdeff points of data. tdeff must be a list as long as the dimensions:
    tdeff = [F1, F2, ..., Fn]
    --------
    Parameters:
    - data: ndarray
        Data to be trimmed
    - tdeff: list of int
        Number of points to be used in each dimension
    """
    datain = np.copy(data)

    def trim(datain, n):
        return datain[...,:n]
    
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
    
    X = tuple(np.roll(np.arange(ndim),1)) # Roll the dimensions to the right
    
    for k in range(ndim):
        if tdeff[k]:
            datain = trim(datain, tdeff[k])
        datain = np.transpose(datain, X)
    
    return datain
    
    
    
def fp(data, wf=None, zf=None, fcor=0.5, tdeff=0):
    """
    Performs the full processing of a 1D NMR FID (data).
    --------
    Parameters:
    - data: 1darray
        Input data
    - wf: dict
        {'mode': function to be used, 'parameters': different from each function}
    - zf: int
        final size of spectrum
    - fcor: float
        weighting factor for the FID first point
    - tdeff: int
        number of points of the FID to be used for the processing.
    ------
    Returns:
    - datap: 1darray
        Processed data
    """
    # Window function
    datap = processing.td_eff(data, tdeff)
    if wf is not None:
        if wf['mode'] == 'qsin':
            datap = processing.qsin(datap, ssb=wf['ssb'])
        if wf['mode'] == 'sin':
            datap = processing.sin(datap, ssb=wf['ssb'])
        if wf['mode'] == 'em':
            datap = processing.em(datap, lb=wf['lb'], sw=wf['sw'])
        if wf['mode'] == 'gm':
            datap = processing.gm(datap, lb=wf['lb_gm'], gb=wf['gb_gm'], sw=wf['sw'], gc=wf['gc'])
        if wf['mode'] == 'gmb':
            datap = processing.gmb(datap, lb=wf['lb'], gb=wf['gb'], sw=wf['sw'])
    # Zero-filling
    if zf is not None:
        datap = processing.zf(datap, zf)
    # FT
    datap = processing.ft(datap, fcor=fcor)
    return datap
    
    
def interactive_fp(fid0, acqus, procs):
    """
    Perform the processing of a 1D NMR spectrum interactively. The GUI offers the opportunity to test different window functions, as well as different tdeff values and final sizes.
    The active parameters appear as blue text.
    -------
    Parameters:
    - fid0: 1darray
        FID to process
    - acqus: dict
        Dictionary of acquisition parameters
    - procs: dict
        Dictionary of processing parameters
    -------
    Returns:
    - pdata: 1darray
        Processed spectrum
    - procs: dict
        Updated dictionary of processing parameters:
    """
    
    def get_apod(size, procs):
        """ Calculate the window function on the basis of 'procs' """
        Y = np.ones(size, dtype='complex64')    # array of ones
        # Process the array of ones and then revert FT to get everything but the processing
        apodf = processing.ift(processing.fp(Y, wf=procs['wf'], zf=procs['zf'], tdeff=procs['tdeff']))
        apodf = apodf.real
        # Adjust the dimension to size
        if apodf.shape[-1] < size:  # if shorter than size, zero-fill
            apodf = processing.zf(apodf, size)
        if apodf.shape[-1] > size:  # if longet than size, trim
            apodf = processing.td_eff(apodf, size)
        return apodf
    
    # Copy initial FID to prevent overwriting
    fid = np.copy(fid0)
    fid0 = np.copy(fid)
    Y = np.ones_like(fid0)
    
    # Calculate starting values
    data = processing.fp(fid, wf=procs['wf'], zf=procs['zf'], tdeff=procs['tdeff'])
    
    
    # Get WF
    apodf = get_apod(fid0.shape[-1], procs)
    
    # Calculate the ppm scale
    fq_scale = processing.make_scale(data.shape[-1], acqus['dw'], rev=True)
    ppm_scale = misc.freq2ppm(fq_scale, acqus['SFO1'], acqus['o1p'])
    
    
    # Make the figure panel
    fig = plt.figure('Interactive Processing')
    fig.set_size_inches(15,9)
    plt.subplots_adjust(left=0.1, bottom=0.05, right=0.8, top=0.95, hspace=0.4)
    ax = fig.add_subplot(4,1,(1,3))     # spectrum
    axf = fig.add_subplot(4,1,4)        # fid
    # Define useful things 
    modes = ['No', 'em', 'sin', 'qsin', 'gm', 'gmb']   # entries for the radiobuttons
    act_keys = {    # Active Parameters:
            'No': [],
            'em': ['lb'],
            'sin': ['ssb'],
            'qsin': ['ssb'],
            'gm': ['lb_gm', 'gb_gm', 'gc'],
            'gmb': ['lb', 'gb'],
            }
    tx = {} # Dictionary of the texts
    
    # Draw boxes for widgets
    SI_box = plt.axes([0.85, 0.85, 0.07, 0.04])
    tdeff_box = plt.axes([0.85, 0.80, 0.07, 0.04])
    mode_box = plt.axes([0.825, 0.5, 0.15, 0.25])
    ssb_box = plt.axes([0.85, 0.40, 0.07, 0.04])
    lb_box = plt.axes([0.85, 0.35, 0.07, 0.04])
    gb_box = plt.axes([0.85, 0.30, 0.07, 0.04])
    lb_gm_box = plt.axes([0.85, 0.25, 0.07, 0.04])
    gb_gm_box = plt.axes([0.85, 0.20, 0.07, 0.04])
    gc_box = plt.axes([0.85, 0.15, 0.07, 0.04])
    
    # Define widgets
    SI_tb = TextBox(SI_box, 'SI', textalignment='center')
    tdeff_tb = TextBox(tdeff_box, 'TDeff', textalignment='center')
    mode_radio = RadioButtons(mode_box, modes, active=0)
    ssb_tb = TextBox(ssb_box, 'SSB', textalignment='center')
    lb_tb = TextBox(lb_box, 'LB', textalignment='center')
    gb_tb = TextBox(gb_box, 'GB', textalignment='center')
    lb_gm_tb = TextBox(lb_box, 'LB_GM', textalignment='center')
    gb_gm_tb = TextBox(gb_box, 'GB_GM', textalignment='center')
    gc_tb = TextBox(gc_box, 'GC', textalignment='center')
    
    
    # Functions connected to widgets
    def update():
        # Redraw the plot
        fid = np.copy(fid0)      # Starting value
        # Process data according to the new values
        data = processing.fp(fid, wf=procs['wf'], zf=procs['zf'], tdeff=procs['tdeff'])
        apodf = get_apod(fid0.shape[-1], procs)     # Get window functions
        # Recalculate the scales
        fq_scale = processing.make_scale(data.shape[-1], acqus['dw'], rev=True)
        ppm_scale = misc.freq2ppm(fq_scale, acqus['SFO1'], acqus['o1p'])
        # Update the plot
        tx['SI'].set_text('{:.0f}'.format(data.shape[-1]))
        line.set_data(ppm_scale, data.real)     # Spectrum
        fidp.set_ydata((fid0 * apodf).real / max(fid0.real))    # FID (blue)
        apodp.set_ydata(apodf)                  # WF (red)
        # Update the limits
        misc.set_ylim(ax, data.real)
        misc.set_ylim(axf, (apodf, -apodf))
        plt.draw()
    
    def update_SI(v):
        nonlocal procs
        try:
            SI = eval(v)
            procs['zf'] = SI
        except:
            pass
        update()
    
    def update_tdeff(v):
        nonlocal procs
        try:
            val = eval(v)
            procs['tdeff'] = int(val)
        except:
            pass
        tx['tdeff'].set_text('{:.0f}'.format(procs['tdeff']))
        update()
    
    def update_mode(label):
        nonlocal procs
        for key, value in tx.items():
            value.set_color('k')
        if label == 'No':
            procs['wf']['mode'] = None
        else:
            procs['wf']['mode'] = label
        for key in act_keys[label]:
            tx[key].set_color('tab:blue')
        update()
    
    def update_lb(v):
        nonlocal procs
        try:
            lb = eval(v)
            procs['wf']['lb'] = lb
        except:
            pass
        tx['lb'].set_text('{:.0f}'.format(procs['wf']['lb']))
        update()
            
    def update_gb(v):
        nonlocal procs
        try:
            gb = eval(v)
            procs['wf']['gb'] = gb
        except:
            pass
        tx['gb'].set_text('{:.2f}'.format(procs['wf']['gb']))
        update()

    def update_lb_gm(v):
        nonlocal procs
        try:
            lb = eval(v)
            procs['wf']['lb_gm'] = lb
        except:
            pass
        tx['lb_gm'].set_text('{:.0f}'.format(procs['wf']['lb_gm']))
        update()
            
    def update_gb_gm(v):
        nonlocal procs
        try:
            gb = eval(v)
            procs['wf']['gb_gm'] = gb
        except:
            pass
        tx['gb_gm'].set_text('{:.2f}'.format(procs['wf']['gb_gm']))
        update()
    
    def update_gc(v):
        nonlocal procs
        try:
            gc = eval(v)
            procs['wf']['gc'] = gc
        except:
            pass
        tx['gc'].set_text('{:.2f}'.format(procs['wf']['gc']))
        update()
    
    def update_ssb(v):
        nonlocal procs
        try:
            ssb = eval(v)
            procs['wf']['ssb'] = ssb
        except:
            pass
        tx['ssb'].set_text('{:.0f}'.format(procs['wf']['ssb']))
        update()
    
    
    
    # Draw the figure panel
    
    ax.axhline(0, c='k', lw=0.4)    # baseline
    axf.axhline(0, c='k', lw=0.4)   # baseline
    line, = ax.plot(ppm_scale, data.real, c='tab:blue')         # Spectrum
    fidp, = axf.plot(np.arange(fid.shape[-1]), fid0.real/max(fid0.real), c='tab:blue', lw=0.6)  # FID
    fidp.set_label('Normalized FID')
    apodp, = axf.plot(np.arange(fid.shape[-1]), apodf, c='tab:red', lw=1.0)     # Window function
    apodp.set_label('Window function')
    
    axf.legend()
    
    def calcy(box):
        """ y_coordinate + (box_height / 2) """
        pos = box.get_position().bounds
        y = round(pos[1] + pos[3]/2, 2)
        return y
    
    # Write text alongside figures
    tx['SI'] = plt.text(0.93, calcy(SI_box), '{:.0f}'.format(data.shape[-1]), ha='left', va='center', transform=fig.transFigure)
    tx['tdeff'] = plt.text(0.93, calcy(tdeff_box), '{:.0f}'.format(procs['tdeff']), ha='left', va='center', transform=fig.transFigure)
    tx['ssb'] = plt.text(0.93, calcy(ssb_box), '{:.0f}'.format(procs['wf']['ssb']), ha='left', va='center', transform=fig.transFigure)
    tx['lb'] = plt.text(0.93, calcy(lb_box), '{:.0f}'.format(procs['wf']['lb']), ha='left', va='center', transform=fig.transFigure)
    tx['gb'] = plt.text(0.93, calcy(gb_box), '{:.2f}'.format(procs['wf']['gb']), ha='left', va='center', transform=fig.transFigure)
    tx['lb_gm'] = plt.text(0.93, calcy(lb_gm_box), '{:.0f}'.format(procs['wf']['lb_gm']), ha='left', va='center', transform=fig.transFigure)
    tx['gb_gm'] = plt.text(0.93, calcy(gb_gm_box), '{:.2f}'.format(procs['wf']['gb_gm']), ha='left', va='center', transform=fig.transFigure)
    tx['gc'] = plt.text(0.93, calcy(gc_box), '{:.2f}'.format(procs['wf']['gc']), ha='left', va='center', transform=fig.transFigure)
    
    # Customize appearance
    ax.set_xlabel(r'$\delta\,$'+misc.nuc_format(acqus['nuc'])+' /ppm')
    ax.set_ylabel('Intensity /a.u.')
    misc.set_ylim(ax, data.real)
    misc.set_ylim(axf, (-1,1))
    misc.mathformat(ax)
    misc.mathformat(axf)
    misc.pretty_scale(ax, (max(ppm_scale), min(ppm_scale)))
    misc.pretty_scale(axf, (0, fid.shape[-1]))
    misc.set_fontsizes(ax, 14)
    misc.set_fontsizes(axf, 14)
    
    # Connect function to widgets
    SI_tb.on_submit(update_SI)
    mode_radio.on_clicked(update_mode)
    tdeff_tb.on_submit(update_tdeff)
    ssb_tb.on_submit(update_ssb)
    lb_tb.on_submit(update_lb)
    gb_tb.on_submit(update_gb)
    lb_gm_tb.on_submit(update_lb_gm)
    gb_gm_tb.on_submit(update_gb_gm)
    gc_tb.on_submit(update_gc)
    
    plt.show()
    
    # Calculate final spectrum, return it
    datap = processing.fp(fid0, wf=procs['wf'], zf=procs['zf'], tdeff=procs['tdeff'])
    
    return datap, procs
    
    
    
    
    
    
def inv_fp(data, wf=None, size=None, fcor=0.5):
    """
    Performs the full inverse processing of a 1D NMR spectrum (data).
    -------
    Parameters:
    - data: 1darray
        Spectrum
    - wf: dict
        {'mode': function to be used, 'parameters': different from each function}
    - size: int
        initial size of the FID
    - fcor: float
        weighting factor for the FID first point
    -------
    Returns:
    - pdata: 1darray
        FID
    """
    # IFT
    data = processing.ift(pdata, fcor=fcor)
    # Reverse zero-filling
    if size is not None:
        pdata = processing.td_eff(pdata, size)
    # Reverse window function
    if wf is not None:
        if wf['mode'] == None:
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
    
    
    
def xfb(data, wf=[None, None], zf=[None, None], fcor=[0.5,0.5], tdeff=[0,0], u=True, FnMODE='States-TPPI'):
    """
    Performs the full processing of a 2D NMR FID (data). 
    The returned values depend on u: it is True, returns a sequence of 2darrays depending on FnMODE, otherwise just the complex/hypercomplex data after FT in both dimensions
    --------
    Parameters:
    - data: 2darray
        Input data
    - wf: sequence of dict
        (F1, F2); {'mode': function to be used, 'parameters': different from each function}
    - zf: sequence of int
        final size of spectrum, (F1, F2)
    - fcor: sequence of float 
        weighting factor for the FID first point, (F1, F2)
    - tdeff: sequence of int
        number of points of the FID to be used for the processing, (F1, F2)
    - u: bool
        choose if to unpack the hypercomplex spectrum into separate arrays or not
    - FnMODE: str
        Acquisition mode in F1
    ------
    Returns:
    - datap: 2darray or tuple of 2darray
        Processed data or tuple of 2darray
    """
    
    data = processing.td_eff(data, tdeff)
    
    # Processing the direct dimension
    # Window function
    if wf[1] is not None:
        if wf[1]['mode'] == 'qsin':
            data = processing.qsin(data, ssb=wf[1]['ssb'])
        if wf[1]['mode'] == 'sin':
            data = processing.sin(data, ssb=wf[1]['ssb'])
        if wf[1]['mode'] == 'em':
            data = processing.em(data, lb=wf[1]['lb'], sw=wf[1]['sw'])
        if wf[1]['mode'] == 'gm':
            data = processing.gm(data, lb=wf[1]['lb'], gb=wf[1]['gb'], sw=wf[1]['sw'])
    # Zero-filling
    if zf[1] is not None:
        data = processing.zf(data, zf[1])
    # FT
    data = processing.ft(data, fcor=fcor[1])
    
    # Processing the indirect dimension
    # If FnMODE is 'QF', do normal transpose instead of hyper
    if FnMODE == 'QF':
        data = data.T
    else:
        data = processing.tp_hyper(data)
    
    # Window function
    if wf[0] is not None:
        if wf[0]['mode'] == 'qsin':
            data = processing.qsin(data, ssb=wf[0]['ssb'])
        if wf[0]['mode'] == 'sin':
            data = processing.sin(data, ssb=wf[0]['ssb'])
        if wf[0]['mode'] == 'em':
            data = processing.em(data, lb=wf[0]['lb'], sw=wf[0]['sw'])
        if wf[0]['mode'] == 'gm':
            data = processing.gm(data, lb=wf[0]['lb'], gb=wf[0]['gb'], sw=wf[0]['sw'])
    # Zero-filling
    if zf[0] is not None:
        data = processing.zf(data, zf[0])
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
        if FnMODE == 'QF':
            return data.real, data.imag
        else:
            return processing.unpack_2D(data)           # rr, ir, ri, ii
    else:
        return data
    
    
    
    
def interactive_xfb(fid0, acqus, procs, lvl0=0.1, show_cnt=True):
    """
    Perform the processing of a 2D NMR spectrum interactively. The GUI offers the opportunity to test different window functions, as well as different tdeff values and final sizes.
    The active parameters appear as blue text.
    When changing the parameters, give it some time to compute. The figure panel is quite heavy.
    -------
    Parameters:
    - fid0: 2darray
        FID to process
    - acqus: dict
        Dictionary of acquisition parameters
    - procs: dict
        Dictionary of processing parameters
    - lvl0: float
        Starting level of the contours
    - show_cnt: bool
        Choose if to display data using contours (True) or heatmap (False)
    -------
    Returns:
    - pdata: 2darray
        Processed spectrum
    - procs: dict
        Updated dictionary of processing parameters
    """
    
    def get_apod(size, procs):
        """ Calculate the window function on the basis of 'procs' """
        Y = np.ones(size, dtype='complex64')    # array of ones
        # Process the array of ones and then revert FT to get everything but the processing
        apodf = processing.ift(processing.fp(Y, wf=procs['wf'], zf=procs['zf'], tdeff=procs['tdeff']))
        apodf = apodf.real
        # Adjust the dimension to size
        if apodf.shape[-1] < size:  # if shorter than size, zero-fill
            apodf = processing.zf(apodf, size)
        if apodf.shape[-1] > size:  # if longet than size, trim
            apodf = processing.td_eff(apodf, size)
        return apodf
    
    CNT = bool(show_cnt)
    
    # Copy initial FID to prevent overwriting and create new variables
    fid = np.copy(fid0)
    fid0 = np.copy(fid)
    A1 = np.ones_like(fid0[:,0])    # WF F1
    A2 = np.ones_like(fid0[0,:])    # WF F2
    
    # Split acqus and procs from 2D version in two 1D-like dictionaries
    acqu1s, acqu2s = misc.split_acqus_2D(acqus)
    proc1s, proc2s = misc.split_procs_2D(procs)

    # Calculate starting values, get only rr
    data = processing.xfb(fid, wf=procs['wf'], zf=procs['zf'], tdeff=procs['tdeff'], FnMODE=acqus['FnMODE'])[0]

    # Get WFs
    apodf1 = get_apod(fid0.shape[0], proc1s)
    apodf2 = get_apod(fid0.shape[1], proc2s)

    # Calculate the ppm scales
    #   F1
    fq1_scale = processing.make_scale(data.shape[0], acqu1s['dw'], rev=True)
    ppm_f1 = misc.freq2ppm(fq1_scale, acqu1s['SFO1'], acqu1s['o1p'])
    #   F2
    fq2_scale = processing.make_scale(data.shape[1], acqu2s['dw'], rev=True)
    ppm_f2 = misc.freq2ppm(fq2_scale, acqu2s['SFO1'], acqu2s['o1p'])

    # Define useful things 
    modes = ['No', 'em', 'sin', 'qsin', 'gm', 'gmb']   # entries for the radiobuttons
    act_keys = {    # Active Parameters:
            'No': [],
            'em': ['lb'],
            'sin': ['ssb'],
            'qsin': ['ssb'],
            'gm': ['lb', 'gb', 'gc'],
            'gmb': ['lb', 'gb'],
            }
    tx = [{},{}] # Dictionary of the texts. [Left column i.e. F2, Right column i.e. F1]

    # Draw boxes for widgets
    SI_box = [  # Sizes
            plt.axes([0.76, 0.90, 0.05, 0.04]),
            plt.axes([0.89, 0.90, 0.05, 0.04])]
    tdeff_box = [   # TDEFF
            plt.axes([0.76, 0.85, 0.05, 0.04]),
            plt.axes([0.89, 0.85, 0.05, 0.04])]
    mode_box = [    # WF mode
            plt.axes([0.76, 0.52, 0.09, 0.30]),
            plt.axes([0.89, 0.52, 0.09, 0.30])]
    ssb_box = [     # SSB
            plt.axes([0.76, 0.45, 0.05, 0.04]),
            plt.axes([0.89, 0.45, 0.05, 0.04])]
    lb_box = [      # LB
            plt.axes([0.76, 0.40, 0.05, 0.04]),
            plt.axes([0.89, 0.40, 0.05, 0.04])]
    gb_box = [      # GB
            plt.axes([0.76, 0.35, 0.05, 0.04]),
            plt.axes([0.89, 0.35, 0.05, 0.04])]
    gc_box = [      # GC
            plt.axes([0.76, 0.3, 0.05, 0.04]),
            plt.axes([0.89, 0.3, 0.05, 0.04])]

    # Define widgets
    SI_tb = [       # SI
            TextBox(SI_box[0], 'SI', textalignment='center'),
            TextBox(SI_box[1], '', textalignment='center')]
    tdeff_tb = [    # TDEFF
            TextBox(tdeff_box[0], 'TDeff', textalignment='center'),
            TextBox(tdeff_box[1], '', textalignment='center')]
    mode_radio = [  # WF mode
            RadioButtons(mode_box[0], modes, active=0),
            RadioButtons(mode_box[1], modes, active=0)]
    ssb_tb = [      # SSB
            TextBox(ssb_box[0], 'SSB', textalignment='center'),
            TextBox(ssb_box[1], '', textalignment='center')]
    lb_tb = [       # LB
            TextBox(lb_box[0], 'LB', textalignment='center'),
            TextBox(lb_box[1], '', textalignment='center')]
    gb_tb = [       # GB
            TextBox(gb_box[0], 'GB', textalignment='center'),
            TextBox(gb_box[1], '', textalignment='center')]
    gc_tb = [       # GC
            TextBox(gc_box[0], 'GC', textalignment='center'),
            TextBox(gc_box[1], '', textalignment='center')]

    # Functions connected to widgets
    def update():
        # Redraws the plot
        nonlocal cnt
        proc1s, proc2s = misc.split_procs_2D(procs)     # split procs for WFs
        fid = np.copy(fid0)      # Starting value
        fid02 = np.copy(fid0[0,:])      # F2 FID
        fid01 = np.copy(fid0[:,0])      # F1 FID
        fidp = np.copy(fid0)            # Whole FID for heatmap

        # Calculate the processed FID before FT, as processing.xfb does but without FTs
        fidp = processing.ift(processing.fp(fidp, wf=proc2s['wf'], zf=fid0.shape[-1], tdeff=procs['tdeff']))
        if acqus['FnMODE'] == 'QF':
            fidp = fidp.T
        else:
            fidp = processing.tp_hyper(fidp)
        fidp = processing.ift(processing.fp(fidp, wf=proc1s['wf'], zf=fid0.shape[0], tdeff=procs['tdeff']))
        if acqus['FnMODE'] == 'QF':
            fidp = fidp.T
        else:
            fidp = processing.tp_hyper(fidp)

        # Process data according to the new values
        data = processing.xfb(fid, wf=procs['wf'], zf=procs['zf'], tdeff=procs['tdeff'], FnMODE=acqus['FnMODE'])[0]

        # Get WFs
        apodf1 = get_apod(fid01.shape[-1], proc1s)
        apodf2 = get_apod(fid02.shape[-1], proc2s)

        # Recalculate the scales
        fq1_scale = processing.make_scale(data.shape[0], acqu1s['dw'], rev=True)
        ppm_f1 = misc.freq2ppm(fq1_scale, acqu1s['SFO1'], acqu1s['o1p'])
        fq2_scale = processing.make_scale(data.shape[1], acqu2s['dw'], rev=True)
        ppm_f2 = misc.freq2ppm(fq2_scale, acqu2s['SFO1'], acqu2s['o1p'])

        # Update SI text with the actual size of data
        tx[0]['SI'].set_text('{:.0f}'.format(data.shape[-1]))
        tx[1]['SI'].set_text('{:.0f}'.format(data.shape[0]))

        # Update the plot
        #   Spectrum
        if CNT:
            cnt, _ = figures.redraw_contours(ax, ppm_f2, ppm_f1, data, lvl0, cnt, Neg=False, Ncnt=None, lw=0.5, cmap=[None, None])
        else:
            cnt.set_data(data)

        #   F2 FID
        fidp2.set_ydata((fid02 * apodf2).real / np.max(fid02.real))    # FID (blue)
        apodp2.set_ydata(apodf2)                  # WF (red)
        #   F1 FID
        fidp1.set_ydata((fid01 * apodf1).real / np.max(fid01.real))    # FID (blue)
        apodp1.set_ydata(apodf1)                  # WF (red)

        #   Whole FID heatmap
        hm.set_data(fidp.real)

        # Update the limits and make figure pretty
        ax.set_xlabel(r'$\delta\,$'+ misc.nuc_format(acqu2s['nuc'])+' /ppm')
        ax.set_ylabel(r'$\delta\,$'+ misc.nuc_format(acqu1s['nuc'])+' /ppm')
        misc.set_ylim(axf2, (apodf2, -apodf2))
        misc.set_ylim(axf1, (apodf1, -apodf1))
        misc.set_fontsizes(ax, 14)

        # Redraw
        fig.canvas.draw()

    # --------------------------------------------------
    # update_SI = [update_SI_f2, update_SI_f1]    
    def update_SI_f2(v):
        nonlocal procs
        try:
            SI = eval(v)
            procs['zf'][1] = SI
        except:
            pass
        update()
    def update_SI_f1(v):
        nonlocal procs
        try:
            SI = eval(v)
            procs['zf'][0] = SI
        except:
            pass
        update()
    update_SI = [update_SI_f2, update_SI_f1]

    # --------------------------------------------------
    # update_tdeff = [update_tdeff_f2, update_tdeff_f1]
    def update_tdeff_f2(v):
        nonlocal procs
        try:
            val = eval(v)
            procs['tdeff'][1] = int(val)
        except:
            pass
        tx[0]['tdeff'].set_text('{:.0f}'.format(procs['tdeff'][1]))
        update()
    def update_tdeff_f1(v):
        nonlocal procs
        try:
            val = eval(v)
            procs['tdeff'][0] = int(val)
        except:
            pass
        tx[1]['tdeff'].set_text('{:.0f}'.format(procs['tdeff'][0]))
        update()
    update_tdeff = [update_tdeff_f2, update_tdeff_f1]

    # --------------------------------------------------
    # update_mode = [update_mode_f2, update_mode_f1]
    def update_mode_f2(label):
        nonlocal procs
        for key, value in tx[0].items():
            value.set_color('k')
        if label == 'No':
            procs['wf'][1]['mode'] = None
        else:
            procs['wf'][1]['mode'] = label
        for key in act_keys[label]:
            tx[0][key].set_color('tab:blue')
        update()
    def update_mode_f1(label):
        nonlocal procs
        for key, value in tx[1].items():
            value.set_color('k')
        if label == 'No':
            procs['wf'][0]['mode'] = None
        else:
            procs['wf'][0]['mode'] = label
        for key in act_keys[label]:
            tx[1][key].set_color('tab:blue')
        update()
    update_mode = [update_mode_f2, update_mode_f1]

    # --------------------------------------------------
    # update_ssb = [update_ssb_f2, update_ssb_f1]
    def update_ssb_f2(v):
        nonlocal procs
        try:
            ssb = eval(v)
            procs['wf'][1]['ssb'] = ssb
        except:
            pass
        tx[0]['ssb'].set_text('{:.0f}'.format(procs['wf'][1]['ssb']))
        update()
    def update_ssb_f1(v):
        nonlocal procs
        try:
            ssb = eval(v)
            procs['wf'][0]['ssb'] = ssb
        except:
            pass
        tx[1]['ssb'].set_text('{:.0f}'.format(procs['wf'][0]['ssb']))
        update()
    update_ssb = [update_ssb_f2, update_ssb_f1]

    # --------------------------------------------------
    # update_lb = [update_lb_f2, update_lb_f1]
    def update_lb_f2(v):
        nonlocal procs
        try:
            lb = eval(v)
            procs['wf'][1]['lb'] = lb
        except:
            pass
        tx[0]['lb'].set_text('{:.0f}'.format(procs['wf'][1]['lb']))
        update()
    def update_lb_f1(v):
        nonlocal procs
        try:
            lb = eval(v)
            procs['wf'][0]['lb'] = lb
        except:
            pass
        tx[1]['lb'].set_text('{:.0f}'.format(procs['wf'][0]['lb']))
        update()
    update_lb = [update_lb_f2, update_lb_f1]
            
    # --------------------------------------------------
    # update_gb = [update_gb_f2, update_gb_f1]
    def update_gb_f2(v):
        nonlocal procs
        try:
            gb = eval(v)
            procs['wf'][1]['gb'] = gb
        except:
            pass
        tx[0]['gb'].set_text('{:.2f}'.format(procs['wf'][1]['gb']))
        update()
    def update_gb_f1(v):
        nonlocal procs
        try:
            gb = eval(v)
            procs['wf'][0]['gb'] = gb
        except:
            pass
        tx[1]['gb'].set_text('{:.2f}'.format(procs['wf'][0]['gb']))
        update()
    update_gb = [update_gb_f2, update_gb_f1]

    # --------------------------------------------------
    # update_gc = [update_gc_f2, update_gc_f1]
    def update_gc_f2(v):
        nonlocal procs
        try:
            gc = eval(v)
            procs['wf'][1]['gc'] = gc
        except:
            pass
        tx[0]['gc'].set_text('{:.2f}'.format(procs['wf'][1]['gc']))
        update()
    def update_gc_f1(v):
        nonlocal procs
        try:
            gc = eval(v)
            procs['wf'][0]['gc'] = gc
        except:
            pass
        tx[1]['gc'].set_text('{:.2f}'.format(procs['wf'][0]['gc']))
        update()
    update_gc = [update_gc_f2, update_gc_f1]

    # ------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------


    # Draw the figure panel
    fig = plt.figure('Interactive Processing')
    fig.set_size_inches(15,9)
    plt.subplots_adjust(left=0.1, bottom=0.05, right=0.725, top=0.95, hspace=0.75, wspace=0.25)
    ax = fig.add_subplot(4,3,(1,9))     # spectrum
    axf2 = fig.add_subplot(4,3,10)        # fid F2
    axf1 = fig.add_subplot(4,3,11)        # fid F1
    axhm = fig.add_subplot(4,3,12)        # fid total


    # Spectrum plot
    ax.set_title('Spectrum')
    if CNT:
        cnt = figures.ax2D(ax, ppm_f2, ppm_f1, data, lvl=lvl0, X_label='', Y_label='', fontsize=14)
    else:
        cnt, axcbar = figures.ax_heatmap(ax, data, zlim='auto', z_sym=True, cmap=None, xscale=ppm_f2, yscale=ppm_f1, rev=(True,True), n_xticks=10, n_yticks=10, n_zticks=10, fontsize=14)
        axcbar.tick_params(axis='y', labelright=False)  # Turn off the ticks of the colorbar otherwise it is ugly as shit

    # FID F2 plot
    axf2.set_title('F2 FID')
    axf2.axhline(0, c='k', lw=0.4)   # baseline
    fidp2, = axf2.plot(np.arange(fid.shape[-1]), fid0[0].real/max(fid0[0].real), c='tab:blue', lw=0.6)  # FID
    fidp2.set_label('Normalized FID')
    apodp2, = axf2.plot(np.arange(fid.shape[-1]), apodf2, c='tab:red', lw=1.0)     # Window function
    apodp2.set_label('Window function')
    axf2.legend()
    
    # FID F1 plot
    axf1.set_title('F1 FID')
    axf1.axhline(0, c='k', lw=0.4)   # baseline
    fidp1, = axf1.plot(np.arange(fid.shape[0]), fid0[:,0].real/max(fid0[:,0].real), c='tab:blue', lw=0.6)  # FID
    fidp1.set_label('Normalized FID')
    apodp1, = axf1.plot(np.arange(fid.shape[0]), apodf1, c='tab:red', lw=1.0)     # Window function
    apodp1.set_label('Window function')
    axf1.legend()

    # Whole FID heatmap plot
    axhm.set_title('FID')
    hm, _ = figures.ax_heatmap(axhm, fid0.real, zlim='auto', z_sym=True, cmap=None, rev=(False, True), n_xticks=5, n_yticks=5, n_zticks=5, fontsize=14)

    # Write text alongside figures
    #   Define a function to calculate the y coordinate given the box coordinates
    def calcy(box):
        """ y_coordinate + (box_height / 2) """
        pos = box.get_position().bounds
        y = round(pos[1] + pos[3]/2, 2)
        return y

    #   Write the text
    #       Header
    plt.text(0.80, 0.97, 'F2', rotation=0, ha='center', va='center', transform=fig.transFigure, fontsize=14)
    plt.text(0.93, 0.97, 'F1', rotation=0, ha='center', va='center', transform=fig.transFigure, fontsize=14)
    #       Left column, F2
    tx[0]['SI'] = plt.text(0.82, calcy(SI_box[0]), '{:.0f}'.format(data.shape[-1]), ha='left', va='center', transform=fig.transFigure)
    tx[0]['tdeff'] = plt.text(0.82, calcy(tdeff_box[0]), '{:.0f}'.format(proc2s['tdeff']), ha='left', va='center', transform=fig.transFigure)
    tx[0]['ssb'] = plt.text(0.82, calcy(ssb_box[0]), '{:.0f}'.format(proc2s['wf']['ssb']), ha='left', va='center', transform=fig.transFigure)
    tx[0]['lb'] = plt.text(0.82, calcy(lb_box[0]), '{:.0f}'.format(proc2s['wf']['lb']), ha='left', va='center', transform=fig.transFigure)
    tx[0]['gb'] = plt.text(0.82, calcy(gb_box[0]), '{:.2f}'.format(proc2s['wf']['gb']), ha='left', va='center', transform=fig.transFigure)
    tx[0]['gc'] = plt.text(0.82, calcy(gc_box[0]), '{:.2f}'.format(proc2s['wf']['gc']), ha='left', va='center', transform=fig.transFigure)
    #       Right column, F1
    tx[1]['SI'] = plt.text(0.95, calcy(SI_box[0]), '{:.0f}'.format(data.shape[0]), ha='left', va='center', transform=fig.transFigure)
    tx[1]['tdeff'] = plt.text(0.95, calcy(tdeff_box[0]), '{:.0f}'.format(proc1s['tdeff']), ha='left', va='center', transform=fig.transFigure)
    tx[1]['ssb'] = plt.text(0.95, calcy(ssb_box[0]), '{:.0f}'.format(proc1s['wf']['ssb']), ha='left', va='center', transform=fig.transFigure)
    tx[1]['lb'] = plt.text(0.95, calcy(lb_box[0]), '{:.0f}'.format(proc1s['wf']['lb']), ha='left', va='center', transform=fig.transFigure)
    tx[1]['gb'] = plt.text(0.95, calcy(gb_box[0]), '{:.2f}'.format(proc1s['wf']['gb']), ha='left', va='center', transform=fig.transFigure)
    tx[1]['gc'] = plt.text(0.95, calcy(gc_box[0]), '{:.2f}'.format(proc1s['wf']['gc']), ha='left', va='center', transform=fig.transFigure)

    # Add other elements to the figure
    #   Vertical line between F1 and F2
    plt.text(0.87, 0.63, '$-$'*55, rotation=90, ha='left', va='center', transform=fig.transFigure, fontsize=10)
    #   Horizontal line below 'F1       F2' header
    plt.text(0.87, 0.95, '$-$'*32, rotation=0, ha='center', va='center', transform=fig.transFigure, fontsize=10)
    #   Horizontal line between the 'Spectrum' plot and the three at the bottom
    plt.text(0.40, 0.235, '$-$'*90, rotation=0, ha='center', va='center', transform=fig.transFigure, fontsize=10)

    # Customize appearance
    #   Spectrum axis labels
    ax.set_xlabel(r'$\delta\,$' + misc.nuc_format(acqu2s['nuc']) + ' /ppm')
    ax.set_ylabel(r'$\delta\,$' + misc.nuc_format(acqu1s['nuc']) + ' /ppm')
    #   Spectrum axes scales
    misc.pretty_scale(ax, (max(ppm_f2), min(ppm_f2)), axis='x')
    misc.pretty_scale(ax, (max(ppm_f1), min(ppm_f1)), axis='y')

    #   FID F2 axes
    #       y
    misc.set_ylim(axf2, (-1,1))
    misc.mathformat(axf2)
    #       x
    misc.pretty_scale(axf2, (0, fid.shape[1]), n_major_ticks=4)
    #   FID F1 y-axis
    #       y
    misc.set_ylim(axf1, (-1,1))
    misc.mathformat(axf1)
    #       x
    misc.pretty_scale(axf1, (0, fid.shape[0]), n_major_ticks=4)

    #   Font sizes
    misc.set_fontsizes(ax, 14)
    misc.set_fontsizes(axf2, 14)
    misc.set_fontsizes(axf1, 14)
    misc.set_fontsizes(axhm, 14)

    # Connect function to widgets
    for i in range(2):
        SI_tb[i].on_submit(update_SI[i])
        mode_radio[i].on_clicked(update_mode[i])
        tdeff_tb[i].on_submit(update_tdeff[i])
        ssb_tb[i].on_submit(update_ssb[i])
        lb_tb[i].on_submit(update_lb[i])
        gb_tb[i].on_submit(update_gb[i])
        gc_tb[i].on_submit(update_gc[i])

    plt.show()
    
    # Calculate final spectrum. Do not unpack the hyperser
    datap = processing.xfb(fid, wf=procs['wf'], zf=procs['zf'], tdeff=procs['tdeff'], FnMODE=acqus['FnMODE'], u=False)

    # Return hyperser and updated procs dictionary
    return datap, procs


def inv_xfb(data, wf=[None, None], size=(None, None), fcor=[0.5,0.5], FnMODE='States-TPPI'):
    """
    Reverts the full processing of a 2D NMR FID (data).
    -------
    Parameters:
    - data: 2darray
        Input data, hypercomplex
    - wf: list of dict
        list of two entries [F1, F2]. Each entry is a dictionary of window functions
    - size: list of int
        Initial size of FID
    - fcor: list of float
        first fid point weighting factor [F1, F2]
    - FnMODE: str
        Acquisition mode in F1
    --------
    Returns:
    - data: 2darray
        Processed data
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
        if wf[0]['mode'] == None:
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
        if wf[1]['mode'] == None:
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
    Computes the frequency scale of the NMR spectrum, given the # of points and the employed dwell time (the REAL one, not the TopSpin one!). 
    "rev"=True is required for the correct frequency arrangement in the NMR sense.
    --------
    Parameters:
    - size: int
        Number of points of the frequency scale
    - dw : float
        Time spacing in the time dimension
    - rev: bool
        Reverses the scale
    -------
    Returns:
    - fqscale: 1darray
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
    fig.set_size_inches(12,8)
    plt.subplots_adjust(left=0.15, bottom=0.15)
    ax = fig.add_subplot(1,1,1)
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
        path = Path(verts, closed=True)
        selected = []
        for i in yscale:
            for j in xscale:
                if path.contains_point((float(j),float(i))):
                    selected.append([j,i])

        # Create convex hull around the raw lasso
        CH = ConvexHull(np.array(selected))
        # Create delimiting wall
        xhull = list(CH.points[CH.vertices,0])
        xhull.append(CH.points[CH.vertices[0],0])
        xhull = np.array(xhull)
        yhull = list(CH.points[CH.vertices,1])
        yhull.append(CH.points[CH.vertices[0],1])
        yhull = np.array(yhull)
    
        # Update the plot
        hull.set(visible=True)
        hull.set_data(xhull, yhull)

        # Fine selection of points on the basis of the hull
        path = Path(CH.points[CH.vertices], closed=True)
        selected = []
        for i in yscale:
            for j in xscale:
                if path.contains_point((float(j),float(i))):
                    selected.append([j,i])
        # Store the selected points in a non-local variable
        sgn_reg = np.array(selected)

        plt.draw()

    def save(event):
        # Function connected to the button

        nonlocal thesignal
        thesignal.append(sgn_reg)       # Save the region 

        CH = ConvexHull(sgn_reg)        # Compute convex hull
        # Create the walls
        xhull = list(CH.points[CH.vertices,0])
        xhull.append(CH.points[CH.vertices[0],0])
        xhull = np.array(xhull)
        yhull = list(CH.points[CH.vertices,1])
        yhull.append(CH.points[CH.vertices[0],1])
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

    hull, = ax.plot(0,0, visible=False)             # Create variable for the lasso selection on screen

    # Set limits
    #ax.set_xlim(data.shape[1], 0)
    ax.set_ylim(data.shape[0], 0)

    # Widgets
    lasso = LassoSelector(ax, onselect)
    button.on_clicked(save)

    plt.show()
    plt.close()

    # Fill the masking matrix on the basis of the selected region
    #   If you selected something, set as '1' the highlighted points
    if len(thesignal) > 0:
        thesignal = np.concatenate(thesignal)
        for k in range(thesignal.shape[0]):
            mask[thesignal[k,1], thesignal[k,0]] = 1
    #   If you did not select anything, the masking matrix does not alter the spectrum
    else:
        mask = np.ones_like(data)
    return mask



# Phase correction
def interactive_phase_1D(ppmscale, S):
    """
    This function allow to adjust the phase of 1D spectra interactively. Use the mouse scroll to regulate the values.
    -------
    Parameters:
    - ppmscale: 1darray
        ppm scale of the spectrum. Used to regulate the pivot position
    - S:  1darray
        Spectrum to be phased. Must be complex!
    -------
    Returns:
    - phased_data: 1darray
        Phased spectrum
    """

    def phase(data, p0=0, p1=0, pivot=None):
        """ This is the actual phase function """
        if data.dtype != 'complex64':
            data = data.astype('complex64')
        size = data.shape[-1]
        # convert to radians
        p0 = p0 * np.pi / 180
        p1 = p1 * np.pi / 180
        # Define axis for pivot that goes from 0 to 1
        pvscale = np.arange(size) / size
        if pivot is None:
            pv = 0.5
        else:
            pv = misc.ppmfind(ppmscale, pivot)[0]/size 
        apod = np.exp(1j * (p0 + p1 * (pvscale - pv))).astype(data.dtype)
        return apod * data

    # Make the figure
    fig = plt.figure('Phase Correction')
    fig.set_size_inches(15,8)
    plt.subplots_adjust(left = 0.075, bottom=0.10, right=0.8, top=0.9)    # Make room for the sliders
    ax = fig.add_subplot(1,1,1)

    # sensitivity
    sens = [5, 5, 0.1]

    # create empty variables for the phases and pivot to be returned
    p0_f = 0
    p1_f = 0
    pivot_f = round(np.mean(ppmscale), 2)

    # Boxes for widgets
    box_us = plt.axes([0.815, 0.825, 0.08, 0.075])      # increase sensitivity
    box_ds = plt.axes([0.905, 0.825, 0.08, 0.075])      # decrease sensitivity
    box_save = plt.axes([0.81, 0.15, 0.085, 0.04])      # save button
    box_reset = plt.axes([1-0.095, 0.15, 0.085, 0.04])  # reset button
    box_sande = plt.axes([0.81, 0.10, 0.18, 0.04])      # save and exit button
    box_radio = plt.axes([0.81, 0.55, 0.18, 0.25])      # radio buttons

    radiolabels = [     # labels for the radio buttons
            '0$^{th}$-order\nphase correction', 
            '1$^{st}$-order\nphase correction', 
            '1$^{st}$-order\npivot'
            ]

    
    # Make widgets
    #   Buttons
    up_button = Button(box_us, r'$\uparrow$', hovercolor='0.975')
    down_button = Button(box_ds, r'$\downarrow$', hovercolor='0.975')
    save_button = Button(box_save, 'SAVE', hovercolor='0.975')
    reset_button = Button(box_reset, 'RESET', hovercolor='0.975')
    saveandexit = Button(box_sande, 'SAVE AND EXIT', hovercolor='0.975')
    #   Radiobuttons
    radio = RadioButtons(box_radio, radiolabels)

    # Array 'status': 1 means active, 0 means inactive.
    stat = np.array([1, 0, 0])
    # values:     p0 p1 pivot
    P = np.array([0, 0, float(round(np.mean(ppmscale), 2))] )

    zoom_adj = True

    def statmod(label):
        # changes the 'stat' array according to the radiobutton
        nonlocal stat
        stat = np.zeros(3)
        for k, L in enumerate(radiolabels):
            if label == L:
                stat[k] = 1

    def roll_up(event):
        # Increase the active value of its 'sens'
        nonlocal P
        for k in range(3):
            if stat[k]:
                P[k] += sens[k]

    def roll_down(event):
        # Decrease the active value of its 'sens'
        nonlocal P
        for k in range(3):
            if stat[k]:
                P[k] -= sens[k]

    def sens_up(event):
        # Doubles the active 'sens'
        nonlocal sens
        for k in range(3):
            if stat[k]:
                sens[k] = sens[k]*2
    def sens_down(event):
        # Halves the active 'sens'
        nonlocal sens
        for k in range(3):
            if stat[k]:
                sens[k] = sens[k]/2

    def on_scroll(event): 
        # When you move the mouse scroll
        if event.button == 'up':
            roll_up(event)
        if event.button == 'down':
            roll_down(event)

        # Print the actual values
        phases_text.set_text('p0={:7.2f} | p1={:7.2f} | pv={:7.2f}'.format(*P))

        # Set the values
        p0 = P[0]
        p1 = P[1]
        pivot = P[2]

        data_inside = phase(S, p0, p1, pivot)   # phase the spectrum
        spectrum.set_ydata(data_inside.real)    # update plot
        pivot_bar.set_xdata((pivot,))              # update pivot bar
        # Interactively update the vertical limits
        if zoom_adj:
            T = max(data_inside.real)   
            B = min(data_inside.real)
            ax.set_ylim(B - 0.05*T, T + 0.05*T)
        # Update
        fig.canvas.draw()


    def update_lim(val):
        # Trim the figure according to the border sliders
        L = l.val
        R = r.val
        ax.set_xlim(L,R)

    def reset(event):
        # Reset the phase and pivot values to their starting point
        nonlocal P
        P = np.array([0, 0, round(np.mean([min(ppmscale),max(ppmscale)])) ] )
        on_scroll(event)

    def save(event):
        # Write the actual P values in the final variables
        nonlocal p0_f, p1_f, pivot_f
        p0_f = P[0]
        p1_f = P[1]
        pivot_f = P[2]

    def save_and_exit(event):
        # Function for the SAVE AND EXIT button:
        # Calls the 'save' function, then closes the figure
        save(event)
        plt.close()

    def zoom_onoff(event):
        nonlocal zoom_adj
        if event.key == 'z':
            zoom_adj = not(zoom_adj)


    # Set borders and scale
    ax.set_xlim(max(ppmscale), min(ppmscale))
    T = max(S.real)
    B = min(S.real)
    ax.set_ylim(B - 0.01*T, T + 0.01*T)
    # Make pretty scale
    misc.pretty_scale(ax, ax.get_xlim(), 'x')
    misc.pretty_scale(ax, ax.get_ylim(), 'y')
    misc.mathformat(ax)
    

    # Write axis label
    plt.text(0.5, 0.05, r'$\delta$ /ppm', ha='center', va='center', fontsize=20, transform=fig.transFigure)

    phases_text = plt.text(0.75, 0.015, 
            'p0={:7.2f} | p1={:7.2f} | pv={:7.2f}'.format(*P),
            ha='center', va='bottom', transform=fig.transFigure, fontsize=10)

    ax.axhline(0, c='k', lw=0.2)    # baseline guide

    spectrum, = ax.plot(ppmscale, S.real, c='b', lw=0.8)        # Plot the data
    pivot_bar = ax.axvline(P[2], c='r', lw=0.5)  # Plot the pivot bar

    # Link widgets to functions
    up_button.on_clicked(sens_up)
    down_button.on_clicked(sens_down)
    radio.on_clicked(statmod)
    reset_button.on_clicked(reset)
    save_button.on_clicked(save)
    saveandexit.on_clicked(save_and_exit)
    scroll = fig.canvas.mpl_connect('scroll_event', on_scroll)
    fig.canvas.mpl_connect('key_press_event', zoom_onoff)


    plt.show()

    phased_data = phase(S, p0=p0_f, p1=p1_f, pivot=pivot_f)
    final_values = p0_f, p1_f, pivot_f
    print('p0: {:.3f}, p1: {:.3f}, pv: {:.3f}\n'.format(*final_values))
    return phased_data, final_values



def interactive_phase_2D(ppm_f1, ppm_f2, S, hyper=True):
    """
    Interactively adjust the phases of a 2D spectrum
    S must be complex or hypercomplex, so BEFORE TO UNPACK
    -------
    Parameters:
    - ppm_f1: 1darray
        ppm scale of the indirect dimension
    - ppm_f2: 1darray
        ppm scale of the direct dimension
    - S: 2darray
        Data to be phase-adjusted
    - hyper: bool
        True if S is hypercomplex, False if S is just complex
    -------
    Returns:
    - S: 2darray
        Phased data
    - final_values_f1: tuple
        (p0_f1, p1_f1, pivot_f1)
    - final_values_f2: tuple
        (p0_f2, p1_f2, pivot_f2)
    """


    # Unpack the hyperser
    if hyper:
        S_rr, S_ri, S_ir, S_ii = processing.unpack_2D(S)
    else:
        S_rr, S_ii = S.real, S.imag

    zoom_adj = True

    def phase(data, p0=0, p1=0, pivot=None, dim='f2'):
        """This is the actual phase function """
        # as 1D
        if data.dtype != 'complex64':
            data = data.astype('complex64')
        size = data.shape[-1]
        # convert to radians
        p0 = p0 * np.pi / 180
        p1 = p1 * np.pi / 180
        # Define axis for pivot that goes from 0 to 1
        pvscale = np.arange(size) / size
        if pivot is None:
            pv = 0.5
        elif dim == 'f2':
            pv = misc.ppmfind(ppm_f2, pivot)[0]/size 
        elif dim == 'f1':
            pv = misc.ppmfind(ppm_f1, pivot)[0]/size 
        apod = np.exp(1j * (p0 + p1 * (pvscale - pv))).astype(data.dtype)
        return apod * data

    def maketraces(coord, S, ppm_f2, ppm_f1, hyper=True):
        # Extract the traces according to the 'coord' list
        if hyper:
            S_rr, S_ri, S_ir, S_ii = processing.unpack_2D(S)
        else:
            S_rr, S_ii = S.real, S.imag
        # Create empty lists for the traces
        f1, f2 = [], []
        npk = len(coord)
        for i in range(npk):
            y = misc.get_trace(S_rr, ppm_f2, ppm_f1, coord[i][0], column=True)
            f1.append(y)
            x = misc.get_trace(S_rr, ppm_f2, ppm_f1, coord[i][1], column=False)
            f2.append(x)
        return f1, f2


    # Get the traces on which to see the effects of phase adjustment
    coord = misc.select_traces(ppm_f1, ppm_f2, S_rr)
    npk = len(coord)

    # Make the figure
    fig = plt.figure('Phase Correction')
    fig.set_size_inches(15,8)
    plt.subplots_adjust(left = 0.125, bottom=0.125, right=0.8, top=0.9, wspace=0.10, hspace=0.20)    # Make room for the sliders

    # Get the traces
    f1, f2 = maketraces(coord, S, ppm_f2, ppm_f1, hyper)

    # Set initial values

    # Create boxes
    #   for sentitivity sliders
    box_us = plt.axes([0.815, 0.825, 0.08, 0.075])      # increase sensitivity
    box_ds = plt.axes([0.905, 0.825, 0.08, 0.075])      # decrease sensitivity
    #   for zoom sliders
    box_l_f2 = plt.axes([0.025, 0.15, 0.015, 0.30])
    box_r_f2 = plt.axes([0.060, 0.15, 0.015, 0.30])
    box_l_f1 = plt.axes([0.025, 0.60, 0.015, 0.30])
    box_r_f1 = plt.axes([0.060, 0.60, 0.015, 0.30])
    #   for buttons
    box_save = plt.axes([0.81, 0.15, 0.085, 0.04])      # save button
    box_reset = plt.axes([1-0.095, 0.15, 0.085, 0.04])  # reset button
    box_sande = plt.axes([0.81, 0.10, 0.18, 0.04])      # save and exit button
    box_radio = plt.axes([0.81, 0.55, 0.18, 0.25])      # radio buttons
    box_dimen = plt.axes([0.81, 0.35, 0.18, 0.18])      # radio buttons

    radiolabels = [     # labels for the radio buttons
            '0$^{th}$-order\nphase correction', 
            '1$^{st}$-order\nphase correction', 
            '1$^{st}$-order\npivot'
            ]

    # Make the sliders
    #   for sensitivity
    up_button = Button(box_us, r'$\uparrow$', hovercolor='0.975')
    down_button = Button(box_ds, r'$\downarrow$', hovercolor='0.975')
    #   for zoom
    l_f2 = Slider(ax=box_l_f2, label='Left', valmin=min(ppm_f2), valmax=max(ppm_f2), valinit=max(ppm_f2), orientation='vertical')
    r_f2 = Slider(ax=box_r_f2, label='Right', valmin=min(ppm_f2), valmax=max(ppm_f2), valinit=min(ppm_f2), orientation='vertical')
    l_f1 = Slider(ax=box_l_f1, label='Left', valmin=min(ppm_f1), valmax=max(ppm_f1), valinit=max(ppm_f1), orientation='vertical')
    r_f1 = Slider(ax=box_r_f1, label='Right', valmin=min(ppm_f1), valmax=max(ppm_f1), valinit=min(ppm_f1), orientation='vertical')
    # Make the buttons
    save_button = Button(box_save, 'SAVE', hovercolor='0.975')
    reset_button = Button(box_reset, 'RESET', hovercolor='0.975')
    saveandexit = Button(box_sande, 'SAVE AND EXIT', hovercolor='0.975')
    #   Radiobuttons
    radio = RadioButtons(box_radio, radiolabels)
    seldim = RadioButtons(box_dimen, ['F2', 'F1'])

    # Array "sensitivity":
    sens = [#p0 p1 pivot
            [5, 5, 0.1],    #F2
            [5, 5, 0.1]     #F1
            ]

    # "status" arrays: 
    stat = np.array([1, 0, 0])  # p0, p1, pivot
    statf = np.array([1, 0])    # f2, f1

    P = np.array([  # Values
        [0, 0, round(np.mean([min(ppm_f2),max(ppm_f2)]), 2) ],  #F2
        [0, 0, round(np.mean([min(ppm_f1),max(ppm_f1)]), 2) ]   #F1
        ])
    # For reset
    P0 = np.copy(P)

    # Initialize final variables with starting values
    p0_f2 = P0[0][0]
    p1_f2 = P0[0][1]
    pivot_f2 = P0[0][2]
    p0_f1 = P0[1][0]
    p1_f1 = P0[1][1]
    pivot_f1 = P0[1][2]

    
    # Functions connected to widgets
    def statmod(label):
        # changes the 'stat' array according to the radiobutton
        nonlocal stat
        stat = np.zeros(3)
        for k, L in enumerate(radiolabels):
            if label == L:
                stat[k] = 1

    def change_dim(label):
        nonlocal statf
        if label == 'F2':
            statf = np.array([1,0])
        if label == 'F1':
            statf = np.array([0,1])

    def roll_up(event):
        # Increase the active value of its 'sens'
        nonlocal P
        for i in range(2):
            for k in range(3):
                if statf[i] and stat[k]:
                    P[i,k] += sens[i][k]
                    # Manage out-of-border
                    if P[0][2] > max(ppm_f2):
                        P[0][2] = round(np.floor(max(ppm_f2)), 2)
                    if P[1][2] > max(ppm_f1):
                        P[1][2] = round(np.floor(max(ppm_f1)), 2)

    def roll_down(event):
        # Decrease the active value of its 'sens'
        nonlocal P
        for i in range(2):
            for k in range(3):
                if statf[i] and stat[k]:
                    P[i][k] -= sens[i][k]
                    # Manage out-of-border
                    if P[0][2] < min(ppm_f2):
                        P[0][2] = round(np.ceil(min(ppm_f2)), 2)
                    if P[1][2] < min(ppm_f1):
                        P[1][2] = round(np.ceil(min(ppm_f1)), 2)

    def sens_up(event):
        # Doubles the active 'sens'
        nonlocal sens
        for i in range(2):
            for k in range(3):
                if statf[i] and stat[k]:
                    sens[i][k] = sens[i][k]*2
    def sens_down(event):
        # Halves the active 'sens'
        nonlocal sens
        for i in range(2):
            for k in range(3):
                if statf[i] and stat[k]:
                    sens[i][k] = sens[i][k]/2

    def on_scroll(event): 
        # When you move the mouse scroll
        if event.button == 'up':
            roll_up(event)
        if event.button == 'down':
            roll_down(event)

        # Print the actual values
        phases_text.set_text(
                'p02={:7.2f} | p12={:7.2f} | pv2={:7.2f} || p01={:7.2f} | p11={:7.2f} | pv1={:7.2f}'.format(*P[0], *P[1]))

        # phase the entire 2D
        Sp = phase(S, p0=P[0][0], p1=P[0][1], pivot=P[0][2], dim='f2')
        if hyper:
            Sp = processing.tp_hyper(Sp)
        else:
            Sp = Sp.T
        Sp = phase(Sp, p0=P[1][0], p1=P[1][1], pivot=P[1][2], dim='f1')
        if hyper:
            Sp = processing.tp_hyper(Sp)
        else:
            Sp = Sp.T

        # Get the traces
        f1, f2 = maketraces(coord, Sp, ppm_f2, ppm_f1, hyper)

        for i in range(npk):
            # take the traces
            y_f1 = f1[i]
            y_f2 = f2[i]
            # update plots
            t_f2[i].set_ydata(y_f2.real)
            t_f1[i].set_ydata(y_f1.real)
            p_f2[i].set_xdata((P[0][2],))
            p_f1[i].set_xdata((P[1][2],))
            # Update zoom
            if zoom_adj:
                misc.set_ylim(ax[2*i], y_f2.real)
                misc.set_ylim(ax[2*i+1], y_f1.real)
        fig.canvas.draw()

    def zoom_onoff(event):
        nonlocal zoom_adj
        if event.key == 'z':
            zoom_adj = not(zoom_adj)

    def update_lim(val):
        # Update zoom
        L2 = l_f2.val
        R2 = r_f2.val
        L1 = l_f1.val
        R1 = r_f1.val
        for i in range(npk):
            ax[2*i].set_xlim(L2,R2)
            ax[2*i+1].set_xlim(L1,R1)

    def reset(event):
        # Reset the sliders
        nonlocal P
        P = np.copy(P0)
        on_scroll(event)

    def save(event):
        # Save the values
        nonlocal p0_f2, p1_f2, pivot_f2, p0_f1, p1_f1, pivot_f1
        p0_f2 = P[0][0]
        p1_f2 = P[0][1]
        pivot_f2 = P[0][2]
        p0_f1 = P[1][0]
        p1_f1 = P[1][1]
        pivot_f1 = P[1][2]

    def save_and_exit(event):
        # Function for the SAVE AND EXIT button:
        # Calls the 'save' function, then closes the figure
        save(event)
        plt.close()

    # Create figure panels: one for each trace
    ax = []
    for i in range(2*npk):
        ax.append(fig.add_subplot(npk, 2, i+1))

    # Set axis limits
    for i in range(2*npk):
        if np.mod(i+1,2)!=0:
            ax[i].set_xlim(max(ppm_f2), min(ppm_f2))
        else:
            ax[i].set_xlim(max(ppm_f1), min(ppm_f1))
    # Set vertical limits
    for i in range(npk):
        for j in range(2):
            if j==0:    # left
                T = max(f2[i].real)
                B = min(f2[i].real)
                panel = 2 * i
                ax[panel].set_title(r'$\delta\,$F1: '+'{:.1f} ppm'.format(coord[i][1]))
            else:       # right
                T = max(f1[i].real)
                B = min(f1[i].real)
                panel = 2 * i + 1
                ax[panel].set_title(r'$\delta\,$F2: '+'{:.1f} ppm'.format(coord[i][0]))
            
            
            ax[panel].set_ylim(B - 0.01*T, T + 0.01*T)
            # Make pretty scale
            xsx, xdx = ax[panel].get_xlim()

            misc.pretty_scale(ax[panel], ax[panel].get_xlim(), axis='x', n_major_ticks=10)

            misc.mathformat(ax[panel])
            # Plot ticks only in the bottom row
            if i != npk-1:
                ax[panel].tick_params(axis='x', labelbottom=False)
    
    # Create empty lists for traces plots
    t_f2 = []
    t_f1 = []
    p_f2 = []
    p_f1 = []
    # Plot the traces and append to the correct list
    for i in range(npk):
        tf2, = ax[2*i].plot(ppm_f2, f2[i], c='b', lw=0.8)        # Plot the data
        t_f2.append(tf2)
        pivot_bar_f2 = ax[2*i].axvline(P[0][2], c='r', lw=0.5)
        p_f2.append(pivot_bar_f2)
        tf1, = ax[2*i+1].plot(ppm_f1, f1[i], c='b', lw=0.8)        # Plot the data
        t_f1.append(tf1)
        pivot_bar_f1 = ax[2*i+1].axvline(P[1][2], c='r', lw=0.5)
        p_f1.append(pivot_bar_f1)
        ax[2*i].axhline(0, c='k', lw=0.2)    # baseline guide
        ax[2*i+1].axhline(0, c='k', lw=0.2)    # baseline guide

    plt.text(0.30, 0.050, r'$\delta$ F2 /ppm', ha='center', va='bottom', fontsize=18, transform=fig.transFigure)
    plt.text(0.65, 0.050, r'$\delta$ F1 /ppm', ha='center', va='bottom', fontsize=18, transform=fig.transFigure)

    phases_text = plt.text(0.975, 0.015, 
            'p02={:7.2f} | p12={:7.2f} | pv2={:7.2f} || p01={:7.2f} | p11={:7.2f} | pv1={:7.2f}'.format(*P[0], *P[1]),
            ha='right', va='bottom', transform=fig.transFigure, fontsize=10)

    # Connect the widgets to the functions
    l_f2.on_changed(update_lim)
    r_f2.on_changed(update_lim)
    l_f1.on_changed(update_lim)
    r_f1.on_changed(update_lim)
    reset_button.on_clicked(reset)
    save_button.on_clicked(save)
    saveandexit.on_clicked(save_and_exit)
    
    up_button.on_clicked(sens_up)
    down_button.on_clicked(sens_down)
    radio.on_clicked(statmod)
    seldim.on_clicked(change_dim)
    scroll = fig.canvas.mpl_connect('scroll_event', on_scroll)
    fig.canvas.mpl_connect('key_press_event', zoom_onoff)

    plt.show()

    # Phase the spectrum with the final Parameters:
    S = phase(S, p0=p0_f2, p1=p1_f2, pivot=pivot_f2, dim='f2')
    if hyper:
        S = processing.tp_hyper(S)
    else:
        S = S.T
    S = phase(S, p0=p0_f1, p1=p1_f1, pivot=pivot_f1, dim='f1')
    if hyper:
        S = processing.tp_hyper(S)
    else:
        S = S.T

    final_values_f1 = p0_f1, p1_f1, pivot_f1
    final_values_f2 = p0_f2, p1_f2, pivot_f2
    print('F2 - p0: {:.3f}, p1: {:.3f}, pv: {:.3f}'.format(*final_values_f2))
    print('F1 - p0: {:.3f}, p1: {:.3f}, pv: {:.3f}\n'.format(*final_values_f1))

    return S, final_values_f1, final_values_f2

def integral(fx, x=None, lims=None):
    """
    Calculates the primitive of fx. If fx is a multidimensional array, the integrals are computed along the last dimension.
    -------
    Parameters:
    - fx: ndarray
        Function (array) to integrate
    - x: 1darray or None
        Independent variable. Determines the integration step. If None, it is the point scale
    - lims: tuple or None
        Integration range. If None, the whole function is integrated.
    -------
    Returns:
    - Fx: ndarray
        Integrated function.
    """

    # Copy variables for check
    fx_in = np.copy(fx)
    if x is None:   # Make the point scale
        x_in = np.arange(fx.shape[-1])
    else:
        x_in = np.copy(x)
    # Integration step
    dx = misc.calcres(x_in)

    if lims is None:    # whole range
        x_tr, fx_tr = np.copy(x_in), np.copy(fx_in)
    else:
        # Trim data according to lims
        x_tr, fx_tr = misc.trim_data(x_in, fx_in, lims)
    
    # Integrate
    Fx = np.cumsum(fx_tr, axis=-1) * dx
    return Fx

def integrate(fx, x=None, lims=None):
    """
    Calculates the definite integral of fx as I = F[-1] - F[0]. If fx is a multidimensional array, the integrals are computed along the last dimension.
    
    -------
    Parameters:
    - fx: ndarray
        Function (array) to integrate
    - x: 1darray or None
        Independent variable. Determines the integration step. If None, it is the point scale
    - lims: tuple or None
        Integration range. If None, the whole function is integrated.
    -------
    Returns:
    - I: float
        Integrated function.
    """
    Fx = processing.integral(fx, x, lims)
    # Calculus fundamental theorem
    I = Fx[...,-1] - Fx[...,0]
    return I

def integral_2D(ppm_f1, t_f1, SFO1, ppm_f2, t_f2, SFO2, u_1=None, fwhm_1=200, utol_1=0.5, u_2=None, fwhm_2=200, utol_2=0.5, plot_result=False):
    """
    Calculate the integral of a 2D peak. The idea is to extract the traces correspondent to the peak center and fit them with a gaussian function in each dimension. Then, once got the intensity of each of the two gaussians, multiply them together in order to obtain the 2D integral. 
    This procedure should be equivalent to what CARA does.
    ---------
    Parameters:
    - ppm_f1: 1darray
        PPM scale of the indirect dimension
    - t_f1: 1darray 
        Trace of the indirect dimension, real part
    - SFO1: float
        Larmor frequency of the nucleus in the indirect dimension
    - ppm_f2: 1darray 
        PPM scale of the direct dimension
    - t_f2: 1darray 
        Trace of the direct dimension, real part
    - SFO2: float
        Larmor frequency of the nucleus in the direct dimension
    - u_1: float
        Chemical shift in F1 /ppm. Defaults to the center of the scale
    - fwhm_1: float
        Starting FWHM /Hz in the indirect dimension
    - utol_1: float
        Allowed tolerance for u_1 during the fit. (u_1-utol_1, u_1+utol_1)
    - u_2: float
        Chemical shift in F2 /ppm. Defaults to the center of the scale
    - fwhm_2: float
        Starting FWHM /Hz in the direct dimension
    - utol_2: float
        Allowed tolerance for u_2 during the fit. (u_2-utol_2, u_2+utol_2)
    - plot_result: bool
        True to show how the program fitted the traces.
    --------
    Returns:
    - I_tot: float
        Computed integral.
    """

    def f2min(param, T, x, SFO1):
        """ Cost function """
        par = param.valuesdict()
        sigma = misc.freq2ppm(par['fwhm'], np.abs(SFO1)) / 2.355    # Convert FWHM to ppm and then to std
        model = sim.f_gaussian(x, par['u'], sigma, A=par['I'])      # Compute gaussian
        par['I'] = fit.fit_int(T, model)                            # Calculate integral
        residual = par['I'] * model - T
        return residual
    
    def fitting(ppm, T, SFO1, u_0, fwhm_0, utol=0.5):
        """ Main function """
        param = l.Parameters()
        param.add('u', value=u_0, min=u_0-utol, max=u_0+utol)
        param.add('fwhm', value=fwhm_0, min=0)
        param.add('I', value=1, vary=False)         # Do not vary as it is adjusted during the fit

        minner = l.Minimizer(f2min, param, fcn_args=(T, ppm, SFO1))
        result = minner.minimize(method='leastsq', max_nfev=10000, xtol=1e-10, ftol=1e-10)
        popt = result.params.valuesdict()
        res = result.residual

        # Calculate the model, update the popt dictionary
        sigma = misc.freq2ppm(popt['fwhm'], np.abs(SFO1)) / 2.355
        model_0 = sim.f_gaussian(ppm, popt['u'], sigma, A=popt['I'])
        popt['I'] = fit.fit_int(T, model_0)
        model_0 *= popt['I']

        return popt, model_0 

    # Calculate u_0 if not given
    if u_1 is None:
        u_1 = np.mean(ppm_f1)
    if u_2 is None:
        u_2 = np.mean(ppm_f2)

    # Fit both traces using the function above
    popt_f2, fit_f2 = fitting(ppm_f2, t_f2, SFO2, u_2, fwhm_2, utol_2)
    popt_f1, fit_f1 = fitting(ppm_f1, t_f1, SFO1, u_1, fwhm_1, utol_1)

    if plot_result: # Do the plot
        xlim = [(max(ppm_f2), min(ppm_f2)),
                (max(ppm_f1), min(ppm_f1))]

        # Make the figure
        fig = plt.figure('Computed Integrals')
        fig.set_size_inches(figures.figsize_large)
        plt.subplots_adjust(left=0.05, right=0.95, bottom=0.10, top=0.90, wspace=0.20)
        
        axes = [fig.add_subplot(1,2,w+1) for w in range(2)]
        axes[0].set_title('FIT F2')
        axes[1].set_title('FIT F1')
        axes[0].plot(ppm_f2, t_f2, c='tab:blue', label='Trace F2')
        axes[0].plot(ppm_f2, fit_f2, c='tab:red', lw=0.9, label='Fit F2')
        axes[0].plot(ppm_f2, t_f2-fit_f2, c='green', lw=0.6, label='residual')
        axes[1].plot(ppm_f1, t_f1, c='tab:blue', label='Trace F1')
        axes[1].plot(ppm_f1, fit_f1, c='tab:red', lw=0.9, label='Fit F1')
        axes[1].plot(ppm_f1, t_f1-fit_f1, c='green', lw=0.6, label='residual')

        # Fancy shit
        for k, ax in enumerate(axes):
            misc.pretty_scale(ax, xlim[k], 'x')
            misc.pretty_scale(ax, ax.get_ylim(), 'y')
            misc.mathformat(ax)
            ax.set_xlabel(r'$\delta$ /ppm')
            ax.legend()
            misc.set_fontsizes(ax, 16)

        plt.show()
        plt.close()

    # Calculate integral
    I_tot = popt_f1['I'] * popt_f2['I']
    return I_tot



def pknl(data, grpdly=0, onfid=False):
    """
    Compensate for the Bruker group delay at the beginning of FID through a first-order phase correction of 
        p1 = 360 * GRPDLY
    This should be applied after apodization and zero-filling.
    -------
    Parameters:
    - data: ndarray
        Input data. Be sure it is complex!
    - grpdly: int
        Number of points that make the group delay.
    - onfid: bool
        If it is True, performs FT before to apply the phase correction, and IFT after.
    -------
    Returns:
    - datap: ndarray
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
    -------
    Parameters:
    - data: ndarray
        FID with digital filter
    - grpdly: int
        Number of points that the digital filter consists of. Key $GRPDLY in acqus file
    - scaling: float
        Scaling factor of the resulting FID. Needed to match TopSpin's intensities.
    -------
    Returns:
    - data_in: ndarray
        FID without the digital filter. It will have grpdly points less than data.
    """
    # Safety copy
    data_in = np.copy(data)

    # Circular shift to put the digital filter at the end of FID
    data_in = np.roll(data_in, -grpdly, axis=-1)
    # Digital filter, reversed to make it look like a FID.
    dig_filt = data_in[..., -grpdly:][::-1]

    # Subtract the digital filter, reversed, from the start of the FID
    data_in[...,:grpdly] -= dig_filt 
    # Trim the digital filter at the end of FID
    data_in = data_in[...,:-grpdly]
    # Correct the intensities
    data_in *= scaling
    return data_in


def calibration(ppmscale, S):
    """
    Scroll the ppm scale of spectrum to make calibration.
    The interface offers two guidelines: the red one, labelled 'reference signal' remains fixed, whereas the green one ('calibration value') moves with the ppm scale.
    The ideal calibration procedure consists in placing the red line on the signal you want to use as reference, and the green line on the ppm value that the reference signal must assume in the calibrated spectrum. Then, scroll with the mouse until the two lines are superimposed.
    -------
    Parameters:
    - ppmscale: 1darray
        The ppm scale to be calibrated
    - S: 1darray
        The spectrum to calibrate
    -------
    Returns:
    - offset: float
        Difference between original scale and new scale. This must be summed up to the original ppm scale to calibrate the spectrum.
    """
        
    #initialize values
    if ppmscale[0] < ppmscale[-1]:
        S = S[::-1]
        ppmscale = ppmscale[::-1]
    ppmscale0 = np.copy(ppmscale)       # save original scale for reset

    offset = 0                          # initialize returned value
    calstep = 0.25                      # calibration step 
    
    radio_flag = 1                          # radiobutton status
    
    # Initialize guidelines positions
    #   Fixed one
    g_idx = len(ppmscale)//2
    g_pos = ppmscale[g_idx]
    #   Mobile one
    d_idx = len(ppmscale)//2
    d_pos = ppmscale[g_idx]

    # Boxes and widgets
    #   Sliders
    box_left = plt.axes([0.1, 0.15, 0.80, 0.02])
    left_slider = Slider(box_left, 'Left', 0, len(ppmscale)-1, 0, valstep=1)
    box_right = plt.axes([0.1, 0.10, 0.80, 0.02])
    right_slider = Slider(box_right, 'Right', 0, len(ppmscale)-1, len(ppmscale)-1, valstep=1)

    #   Buttons
    box_save = plt.axes([0.905, 0.475, 0.07, 0.08])
    button = Button(box_save, 'SAVE\nAND\nEXIT', hovercolor='0.975')
    box_reset = plt.axes([0.825, 0.475, 0.07, 0.08])
    reset_button = Button(box_reset, 'RESET', hovercolor='0.975')
    box_up = plt.axes([0.905, 0.675, 0.07, 0.08])
    up_button = Button(box_up, r'$\uparrow$', hovercolor='0.975')
    box_down = plt.axes([0.825, 0.675, 0.07, 0.08])
    down_button = Button(box_down, r'$\downarrow$', hovercolor='0.975')

    # RadioButtons
    box_radio = plt.axes([0.825, 0.25, 0.15, 0.2])
    radio_labels = ['Reference signal', 'Calibration value']
    radio = RadioButtons(box_radio, radio_labels, active=0)
    
    
    # Functions connected to the widgets
    def radio_val(label):
        # Switch the status of the radiobutton
        nonlocal radio_flag
        if label==radio_labels[0]:
            radio_flag = 1
        elif label==radio_labels[1]:
            radio_flag = 0

    def increase_step(event):
        # up
        nonlocal calstep
        calstep *= 2

    def decrease_step(event):
        # down
        nonlocal calstep
        calstep /= 2

    def update(val):
        left = left_slider.val
        right = right_slider.val
        ppm_in = ppmscale[left], ppmscale[right]
        if np.abs(ppm_in[0]-ppm_in[1]) > 1:
            misc.pretty_scale(ax, ppm_in)
        else:
            ax.set_xlim(ppm_in)

        S_in = S[min(left,right):max(left,right)]
        T = np.max(np.array(S_in).real)
        B = np.min(np.array(S_in).real)
        ax.set_ylim(B - 0.01*T, T + 0.01*T)

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
        if radio_flag:
            move_fixed(event)
        else:
            move_mobile(event)
            
    def move_fixed(event):
        # set position of the red bar
        x = event.xdata
        if event.dblclick and str(event.button) == '1':
            nonlocal g_pos, g_idx
            g_pos = x 
            g_idx = misc.ppmfind(ppmscale, g_pos)[0]
            guide.set_xdata((x,))
        gtext.set_text('Ref: {: 9.3f}'.format(g_pos))
        fig.canvas.draw()
        
    def move_mobile(event):
        # set position of the green bar
        x = event.xdata
        if x is not None:
            if event.dblclick and str(event.button) == '1':
                nonlocal d_pos, d_idx
                d_pos = x 
                d_idx = misc.ppmfind(ppmscale, d_pos)[0]
                dguide.set_xdata((x,))
            dtext.set_text('Cal: {: 9.3f}'.format(d_pos))
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
        gtext.set_text('Ref: {: 9.3f}'.format(ppmscale[g_idx]))
        dtext.set_text('Cal: {: 9.3f}'.format(d_pos))
        update(0)
        fig.canvas.draw()
        
    # Make the figure 
    fig = plt.figure('Calibration')
    fig.set_size_inches(15,8)
    plt.subplots_adjust(left = 0.1, bottom=0.25, right=0.80, top=0.90)
    ax = fig.add_subplot(1,1,1)

    spect, = ax.plot(ppmscale, S.real, c='tab:blue', lw=0.8)    # plot spectrum

    # Plot the guidelines
    guide = ax.axvline(x=g_pos, lw=0.7, c='tab:red')        # static
    dguide = ax.axvline(x=d_pos, lw=0.7, c='tab:green')     # dynamic
    #   green and red lines position
    gtext = plt.text(0.925, 0.89, 'Ref: {: 9.3f}'.format(g_pos), ha='right', va='top', fontsize=20, transform=fig.transFigure, c='tab:red')
    dtext = plt.text(0.925, 0.85, 'Cal: {: 9.3f}'.format(d_pos), ha='right', va='top', fontsize=20, transform=fig.transFigure, c='tab:green')

    # Make cool figure
    T = np.max(np.array(S).real)
    B = np.min(np.array(S).real)
    ax.set_ylim(B - 0.01*T, T + 0.01*T)

    ax.ticklabel_format(axis='y', style='scientific', scilimits=(-2,2), useMathText=True)
    misc.pretty_scale(ax, (max(ppmscale), min(ppmscale)))

    # Connect widgets to functions
    left_slider.on_changed(update)
    right_slider.on_changed(update)
    button.on_clicked(save)  
    reset_button.on_clicked(reset)
    up_button.on_clicked(increase_step)
    down_button.on_clicked(decrease_step)
    radio.on_clicked(radio_val)
    cursor = Cursor(ax, useblit=True, horizOn=False, color='k', linewidth=0.4)
    mouse = fig.canvas.mpl_connect('button_press_event', mouse_click)
    scroll = fig.canvas.mpl_connect('scroll_event', on_scroll)
    
    plt.show()
    plt.close(1)

    print('Offset: {: .3f} /ppm'.format(offset))

    return offset

#-----------------------------------------------------------------------------------------
# MCR and related
def mcr_stack(input_data, P='H'):
    """
    Performs matrix augmentation by assembling input_data according to the positioning matrix P.
    P has two default modes: 'H' = horizontal stacking; 'V' = vertical stacking. Otherwise, a custom P matrix can be given as follows.
    The entries of the P matrix are the indices of the data in input_data. The shape of the matrix determines the final arrangement.
    Example: if input_data is [a, b, c, d, e, f], and one wants to obtain [[a, b], [d,c], [f, e]] the correspondant P matrix is:
    [[0, 1], [3, 2], [5, 4]]
    If each dataset in input_data has dimensions (m, n) and P has dimensions (u,v), then the returned data matrix will have dimensions (mu, nv).
    -------
    Parameters:
    - input_data: 3darray
        Contains the spectra to be stacked together. The index that runs on the datasets must be the first one.
    - P: str or 2darray
        'H' for horizontal stacking, 'V' for vertical stacking, or custom matrix as explained in the description
    -------
    Returns:
    - data: 2darray
        Augmented data matrix.
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
            P = np.arange(nds).reshape(1,-1)
        elif P == 'V':  # Vertical
            P = np.arange(nds).reshape(-1,1)
        else:   # Unknown
            raise ValueError('Unrecognized P type')
    elif isinstance(P, np.ndarray): # Check if the dimensions are compatible
        assert np.prod(P.shape) == nds, 'Wrong P shape'

    # Assemble the data
    for k in range(nds):
        # Compute mask matrix
        Mk = np.zeros_like(P)
        # Find the position of the k-th spectrum in P
        i, j = np.where(P == k)
        # Set that position as a 1 in Mk, all the rest is 0
        Mk[i[-1],j[-1]] = 1     # np.where returns lists of 1 number each
        if k == 0:  # Make the variable
            data = np.kron(Mk, Q[k])
        else:       # Add it
            data += np.kron(Mk, Q[k])
    return data


def mcr_unpack(C, S, nds, P='H'):
    """
    Reverts matrix augmentation of mcr_stack.
    The denoised spectra can be calculated by matrix multiplication: D[k] = C_f[k] S_f[k], for k = 0,..., nds-1
    --------
    Parameters:
    - C: 2darray
        MCR C matrix
    - S: 2darray
        MCR S matrix
    - nds: int
        number of experiments
    - P: str or 2darray
        'H' for horizontal stacking, 'V' for vertical stacking, or custom matrix as explained in the description of mcr_stack
    ---------
    Returns:
    - C_f: list of 2darray
        Disassembled MCR C matrix
    - S_f: list of 2darray
        Disassembled MCR C matrix
    """
    # Compute the P matrix
    if isinstance(P, str):  # default options
        if P == 'H':    # Horizontal
            P = np.arange(nds).reshape(1,-1)
        elif P == 'V':  # Vertical
            P = np.arange(nds).reshape(-1,1)
        else:   # Unknown
            raise ValueError('Unrecognized P type')
    elif isinstance(P, np.ndarray): # Check if the dimensions are compatible
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
        S_f.append(S[...,cslice])

    return np.array(C_f), np.array(S_f)

def calc_nc(data, s_n):
    """
    Calculates the optimal number of components, given the standard deviation of the noise.
    The threshold value is calculated as stated in Theorem 1 of reference: https://arxiv.org/abs/1710.09787v2
    -------
    Parameters:
    - data: 2darray
        Input data
    - s_n: float
        Noise standard deviation
    -------
    Returns:
    - n_c: int
        Number of components
    """
    M, N = data.shape

    S = slinalg.svdvals(data)
    
    b = M/N
    c = (1/2**0.5) * ( 1 + b + (1 + 14*b + b**2)**0.5 )**0.5
    threshold = s_n * ( (c + 1/c) * (c + b/c))**0.5

    threshold *= S[0]
    for k in range(len(S)):
        if S[k] < threshold:
            n_c = k+1
            break

    return n_c


def simplisma(D, nc, f=10, oncols=True):
    """
    Finds the first nc purest components of matrix D using the simplisma algorithm, proposed by Windig and Guilment (DOI: 10.1021/ac00014a016 ). If oncols=True, this function estimates S with simplisma, then calculates C = DS+ . If oncols=False, this function estimates C with simplisma, then calculates S = C+ D. f defines the percentage of allowed noise.
    -------
    Parameters:
    - D: 2darray
        Input data, of dimensions m x n
    - nc: int
        Number of components to be found. This determines the final size of the C and S matrices.
    - f: float
        Percentage of allowed noise.
    - oncols: bool
        If True, simplisma estimates the S matrix, otherwise estimates C.
    -------
    Returns:
    - C: 2darray
        Estimation of the C matrix, of dimensions m x nc.
    - S: 2darray
        Estimation of the S matrix, of dimensions nc x n.
    """

    rows = D.shape[0]       # number of rows of D
    cols = D.shape[1]       # number of columns of D

    if oncols:
        # on columns
        m = np.zeros(rows).astype(D.dtype)
        s = np.zeros(rows).astype(D.dtype)

        for i in range(rows):
            m[i] = np.mean(D[i,:])      # mean of the i-th row
            s[i] = np.std(D[i,:])       # STD of the i-th row

        # Correction factor for the noise 'alpha'
        a = 0.01 * f * max(m)

        print('Computing 1° purest variable...', end='\r')
        # Rescaling of data for lambda: makes determinant of COO
        # proportional only to the independance between variables
        l = ( s**2 + (m + a)**2 )**0.5  # lambda corrected for alpha
        Dl = np.zeros_like(D)
        for i in range(rows):
            Dl[i,:] = D[i,:] / l[i]

        Q = (1/cols) * Dl @ Dl.T      # Correlation-around-origin matrix

        # Calculation of the weighting factors:
        # express the independency between the variables
        w = np.zeros((rows, nc)).astype(D.dtype)       # Weights
        p_s = np.zeros((rows, nc)).astype(D.dtype)     # Pure components spectra
        s_s = np.zeros((rows, nc)).astype(D.dtype)     # STD spectra

        # First weight
        w[:,0] =  (s**2 + m**2) / (s**2 + (m + a)**2)
        p0 = s / (m + a)    # First purity spectrum
        pv, ipv = [], []    # Purest variables and correspondant index

        p_s[:,0] = w[:,0] * p0
        s_s[:,0] = w[:,0] * s

        # 1st purest variable
        pv.append(max(p_s[:,0]))
        ipv.append(np.argmax(p_s[:,0]))

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
                W = np.zeros((c+1,c+1)).astype(D.dtype)
                W[0,0] = Q[i,i]
                for k in range(1, c+1):                 # cycles inside W
                    W[0,k] = Q[i,ipv[k-1]]              # first row \{0,0}
                    W[k,0] = Q[ipv[k-1],i]              # first column \{0,0}
                    for q in range(1, c+1):
                        W[k,q] = Q[ipv[k-1],ipv[q-1]]   # all the rest, going row per row
                w[i,c] = linalg.det(W)

            p_s[:,c] = p0 * w[:,c]              # Create pure spectrum of c-th component
            s_s[:,c] = s_s[:,0] * w[:,c]        # Create STD spectrum of c-th component
            pv.append(max(p_s[:,c]))            # Update pure component
            ipv.append(np.argmax(p_s[:,c]))     # Update pure variable

        print('Purest variables succesfully found.\n')
        for c in range(nc):
            print('{}° purest variable:\t\t{}'.format(c+1, ipv[c]))

        # MCR "S" matrix (D = CS + E)
        S = np.zeros((nc, cols)).astype(D.dtype)     
        for c in range(nc):
            S[c,:] = D[ipv[c],:]
        C = D @ linalg.pinv(S)

    else:
        # on rows
        m = np.zeros((cols)).astype(D.dtype)
        s = np.zeros((cols)).astype(D.dtype)

        for j in range(cols):
            m[j] = np.mean(D[:,j])      # mean of the i-th row
            s[j] = np.std(D[:,j])       # STD of the i-th row

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
        l = ( s**2 + (m + a)**2 )**0.5  # lambda corrected for alpha
        Dl = np.zeros_like(D)
        for j in range(cols):
            Dl[:,j] = D[:,j] / l[j]

        Q = (1/rows) * Dl.T @ Dl      # Correlation-around-origin matrix

        # Calculation of the weighting factors: 
        # express the independency between the variables

        w = np.zeros((cols, nc)).astype(D.dtype)       # Weights
        p_s = np.zeros((cols, nc)).astype(D.dtype)     # Pure components spectra
        s_s = np.zeros((cols, nc)).astype(D.dtype)     # STD spectra

        # First weight
        w[:,0] =  (s**2 + m**2) / (s**2 + (m + a)**2)
        p_s[:,0] = w[:,0] * p1
        s_s[:,0] = w[:,0] * s

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
                W = np.zeros((c+1,c+1)).astype(D.dtype)
                W[0,0] = Q[j,j]
                for k in range(1, c+1): # cycles inside W
                    W[0,k] = Q[j,ipv[k-1]]        # first row \{0,0}
                    W[k,0] = Q[ipv[k-1],j]        # first column \{0,0}
                    for q in range(1, c+1):
                        W[k,q] = Q[ipv[k-1],ipv[q-1]] # all the rest, going row per row
                w[j,c] = linalg.det(W)

            p_s[:,c] = p_s[:,0] * w[:,c]      # Create pure spectrum of c-th component
            s_s[:,c] = s_s[:,0] * w[:,c]      # Create STD spectrum of c-th component
            pv.append(max(p_s[:,c]))          # Update pure component
            ipv.append(np.argmax(p_s[:,c]))   # Update pure variable
            
        print('Purest variables succesfully found.\n')
        for c in range(nc):
            print('{}° purest variable:\t\t{}'.format(c+1, ipv[c]))

        # MCR "C" matrix (D = CS + E)
        C = np.zeros((rows, nc)).astype(D.dtype)    
        for c in range(nc):
            C[:,c] = D[:,ipv[c]]
        S = linalg.pinv(C) @ D

    return C, S


def mcr_als(D, C, S, itermax=10000, tol=1e-5):
    """
    Performs alternating least squares to get the final C and S matrices. Being the fundamental MCR equation:
        D = CS + E
    At the k-th step of the iterative cycle:
        1. C(k) = DS+(k-1)
        2. S(k) = C+(k) D
        3. E(k) = D - C(k) S(k)
    Defined rC and rS as the Frobenius norm of the difference of C and S matrices between two subsequent steps:
        rC = || C(k) - C(k-1) ||
        rS = || S(k) - S(k-1) ||
    The convergence is reached when:
        rC <= tol && rS <= tol
    -------
    Parameters:
    - D: 2darray
        Input data, of dimensions m x n
    - C: 2darray
        Estimation of the C matrix, of dimensions m x nc.
    - S: 2darray
        Estimation of the S matrix, of dimensions nc x n.
    - itermax: int
        Maximum number of iterations
    - tol: float
        Threshold for the arrest criterion.
    -------
    Returns:
    - C: 2darray
        Optimized C matrix, of dimensions m x nc.
    - S: 2darray
        Optimized S matrix, of dimensions nc x n.
    """

    itermax = int(itermax)
    E = D - C @ S

    start_time = datetime.now()
    print('\n-----------------------------------------------------\n')
    print('             MCR optimization running...             \n')

    convergence_flag = 0
    print( '#   \tC convergence\tS convergence')
    for kk in range(itermax):
        # Copy from previous cycle
        C0 = np.copy(C)
        E0 = np.copy(E)
        S0 = np.copy(S)

        # Compute new C, S and E
        C = D @ linalg.pinv(S)
        S = linalg.pinv(C) @ D
        E = D - C @ S

        # Compute the Frobenius norm of the difference matrices
        # between two subsequent cycles
        rC = linalg.norm(C - C0)
        rS = linalg.norm(S - S0)

        # Ongoing print of the residues
        print(str(kk+1)+' \t{:.5e}'.format(rC)+ '\t'+'{:.5e}'.format(rS), end='\r')

        # Arrest criterion
        if (rC < tol) and (rS < tol) and kk:
            end_time = datetime.now()
            print( '\n\n\tMCR converges in '+str(kk+1)+' steps.')
            convergence_flag = 1    # Set to 1 if the arrest criterion is reached
            break

    if not convergence_flag:
        print ('\n\n\tMCR does not converge.')
    end_time = datetime.now()
    print( '\tTotal runtime: {}'.format(end_time - start_time))

    return C, S
    
def new_MCR_ALS(D, C, S, itermax=10000, tol=1e-5, reg_f=None, reg_fargs=[]):
    """
    Modified function to do ALS
    """

    itermax = int(itermax)
    E = D - C @ S

    start_time = datetime.now()
    print('\n-----------------------------------------------------\n')
    print('             MCR optimization running...             \n')

    convergence_flag = 0
    print( '#   \tC convergence\tS convergence')
    reg_fargs.append(None)
    for kk in range(itermax):
        # Copy from previous cycle
        C0 = np.copy(C)
        E0 = np.copy(E)
        S0 = np.copy(S)
        
            
        # Compute new C, S and E
        C = D @ linalg.pinv(S)

        # Regularization
        if reg_f is None:
            pass
        else:
            C, S, prev_param = reg_f(C, S, *reg_fargs, cycle=kk)
            reg_fargs[-1] = prev_param

        S = linalg.pinv(C) @ D
        if reg_f is not None:
            for i in range(S.shape[0]):
                S[i] /= 1#np.max(S[i])
        E = D - C @ S

        # Compute the Frobenius norm of the difference matrices
        # between two subsequent cycles
        rC = linalg.norm(C - C0)
        rS = linalg.norm(S - S0)

        # Ongoing print of the residues
        print(str(kk+1)+' \t{:.5e}'.format(rC)+ '\t'+'{:.5e}'.format(rS), end='\r')

        # Arrest criterion
        if (rC < tol) and (rS < tol):
            end_time = datetime.now()
            print( '\n\n\tMCR converges in '+str(kk+1)+' steps.')
            convergence_flag = 1    # Set to 1 if the arrest criterion is reached
            break

    if not convergence_flag:
        print ('\n\n\tMCR does not converge.')
    end_time = datetime.now()
    print( '\tTotal runtime: {}'.format(end_time - start_time))

    return C, S


def mcr(input_data, nc, f=10, tol=1e-5, itermax=1e4, P='H', oncols=True):
    """
    This is an implementation of Multivariate Curve Resolution for the denoising of 2D NMR data.
    Let us consider a matrix D, of dimensions m x n, where the starting data are stored. The final purpose of MCR is to decompose the D matrix as follows:
        D = CS + E
    where C and S are matrices of dimension m x nc and nc x n, respectively, and E contains the part of the data that are not reproduced by the factorization.
    Being D the FID of a NMR spectrum, C will contain time evolutions of the indirect dimension, and S will contain transients in the direct dimension.

    The total MCR workflow can be separated in two parts: a first algorithm that produces an initial guess for the three matrices C, S and E (simplisma), and an optimization step that aims at the removal of the unwanted features of the data by iteratively filling the E matrix (MCR ALS).
    This function returns the denoised datasets, CS, and the single C and S matrices.
    -------
    Parameters:
    - input_data: 2darray or 3darray
        a 3D array containing the set of 2D NMR datasets to be coprocessed stacked along the first dimension. A single 2D array can be passed, if the denoising of a single dataset is desired.
    - nc: int
        number of purest components to be looked for;
    - f: float
        percentage of allowed noise;
    - tol: float
        tolerance for the arrest criterion;
    - itermax: int
        maximum number of allowed iterations
    - P: str or 2darray
        'H' for horizontal stacking, 'V' for vertical stacking, or custom matrix as explained in the description of mcr_stack
    - oncols: bool
        True to estimate S with processing.simplisma, False to estimate C.
    -------
    Returns:
    - CS_f: 2darray or 3darray
        Final denoised data matrix
    - C_f: 2darray or 3darray
        Final C matrix
    - S_f: 2darray or 3darray
        Final S matrix
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
            print('Input data is not a matrix!')
            exit()
            

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



def new_MCR(input_data, nc, f=10, tol=1e-5, itermax=1e4, H=True, oncols=True, our_function=None, fargs=[], our_function2=None, f2args=[]):
    """
    # This is an implementation of Multivariate Curve Resolution
    # for the denoising of 2D NMR data. It requires:
    # - input_data: a tensor containing the set of 2D NMR datasets to be coprocessed
    #   stacked along the first dimension;
    # - nc      : number of purest components;
    # - f       : percentage of allowed noise;
    # - tol     : tolerance for the arrest criterion;
    # - itermax : maximum number of allowed iterations, default 10000
    # - H       : True for horizontal stacking of data (default), False for vertical;
    # - oncols  : True to estimate S with purest components, False to estimate C
    # This function returns the denoised datasets, 'CS', and the 'C' and 'S' matrices.
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
            print('Input data is not a matrix!')
            exit()
            

    print('\n*****************************************************')
    print('*                                                   *')
    print('*           Multivariate Curve Resolution           *')
    print('*                                                   *')
    print('*****************************************************\n')

    D = processing.mcr_stack(input_data, H=H)           # Matrix augmentation

    # Get initial estimation of C, S and E
    if our_function is None:
        C0, S0 = processing.simplisma(D, nc, f, oncols=oncols) 
    else:
        C0, S0, nc = our_function(D, *fargs)

    # Optimize C and S matrix through Alternating Least Squares
    if our_function2 is None:
        C, S = processing.mcr_als(D, C0, S0, itermax=itermax, tol=tol)
    else:
        C, S = processing.new_MCR_ALS(D, C0, S0, itermax, tol, our_function2, f2args)

    # Revert matrix augmentation
    C_f, S_f = processing.mcr_unpack(C, S, nds, H)
        
    # Obtain the denoised data of the same shape as the input
    if isinstance(input_data, list):
        CS_f = []
        for j in range(nds):
            CS_f.append(C_f[j] @ S_f[j])
    else:
        CS_f = np.zeros_like(input_data).astype(input_data.dtype)
        for j in range(nds):
            CS_f[j] = C_f[j] @ S_f[j]

    # Reshape if no matrix augmentation is performed
    if nds == 1:
        CS_f = CS_f[0]
        C_f = C_f[0]
        S_f = S_f[0]
  
    print('\n*****************************************************\n')

    return CS_f, C_f, S_f

def lrd(data, nc):
    """
    Denoising method based on Low-Rank Decomposition.
    The algorithm performs a singular value decomposition on data, then keeps only the first nc singular values while setting all the others to 0.
    Finally, rebuilds the data matrix using the modified singular values.
    -------
    Parameters:
    - data: 2darray
        Data to be denoised
    - nc: int
        Number of components, i.e. number of singular values to keep
    -------
    Returns:
    - data_out: 2darray
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
    print('Performing SVD. This might take a while...')
    U, svals, V = linalg.svd(data)
    print('Done.\n')
    # Apply hard-thresholding
    svals_p = np.zeros_like(svals)
    svals_p[:nc] = svals[:nc]
    # Reconstruct the denoised data
    data_out = U @ slinalg.diagsvd(svals_p, U.shape[1], V.shape[0]) @ V
    print('Low-Rank Denosing completed.')
    print('\n*****************************************************\n')
    return data_out

def cadzow(data, n, nc, print_head=True):
    """
    Performs Cadzow denoising on data, which is a 1D array of N points.
    The algorithm works as follows:
    1. Transform data in a Hankel matrix H of dimensions (N-n, n)
    2. Make SVD on H = U S V
    3. Keep only the first nc singular values, and put all the rest to 0 (S -> S')
    4. Rebuild H' = U S' V
    5. Average the antidiagonals to rebuild the Hankel-type structure, then make 1D array

    Set print_head=True to display the fancy heading.
    -------------
    Parameters:
    - data: 1darray
        Input data
    - n: int
        Number of columns of the Hankel matrix.
    - nc: int
        Number of singular values to keep.
    - print_head: bool
        Set it to True to display the fancy heading.
    -------------
    Returns:
    - datap: 1darray
        Denoised data
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
    """
    Performs Cadzow denoising on data, which is a 1D array of N points, in an iterative manner.
    The algorithm works as follows:
    1. Transform data in a Hankel matrix H of dimensions (N-n, n)
    2. Make SVD on H = U S V
    3. Keep only the first nc singular values, and put all the rest to 0 (S -> S')
    4. Rebuild H' = U S' V
    5. Average the antidiagonals to rebuild the Hankel-type structure, then make 1D array
    6. Check arrest criterion: if it is not reached, go to 1, else exit.

    The arrest criterion is:
    | S(step k-1)[nc-1] / S(step k-1)[0] - S(step k)[nc-1] / S(step k)[0] | < f * S(step 0)[nc-1] / S(step 0)[0]

    --------
    Parameters:
    - data: 1darray
        Data to be processed
    - n: int
        Number of columns of the Hankel matrix
    - nc: int
        Number of singular values to preserve
    - itermax: int
        max number of iterations allowed
    - f: float
        factor that appears in the arrest criterion
    - print_time: bool
        set it to True to show the time it took
    - print_head: bool
        set it to True to display the fancy heading.
    --------
    Returns:
    - datap: 1darray
        Denoised data
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
            print(f'\nCadzow converges in {k+1} steps.')
            break
        else:
            s0 = s 
            data0 = datap

    end_time = datetime.now()
    if k+1 == itermax:
        print('\tCadzow does not converge.')
    if print_time is True:
        print( 'Total runtime: {}'.format(end_time - start_time))
    # Add empty line for aesthetic purposes
    print()

    return datap

def cadzow_2D(data, n, nc, i=True, f=0.005, itermax=100, print_time=True):
    """
    Performs the Cadzow denoising method on a 2D spectrum, one transient at the time. 
    This function calls either Cadzow or iterCadzow, depending on the parameter 'i': 
    True for iterCadzow, False for normal Cadzow.
    ----------
    Parameters:
    - data: 2darray
        Input data
    - n: int
        Number of columns of the Hankel matrix.
    - nc: int
        Number of singular values to keep.
    - i: bool
        Calls processing.cadzow if i=False, or processing.iterCadzow if i=True.
    - itermax: int
        Maximum number of iterations allowed.
    - f: float
        Factor for the arrest criterion.
    - print_time: bool
        Set it to True to display the time spent.
    ----------
    Returns:
    - datap: 2darray
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
        print('Processing of transient '+str(k+1)+' of '+str(data.shape[0]))
        if i:
            datap[k] = processing.iterCadzow(data[k], n=n, nc=nc, f=f, itermax=itermax, print_head=False, print_time=False)
        else:
            datap[k] = processing.cadzow(data[k], n=n, nc=nc, print_head=False)
    print('Processing has ended!')
    end_time = datetime.now()
    if print_time is True:
        print( 'Total runtime: {}'.format(end_time - start_time))

    return datap










#-------------------------------------------------------------------------------------------------------------------

# BASELINE


def interactive_basl_windows(ppm, data):
    """
    Allows for interactive partitioning of a spectrum in windows. 
    Double left click to add a bar, double right click to remove it.
    Returns the location of the red bars as a list.
    -------
    Parameters:
    - ppm: 1darray
        PPM scale of the spectrum
    - data: 1darray
        Spectrum to be partitioned
    -------
    Returns:
    - coord: list
        List containing the coordinates of the windows, plus ppm[0] and ppm[-1]
    """

    # Make the figure
    fig = plt.figure('Manual Computation of Polynomial Baseline')
    fig.set_size_inches(15,8)
    ax = fig.add_subplot(1,1,1)
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.95)

    ax.set_title('Divide the spectrum into windows. Double click to set a wall, right click to remove it')

    # Set figure borders

    spectrum = figures.ax1D(ax, ppm, data)

    # Parameters to save coordinates
    coord = []          # Final list of coordinates
    dotvline = []       # Vertical lines

    def on_click(event):
        # What happens if you click?
        if event.inaxes == ax:
            pass
        else:
            return None

        x = event.xdata     # x,y position of cursor
        if x is not None:     # You are inside the figure
            idx, ix = misc.ppmfind(ppm, x) 
            if str(event.button) == '1' and event.dblclick:     # Left click: add point
                if ix not in coord:       # Avoid superimposed peaks
                    coord.append(ix)       # Update list
                    # Update figure:
                    #   add bullet
                    dotvline.append(ax.axvline(ix, c='r', lw=0.4))
            if str(event.button) == '3':    # Right click: remove point
                if ix in coord:       # only if the point is already selected
                    # Remove coordinates and all figure elements
                    i = coord.index(ix)
                    coord.remove(ix)
                    killv = dotvline.pop(i)
                    killv.remove()

        fig.canvas.draw()
    
    misc.set_fontsizes(ax, 14)
    # Widgets
    cursor = Cursor(ax, useblit=True, color='k', linewidth=0.2)
    mouse = fig.canvas.mpl_connect('button_press_event', on_click)

    plt.show()
    plt.close()


    # Append initial and final values of the ppm scale
    coord.append(ppm[0])
    coord.append(ppm[-1])
    # Sort the coordinates
    coord = sorted(coord)

    return coord


def make_polynomion_baseline(ppm, data, limits):
    """
    Interactive baseline correction with 4th degree polynomion.
    -------
    Parameters:
    - ppm: 1darray
        PPM scale of the spectrum
    - data: 1darray
        spectrum
    - limits: tuple
        Window limits (left, right).
    -------
    Returns:
    - mode: str
        Baseline correction mode: 'polynomion' as default, 'spline' if you press the button
    - C_f: 1darray or str
        Baseline polynomion coefficients, or 'callintsmooth' if you press the spline button
    """

    # Initialize mode
    mode = 'polynomion'

    # Lenght of data
    N = data.shape[-1]

    # Get index for the limits
    lim1 = misc.ppmfind(ppm, limits[0])[0]
    lim2 = misc.ppmfind(ppm, limits[1])[0]
    lim1, lim2 = min(lim1, lim2), max(lim1, lim2)
    
    # make boxes for widgets
    poly_box = plt.axes([0.87, 0.10, 0.10, 0.3])
    su_box = plt.axes([0.815, 0.825, 0.08, 0.075])
    giu_box = plt.axes([0.894, 0.825, 0.08, 0.075])
    callspline_box = plt.axes([0.825, 0.625, 0.15, 0.075])
    save_box = plt.axes([0.88, 0.725, 0.085, 0.04])
    reset_box = plt.axes([0.88, 0.765, 0.085, 0.04])
    
    # Make widgets
    #   Buttons
    up_button = Button(su_box, r'$\uparrow$', hovercolor = '0.975')    
    down_button = Button(giu_box, r'$\downarrow$', hovercolor = '0.975')
    save_button = Button(save_box, 'SAVE', hovercolor = '0.975')
    reset_button = Button(reset_box, 'RESET', hovercolor = '0.975')
    callspline_button = Button(callspline_box, 'SPLINE BASELINE\nCORRECTION', hovercolor = '0.975')
    
    #   Radio
    poly_name = ['a', 'b', 'c', 'd', 'e']
    poly_radio = RadioButtons(poly_box, poly_name, activecolor='tab:orange')       # Polynomion
    
    # Create variable for the 'active' status
    stats = np.zeros(len(poly_name))
    #    a   b   c   d   e
    stats[0] = 1

    # Initial values
    #   Polynomion coefficients
    C = np.zeros(len(poly_name))  
    #   Increase step for the polynomion (order of magnitude)
    om = np.zeros(len(poly_name))
    
    # Functions connected to the widgets
    def statmod(label):
        # Sets 'label' as active modifying 'stats'
        nonlocal stats
        if label in poly_name:    # if baseline
            stats = np.zeros(len(poly_name))
            for k, L in enumerate(poly_name):
                if label == L:
                    stats[k] = 1
        update(0)       # Call update to redraw the figure
                
    def roll_up_p(event):
        # Increase polynomion with mouse scroll
        nonlocal C
        for k in range(len(poly_name)):
            if stats[k]:
                C[k]+=10**om[k]
                
    def roll_down_p(event):
        # Decrease polynomion with mouse scroll
        nonlocal C
        for k in range(len(poly_name)):
            if stats[k]:
                C[k]-=10**om[k]
    
    def up_om(event):
        # Increase the om of the active coefficient by 1
        nonlocal om
        for k in range(len(poly_name)):
            if stats[k]:
                om[k] += 1
        
    def down_om(event):
        # Decrease the om of the active coefficient by 1
        nonlocal om
        for k in range(len(poly_name)):
            if stats[k]:
                om[k] -= 1
                
    def on_scroll(event):
        # Mouse scroll
        if event.button == 'up':
            roll_up_p(event)
        elif event.button == 'down':
            roll_down_p(event)
        update(0)
                
    # polynomion
    x = np.linspace(0, 1, ppm[lim1:lim2].shape[-1])[::-1]
    y = np.zeros_like(x)


    # Initial figure
    fig = plt.figure('Manual Computation of Polynomial Baseline')
    fig.set_size_inches(15,8)
    plt.subplots_adjust(bottom=0.10, top=0.90, left=0.05, right=0.80)
    ax = fig.add_subplot(1,1,1)

    ax.plot(ppm[lim1:lim2], data[lim1:lim2], label='Spectrum', lw=1.0, c='tab:blue')  # experimental

    poly_plot, = ax.plot(ppm[lim1:lim2], y, label = 'Baseline', lw=0.8, c='tab:orange') # Polynomion

    # make pretty scale
    ax.set_xlim(max(limits),min(limits))
    misc.pretty_scale(ax, ax.get_xlim(), axis='x', n_major_ticks=10)
    misc.set_ylim(ax, data[lim1:lim2])


    def update(val):
        # Calculates and draws all the figure elements
        y = misc.polyn(x, C)
        poly_plot.set_ydata(y)
        values_print.set_text('{:+5.2e}, {:+5.2e}, {:+5.2e}, {:+5.2e}, {:+5.2e}'.format(C[0], C[1], C[2], C[3], C[4]))
        plt.draw()
    
    def reset(event):
        # Sets all the widgets to their starting values
        nonlocal C, om 
        C = np.zeros(len(poly_name))
        om = np.zeros_like(C)
        update(0)       # to update the figure
    
    # Declare variables to store the final values
    C_f = np.zeros_like(C)
    def save(event):
        # Put current values in the final variables that are returned
        nonlocal C_f
        C_f = np.copy(C)

    def use_spline_instead(X):
        # Close everything and return
        nonlocal mode, C_f
        plt.close()
        mode = 'spline'
        C_f = 'callintsmooth'

    # Header for current values print
    plt.text(0.1, 0.04,
            '{:_^11}, {:_^11}, {:_^11}, {:_^11}, {:_^11}'.format('a', 'b', 'c', 'd', 'e'),
            ha='left', va='bottom', transform=fig.transFigure, fontsize=10)
    values_print = plt.text(0.1, 0.01,
            '{:+5.2e}, {:+5.2e}, {:+5.2e}, {:+5.2e}, {:+5.2e}'.format(*C),
            ha='left', va='bottom', transform=fig.transFigure, fontsize=10)
    misc.set_fontsizes(ax, 14)

    # Connect widgets to functions
    poly_radio.on_clicked(statmod)
    up_button.on_clicked(up_om)
    down_button.on_clicked(down_om)
    scroll = fig.canvas.mpl_connect('scroll_event', on_scroll)
    save_button.on_clicked(save)
    reset_button.on_clicked(reset)
    callspline_button.on_clicked(use_spline_instead)

    ax.legend()
    plt.show()
    plt.close()
   
    return mode, C_f


def write_basl_info(f, limits, mode, data):
    """
    Writes the baseline parameters of a certain window in a file.
    --------
    Parameters:
    - f: TextIO object
        File where to write the parameters
    - limits: tuple
        Limits of the spectral window. (left, right)
    - mode: str
        Baseline correction mode: 'polynomion' or 'spline'
    - data: float or 1darray
        It can be either the spline smoothing factor or the polynomion coefficients
    """
    f.write('***{:^54}***\n'.format('WINDOW LIMITS /PPM'))
    f.write('{: 8.3f}\t{: 8.3f}\n'.format(limits[0], limits[1]))
    f.write('***{:^54}***\n'.format('BASELINE CORRECTION MODE'))
    f.write('{}\n'.format(mode))
    f.write('***{:^54}***\n'.format('POLYNOMION COEFFICIENTS'))
    if mode == 'polynomion':
        N = len(data)
        for k, c in enumerate(data):
            if k < N - 1: 
                f.write('{: 5.2e}\t'.format(c))
            else:
                f.write('{: 5.2e}\n'.format(c))
                break
    else:
        N = 5
        for k, c in enumerate(np.zeros(5)):
            if k < N - 1: 
                f.write('{: 5.2e}\t'.format(c))
            else:
                f.write('{: 5.2e}\n'.format(c))
                break
    f.write('***{:^54}***\n'.format('SPLINE SMOOTHING FACTOR'))
    if mode == 'spline':
        f.write('{:5.3e}\n'.format(data))
    else:
        f.write('{:5.3e}\n'.format(0))
    f.write('***{:^54}***\n'.format('-'*50))


def baseline_correction(ppm, data, basl_file='spectrum.basl', winlim=None):
    """
    Interactively corrects the baseline of a given spectrum and saves the parameters in a file.
    The program starts with an interface to partition the spectrum in windows to correct separately.
    Then, for each window, an interactive panel opens to allow the user to compute the baseline.
    --------
    Parameters:
    - ppm: 1darray
        PPM scale of the spectrum
    - data: 1darray
        The spectrum of which to adjust the baseline
    - basl_file: str
        Name for the baseline parameters file
    - winlim: list or str or None
        List of the breakpoints for the window. If it is str, indicates the location of a file to be read with np.loadtxt. If it is None, the partitioning is done interactively.
    """

    # Check if winlim is passed as list
    if isinstance(winlim, list):
        coord = winlim
    elif isinstance(winlim, str):
        # It means it is a file. Try to read it
        if os.path.exists(winlim):
            coord = list(np.loadtxt(winlim))
        else:
            raise NameError('File {} not found.'.format(winlim)) 
    else:
        # Interactive partitioning
        coord = processing.interactive_basl_windows(ppm, data)

    # Clear the file
    if os.path.exists(basl_file):
        os.remove(basl_file)

    # Open the file
    F = open(basl_file, 'a')
    for i, _ in enumerate(coord):
        if i == len(coord) - 1:
            break       # Stop before it raises error
        limits = coord[i], coord[i+1]
        mode, C_f = processing.make_polynomion_baseline(ppm, data, limits)        # Interactive polynomion
        if isinstance(C_f, str):    # If you press "use spline" in the polynomion interactive figure
            # Get the limits
            lim1 = misc.ppmfind(ppm, limits[0])[0]
            lim2 = misc.ppmfind(ppm, limits[1])[0]
            lim1, lim2 = min(lim1, lim2), max(lim1, lim2)
            # trim ppm and data
            xdata, ydata = ppm[lim1:lim2], data[lim1:lim2]
            # Calculate the spline
            _, C_f = fit.interactive_smoothing(xdata, ydata)
        # Write the section in the file
        processing.write_basl_info(F, limits, mode, C_f)
    F.close()

def load_baseline(filename, ppm, data):
    """
    Read the baseline parameters from a file and builds the baseline itself.
    -------
    Parameters:
    - filename: str
        Location of the baseline file
    - ppm: 1darray
        PPM scale of the spectrum
    - data: 1darray
        Spectrum of which to correct the baseline
    -------
    Returns:
    - baseline: 1darray
        Computed baseline
    """
    
    # Opens the file
    f = open(filename, 'r')
    r = f.readlines()

    # Initialize the lists of the variables
    limits = []     # Window limits
    mode = []       # Baseline correction mode
    C = []          # Polynomion coefficients
    S = []          # Spline smoothing factor

    tmpmode = None      # Correction mode for the active section
    for k, line in enumerate(r):
        # Read the limits
        if 'WINDOW LIMITS /PPM' in line:
            Q = r[k+1]
            Q = Q.replace('\t', ', ')
            limits.append(eval(Q))
            continue
        # Read mode
        if 'BASELINE CORRECTION MODE' in line:
            tmpmode = r[k+1].strip()
            mode.append(tmpmode)
            continue
        # Read the polynomion coefficients
        if 'POLYNOMION COEFFICIENTS' in line:
            if tmpmode == 'polynomion':
                Q = r[k+1]
                Q = Q.replace('\t', ',')
                C.append(np.array(eval('['+Q+']')))
            else:
                C.append(np.zeros(5))
            continue
        # Read the spline smoothing factor
        if 'SPLINE SMOOTHING FACTOR' in line:
            if tmpmode == 'spline':
                Q = r[k+1]
                S.append(eval(Q))
            else:
                S.append(0)
            continue
        # Reset tmpmode 
        if '-----' in line:
            tmpmode = None
            continue

    # Now, make the baseline

    # Initialize flat baseline
    baseline = np.zeros_like(ppm)
    n_w = len(limits)   # Number of windows

    for k in range(n_w):
        # Translate the limits in points
        lim1 = misc.ppmfind(ppm, limits[k][0])[0]
        lim2 = misc.ppmfind(ppm, limits[k][1])[0]
        lim1, lim2 = min(lim1, lim2), max(lim1, lim2)

        if mode[k] == 'polynomion': # Compute polynomion in the active region
            x = np.linspace(0, 1, ppm[lim1:lim2].shape[-1])[::-1]
            tmpbasl = misc.polyn(x, C[k])
        elif mode[k] == 'spline': # Fit the spectrum in the active region with a spline
            y = data[lim1:lim2]
            tmpbasl = fit.smooth_spl(y, S[k])
        # Put the just computed baseline in the corresponding region
        baseline[lim1:lim2] = tmpbasl

    return baseline

def qfil(ppm, data, u, s):
    """
    Suppress signals in the spectrum using a gaussian filter.
    ---------
    Parameters:
    - ppm: 1darray
        Scale on which to build the filter
    - data: ndarray
        Data to be processed. The filter is applied on the last dimension
    - u: float
        Position of the filter
    - s: float
        Width of the filter (standard deviation)
    --------
    Returns:
    - pdata: ndarray
        Filtered data
    """
    G = sim.gaussian_filter(ppm, u, s)
    datap = np.zeros_like(data)
    datap[...,:] = data[...,:] * G
    return datap

def interactive_qfil(ppm, data_in):
    """ 
    Interactive function to design a gaussian filter with the aim of suppressing signals in the spectrum.
    You can adjust position and width of the filter scrolling with the mouse.
    ---------
    Parameters:
    - ppm: 1darray
        Scale on which the filter will be built
    - data_in: 1darray
        Spectrum on which to apply the filter.
    ---------
    Returns:
    - u: float
        Position of the gaussian filter
    - s: float
        Width of the gaussian filter (Standard deviation)
    """

    # Safe copy
    data = np.copy(data_in.real)

    # Initialize the values: u at the center of the spectrum, s as 100 points
    u = np.mean(ppm)
    s = 100 * misc.calcres(ppm)

    sens = 0.2  # one mouse 'tick'
    stat = 0    # move u

    # Make the filter with start values
    G = sim.f_gaussian(ppm, u, s)
    G /= max(G)     # Normalize it to preserve intensities

    # Make the figure
    fig = plt.figure('Adjust Position and Width for QFIL')
    fig.set_size_inches(figures.figsize_large)
    plt.subplots_adjust(left=0.10, bottom=0.15, right=0.85, top=0.90)
    ax = fig.add_subplot(1,1,1)

    # Plot 
    #   Original spectrum
    figures.ax1D(ax, ppm, data, c='tab:blue', lw=0.8, X_label=r'$\delta\, $/ppm', Y_label='Intensity /a.u.', label='Original')
    #   Filter
    G_plot, = ax.plot(ppm, G*np.max(data), c='tab:orange', lw=0.6, ls='--', label='Filter')
    #   Processed data
    pdata = data * (1 - G)      # Compute it
    p_spect, = ax.plot(ppm, pdata, c='tab:red', lw=0.7, label='Processed')

    # --------------------------------------------------

    # WIDGETS
    #   Radio-buttons to select which value to modify
    radio_box = plt.axes([0.875, 0.40, 0.10, 0.20]) 
    radio_labels = ['u', 's']
    radio = RadioButtons(radio_box, radio_labels)

    # Modify sensitivity buttons
    up_box = plt.axes([0.875, 0.70, 0.05, 0.05])
    up_button = Button(up_box, r'$\uparrow$')
    dn_box = plt.axes([0.925, 0.70, 0.05, 0.05])
    dn_button = Button(dn_box, r'$\downarrow$')

    # FUNCTIONS CONNECTED TO WIDGETS
    def up_sens(event):
        """ Double sens """
        nonlocal sens
        sens *= 2
    def dn_sens(event):
        """ Halves sens """
        nonlocal sens
        sens /= 2

    def radio_func(label):
        """ Change the variable 'stats' according to the radiobutton """
        nonlocal stat
        if label == radio_labels[0]:    # u
            stat = 0
        elif label == radio_labels[1]:  # s
            stat = 1
            
    def on_scroll(event):
        """ On mouse scroll, modify the correspondant value, then redraw the figure """
        nonlocal u, s
        if event.button == 'up':
            if stat:    # s
                s += sens
            else:       # u
                u += sens
        elif event.button == 'down':
            if stat:    # s
                s -= sens
                if s < 0:   # Safety check
                    s = 0
            else:       # u
                u -= sens
        update()

    def update():
        """ Redraw the figure """
        # Compute the filter with the new values
        G_in = sim.f_gaussian(ppm, u, s)
        G_in /= max(G_in)
        # Multiply * max(data) to make it visible
        G_plot.set_ydata(G_in*np.max(data))
        # Compute processed data
        pdata = data * (1 - G_in)
        p_spect.set_ydata(pdata)
        plt.draw()

    # --------------------------------------------------

    # CONNECT WIDGETS TO THE FUNCTIONS
    up_button.on_clicked(up_sens)
    dn_button.on_clicked(dn_sens)
    radio.on_clicked(radio_func)
    fig.canvas.mpl_connect('scroll_event', on_scroll)

    # --------------------------------------------------

    # Adjust figure appearence
    ax.legend(loc='upper right', fontsize=12)
    misc.mathformat(ax)
    misc.set_fontsizes(ax, 14)
    plt.show()
    plt.close()

    return u, s

def acme(data, m=1, a=5e-5):
    """
    Automated phase Correction based on Minimization of Entropy.
    This algorithm allows for automatic phase correction by minimizing the entropy of the m-th derivative of the spectrum, as explained in detail by L. Chen et.al. in Journal of Magnetic Resonance 158 (2002) 164-168.
    
    Defined the entropy of h as:
        S = - sum_j h_j ln(h_j)
    and 
        h = | R_j^(m) | / sum_j | R_j^(m) |
    where 
        R = Re{ spectrum * e^(-i phi) }
    and R^(m) is the m-th derivative of R, the objective function to minimize is:
        S + P(R)
    where P(R) is a penalty function for negative values of the spectrum.

    The phase correction is applied using processing.ps. The values p0 and p1 are fitted using Nelder-Mead algorithm.
    ----------
    Parameters:
    - data: 1darray
        Spectrum to be phased, complex
    - m: int
        Order of the derivative to be computed
    - a: float
        Weighting factor for the penalty function
    ----------
    Returns:
    - p0f: float
        Fitted zero-order phase correction, in degrees
    - p1f: float
        Fitted first-order phase correction, in degrees
    """

    def entropy(data):
        """
        Compute entropy of data.
        --------
        Parameters:
        - data: ndarray
            Input data
        --------
        Returns:
        - S: float
            Entropy of data
        """
        data_in = np.copy(data)
        if not data_in.all():
            zero_ind = np.flatnonzero(data_in==0)
            for i in zero_ind:
                data_in[i] = 1e-15

        return - np.sum(data_in * np.log(data_in))

    def mth_derivative(data, m):
        """
        Computes the m-th derivative of data by applying np.gradient m times.
        -------
        Parameters:
        - data: 1darray
            Input data
        - m: int
            Order of the derivative to be computed
        -------
        Returns:
        - pdata: 1darray
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
        --------
        Parameters:
        - data: 1darray
            Input data
        - a: float
            Weighting factor
        --------
        Returns:
        - p_fun: float
            a * sum_j F(y_j) y_j^2
        """
        signs = - np.sign(data)     # 1 for negative entries, -1 for positive entries
        p_arr = np.array([0 if j<1 else 1 for j in signs])  # replace all !=1 values in signs with 0
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
    param = l.Parameters()
    param.add('p0', value=0, min=-360, max=360)
    param.add('p1', value=0, min=-720, max=720)

    # Minimize using simplex method because the residue is a scalar
    minner = l.Minimizer(f2min, param, fcn_args=(np.copy(data), m, a))
    result = minner.minimize(method='nelder', tol=1e-15)
    popt = result.params.valuesdict()

    return popt['p0'], popt['p1']


def whittaker_smoother(data, n=2, s_f=1, w=None):
    """
    Adapted from P.H.C. Eilers, Anal. Chem 2003, 75, 3631-3636.
    Implementation of the smoothing algorithm proposed by Whittaker in 1923.
    --------
    Parameters:
    - data: 1darray
        Data to be smoothed
    - n: int
        Order of the difference to be computed
    - s_f: float
        Smoothing factor
    - w: 1darray or None
        Array of weights. If None, no weighting is applied.
    --------
    Returns:
    - z: 1darray
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

    The baseline is computed using a low-order polynomion, built on a scale that goes from -1 to 1, whose coefficients are obtained minimizing a non-quadratic cost function. It is recommended to use either "tq" (truncated quadratic, much faster) or "huber" (Huber function, slower but sometimes more accurate). The user is requested to choose between separating the real and imaginary channel in this step. The order of the polynomion and the threshold value are the key parameters for obtaining a good baseline. The used function is processing.polyn_basl

    The phase correction is computed on the baseline-subtracted complex data as described in the SINC algorithm (ref.). The default parameters are generally fine, but in case of data with poor SNR (approximately SNR < 10) better results can be obtained by increasing the value of the e1 parameter. The employed function is processing.SINC_phase
    -----------
    Parameters:
    - data: 1darray
        Data to be processed, complex-valued
    - split_imag: bool
        If True, computes the baseline on the real and imaginary part separately; else, the set of polynomion coefficients are forced to be the same for both
    - n: int
        Number of coefficients of the polynomion, i.e. it will be of degree n-1
    - basl_method: str
        Cost function to be minimized for the baseline computation. Look for fit.CostFunc, "method" attribute
    - basl_thresh: float
        Relative threshold value for the non-quadratic behaviour of the cost function. Look for fit.CostFunc, "s" attribute
    - basl_itermax: int
        Maximun number of iterations allowed during the baseline fitting procedure
    - phase_kws: keyworded arguments
        Optional arguments for the phase correction. Look for fit.SINC_phase keyworded arguments for details.
    -----------
    Returns:
    - y: 1darray
        Processed data
    - p0: float
        Zero-order phase correction angle, in degrees
    - p1: float
        First-order phase correction angle, in degrees
    - c: 1darray
        Set of coefficients to be used for the baseline computation, starting from the 0-order coefficient
    """

    # Check if the data is actually complex
    if np.iscomplexobj(data):
        y = np.copy(data)
    else:
        raise ValueError('Input data is not complex. Aborting...')

    ## BASELINE COMPUTATION AND SUBTRACTION
    if not n: # Do not correct the baseline
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

    ## PHASE CORRECTION
    # Compute the phase angles
    p0, p1 = fit.sinc_phase(y, **phase_kws)
    # Apply it
    y, *_ = processing.ps(y, p0=p0, p1=p1)

    return y, p0, p1, c




def align(ppm_scale, data, lims, u_off=0.5, ref_idx=0):
    """
    Performs the calibration of a pseudo-2D experiment by circular-shifting the spectra of an appropriate amount.
    The target function aims to minimize the superimposition between a reference spectrum and the others using a brute-force method.
    ----------
    Parameters:
    - ppm_scale: 1darray
        ppm scale of the spectrum to calibrate
    - data: 2darray
        Complex-valued spectrum
    - lims: tuple
        (ppm sx, ppm dx) of the calibration region
    - u_off: float
        Maximum offset for the circular shift, in ppm
    - ref_idx: int
        Index of the spectrum to be used as reference
    ----------
    Returns:
    - data_roll: 2darray
        Calibrated data
    - u_cal: list
        Number of point of which the spectra have been circular-shifted
    - u_cal_ppm: list
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
            param = l.Parameters()
            param.add('u', value=0, max=npoints, min=-npoints)
            param['u'].set(brute_step = 1)      # Discrete step of one point
            
            # Fit
            minner = l.Minimizer(f2min, param, fcn_args=(s_ref, s_i, cal_reg))
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
        D a = d
    where 'a' is the array of lp coefficients.
    -----------
    Parameters:
    - data: 1darray
        FID to be linear-predicted
    - pred: int
        Number of points to predict
    - order: int
        Number of coefficients to use for the prediction
    - mode: str
        'f' for forward linear prediction, 'b' for backward linear prediction
    -----------
    Returns:
    - newdata: 1darray
        FID with linear prediction applied.
    """
    def make_Dd(x, order, mode):
        L = len(x) - order      # Dimension of the coefficient array
        if mode == 'f':         # Forward lp
            # Characteristic matrix
            D = misc.hankel(x[:-1], order)
            # Coefficient vector
            d = x[order:].reshape(L,1)
        elif mode == 'b':       # Backward lp
            # Characteristic matrix
            D = misc.hankel(x[1:], order)
            # Coefficient vector
            d = x[:L].reshape(L,1)
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
            ntrace[-M:] = trace # Copy the "old" fid at the end
            for i in range(pred):   # Fill the rest point-by-point, going backwards
                ntrace[pred-i-1] = np.sum(np.multiply(ntrace[pred-i: pred+m-i], a.flat))
        return ntrace
    if len(data.shape) != 1:
        raise NotImplementedError('LP available for only 1D datasets')

    N = data.shape[-1]      # Length of the dataset
    # Cut the data to ease the computational workload
    t = min( N, max(10*pred, 2048) ) # 500 points or 10x the number of points
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
    Applies backward linear prediction by calling processing.lp with mode='b'.
    -----------------
    Parameters:
    - data: 1darray
        FID to be linear-predicted
    - pred: int
        Number of points to predict
    - order: int
        Number of coefficients to use for the prediction
    -----------
    Returns:
    - lpdata: 1darray
        FID with linear prediction applied.
    """
    lpdata = lp(data, pred, order, mode='b')
    return lpdata

def blp_ng(data, pred=1, order=8, N=2048):
    """
    Performs backwards linear prediction on data.
    This function calls nmrglue.process.proc_lp.lp with most of the parameters set automatically.
    The algorithm predicts "pred" points of the FID using "order" coefficient for the linear interpolation.
    Only the first N points of the FID are used in the LP equation, because the computational cost scales with n**2, making the use of more than 8k points not effective: using more points brings negligible contiribution to the final result.
    For Oxford spectra, set "pred" to half the value written in "TDoff".
    ------------
    Parameters:
    - data: ndarray
        Data on which to perform the linear prediction. For 2d data, it is performed row-by-row
    - pred: int 
        Number of points to be predicted
    - order: int
        Number of coefficients to be used for the prediction
    - N: int
        Number of points of the FID to be used in the calculation
    -------------
    Returns:
    - datap: ndarray
        Data with the predicted points appended at the beginning
    """
    # Compute a slice to trim the data to decrease the computation time
    if data.shape[-1] >= N:     # Slice from 0 to N
        data_sl = slice(0, N)
    else:                       # Leave unchanged
        data_sl = None
    datap = ng.process.proc_lp.lp(data, 
            pred=pred,          # Number of points to predict
            order=order,        # Number of coefficients to use
            slice=data_sl,      # Slicing
            mode='b',           # Backwards
            append='before',    # Append points before
            bad_roots='decr',   # Default option for mode='b'
            method='svd')       # Choice of method basically uninfluent
    return datap

def stack_fids(*fids, filename=None):
    """
    Stacks together FIDs in order to create a pseudo-2D experiment.
    This function can handle either arrays or Spectrum_1D objects.
    -----------
    Parameters:
    - fids: sequence of 1darrays or Spectrum_1D objects
        Input data.
    - filename: str
        Location for a .npy file to be saved. If None, no file is created.
    -----------
    Returns:
    - p2d: 2darray
        Stacked FIDs.
    """
    p2d_fid = []    # Placeholder
    # Append the FIDs to this list
    for k, fid in enumerate(fids):
        # If 1darray, append it as is
        if isinstance(fid, np.ndarray) and len(fid.shape)==1:
            p2d_fid.append(fid)
        # If Spectrum_1D, append the "fid" attribute
        elif isinstance(fid, Spectrum_1D):
            p2d_fid.append(fid.fid)
        else:   # Raise an error
            raise ValueError(f'There was a problem in reading the {k+1}° fid.')

    # Pile up the FIDs
    p2d = np.stack(p2d_fid, axis=0)

    # Save the .npy file
    if isinstance(filename, str):
        np.save(filename, p2d)

    return p2d


def hilbert(f):
    """
    Computes the Hilbert transform of real vector f in order to retrieve its imaginary part.
    Make sure that the original spectrum was zero-filled to at least twice the original size of the FID.
    The algorithm computes the convolution by means of FT, as follows:
    > make IFT of f = a
    > compute h = [1j for x in range(N) if x<N/2 else -1j]
    > Compute b = ha
    > Build d = a + 1j*b
    > make FT of d = F
    > replace Re(F) with f
    ---------
    Parameters:
    - f: ndarray
        Array of which you want to compute the imaginary part
    ---------
    Returns:
    - f_cplx: ndarray
        Complex version of f
    """
    # Suppress warnings for ft of real data
    warnings.simplefilter("ignore")
    # Get the number of points of f
    N = f.shape[-1]
    # Apply Hilbert transform:
    #   make ift of the data
    a = np.fft.ifft(f.real)
    #   compute vector h: i for the first half, -i for the other half
    h = 1j * np.ones_like(a)
    h[...,N//2:] *= -1
    #   retrieve imaginary part
    b = a * h
    #   make the full, complex signal
    d = a + 1j * b
    #   do ft 
    f_i = np.fft.fft(d)
    #   replace real part of the data with the original one
    f_cplx = f.real + 1j*f_i.imag
    # Re-enable warnings
    warnings.simplefilter("default")
    return f_cplx

def convolve(in1, in2):
    """
    Perform the convolution of the two array by multiplying their inverse Fourier transform.
    The two arrays must have the same dimension.
    -----------
    Parameters:
    - in1: ndarray
        First array
    - in2: ndarray
        Second array
    -----------
    Returns:
    - cnv: ndarray
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
    -----------
    Parameters:
    - in1: ndarray
        First array
    - in2: ndarray
        Second array
    -----------
    Returns:
    - cnv: ndarray
        Convolved array
    """
    assert in1.shape[-1] == in2.shape[-1], 'The two arrays have different dimensions!'
    size = in1.shape[-1]
    in1t = np.fft.ifft(np.fft.ifftshift(in1))
    in2t = np.fft.ifft(np.fft.ifftshift(in2))
    cnvt = in1t * np.linalg.pinv(in2t.reshape(-1,1)).reshape(-1)
    # factor size is needed to correct the intensity
    cnv = size * np.fft.fftshift(np.fft.fft(cnvt))
    return cnv
