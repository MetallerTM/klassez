#! /usr/bin/env python3

import os
import sys
import numpy as np
from numpy import linalg
from scipy import stats
from scipy.spatial import ConvexHull
from scipy.signal import find_peaks, peak_widths
from scipy import interpolate
from csaps import csaps
import random
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.widgets import Slider, Button, RadioButtons, TextBox, CheckButtons, Cursor, LassoSelector, SpanSelector
from matplotlib.path import Path
import seaborn as sns
import nmrglue as ng
import lmfit as l
from datetime import datetime
import warnings
from copy import deepcopy

from . import fit, misc, sim, figures, processing
#from .__init__ import CM
from .config import CM, COLORS, cron


"""
Functions for performing fits.
"""

s_colors=[ 'tab:cyan', 'tab:red', 'tab:green', 'tab:purple', 'tab:pink', 'tab:gray', 'tab:brown', 'tab:olive', 'salmon', 'indigo' ]

def histogram(data, nbins=100, density=True, f_lims= None, xlabel=None, x_symm=False, fitG=True, barcolor='tab:blue', fontsize=10, name=None, ext='tiff', dpi=600):
    """
    Computes an histogram of 'data' and tries to fit it with a gaussian lineshape.
    The parameters of the gaussian function are calculated analytically directly from 'data' using 'scipy.stats.norm'
    --------
    Parameters:
    - data : ndarray
        the data to be binned
    - nbins : int
        number of bins to be calculated
    - density : bool
        True for normalize data
    - f_lims : tuple or None
        limits for the x axis of the figure
    - xlabel : str or None
        Text to be displayed under the x axis
    - x_symm : bool
        set it to True to make symmetric x-axis with respect to 0
    - fitG: bool
        Shows the gaussian approximation
    - barcolor: str
        Color of the bins
    - fontsize: float
        Biggest fontsize in the figure
    - name : str
        name for the figure to be saved
    - ext: str
        Format of the image
    - dpi: int
        Resolution of the image in dots per inches
    -------
    Returns:
    - m : float
        Mean of data
    - s : float
        Standard deviation of data.
    """

    fig = plt.figure('Histogram')
    ax = fig.add_subplot(1,1,1)
    plt.subplots_adjust(left=0.10, bottom=0.10, top=0.90, right=0.95)
    fig.set_size_inches(figures.figsize_large)

    m, s = fit.ax_histogram(ax, data, nbins=nbins, density=density, f_lims=f_lims, xlabel=xlabel, x_symm=x_symm, fitG=fitG, barcolor=barcolor, fontsize=fontsize)

    if name:
        print(f'Saving {name}.{ext}...')
        plt.savefig(f'{name}.{ext}', format=f'{ext}', dpi=dpi)
    else:
        plt.show()
    plt.close()
    print('Done.')

    return m, s

def ax_histogram(ax, data0, nbins=100, density=True, f_lims=None, xlabel=None, x_symm=False, fitG=True, barcolor='tab:blue', fontsize=10):
    """
    Computes an histogram of 'data' and tries to fit it with a gaussian lineshape.
    The parameters of the gaussian function are calculated analytically directly from 'data' using 'scipy.stats.norm'
    --------
    Parameters:
    - ax : matplotlib.subplot Object
        panel of the figure where to put the histogram
    - data0 : ndarray
        the data to be binned
    - nbins : int
        number of bins to be calculated
    - density : bool
        True for normalize data
    - f_lims : tuple or None
        limits for the x axis of the figure
    - xlabel : str or None
        Text to be displayed under the x axis
    - x_symm : bool
        set it to True to make symmetric x-axis with respect to 0
    - fitG: bool
        Shows the gaussian approximation
    - barcolor: str
        Color of the bins
    - fontsize: float
        Biggest fontsize in the figure
    -------
    Returns:
    - m : float
        Mean of data
    - s : float
        Standard deviation of data.
    """

    if len(data0.shape) > 1:
        data = data0.real.flatten()
    else:
        data = np.copy(data0.real)

    if x_symm:
        lims = ( -max(np.abs(data)), max(np.abs(data)) )
    else:
        lims = (min(data), max(data))
    
    hist, bin_edges = np.histogram(data, bins=nbins, range=lims, density=density)   # Computes the bins for the histogram

    x = np.linspace(lims[0], lims[1], len(data))        # Scale for a smooth gaussian
    m, s = np.mean(data), np.std(data)                  # Get mean and standard deviation of 'data'
   
    if density:
        A = 1
    else:
        A = np.trapz(hist, dx=bin_edges[1]-bin_edges[0])    # Integral

    if fitG:
        fit_g = sim.f_gaussian(x, m, s, A)  # Gaussian lineshape

    ax.hist(data, color=barcolor, density=density, bins=bin_edges) 
    if fitG:
        ax.plot(x, fit_g, c='r', lw=0.6, label='Gaussian approximation') 
        ax.legend(loc='upper right')


    if density:
        ax.set_ylabel('Normalized count')
    else:
        ax.set_ylabel('Count')
    if xlabel:
        ax.set_xlabel(xlabel)

    if f_lims:
        misc.pretty_scale(ax, f_lims, 'x')
    else:
        misc.pretty_scale(ax, ax.get_xlim(), 'x')
    misc.pretty_scale(ax, ax.get_ylim(), 'y')

    misc.mathformat(ax, axis='both', limits=(-3,3))


    misc.set_fontsizes(ax, fontsize)

    return m, s

def bin_data(data0, nbins=100, density=True, x_symm=False):
    """
    Computes the histogram of data, sampling it into nbins bins.
    --------
    Parameters:
    - data : ndarray
        the data to be binned
    - nbins : int
        number of bins to be calculated
    - density : bool
        True for normalize data
    - x_symm : bool
        set it to True to make symmetric x-axis with respect to 0
    -------
    Returns:
    - hist: 1darray
        The bin intensity
    - bin_scale: 1darray
        Scale built with the mean value of the bin widths.
    """
    if len(data0.shape) > 1:
        data = data0.flatten()
    else:
        data = np.copy(data0)

    if x_symm:
        lims = (- max(np.abs(data)), max(np.abs(data)) )
    else:
        lims = (min(data), max(data))

    hist, bin_edges = np.histogram(data, bins=nbins, range=lims, density=density)   # Computes the bins for the histogram
    bin_scale = np.array( [np.mean((bin_edges[k], bin_edges[k+1])) for k in range(len(bin_edges) - 1) ])
    return hist, bin_scale


def lr(y, x=None, force_intercept=False):
    """ 
    Performs a linear regression of y with a model y_c = mx + q.
    ---------
    Parameters:
    - y: 1darray
        Data to be fitted
    - x: 1darray
        Independent variable. If None, the point indexes are used.
    - force_intercept: bool
        If True, forces the intercept to be zero.
    ---------
    Returns:
    - y_c: 1darray
        Fitted trend
    - values: tuple
        (m, q)
    """
    # Make the scale of points, if not given
    if x is None:
        x = np.arange(y.shape[-1])

    # Create the Vandermonde matrix of x:
    if force_intercept:     # It is x
        T = np.copy(x).reshape(-1,1)
    else:                   # it is [1, x]
        T = np.array(
            [x**k for k in range(2)]
            ).T

    # Pseudo-invert it
    Tpinv = np.linalg.pinv(T)
    # Solve the system
    c = Tpinv @ y

    if force_intercept:
        m = float(c)        # It is just a number
        q = 0
    else:                   # unpack the array
        q, m = c
    # Compute the model
    y_c = m * x + q 
    return y_c, (m, q)

def calc_R2(y, y_c):
    """
    Computes the R-squared coefficient of a linear regression as:
        R^2 = 1 - [ \sum (y - y_mean)^2 ] / [ \sum (y - y_c)^2 ]
    -------
    Parameters:
    - y: 1darray
        Experimental data
    - y_c: 1darray
        Calculated data
    -------
    Returns:
    - R2: float
        R-squared coefficient
    """
    sst = np.sum( (y - np.mean(y))**2 )
    sse = np.sum( (y - y_c)**2 )
    R2 = 1 - sse/sst
    return R2



def fit_int(y, y_c, q=True):
    """
    Computes the optimal intensity and intercept of a linear model in the least squares sense.
    Let y be the experimental data and y_c the model, and let <w> the mean of variable w.
    Then:
        A = ( <y_c y> - <y_c><y> ) / ( <y_c^2> - <y_c>^2 )
        q = ( <y_c>^2<y> - <y_c><y_c y> ) / ( <y_c^2> - <y_c>^2 )
    ----------
    Parameters:
    - y: 1darray
        Experimental data
    - y_c: 1darray
        Model data
    - q: bool
        If True, includes the offset in the calculation. If False, only the intensity factor is computed.
    ----------
    Returns:
    - A: float
        Optimized intensity
    - q: float
        Optimized intercept
    """

    if q:
        # Apply the formulaes, numerator only
        A = np.mean(y_c * y) - np.mean(y_c) * np.mean(y)
        q = np.mean(y_c**2) * np.mean(y) - np.mean(y_c) * np.mean(y * y_c)

        # Compute denominator
        Q = np.mean(y_c**2) - np.mean(y_c)**2
        # Divide
        A /= Q
        q /= Q
    else:
        # Apply other formula
        A = np.mean(y_c * y) / np.mean(y_c**2)
        q = 0

    return A, q

def get_region(ppmscale, S, rev=True):
    """
    Interactively select the spectral region to be fitted. 
    Returns the border ppm values.
    -------
    Parameters:
    - ppmscale: 1darray
        The ppm scale of the spectrum
    - S: 1darray
         The spectrum to be trimmed
    - rev: bool
        Choose if to reverse the ppm scale and data (True) or not (False).
    -------
    Returns:
    - left: float
        Left border of the selected spectral window
    - right: float
        Right border of the selected spectral window
    """

    # Set the slider initial values
    if rev:
        left = max(ppmscale)
        right = min(ppmscale)
    else:
        right = max(ppmscale)
        left = min(ppmscale)
    res = misc.calcres(ppmscale)

    zoom_toggle = False

    # Make the boxes
    #   for sliders
    box_left = plt.axes([0.1, 0.15, 0.80, 0.02])
    box_t_left = plt.axes([0.1, 0.05, 0.05, 0.03])
    box_right = plt.axes([0.1, 0.10, 0.80, 0.02])
    box_t_right = plt.axes([0.85, 0.05, 0.05, 0.03])
    #   for buttons
    box_button = plt.axes([0.45, 0.925, 0.1, 0.04])
    # Make the sliders
    left_slider = Slider(ax=box_left, label='Left', valmin=min(ppmscale), valmax=max(ppmscale), valinit=left, valstep=res, color='tab:red')
    right_slider = Slider(ax=box_right, label='Right', valmin=min(ppmscale), valmax=max(ppmscale), valinit=right, valstep=res, color='tab:green')
    # Make the buttons
    button = Button(box_button, 'SAVE', hovercolor='0.975')
    l_tbox = TextBox(box_t_left, '', textalignment='center')
    r_tbox = TextBox(box_t_right, '', textalignment='center')


    # Definition of the 'update' functions
    # 
    def update_region(val):
        # updates the value for the range selectors
        left = left_slider.val
        right = right_slider.val
        LB, RB = misc.ppmfind(ppmscale, left)[0], misc.ppmfind(ppmscale, right)[0]
        data_inside = S[min(LB,RB):max(LB,RB)]

        L.set_xdata((left),)
        R.set_xdata((right),)
        if rev:
            ax.set_xlim(left+25*res, right-25*res)
        else:
            ax.set_xlim(left-25*res, right+25*res)
        T = max(data_inside.real)
        B = min(data_inside.real)
        if zoom_toggle:
            ax.set_ylim(B - 0.05*T, T + 0.05*T)



    def on_submit_l(v):
        V = eval(v)
        left_slider.set_val(V)
        update_region(0)
    def on_submit_r(v):
        V = eval(v)
        right_slider.set_val(V)
        update_region(0)

    def save(event):
        # Saves the values to be returned
        nonlocal left, right
        left = left_slider.val
        right = right_slider.val

    def key_press(event):
        nonlocal zoom_toggle
        if event.key == 'z':
            zoom_toggle = not(zoom_toggle)

    # Creation of interactive figure panel
    fig = plt.figure('Region Selector')
    fig.set_size_inches(15,8)
    plt.subplots_adjust(left = 0.10, bottom=0.25, right=0.90, top=0.90)    # Make room for the sliders
    ax = fig.add_subplot(1,1,1)

    misc.pretty_scale(ax, (left, right))
    if rev:
        ax.set_xlim(left+25*res, right-25*res)
    else:
        ax.set_xlim(left-25*res, right+25*res)

    ax.plot(ppmscale, S.real, c='b', lw=0.8)        # Plot the data
    misc.mathformat(ax, 'y')
    ax.set_xlabel(r'$\delta\,$ /ppm')
    ax.set_ylabel('Intensity /a.u.')
    ax.set_title('Press Z to toggle the automatic zoom')
    L = ax.axvline(x=left, lw=0.5, c='r')           # Left selector
    R = ax.axvline(x=right, lw=0.5, c='g')          # Right selector

    # Call the 'update' functions upon interaction with the widgets
    left_slider.on_changed(update_region)
    right_slider.on_changed(update_region)
    button.on_clicked(save)
    l_tbox.on_submit(on_submit_l) 
    r_tbox.on_submit(on_submit_r) 
    fig.canvas.mpl_connect('key_press_event', key_press)

    misc.set_fontsizes(ax, 14)

    plt.show()
    plt.close(1)

    return left, right



def make_signal(t, u, s, k, b, phi, A, SFO1=701.125, o1p=0, N=None):
    """
    Generates a voigt signal on the basis of the passed parameters in the time domain. Then, makes the Fourier transform and returns it.
    -------
    Parameters:
    - t : ndarray
        acquisition timescale
    - u : float
        chemical shift /ppm
    - s : float
        full-width at half-maximum /Hz
    - k : float
        relative intensity
    - b : float
        fraction of gaussianity
    - phi : float
        phase of the signal, in degrees
    - A : float
        total intensity
    - SFO1 : float
        Larmor frequency /MHz
    - o1p : float
        pulse carrier frequency /ppm
    - N : int or None
        length of the final signal. If None, signal is not zero-filled before to be transformed.
    -------
    Returns:
    - sgn : 1darray
        generated signal in the frequency domain
    """
    U = misc.ppm2freq(u, SFO1, o1p)         # conversion to frequency units
    S = s * 2 * np.pi                       # conversion to radians
    phi = phi * np.pi / 180                 # conversion to radians
    sgn = sim.t_voigt(t, U, S, A=A*k, phi=phi, b=b) # make the signal
    if isinstance(N, int):
        sgn = processing.zf(sgn, N)         # zero-fill it
    sgn = processing.ft(sgn)                # transform it
    return sgn


def plot_fit(S, ppm_scale, regions, t_AQ, SFO1, o1p, show_total=False, show_res=False, res_offset=0, X_label=r'$\delta$ /ppm', labels=None, filename='fit', ext='tiff', dpi=600):
    """
    Plots either the initial guess or the result of the fit, and saves all the figures. Calls fit.plot_fit.
    The figure <filename>_full will show the whole model and the whole spectrum. 
    The figures labelled with _R<k> will depict a detail of the fit in the k-th fitting region.
    Optional labels for the components can be given: in this case, the structure of 'labels' should match the structure of 'regions'. This means that the length of the outer list must be equal to the number of fitting region, and the length of the inner lists must be equal to the number of peaks in that region.
    ------------
    Parameters:
    - S: 1darray
        Spectrum to be fitted
    - ppm_scale: 1darray
        ppm scale of the spectrum
    - regions: dict
        Generated by fit.read_vf
    - t_AQ: 1darray
        Acquisition timescale
    - SFO1: float
        Larmor frequency of the observed nucleus, in MHz
    - o1p: float
        Carrier position, in ppm
    - nuc: str
        Observed nucleus. Used to customize the x-scale of the figures.
    - show_total: bool
        Show the total trace (i.e. sum of all the components) or not
    - show_res: bool
        Show the plot of the residuals
    - res_offset: float
        Displacement of the residuals plot from 0, to be given as a fraction of the height of the experimental spectrum. res_offset > 0 will move the residuals BELOW the zero-line!
    - X_label: str
        Text to show as label for the chemical shift axis
    - labels: list of list
        Optional labels for the components. The structure of this parameter must match the structure of self.result
    - filename: str
        Root of the name of the figures that will be saved. If None, <self.filename> is used
    - ext: str
        Format of the saved figures
    - dpi: int
        Resolution of the figures, in dots per inches
    """


    def calc_total(peaks, A=1):
        """
        Calculates the sum trace from a collection of peaks stored in a dictionary.
        If the dictionary is empty, returns an array of zeros.
        ---------
        Parameters:
        - peaks: dict
            Components
        - A: float
            Absolute intensity
        --------
        Returns:
        - total: 1darray
            Sum spectrum
        """
        # Get the arrays from the dictionary
        T = [p(A) for _, p in peaks.items()]
        if len(T) > 0:  # Check for any peaks
            total = np.sum(T, axis=0)
            return total.real
        else:
            return np.zeros_like(ppm_scale)

    print('Saving figures...')
    # Shallow copy of the real part of the experimental spectrum
    S_r = np.copy(S.real)
    N = S_r.shape[-1]       # For (eventual) zero-filling
    # Make the acqus dictionary to be fed into the fit.Peak objects
    acqus = { 't1': t_AQ, 'SFO1': SFO1, 'o1p': o1p, }


    ## Single regions
    for k, region in enumerate(regions):        # Loop on the regions
        # Shallow copy of the region dict
        in_region = dict(region)
        # Calculate the slice that delimits the fit region
        #   Remove from in_region
        limits = in_region.pop('limits')
        #   Convert to points on the ppm scale
        limits_pt = misc.ppmfind(ppm_scale, limits[0])[0], misc.ppmfind(ppm_scale, limits[1])[0]
        #   Make the slice
        lims = slice(min(limits_pt), max(limits_pt))
        # Get the absolute intensity
        I = in_region.pop('I')

        # Create a dictionary of fit.Peak objects with the same structure of in_region
        peaks = {key: fit.Peak(acqus, N=N, **peakval) for key, peakval in in_region.items()}
        # Get the total trace
        total = calc_total(peaks, I)

        # Trim the ppm scale according to the fitting region
        t_ppm = ppm_scale[lims]

        # Make the figure
        fig = plt.figure('Fit')
        fig.set_size_inches(figures.figsize_large)
        ax = fig.add_subplot()
        plt.subplots_adjust(left=0.10, bottom=0.10, top=0.90, right=0.95)

        # Plot the experimental dataset
        ax.plot(t_ppm, S_r[lims], c='k', lw=1, label='Experimental')

        if show_total is True:  # Plot the total trace in blue
            ax.plot(t_ppm, total[lims], c='b', lw=0.5, label='Fit')

        for key, peak in peaks.items(): # Plot the components
            p_sgn, = ax.plot(t_ppm, peak(I)[lims], lw=0.6, label=f'{key}')
            if labels is not None:  # Set the custom label
                p_sgn.set_label(labels[k][key-1])

        if show_res is True:    # Plot the residuals
            # Compute the absolute value of the offset
            r_off = min(S_r[lims]) + res_offset * (max(S_r[lims])-min(S_r[lims]))
            ax.plot(t_ppm, (S_r - total)[lims] - r_off, c='g', ls=':', lw=0.6, label='Residuals')

        # Visual adjustments
        ax.set_xlabel(X_label)
        ax.set_ylabel('Intensity /a.u.')
        misc.pretty_scale(ax, (max(t_ppm), min(t_ppm)), axis='x')
        misc.pretty_scale(ax, ax.get_ylim(), axis='y')
        misc.mathformat(ax)
        ax.legend()
        misc.set_fontsizes(ax, 20)
        # Save the figure
        plt.savefig(f'{filename}_R{k+1}.{ext}', dpi=dpi)
        plt.close()


    ## Total
    # Make the figure
    fig = plt.figure('Fit')
    fig.set_size_inches(figures.figsize_large)
    plt.subplots_adjust(left=0.10, bottom=0.10, top=0.90, right=0.95)
    ax = fig.add_subplot()

    # Placeholder for total trace and for the limits
    total = np.zeros_like(ppm_scale)
    lims_pt = []

    # Plot the experimental spectrum
    ax.plot(ppm_scale, S_r, c='k', lw=1, label='Experimental', zorder=1)
    # Plot the components, region by region
    for k, region in enumerate(regions):
        # Shallow copy of in_region
        in_region = dict(region)
        # Make the slice
        limits = in_region.pop('limits')
        limits_pt = misc.ppmfind(ppm_scale, limits[0])[0], misc.ppmfind(ppm_scale, limits[1])[0]
        lims = slice(min(limits_pt), max(limits_pt))
        lims_pt.append(lims)    # Save it in the list
        # Get the absolute intensity
        I = in_region.pop('I')
        # Make the dictionary of fit.Peak objects
        peaks = {key: fit.Peak(acqus, N=N, **peakval) for key, peakval in in_region.items()}
        # Plot the components
        for idx, peak in peaks.items():  
            p_sgn, = ax.plot(ppm_scale, peak(I), lw=0.6, label=f'Win. {k+1}, Comp. {idx}', zorder=10)
            if labels is not None:  # Set custom label
                p_sgn.set_label(labels[k][idx-1])

        # Add these contributions to the total trace
        total += calc_total(peaks, I)

    # Residuals
    R = S_r - total
    
    if show_total is True:  # Plot the total trace
        ax.plot(ppm_scale, total, c='b', lw=0.5, label='Fit', zorder=2)

    # Visual adjustments
    ax.set_xlabel(X_label)
    ax.set_ylabel('Intensity /a.u.')
    misc.pretty_scale(ax, (max(ppm_scale), min(ppm_scale)), axis='x')
    misc.pretty_scale(ax, ax.get_ylim(), axis='y')
    misc.mathformat(ax)
    ax.legend()
    misc.set_fontsizes(ax, 20)
    # Save the figure
    plt.savefig(f'{filename}_full.{ext}', dpi=dpi)
    plt.close()
    print('Done.')


def voigt_fit_indep(S, ppm_scale, regions, t_AQ, SFO1, o1p, u_lim=1, f_lim=10, k_lim=(0, 3), vary_phase=False, vary_b=True, itermax=10000, fit_tol=1e-8, filename='fit', method='leastsq'):
    """
    Performs a lineshape deconvolution fit using a Voigt model.
    The initial guess must be read from a .ivf file. All components are treated as independent, regardless from the value of the "group" attribute.
    The fitting procedure operates iteratively one window at the time.
    ------------
    Parameters:
    - S: 1darray
        Experimental spectrum
    - ppm_scale: 1darray
        PPM scale of the spectrum
    - regions: dict
        Generated by fit.read_vf
    - t_AQ: 1darray
        Acquisition timescale
    - SFO1: float
        Nucleus Larmor frequency /MHz
    - o1p: float
        Carrier frequency /ppm
    - u_lim: float
        Maximum allowed displacement of the chemical shift from the initial value /ppm
    - f_lim: float
        Maximum allowed displacement of the linewidth from the initial value /ppm
    - k_lim: float or tuple
        If tuple, minimum and maximum allowed values for k during the fit. If float, maximum displacement from the initial guess
    - vary_phase: bool
        Allow the peaks to change phase
    - vary_b: bool
        Allow the peaks to change Lorentzian/Gaussian ratio
    - itermax: int
        Maximum number of allowed iterations
    - fit_tol: float
        Target value to be set for x_tol and f_tol
    - filename: str
        Name of the file where the fitted values will be saved. The .fvf extension is added automatically
    - method: str
        Method to be used for the optimization. See lmfit for details.
    """

    ## USED FUNCTIONS

    def calc_total(peaks, A=1):
        """
        Calculates the sum trace from a collection of peaks stored in a dictionary.
        If the dictionary is empty, returns an array of zeros.
        ---------
        Parameters:
        - peaks: dict
            Components
        - A: float
            Absolute intensity
        --------
        Returns:
        - total: 1darray
            Sum spectrum
        """
        # Get the arrays from the dictionary
        T = [p(A) for _, p in peaks.items()]
        if len(T) > 0:  # Check for any peaks
            total = np.sum(T, axis=0)
            return total
        else:
            return np.zeros_like(ppm_scale)

    def peaks_frompar(peaks, par):
        """
        Replaces the values of a "peaks" dictionary, which contains a fit.Peak object for each key "idx", with the values contained in the "par" dictionary.
        The par dictionary keys must have keys of the form <parameter>_<idx>, where <parameter> is in [u, fwhm, k, 'b', 'phi'], and <idx> are the keys of the peaks dictionary.
        -----------
        Parameters:
        - peaks: dict
            Collection of fit.Peak objects
        - par: dict
            New values for the peaks
        ----------
        Returns:
        - peaks: dict
            Updated peaks dictionary with the new values
        """
        for idx, peak in peaks.items():
            peak.u = par[f'u_{idx}']
            peak.fwhm = par[f'fwhm_{idx}']
            peak.k = par[f'k_{idx}']
            peak.b = par[f'b_{idx}']
            peak.phi = par[f'phi_{idx}']
        return peaks

    def f2min(param, S, fit_peaks, I, lims, first_residual=1):
        """
        Function that calculates the residual to be minimized in the least squares sense.
        This function requires a set of pre-built fit.Peak objects, stored in a dictionary. The parameters of the peaks are replaced on this dictionary according to the values in the lmfit.Parameter object. At this point, the total trace is computed and the residual is returned as the difference between the experimental spectrum and the total trace, only in the region delimited by the "lims" tuple.
        ------------
        Parameters:
        - param: lmfit.Parameters object
            Usual lmfit stuff
        - S: 1darray
            Experimental spectrum
        - fit_peaks: dict
            Collection of fit.Peak objects
        - I: float
            Absolute intensity.
        - lims: slice
            Trimming region corresponding to the fitting window, in points
        - first_residual: float
            Target value at the first call of this function. Used to compute the relative target function.
        -----------
        Returns:
        - residual: 1darray
            Experimental - calculated, in the fitting window
        """
        param['count'].value += 1
        # Unpack the lmfit.Parameters object
        par = param.valuesdict()
        # create a shallow copy of the fit_peaks dictionary to prevent overwriting
        peaks = deepcopy(fit_peaks)
        # Update the peaks dictionary according to how lmfit is moving the fit parameters
        peaks = peaks_frompar(peaks, par)
        # Compute the total trace and the residuals
        total = calc_total(peaks, 1)
        exp = S[lims]/I
        calc = total[lims]
        correction_factor, _ = fit.fit_int(exp, calc, q=False)
        residual = exp - calc * correction_factor
        param['correction_factor'].set(value=correction_factor)
        print(f'Step: {par["count"]:6.0f} | Target: {np.sum(residual**2)/first_residual:10.5e}', end='\r')
        return residual 

    def gen_reg(regions):
        """
        Generator function that loops on the regions and extracts the limits of the fitting window, the limits, and the dictionary of peaks.
        """
        for k, region in enumerate(regions):
            # Get limits and total intensity from the dictionary
            limits = region['limits']
            I = region['I']
            if 1:   # Switch: turn this print on and off
                print(f'Fitting of region {k+1}/{Nr}. [{limits[0]:.3f}:{limits[1]:.3f}] ppm')
            # Make a copy of the region dictionary and remove what is not a peak
            peaks = deepcopy(region)
            peaks.pop('limits')
            peaks.pop('I')
            yield limits, I, peaks

    # -----------------------------------------------------------------------------
    # Make the acqus dictionary to be fed into the fit.Peak objects
    acqus = { 't1': t_AQ, 'SFO1': SFO1, 'o1p': o1p, }

    N = S.shape[-1]     # Number of points of the spectrum
    Nr = len(regions)   # Number of regions to be fitted

    # Write info on the fit in the output file
    with open(f'{filename}.fvf', 'a', buffering=1) as f:
        now = datetime.now()
        date_and_time = now.strftime("%d/%m/%Y at %H:%M:%S")
        f.write('! Fit performed by {} on {}\n\n'.format(os.getlogin(), date_and_time))

    # Generate the values from the regions dictionary with the gen_reg generator
    Q = gen_reg(regions)
    with open('cnvg', 'w') as f:
        pass

    # Start fitting loop
    prev = 0
    for q in Q:
        limits, I, peaks = q    # Unpack
        Np = len(peaks.keys())  # Number of Peaks

        # Create a dictionary which contains Peak objects
        fit_peaks = {}
        for key, peakval in peaks.items():  
            # Same keys of the input dictionary
            fit_peaks[key] = fit.Peak(acqus, N=N, **peakval)

        # Add the peaks' parameters to a lmfit Parameters object
        peak_keys = ['u', 'fwhm', 'k', 'b', 'phi']
        param = l.Parameters()
        for idx, peak in fit_peaks.items():
            # Name of the object: <parameter>_<index>
            p_key = f'_{idx}'

            # Fill the Parameters object
            for key in peak_keys:
                par_key = f'{key}{p_key}'   # Add the parameter to the label
                val = peak.par()[key]       # Get the value from the input dictionary
                param.add(par_key, value=val)   # Make the Parameter object
                # Set the limits for each parameter, and fix the ones that have not to be varied during the fit
                if 'u' in key:  # u: [u-u_tol, u+u_tol]
                    param[par_key].set(min=max(val-u_lim, min(limits)), max=min(val+u_lim, max(limits)))
                elif 'fwhm' in key: # fwhm: [max(0, fwhm-f_tol), fwhm+f_tol] (avoid negative fwhm)
                    param[par_key].set(min=max(0, val-f_lim), max=val+f_lim)
                elif 'k' in key:    # k: [0, 3]
                    if isinstance(k_lim, float):
                        param[par_key].set(min=param[par_key].value-k_lim, max=param[par_key].value+k_lim)
                    else:
                        param[par_key].set(min=min(k_lim), max=max(k_lim))
                elif 'phi' in key:  # phi: [-180°, +180°]
                    param[par_key].set(min=-180, max=180, vary=vary_phase)
                elif 'b' in key:  # b: [0, 1]
                    param[par_key].set(min=0, max=1, vary=vary_b)

        # Convert the limits from ppm to points and make the slice
        limits_pt = misc.ppmfind(ppm_scale, limits[0])[0], misc.ppmfind(ppm_scale, limits[1])[0]
        lims = slice(min(limits_pt), max(limits_pt))

        # Wrap the fitting routine in a function in order to use @cron for measuring the runtime of the fit
        @cron
        def start_fit():
            param.add('count', value=0, vary=False)
            param.add('correction_factor', value=1, vary=False)
            with open(os.devnull, 'w') as sys.stdout:
                first_residual = np.sum(f2min(param, S, fit_peaks, I, lims, 1)**2)
                param['count'].set(value=0)
            sys.stdout = sys.__stdout__
            minner = l.Minimizer(f2min, param, fcn_args=(S, fit_peaks, I, lims, first_residual))
            if method == 'leastsq' or method == 'least_squares':
                result = minner.minimize(method='leastsq', max_nfev=int(itermax), ftol=fit_tol)
            else:
                result = minner.minimize(method=method, max_nfev=int(itermax), tol=fit_tol)
            print(f'{result.message} Number of function evaluations: {result.nfev}.')
            return result
        # Do the fit
        result = start_fit()
        # Unpack the fitted values
        popt = result.params.valuesdict()

        # Replace the initial values with the fitted ones
        fit_peaks = peaks_frompar(fit_peaks, popt)
        # Correct the intensities
        #   Get the correct ones
        r_i, I_corr = misc.molfrac([peak.k for _, peak in fit_peaks.items()])
        I *= I_corr * popt['correction_factor']
        #   Replace them
        for k, idx in enumerate(fit_peaks.keys()):
            fit_peaks[idx].k = r_i[k]

        # Write a section of the output file
        fit.write_vf(f'{filename}.fvf', fit_peaks, limits, I, prev)
        prev += Np

@cron
def voigt_fit_2D(x_scale, y_scale, data, parameters, lim_f1, lim_f2, acqus, N=None, procs=None, utol=(1,1), s1tol=(0,500), s2tol=(0,500), vary_b=False, logfile=None):
    """
    Function that performs the fit of a 2D peak using multiple components. 
    The program reads a parameter matrix, that contains:
        u1 /ppm, u2 /ppm, fwhm1 /Hz, fwhm2 /Hz, I /a.u., b
    in each row. The number of rows corresponds to the number of components used for the computation of the final signal.
    The function returns the analogue version of the parameters matrix, but with the optimized values.
    --------
    Parameters:
    - x_scale: 1darray
        ppm_f2 of the spectrum, full
    - y_scale: 1darray
        ppm_f1 of the spectrum, full
    - data: 2darray
        spectrum, full
    - parameters: 1darray or 2darray
        Matrix (# signals, 6). Read main caption.
    - lim_f2: tuple
        Trimming limits for x_scale
    - lim_f1: tuple
        Trimming limits for y_scale
    - acqus: dict
        Dictionary of acquisition parameters.
    - N: tuple of ints
        len(y_scale), len(x_scale). Used only if procs is None
    - procs: dict
        Dictionary of processing parameters.
    - utol: tuple of floats
        Tolerance for the chemical shifts (utol_f1, utol_f2). Values will be set to u1 +/- utol_f1, u2 +/- utol_f2
    - s1tol: tuple of floats
        Range of variations for the fwhm in f1, in Hz
    - s2tol: tuple of floats
        Range of variations for the fwhm in f2, in Hz
    - vary_b: bool
        Choose if to fix the b value or not
    - logfile: str or None
        Path to a file where to write the fit information. If it is None, they will be printed into standard output.
    -------
    Returns:
    - out_parameters: 2darray
        parameters, but with the optimized values.
    """

    def f2min(param, n_sgn, x_scale, y_scale, data_exp, lim_f2, lim_f1):
        """ 
        Cost function.
        --------
        Parameters:
        - param: lmfit.Parameters object
            Fit parameters. See fit_2D caption.
        - n_sgn: int
            Number of signals
        - x_scale: 1darray
            ppm_f2 of the spectrum, full
        - y_scale: 1darray
            ppm_f1 of the spectrum, full
        - data_exp: 2darray
            spectrum trimmed around the peak of interest
        - lim_f2: tuple
            Trimming limits for x_scale
        - lim_f1: tuple
            Trimming limits for y_scale
        --------
        Returns:
        - res: 2darray
            Experimental -  calculated
        """
        # Extract dictionary of values from param
        P = param.valuesdict()

        # Organize the parameters into a matrix
        in_parameters = np.array([[
                P[f'u1_{i+1}'],
                P[f'u2_{i+1}'],
                P[f's1_{i+1}'],
                P[f's2_{i+1}'],
                P[f'k_{i+1}'] * P['A'],
                P[f'b_{i+1}'],
                ] for i in range(n_sgn)])

        # Feed the peaks to build_2D_sgn to make the calculated spectrum
        calc_peak = fit.build_2D_sgn(in_parameters, acqus, N=N, procs=procs)
        # Trim the calculated spectrum to the experimental one's sizes
        xtrim, ytrim, data_calc = misc.trim_data_2D(x_scale, y_scale, calc_peak, lim_f2, lim_f1)
        # Compute the residuals
        res = data_exp - data_calc
        return res

    #---------------------------------------------------------------------------------------------

    # Redirect the output to logfile, if logfile is given
    if isinstance(logfile, str):    # Open the file in "append" mode
        sys.stdout = open(logfile, 'a', buffering=1)
    elif isinstance(logfile, io.TextIOWrapper): # Just redirect
        sys.stdout = logfile

    # Trim the spectrum according to the given limits
    data_exp = misc.trim_data_2D(x_scale, y_scale, data, lim_f2, lim_f1)[-1]


    # Organize parameters
    parameters = np.array(parameters)
    if len(parameters.shape) == 1:  # it means it is only one signal
        parameters = parameters.reshape(1,-1)   # therefore transform in 1 x n matrix
    n_sgn = parameters.shape[0]     # Number of signals: number of rows of parameters

    # Express relative intensities in "molar fractions" and adjust the absolute intensity accordingly
    k_values, A = misc.molfrac(parameters[...,4])   

    # Initialize the Parameters object
    param = l.Parameters()

    param.add('A', value=A, vary=False)     # Absolute intensity
    for i in range(n_sgn):
        param.add(f'u1_{i+1}', value=parameters[i,0])   # chemical shift f1 /ppm
        param.add(f'u2_{i+1}', value=parameters[i,1])   # chemical shift f2 /ppm
        param.add(f's1_{i+1}', value=parameters[i,2])   # fwhm f1 /Hz
        param.add(f's2_{i+1}', value=parameters[i,3])   # fwhm f2 /Hz
        param.add(f'k_{i+1}', value=k_values[i])        # relative intensity
        param.add(f'b_{i+1}', value=parameters[i,5], min=0-1e-5, max=1+1e-5)   # Fraction of gaussianity
    
    # Set limits to u and s
    u1tol, u2tol = utol # Unpack tolerances for chemical shifts
    [param[f'u1_{i+1}'].set(min=param[f'u1_{i+1}'].value - u1tol) for i in range(n_sgn)]    # min u1
    [param[f'u1_{i+1}'].set(max=param[f'u1_{i+1}'].value + u1tol) for i in range(n_sgn)]    # max u1
    [param[f'u2_{i+1}'].set(min=param[f'u2_{i+1}'].value - u2tol) for i in range(n_sgn)]    # min u2
    [param[f'u2_{i+1}'].set(max=param[f'u2_{i+1}'].value + u2tol) for i in range(n_sgn)]    # max u2
    [param[f's1_{i+1}'].set(min=min(s1tol)) for i in range(n_sgn)]  # min fwhm1
    [param[f's1_{i+1}'].set(max=max(s1tol)) for i in range(n_sgn)]  # max fwhm1
    [param[f's2_{i+1}'].set(min=min(s2tol)) for i in range(n_sgn)]  # min fwhm2
    [param[f's2_{i+1}'].set(max=max(s2tol)) for i in range(n_sgn)]  # max fwhm2
    [param[f'k_{i+1}'].set(min=0) for i in range(n_sgn)]    # min rel int to 0
    [param[f'k_{i+1}'].set(max=5) for i in range(n_sgn)]    # max rel int to 5
    if vary_b is False:    # fix it
        [param[f'b_{i+1}'].set(vary=False) for i in range(n_sgn)]
        
    # Initialize the fit
    minner = l.Minimizer(f2min, param, fcn_args=(n_sgn, x_scale, y_scale, data_exp, lim_f2, lim_f1))
    result = minner.minimize(method='leastsq', max_nfev=10000, xtol=1e-10, ftol=1e-10)

    # Adjust relative and absolute intensities, then update the optimized Parameters object
    k_opt, A_opt = misc.molfrac([result.params.valuesdict()[f'k_{i+1}'] * param['A'] for i in range(n_sgn)])
    [result.params[f'k_{i+1}'].set(value=k_opt[i]) for i in range(n_sgn)]
    result.params['A'].set(value=A_opt) 

    def calc_uncert(param, n_sgn, keys_list=None):
        """ Get the stderr from Parameters and organize them in a matrix. """
        if keys_list is None:   # Take all of them
            keys_list = list(set([f'{key.rsplit("_", 1)[0]}' for key in param.keys()]))

        uncert = []
        for i in range(n_sgn):
            tmp_stderr = []
            for key in keys_list:
                tmp_key = f'{key}_{i+1}'
                try:    # There might not be stderr (NaN)
                    tmp_stderr.append(param[tmp_key].stderr)
                except Exception as E: # Therefore append None instead of NaN
                    print(E, tmp_key)
                    tmp_stderr.append(None)
            uncert.append(tmp_stderr)
        uncert = np.array(uncert)
        return uncert

    # Parameters
    keys_list = 'u1', 'u2', 's1', 's2', 'k', 'b'
    stderr = calc_uncert(result.params, n_sgn, keys_list)   # Uncertainties

    # Get dictionary of parameters
    popt = result.params.valuesdict()
    # Organize them in a matrix
    out_parameters = np.array([[
        popt[f'u1_{i+1}'],
        popt[f'u2_{i+1}'],
        popt[f's1_{i+1}'],
        popt[f's2_{i+1}'],
        popt[f'k_{i+1}'] * popt['A'],   # Put absolute intensity of each component
        popt[f'b_{i+1}'],
        ] for i in range(n_sgn)])

    # Print the outcome of the fit
    print(f'{result.message} Number of function evaluations: {result.nfev:5.0f}.\nEstimated uncertainties:')
    print(f'{"#":<4s},', *[f'{key:>8s},' for key in keys_list]) 
    for k in range(stderr.shape[0]):
        print(f'{k+1:<4},', *[f'{value:8.3g},' if value is not None else f'{"None":>8s}' for value in stderr[k]]) 
    print('')

    return out_parameters

# ----------------------------------------------------------------------------------------------------------


def smooth_spl(x, y, s_f=1, size=0, weights=None):
    """
    Fit the input data with a 3rd-order spline, given the smoothing factor to be applied.
    -------
    Parameters:
    - x: 1darray
        Location of the experimental points
    - y: 1darray
        Input data to be fitted
    - s_f: float
        Smoothing factor of the spline. 0=best straight line, 1=native spline.
    - size: int
        Size of the spline. If size=0, the same dimension as y is chosen.
    -------
    Returns:
    - x_s: 1darray
        Location of the spline data points.
    - y_s: 1darray
        Spline that fits the data.
    """
    # Reverse x and y if x is descending
    if x[0] > x[-1]:
        x_o = np.copy(x[::-1])
        y_o = np.copy(y[::-1])
        if weights is not None:
            weights = weights[::-1]
    else:
        x_o = np.copy(x)
        y_o = np.copy(y)

    # If size is not given, make the spline with the same size as the observed data
    if size:
        x_s = np.linspace(x_o[0], x_o[-1], size)
    else:
        x_s = np.linspace(x_o[0], x_o[-1], x.shape[-1])

    # Compute the spline
    if np.iscomplexobj(y_o):    # Treat real and imaginary part separately, then join them together
        y_sr = csaps(x_o, y_o.real, x_s, weights=weights, smooth=s_f)
        y_si = csaps(x_o, y_o.imag, x_s, weights=weights, smooth=s_f)
        y_s = y_sr + 1j*y_si
    else:   # Normal spline smoothing
        y_s = csaps(x_o, y_o, x_s, weights=weights, smooth=s_f)

    # Reverse the spline if you reversed the observed data
    if x[0] > x[-1]:
        x_s = x_s[::-1]
        y_s = y_s[::-1]
    return x_s, y_s

def interactive_smoothing(x, y, cmap='RdBu'):
    """
    Interpolate the given data with a 3rd-degree spline. Type the desired smoothing factor in the box and see the outcome directly on the figure.
    When the panel is closed, the smoothed function is returned.
    -------
    Parameters:
    - x: 1darray
        Scale of the data
    - y: 1darray
        Data to be smoothed
    - cmap: str
        Name of the colormap to be used to represent the weights
    --------
    Returns:
    - sx: 1darray
        Location of the spline points        
    - sy: 1darray
        Smoothed y
    - s_f: float
        Employed smoothing factor for the spline
    - weights: 1darray
        Weights vector
    """
    cmap = CM[f'{cmap}']        # Read the colormap

    # Get the limits for the figure
    lims = x[0], x[-1]

    # Initialize data 
    s_f = 0.95                      # Smoothing factor
    size = x.shape[-1]              # Spline size
    weights = np.ones_like(x) * 0.5 # Weights vector
    sx, sy = fit.smooth_spl(x, y, size=size, s_f=s_f, weights=weights)  # Calculate starting spline

    # Make the widgets
    #   Smoothing factor textbox
    sf_box = plt.axes([0.25, 0.04, 0.1, 0.06])
    sf_tb = TextBox(sf_box, 'Insert\nSmoothing factor', textalignment='center')

    #   Size textbox
    size_box = plt.axes([0.60, 0.04, 0.1, 0.06])
    size_tb = TextBox(size_box, 'Insert\nSize', textalignment='center')

    #   Weights slider
    slider_box = plt.axes([0.90, 0.15, 0.01, 0.8])
    weight_slider = Slider(
        ax=slider_box,
        label = 'Weight',
        valmin = 1e-5,
        valmax = 1,
        valinit = 0.5,
        valstep = 0.05, 
        orientation = 'vertical'
        )

    #   Colorbar for the weights
    cbar_box = plt.axes([0.94, 0.15, 0.02, 0.8])
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)  # Dummy values to plot the colorbar
    plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_box, orientation='vertical')

    # --------------------------------------------------------------------------------------------------------
    # Functions connected to the widgets

    def update_plot():
        """ Redraw the spline """
        sx, sy = fit.smooth_spl(x, y, size=size, s_f=s_f, weights=weights)
        s_plot.set_data(sx, sy.real)
        plt.draw()

    def update_size(text):
        """ Update size, write it, call update_plot"""
        nonlocal size
        try:
            size = int(eval(text))
        except:
            pass
        size_text.set_text('Size:\n{:.0f}'.format(size))
        update_plot()

    def update_sf(text):
        """ Update s_f, write it, call update_plot"""
        nonlocal s_f
        try:
            s_f = eval(text)
        except:
            pass
        s_text.set_text('Smoothing factor:\n{:.4f}'.format(s_f))
        update_plot()

    def onselect(*event):
        """ Stupid function connected to both mouse click and spanselector """
        if len(event) > 1:  # = Selector, hence OK
            span.set_visible(True)
        else:   # = Mousebutton?
            event, = event  # It is a tuple! Unpack
            if event.inaxes == ax and event.button == 1:
                # Only if you click inside the figure with left button
                span.set_visible(True)
        fig.canvas.draw()

    def update_bg_color(weights):
        """ Draw the figure background according to the weight vector """
        [fill.set_fc(cmap(q)) for fill, q in zip(tmp_fill, weights)]
        fig.canvas.draw()
        
    def press_space(key):
        """ When you press 'space' """
        if key.key == ' ':
            nonlocal weights
            span.set_visible(False)                         # Hide the spanselector
            xmin, xmax = span.extents                       # Get the shaded area
            # Get indexes on x of the shaded area, and sort them
            imin, _ = misc.ppmfind(x, xmin)                 
            imax, _ = misc.ppmfind(x, xmax)
            imin, imax = min(imin, imax), max(imin, imax)
            # Set the weights according to the value set on the slider
            weights[imin:imax] = weight_slider.val
            # Draw the background and the spline
            update_bg_color(weights)
            update_plot()

    def mouse_scroll(event):
        """ Control slider with the mouse scroll """
        valstep = 0.05
        sl_lims = 1e-5, 1
        if event.button == 'up':
            if weight_slider.val < sl_lims[1]:
                weight_slider.set_val(weight_slider.val + valstep)
            else:
                weight_slider.set_val(sl_lims[1])
        elif event.button == 'down':
            if weight_slider.val > sl_lims[0]:
                weight_slider.set_val(weight_slider.val - valstep)
            else:
                weight_slider.set_val(sl_lims[0])

    # --------------------------------------------------------------------------------------------------------

    # Make the figure
    fig = plt.figure('Interactive Smoothing with spline')
    fig.set_size_inches(figures.figsize_large)
    plt.subplots_adjust(left=0.1, right=0.85, top=0.95, bottom=0.15)
    ax = fig.add_subplot()

    ax.set_title('Press SPACE to set the weights')

    # Background
    tmp_fill = [ax.axvspan(x[k-1], x[k], ec=None, fc=cmap(q), alpha=0.25) for k, q in enumerate(weights) if k != 0]

    # Plot things
    ax.plot(x, y.real, c='tab:blue', lw=0.9, label='Original')
    s_text = plt.text(0.45, 0.07, 'Smoothing factor:\n{:.5f}'.format(s_f), fontsize=16, ha='center', va='center', transform=fig.transFigure)
    size_text = plt.text(0.75, 0.07, 'Size:\n{:.0f}'.format(size), fontsize=16, ha='center', va='center', transform=fig.transFigure)
    s_plot, = ax.plot(sx, sy.real, c='tab:red', lw=0.8, label='Smoothed')

    # Adjust figure display
    misc.pretty_scale(ax, lims, 'x')
    misc.pretty_scale(ax, ax.get_ylim(), 'y')
    misc.mathformat(ax)
    misc.set_fontsizes(ax, 16)
    ax.legend(fontsize=12)

    # Connect widget to function
    sf_tb.on_submit(update_sf)
    size_tb.on_submit(update_size)

    # Declare span selector
    span = SpanSelector(ax, onselect, "horizontal", useblit=True,
        props=dict(alpha=0.25, facecolor="tab:blue"), interactive=True, drag_from_anywhere=True)
    
    # Press space and mouse left button
    fig.canvas.mpl_connect('key_press_event', press_space)
    fig.canvas.mpl_connect('button_press_event', onselect)
    fig.canvas.mpl_connect('scroll_event', mouse_scroll)

    plt.show()

    # Compute final output
    sx, sy = fit.smooth_spl(x, y, size=size, s_f=s_f, weights=weights)

    return sx, sy, s_f, weights

def build_baseline(ppm_scale, C, L=None):
    """
    Builds the baseline calculating the polynomion with the given coefficients, and summing up to the right position.
    -------
    Parameters:
    - ppm_scale: 1darray
        ppm scale of the spectrum
    - C: list
        Parameters coefficients. No baseline corresponds to False.
    - L: list
        List of window regions. If it is None, the baseline is built on the whole ppm_scale
    -------
    Returns:
    - baseline: 1darray
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
        lims = misc.ppmfind(ppm_scale, L[k][0])[0], misc.ppmfind(ppm_scale, L[k][1])[0] # Find the indexes on ppm_scale
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
    -------
    Parameters:
    - filenames: list
        List of directories of the input files.
    - ppm_scale: 1darray
        ppm scale of the spectrum. Used to build the baseline
    - joined_name: str or None
        If it is not None, concatenates the files in the list 'filenames' and saves them in a single file named 'joined_name'.
    -------
    Returns:
    - V: 2darray
        Array of joined signal parameters        
    - C: list
        Parameters coefficients. No baseline corresponds to False.
    - L: list
        List of window regions.
    - baseline: 1darray
        Baseline built from C and L.
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
        lims = misc.ppmfind(ppm_scale, L[k][0])[0], misc.ppmfind(ppm_scale, L[k][1])[0] # Find the indexes on ppm_scale
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
    -------
    Parameters:
    - ppm_scale: 1darray
        PPM scale of the spectrum
    - limits: tuple
        (left, right) in ppm
    - t_AQ: 1darray
        Acquisition timescale
    - SFO1: float
        Larmor frequency of the nucleus /ppm
    - o1p: float
        Pulse carrier frequency /ppm
    - N: int
        Size of the final spectrum.
    - V: 2darray
        Matrix containing the values to build the signals.
    - C: 1darray
        Baseline polynomion coefficients. False to not use the baseline
    -------
    Returns:
    - sgn: list
        Voigt signals built using V
    - Total: 1darray
        sum of all the sgn
    - baseline: 1darray
        Polynomion built using C. False if C is False.
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
        sgn.append(fit.make_signal(t_AQ, V[i,0], V[i,1], V[i,2], V[i,3], V[i,4], V[i,5], SFO1=SFO1, o1p=o1p, N=N))
        Total += sgn[i][lim1:lim2]

    return sgn, Total, baseline

def integrate(ppm0, data0, X_label=r'$\delta\,$F1 /ppm'):
    """
    Allows interactive integration of a NMR spectrum through a dedicated GUI. Returns the values as a dictionary, where the keys are the selected regions truncated to the 2nd decimal figure.
    The returned dictionary contains pre-defined keys, as follows:
        > total:    total integrated area
        > ref_pos:  location of the reference peak /ppm1:ppm2
        > ref_int:  absolute integral of the reference peak
        > ref_val:  for how many nuclei the reference peak integrates
    The absolute integral of the x-th peak, I_x, must be calculated according to the formula:
        I_x = I_x(relative) * ref_int / ref_val
    --------
    Parameters:
    - ppm: 1darray
        PPM scale of the spectrum
    - data: 1darray
        Spectrum to be integrated.
    - X_label: str
        Label of the x-axis
    -------
    Returns:
    - f_vals: dict
        Dictionary containing the values of the integrated peaks.
    """

    # Copy to prevent overwriting
    ppm = np.copy(ppm0)
    data = np.copy(data0)

    # Calculate the total integral function
    int_f = processing.integral(data, ppm)

    # Make the figure
    fig = plt.figure('Spectrum Integration')
    fig.set_size_inches(figures.figsize_large)
    plt.subplots_adjust(left=0.10, bottom=0.15, top=0.90, right=0.80)
    ax = fig.add_subplot()

    # Make boxes for buttons
    add_box = plt.axes([0.875, 0.80, 0.05, 0.06])
    setref_box = plt.axes([0.825, 0.72, 0.075, 0.06])
    save_box = plt.axes([0.875, 0.20, 0.05, 0.06])
    # Make box for tbox
    reftb_box = plt.axes([0.925, 0.72, 0.05, 0.06])
    # Make buttons
    add_button = Button(add_box, 'ADD', hovercolor='0.875')
    save_button = Button(save_box, 'SAVE', hovercolor='0.875')
    setref_button = Button(setref_box, 'SET REF', hovercolor='0.875')
    # Make tbox
    ref_tbox = TextBox(ax=reftb_box, label='', initial='{}'.format(1), textalignment='center')

    # Declare variables
    f_vals = {      # Initialize output variable
            'total': float(0),              # Total integrated area
            'ref_pos': '{:.2f}:{:.2f}'.format(ppm[0], ppm[-1]), # Position of the reference signal /ppm1:ppm2
            'ref_val': float(1),            # For how many nuclei the reference peak integrates
            'ref_int': int_f[-1]-int_f[0],            # Reference peak integral, absolute value
            }      
    abs_vals = {}                           # dictionary: integrals of the peaks, absolute values
    text_integrals={}                       # dictionary: labels to keep record of the integrals

    # ---------------------------------------------------------------------------------------
    # Functions connected to the widgets
    def redraw_labels(f_vals):
        """ Computes the relative integrals and updates the texts on the plot """
        corr_func = f_vals['ref_val'] / f_vals['ref_int']   # Correction 

        # Update all the integral texts according to the new total_integral value
        tmp_text.set_text('{:.5f}'.format(tmp_plot.get_ydata()[-1] * corr_func))      # Relative value of the integral: under the red label on the right
        for key, value in abs_vals.items():
            text_integrals[key].set_text('{:.4f}'.format(abs_vals[key] * corr_func))

        fig.canvas.draw()

    def set_ref_val(xxx):
        """ Function of the textbox """
        nonlocal f_vals
        f_vals['ref_val'] = eval(xxx)
        redraw_labels(f_vals)

    def set_ref_int(event):
        nonlocal f_vals
        tmp_plot.set_visible(False)                     # Set the integral function as invisible so that it does not overlay with the permanent one
        xdata, ydata = tmp_plot.get_data()              # Get the data from the red curve

        f_vals['ref_int'] = ydata[-1]                   # Calculate the integral and cast it to the correct entry in f_vals
        f_vals['ref_pos'] = '{:.2f}:{:.2f}'.format(xdata[0], xdata[-1]) # Get reference peak position from the plot and save it in f_vals

        # Update the plot
        ref_plot.set_data(xdata, ydata) # Draw permanent integral function, in blue
        ref_plot.set_visible(True)      # Because at the beginning it is invisible

        ref_text.set_text('{:.4e}'.format(f_vals['ref_int']))   # Update label under the blue label on the right
        redraw_labels(f_vals)


    def onselect(vsx, vdx):
        """ When you drag and release """
        tmp_total_integral = np.copy(f_vals['total'])   # Copy total_integral inside

        corr_func = f_vals['ref_val'] / f_vals['ref_int']   # Correction function

        sx, dx = max(vsx, vdx), min(vsx, vdx)           # Sort the borders of the selected window
        # Take indexes of the borders of the selected window and sort them
        isx = misc.ppmfind(ppm, sx)[0]
        idx = misc.ppmfind(ppm, dx)[0]
        isx, idx = min(isx, idx), max(isx, idx)

        # Compute the integral
        int_fun = processing.integral(data, ppm, (sx, dx))  # Integral function
        int_val = int_fun[-1] - int_fun[0]                                  # Value of the integral
        tmp_total_integral += int_val                                       # Update the total integral, but only inside

        # Update the plot
        tmp_plot.set_data(ppm[isx:idx], int_fun)                            # Plot the integral function on the peak, normalized
        tmp_plot.set_visible(True)                                          # Set this plot as visible, if it is not

        tmp_text.set_text('{:.5f}'.format(int_val * corr_func))             # Relative value of the integral: under the red label on the right
        tot_text.set_text('{:.4e}'.format(tmp_total_integral))              # Total integrated area: under the green label on the right
        fig.canvas.draw()

    def f_add(event):
        """ When you click 'ADD' """
        nonlocal f_vals, abs_vals

        tmp_plot.set_visible(False)                     # Set the integral function as invisible so that it does not overlay with the permanent one
        xdata, ydata = tmp_plot.get_data()              # Get the data from the red curve

        # Update the variables
        f_vals['total'] += ydata[-1]
        abs_vals['{:.2f}:{:.2f}'.format(xdata[0], xdata[-1])] = ydata[-1]

        # Update the plot
        ax.plot(xdata, ydata, c='tab:green', lw=0.8)    # Draw permanent integral function
        tot_text.set_text('{:.4e}'.format(f_vals['total']))  # Text under green label on the right

        xtext = (xdata[0] + xdata[-1]) / 2      # x coordinate of the text: centre of the selected window
        text_integrals['{:.2f}:{:.2f}'.format(xdata[0], xdata[-1])] = ax.text(xtext, ax.get_ylim()[-1], '{:.5f}'.format(ydata[-1]), horizontalalignment='center', verticalalignment='bottom', fontsize=10, rotation=60)     # Add whatever to the label

        # Update all the integral texts according to the new total_integral value
        redraw_labels(f_vals)
        
    def f_save(event):
        """ When you click 'SAVE' """
        nonlocal f_vals     # to update the output variable
        # Append in the dictionary the relative values of the integrals
        for key, value in abs_vals.items():     
            f_vals[key] = value * f_vals['ref_val'] / f_vals['ref_int']
        plt.close()

    # ---------------------------------------------------------------------------------------
    
    # Add things to the figure panel

    ax.plot(ppm, data, c='tab:blue', lw=0.8)        # Spectrum
    tmp_plot, = ax.plot(ppm, int_f/max(int_f)*max(data), c='tab:red', lw=0.8, visible=False)    # Draw the total integral function but set to invisible because it is useless, needed as placeholder for the red curve
    ref_plot, = ax.plot(ppm, int_f/max(int_f)*max(data), c='b', lw=0.8, visible=False)    # Draw the total integral function but set to invisible because it is useless, needed as placeholder for the blue curve

    # Draw text labels in the figure, on the right
    ax.text(0.90, 0.68, 'Current integral (normalized)', horizontalalignment='center', verticalalignment='center', transform=fig.transFigure, fontsize=14, color='tab:red')
    tmp_text = ax.text(0.90, 0.65, '0', horizontalalignment='center', verticalalignment='center', transform=fig.transFigure, fontsize=14)
    ax.text(0.90, 0.60, 'Total integral', horizontalalignment='center', verticalalignment='center', transform=fig.transFigure, fontsize=14, color='tab:green')
    tot_text = ax.text(0.90, 0.55, '0', horizontalalignment='center', verticalalignment='center', transform=fig.transFigure, fontsize=14)
    ax.text(0.90, 0.50, 'Reference integral', horizontalalignment='center', verticalalignment='center', transform=fig.transFigure, fontsize=14, color='b')
    ref_text = ax.text(0.90, 0.45, '{:.4e}'.format(f_vals['ref_int']), horizontalalignment='center', verticalalignment='center', transform=fig.transFigure, fontsize=14)

    # Fancy shit
    ax.set_xlim(ppm[0], ppm[-1])
    ax.set_xlabel(X_label)
    ax.set_ylabel('Intensity /a.u.')
    misc.pretty_scale(ax, ax.get_xlim(), 'x')
    misc.pretty_scale(ax, ax.get_ylim(), 'y')
    misc.mathformat(ax, 'y')
    misc.set_fontsizes(ax, 14)

    # Add more widgets and connect the buttons to their functions
    cursor = Cursor(ax, c='tab:red', lw=0.8, horizOn=False)                                     # Vertical line that follows the cursor
    span = SpanSelector(ax, onselect, 'horizontal', props=dict(facecolor='tab:red', alpha=0.5)) # Draggable window
    add_button.on_clicked(f_add)
    save_button.on_clicked(f_save)
    setref_button.on_clicked(set_ref_int)
    ref_tbox.on_submit(set_ref_val)

    # Show the figure
    plt.show()

    return f_vals


def integrate_2D(ppm_f1, ppm_f2, data, SFO1, SFO2, fwhm_1=200, fwhm_2=200, utol_1=0.5, utol_2=0.5, plot_result=False):
    """ 
    Function to select and integrate 2D peaks of a spectrum, using dedicated GUIs.
    Calls integral_2D to do the dirty job.
    ---------
    Parameters:
    - ppm_f1: 1darray
        PPM scale of the indirect dimension
    - ppm_f2: 1darray 
        PPM scale of the direct dimension
    - data: 2darray 
        real part of the spectrum
    - SFO1: float
        Larmor frequency of the nucleus in the indirect dimension
    - SFO2: float
        Larmor frequency of the nucleus in the direct dimension
    - fwhm_1: float
        Starting FWHM /Hz in the indirect dimension
    - fwhm_2: float
        Starting FWHM /Hz in the direct dimension
    - utol_1: float
        Allowed tolerance for u_1 during the fit. (u_1-utol_1, u_1+utol_1)
    - utol_2: float
        Allowed tolerance for u_2 during the fit. (u_2-utol_2, u_2+utol_2)
    - plot_result: bool
        True to show how the program fitted the traces.
    --------
    Returns:
    - I: dict
        Computed integrals. The keys are '<ppm f1>:<ppm f2>' with 2 decimal figures.
    """

    # Get all the information that integral_2D needs
    peaks = misc.select_for_integration(ppm_f1, ppm_f2, data, Neg=True)

    I = {}      # Declare empty dictionary
    for P in peaks:
        # Extract trace F1
        T1 = misc.get_trace(data, ppm_f2, ppm_f1, P['f2']['u'], column=True)
        x_T1, y_T1 = misc.trim_data(ppm_f1, T1, *P['f1']['lim'])    # Trim according to the rectangle
        # Extract trace F2
        T2 = misc.get_trace(data, ppm_f2, ppm_f1, P['f1']['u'], column=False)
        x_T2, y_T2 = misc.trim_data(ppm_f2, T2, *P['f2']['lim'])    # Trim according to the rectangle

        # Compute the integrals
        I_p = processing.integral_2D(x_T1, y_T1, SFO1, x_T2, y_T2, SFO2,
                u_1=P['f1']['u'], fwhm_1=fwhm_1, utol_1=utol_1, 
                u_2=P['f2']['u'], fwhm_2=fwhm_2, utol_2=utol_2,
                plot_result=plot_result)

        # Store the integral in the dictionary
        I[f'{P["f2"]["u"]:.2f}:{P["f1"]["u"]:.2f}'] = I_p
    return I



class Peak:
    """
    Class to represent the characteristic parameters of an NMR peak, and to compute it.
    ----------
    Attributes:
    - t: 1darray
        Timescale for the FID
    - SFO1: float
        Nucleus Larmor frequency
    - o1p: float
        Carrier position
    - N: int
        Number of points of the spectrum, i.e. after eventual zero-filling
    - u: float
        Chemical shift /ppm
    - fwhm: float
        Linewidth /Hz
    - k: float
        Intensity, relative
    - b: float
        Fraction of gaussianity (b=0 equals pure lorentzian)
    - phi: float
        Phase /degrees
    - group: int
        Identifier for the component of a multiplet
    """
    def __init__(self, acqus, u=None, fwhm=5, k=1, b=0, phi=0, N=None, group=0):
        """
        Initialize the class with the configuration parameters, and with defauls values, if not given.
        ----------
        Parameters:
        - acqus: dict
            It should contain "t", "SFO1", "o1p", and "N"
        - u: float
            Chemical shift /ppm
        - fwhm: float
            Linewidth /Hz
        - k: float
            Intensity, relative
        - b: float
            Fraction of gaussianity (b=0 equals pure lorentzian)
        - phi: float
            Phase /degrees
        - N: int
            Number of points of the spectrum, i.e. after eventual zero-filling. None means to not zero-fill
        - group: int
            Identifier for the component of a multiplet
        """
        # Unpack the acqus dictionary
        self.t = misc.extend_taq(acqus['t1'], N)
        self.SFO1 = acqus['SFO1']
        self.o1p = acqus['o1p']
        self.N = N
        #self.I = I

        # Set the values as attributes
        if u is None:
            u = self.o1p
        self.u = u
        self.fwhm = fwhm
        self.k = k
        self.b = b
        self.phi = phi
        self.group = int(group)

    def __call__(self, A=1, cplx=False, get_fid=False):
        """
        Generates a voigt signal on the basis of the stored attributes, in the time domain. Then, makes the Fourier transform and returns it after the eventual zero-filling.
        ----------
        Parameters:
        - A: float
            Absolute intensity value
        - cplx: bool
            Returns the complex (True) or only the real part (False) of the signal
        - get_fid: bool
            If True, returns the FID instead of the transformed signal
        ----------
        Returns:
        - sgn : 1darray
            generated signal in the frequency domain
        """
        sgn = self.get_fid(A=A)
        if not get_fid:
            sgn = processing.ft(sgn)                # transform it
        if cplx or get_fid:
            return sgn
        else:
            return sgn.real

    def get_fid(self, A=1):
        """ 
        Compute and returns the FID encoding for that signal.
        ----------
        Parameters:
        - A: float
            Absolute intensity value
        ----------
        Returns:
        - sgn : 1darray
            generated signal in the time domain
        """
        v = misc.ppm2freq(self.u, self.SFO1, self.o1p)         # conversion to frequency units
        fwhm = self.fwhm * 2 * np.pi                    # conversion to radians
        phi = self.phi * np.pi / 180                    # conversion to radians
        sgn = sim.t_voigt(self.t, v, fwhm, A=A*self.k, phi=phi, b=self.b) # make the signal
        return sgn

    def par(self):
        """
        Creates a dictionary with the currently stored attributes and returns it.
        -----------
        Returns:
        - dic: dict
            Dictionary of parameters
        """
        dic = {
                'u': self.u,
                'fwhm': self.fwhm,
                'k': self.k,
                'b': self.b,
                'phi': self.phi,
                'group': self.group
                }
        return dict(dic)



def make_iguess(S_in, ppm_scale, t_AQ, SFO1=701.125, o1p=0, filename='i_guess'):
    """
    Creates the initial guess for a lineshape deconvolution fitting procedure, using a dedicated GUI.
    The GUI displays the experimental spectrum in black and the total function in blue.
    First, select the region of the spectrum you want to fit by focusing the zoom on it using the lens button.
    Then, use the "+" button to add components to the spectrum. The black column of text under the textbox will be colored with the same color of the active peak.
    Use the mouse scroll to adjust the parameters of the active peak. Write a number in the "Group" textbox to mark the components of the same multiplet.
    Group 0 identifies independent peaks, not part of a multiplet (default).
    The sensitivity of the mouse scroll can be regulated using the "up arrow" and "down arrow" buttons. 
    The active peak can be changed in any moment using the slider.

    When you are satisfied with your fit, press "SAVE" to write the information in the output file. Then, the GUI is brought back to the initial situation, and the region you were working on will be marked with a green rectangle. You can repeat the procedure as many times as you wish, to prepare the guess on multiple spectral windows.

    Keyboard shortcuts:
    > "increase sensitivity" : '>'
    > "decrease sensitivity" : '<'
    > mouse scroll up: 'up arrow key'
    > mouse scroll down: 'down arrow key'
    > "add a component": '+'
    > "remove the active component": '-'
    > "change component, forward": 'page up'
    > "change component, backward": 'page down'

    ------------
    Parameters:
    - S_in: 1darray
        Experimental spectrum
    - ppm_scale: 1darray
        PPM scale of the spectrum
    - t_AQ: 1darray
        Acquisition timescale
    - SFO1: float
        Nucleus Larmor frequency /MHz
    - o1p: float
        Carrier frequency /ppm
    - filename: str
        Path to the filename where to save the information. The '.ivf' extension is added automatically.
    """

    #-----------------------------------------------------------------------
    ## USEFUL STRUCTURES
    def rename_dic(dic, Np):
        """
        Change the keys of a dictionary with a sequence of increasing numbers, starting from 1.
        ----------
        Parameters:
        - dic: dict
            Dictionary to edit
        - Np: int
            Number of peaks, i.e. the sequence goes from 1 to Np
        ----------
        Returns:
        - new_dic: dict
            Dictionary with the changed keys
        """
        old_keys = list(dic.keys())         # Get the old keys
        new_keys = [int(i+1) for i in np.arange(Np)]    # Make the new keys
        new_dic = {}        # Create an empty dictionary
        # Copy the old element in the new dictionary at the correspondant point
        for old_key, new_key in zip(old_keys, new_keys):
            new_dic[new_key] = dic[old_key]
        del dic
        return new_dic

    def calc_total(peaks):
        """
        Calculate the sum trace from a collection of peaks stored in a dictionary.
        If the dictionary is empty, returns an array of zeros.
        ---------
        Parameters:
        - peaks: dict
            Components
        --------
        Returns:
        - total: 1darray
            Sum spectrum
        """
        # Get the arrays from the dictionary
        T = [p(A) for _, p in peaks.items()]
        if len(T) > 0:  # Check for any peaks
            total = np.sum(T, axis=0)
            return total
        else:
            return np.zeros_like(ppm_scale)

    #-------------------------------------------------------------------------------
    # Write the info on the file
    with open(f'{filename}.ivf', 'a', buffering=1) as f:
        now = datetime.now()
        date_and_time = now.strftime("%d/%m/%Y at %H:%M:%S")
        f.write('! Initial guess computed by {} on {}\n\n'.format(os.getlogin(), date_and_time))

    # Remove the imaginary part from the experimental data and make a shallow copy
    if np.iscomplexobj(S_in):
        S = np.copy(S_in).real
    else:
        S = np.copy(S_in)

    N = S.shape[-1]     # Number of points
    Np = 0              # Number of peaks
    lastgroup = 0       # Placeholder for last group added
    prev = 0            # Number of previous peaks

    # Make an acqus dictionary based on the input parameters.
    acqus = {'t1': t_AQ, 'SFO1': SFO1, 'o1p': o1p}

    # Set limits
    limits = [max(ppm_scale), min(ppm_scale)]
    
    # Get point indices for the limits
    lim1 = misc.ppmfind(ppm_scale, limits[0])[0]
    lim2 = misc.ppmfind(ppm_scale, limits[1])[0]
    # Calculate the absolute intensity (or something that resembles it)
    A = np.trapz(S[lim1:lim2], dx=misc.calcres(ppm_scale*SFO1))*2*misc.calcres(acqus['t1'])
    _A = 1 * A
    # Make a sensitivity dictionary
    sens = {
            'u': np.abs(limits[0] - limits[1]) / 50,    # 1/50 of the SW
            'fwhm': 2.5,
            'k': 0.05,
            'b': 0.1,
            'phi': 10,
            'A': 10**(np.floor(np.log10(A)-1))    # approximately
            }
    _sens = dict(sens)                          # RESET value
    # Peaks dictionary
    peaks = {}

    # Initial figure
    fig = plt.figure('Manual Computation of Inital Guess')
    fig.set_size_inches(15,8)
    plt.subplots_adjust(bottom=0.10, top=0.90, left=0.05, right=0.65)
    ax = fig.add_subplot(1,1,1)
    
    # make boxes for widgets
    slider_box = plt.axes([0.68, 0.10, 0.01, 0.65])     # Peak selector slider
    peak_box = plt.axes([0.72, 0.45, 0.10, 0.30])       # Radiobuttons
    up_box = plt.axes([0.815, 0.825, 0.08, 0.075])      # Increase sensitivity button
    down_box = plt.axes([0.894, 0.825, 0.08, 0.075])    # Decrease sensitivity button
    save_box = plt.axes([0.7, 0.825, 0.085, 0.04])      # Save button
    reset_box = plt.axes([0.7, 0.865, 0.085, 0.04])     # Reset button
    group_box = plt.axes([0.76, 0.40, 0.06, 0.04])      # Textbox for the group selection
    plus_box = plt.axes([0.894, 0.65, 0.08, 0.075])     # Add button
    minus_box = plt.axes([0.894, 0.55, 0.08, 0.075])    # Minus button
    
    # Make widgets
    #   Buttons
    up_button = Button(up_box, r'$\uparrow$', hovercolor = '0.975')    
    down_button = Button(down_box, r'$\downarrow$', hovercolor = '0.975')
    save_button = Button(save_box, 'SAVE', hovercolor = '0.975')
    reset_button = Button(reset_box, 'RESET', hovercolor = '0.975')
    plus_button = Button(plus_box, '$+$', hovercolor='0.975')
    minus_button = Button(minus_box, '$-$', hovercolor='0.975')

    #   Textbox
    group_tb = TextBox(group_box, 'Group', textalignment='center')
    
    #   Radiobuttons
    peak_name = [r'$\delta$ /ppm', r'$\Gamma$ /Hz', '$k$', '$x_{g}$', r'$\phi$', '$A$']
    peak_radio = RadioButtons(peak_box, peak_name, activecolor='tab:blue')      # Signal parameters
    
    #   Slider
    slider = Slider(ax=slider_box, label='Active\nSignal', valmin=0, valmax=1-1e-3, valinit=0, valstep=1e-10, orientation='vertical', color='tab:blue')


    #-------------------------------------------------------------------------------
    ## SLOTS

    def redraw():
        misc.pretty_scale(ax, ax.get_xlim(), 'x')
        plt.draw()

    def radio_changed(event):
        """ Change the printed value of sens when the radio changes """
        active = peak_name.index(peak_radio.value_selected)
        param = list(sens.keys())[active]
        write_sens(param)

    def up_sens(event):
        """ Doubles sensitivity of active parameter """
        nonlocal sens
        active = peak_name.index(peak_radio.value_selected)
        param = list(sens.keys())[active]
        sens[param] *= 2
        write_sens(param)

    def down_sens(event):
        """ Halves sensitivity of active parameter """
        nonlocal sens
        active = peak_name.index(peak_radio.value_selected)
        param = list(sens.keys())[active]
        sens[param] /= 2
        write_sens(param)

    def up_value(param, idx):
        """ Increase the value of param of idx-th peak """
        if param == 'A':        # It is outside the peaks dictionary!
            nonlocal A
            A += sens['A']
        else:
            nonlocal peaks
            peaks[idx].__dict__[param] += sens[param]
            # Make safety check for b
            if peaks[idx].b > 1:
                peaks[idx].b = 1

    def down_value(param, idx):
        """ Decrease the value of param of idx-th peak """
        if param == 'A':    # It is outside the peaks dictionary!
            nonlocal A
            A -= sens['A']
        else:
            nonlocal peaks
            peaks[idx].__dict__[param] -= sens[param]
            # Safety check for fwhm
            if peaks[idx].fwhm < 0:
                peaks[idx].fwhm = 0
            # Safety check for b
            if peaks[idx].b < 0:
                peaks[idx].b = 0

    def scroll(event):
        """ Connection to mouse scroll """
        if Np == 0: # No peaks!
            return
        # Get the active parameter and convert it into Peak's attribute
        active = peak_name.index(peak_radio.value_selected)
        param = list(sens.keys())[active]
        # Get the active peak
        idx = int(np.floor(slider.val * Np) + 1)
        # Fork for up/down
        if event.button == 'up':
            up_value(param, idx)
        if event.button == 'down':
            down_value(param, idx)

        # Recompute the components
        for k, _ in enumerate(peaks):
            p_sgn[k+1].set_ydata(peaks[k+1](A)[lim1:lim2])
        # Recompute the total trace
        p_fit.set_ydata(calc_total(peaks)[lim1:lim2])
        # Update the text
        write_par(idx)
        redraw()

    def write_par(idx):
        """ Write the text to keep track of your amounts """
        if idx:     # Write the things
            dic = dict(peaks[idx].par())
            dic['A'] = A
            # Update the text
            values_print.set_text('{u:+7.3f}\n{fwhm:5.3f}\n{k:5.3f}\n{b:5.3f}\n{phi:+07.3f}\n{A:5.2e}\n{group:5.0f}'.format(**dic))
            # Color the heading line of the same color of the trace
            head_print.set_color(p_sgn[idx].get_color())
        else:   # Clear the text and set the header to be black
            values_print.set_text('')
            head_print.set_color('k')

    def write_sens(param):
        """ Updates the current sensitivity value in the text """
        text = f'Sensitivity: $\\pm${sens[param]:10.4g}'
        # Update the text
        sens_print.set_text(text)
        # Redraw the figure
        plt.draw()


    def set_group(text):
        """ Set the attribute 'group' of the active signal according to the textbox """
        nonlocal peaks
        if not Np:  # Clear the textbox and do nothing more
            group_tb.text_disp.set_text('')
            plt.draw()
            return
        # Get active peak
        idx = int(np.floor(slider.val * Np) + 1)
        try:
            group = int(eval(text))
        except:
            group = peaks[idx].group
        group_tb.text_disp.set_text('')
        peaks[idx].group = group
        write_par(idx)
        redraw()

    def selector(event):
        """ Update the text when you move the slider """
        idx = int(np.floor(slider.val * Np) + 1)
        if Np:
            for key, line in p_sgn.items():
                if key == idx:
                    line.set_lw(3)
                else:
                    line.set_lw(0.8)
            write_par(idx)
        redraw()

    def key_binding(event):
        """ Keyboard """
        key = event.key
        if key == '<':
            down_sens(0)
        if key == '>':
            up_sens(0)
        if key == '+':
            add_peak(0)
        if key == '-':
            remove_peak(0)
        if key == 'pagedown':
            if slider.val - slider.valstep >= 0:
                slider.set_val(slider.val - slider.valstep)
            selector(0)
        if key == 'pageup':
            if slider.val + slider.valstep < 1:
                slider.set_val(slider.val + slider.valstep)
            selector(0)
        if key == 'up' or key == 'down':
            event.button = key
            scroll(event)

    def reset(event):
        """ Return everything to default """
        nonlocal Np, peaks, p_sgn, A, sens
        Q = Np
        for k in range(Q):
            remove_peak(event)
        A = _A
        sens = dict(_sens)
        ax.set_xlim(*_xlim)
        ax.set_ylim(*_ylim)
        redraw()

    def add_peak(event):
        """ Add a component """
        nonlocal Np, peaks, p_sgn
        # Increase the number of peaks
        Np += 1 
        # Add an entry to the dictionary labelled as last
        peaks[Np] = fit.Peak(acqus, u=np.mean(ax.get_xlim()), N=N, group=lastgroup)
        # Plot it and add the trace to the plot dictionary
        p_sgn[Np] = ax.plot(ppm_scale[lim1:lim2], peaks[Np](A)[lim1:lim2], lw=0.8)[-1]
        # Move the slider to the position of the new peak
        slider.set_val( (Np - 1) / Np )
        # Recompute the step of the slider
        slider.valstep = 1 / Np
        # Calculate the total trace with the new peak
        total = calc_total(peaks)
        # Update the total trace plot
        p_fit.set_ydata(total[lim1:lim2])
        # Update the text
        write_par(Np)
        redraw()

    def remove_peak(event):
        """ Remove the active component """
        nonlocal Np, peaks, p_sgn
        if Np == 0:
            return
        # Get the active peak
        idx = int(np.floor(slider.val * Np) + 1)
        # Decrease Np of 1
        Np -= 1
        # Delete the entry from the peaks dictionary
        _ = peaks.pop(idx)
        # Remove the correspondant line from the plot dictionary
        del_p = p_sgn.pop(idx)
        # Set it invisible because I cannot truly delete it
        del_p.set_visible(False)
        del del_p   # ...at least clear some memory
        # Change the labels to the dictionary
        peaks = rename_dic(peaks, Np)
        p_sgn = rename_dic(p_sgn, Np)
        # Calculate the total trace without that peak
        total = calc_total(peaks)
        # Update the total trace plot
        p_fit.set_ydata(total[lim1:lim2])
        # Change the slider position
        if Np == 0: # to zero and do not make it move
            slider.set_val(0)
            slider.valstep = 1e-10
            write_par(0)
        elif Np == 1:   # To zero and that's it
            slider.set_val(0)
            slider.valstep = 1 / Np
            write_par(1)
        else:   # To the previous point
            if idx == 1:
                slider.set_val(0)
            else:
                slider.set_val( (idx - 2) / Np)     # (idx - 1) -1
            slider.valstep = 1 / Np
            write_par(int(np.floor(slider.val * Np) + 1))
        redraw()

    def save(event):
        """ Write a section in the output file """
        nonlocal prev
        # Adjust the intensities
        fit.write_vf(f'{filename}.ivf', peaks, ax.get_xlim(), A, prev)
        prev += len(peaks)
        
        # Mark a region as "fitted" with a green box
        ax.axvspan(*ax.get_xlim(), color='tab:green', alpha=0.1)
        # Call reset to return at the initial situation
        reset(event)

    #-------------------------------------------------------------------------------


    ax.plot(ppm_scale[lim1:lim2], S[lim1:lim2], label='Experimental', lw=1.0, c='k')  # experimental
    p_fit = ax.plot(ppm_scale[lim1:lim2], np.zeros_like(S)[lim1:lim2], label='Fit', lw=0.9, c='b')[-1]  # Total trace
    p_sgn = {}  # Components
    
    # Header for current values print
    head_print = ax.text(0.75, 0.35, 
            '{:>7s}\n{:>5}\n{:>5}\n{:>5}\n{:>7}\n{:>7}\n{:>7}'.format(
                r'$\delta$', r'$\Gamma$', '$k$', '$b$', 'Phase', '$A$', 'Group'),
            ha='right', va='top', transform=fig.transFigure, fontsize=14, linespacing=1.5)
    # Text placeholder for the values - linspacing is different to align with the header
    values_print = ax.text(0.85, 0.35, '',
            ha='right', va='top', transform=fig.transFigure, fontsize=14, linespacing=1.55)
    # Text to display the active sensitivity values
    sens_print = ax.text(0.875, 0.775, f'Sensitivity: $\\pm${sens["u"]:10.4g}',
            ha='center', va='bottom', transform=fig.transFigure, fontsize=14)
    # Text to remind keyboard shortcuts
    t_uparrow = r'$\uparrow$'
    t_downarrow = r'$\downarrow$'
    keyboard_text = '\n'.join([
        f'{"KEYBOARD SHORTCUTS":^50s}',
        f'{"Key":>5s}: Action',
        f'-'*50,
        f'{"<":>5s}: Decrease sens.',
        f'{">":>5s}: Increase sens.',
        f'{"+":>5s}: Add component',
        f'{"-":>5s}: Remove component',
        f'{"Pg"+t_uparrow:>5s}: Change component, up',
        f'{"Pg"+t_downarrow:>5s}: Change component, down',
        f'{t_uparrow:>5s}: Increase value',
        f'{t_downarrow:>5s}: Decrease value',
        f'-'*50,
        ])
    keyboard_print = ax.text(0.86, 0.025, keyboard_text, 
            ha='left', va='bottom', transform=fig.transFigure, fontsize=8, linespacing=1.55)

    # make pretty scales
    ax.set_xlim(max(limits[0],limits[1]),min(limits[0],limits[1]))
    misc.pretty_scale(ax, ax.get_xlim(), axis='x', n_major_ticks=10)
    misc.pretty_scale(ax, ax.get_ylim(), axis='y', n_major_ticks=10)
    misc.mathformat(ax)

    # RESET values for xlim and ylim
    _xlim = ax.get_xlim()
    _ylim = ax.get_ylim()

    # Connect the widgets to their slots
    plus_button.on_clicked(add_peak)
    minus_button.on_clicked(remove_peak)
    up_button.on_clicked(up_sens)
    down_button.on_clicked(down_sens)
    slider.on_changed(selector)
    group_tb.on_submit(set_group)
    reset_button.on_clicked(reset)
    save_button.on_clicked(save)
    peak_radio.on_clicked(radio_changed)
    fig.canvas.mpl_connect('scroll_event', scroll)
    fig.canvas.mpl_connect('key_press_event', key_binding)

    plt.show()  # Start event loop
    plt.close()


def make_iguess_auto(ppm, data, SW, SFO1, o1p, filename='iguess'):
    """
    GUI to create a .ivf file, used as initial guess for Voigt_Fit.
    The computation of the peak positions and linewidths employs scipy.signal.find_peaks and scipy.signal.peak_widths, respectively.
    In addition, peak features may be added manually by clicking with the left button twice. Unwanted features can be removed with right clicks.
    If the FWHM of a peak cannot be computed automatically, a dummy FWHM of 1 Hz is assigned automatically.
    The file <filename>.ivf is written upon pressing the SAVE button.
    Press Z to activate/deactivate the cursor snap.
    ---------------------
    Parameters:
    - ppm: 1darray
        PPM scale of the spectrum
    - data: 1darray
        real part of the spectrum to fit
    - SW: float
        Spectral width /Hz
    - SFO1: float
        Nucleus Larmor Frequency /MHz
    - o1p: float
        Carrier position /ppm
    - filename: str
        Path to the file where to save the initial guess. The .ivf extension is added automatically.
    """

    ## MISCELLANEOUS FUNCTIONS
    def is_in(x, lims):
        """ Checks if x is inside the lims interval """
        return min(lims) <= x and x <= max(lims)

    def check_values():
        """ Handles event when P is negative and IW is less than 1 """
        nonlocal P, IW
        if P < 0:
            P = 0
        if IW < 1:
            IW = 1

    def get_pos(x, y, H, P):
        """ 
        Find the position of the peaks given height and prominence with scipy.signal.find_peaks 
        ------------
        Parameters:
        - x: 1darray
            array of x values
        - y: 1darray
            array of y values
        - H: float
            Threshold values (height)
        - P: float
            Threshold values (prominence)
        ------------
        Returns:
        - ks: list
            List of indices where the program found peaks
        """
        ks, *_ = find_peaks(y, height=H, prominence=P)
        return ks

    def maketotal(xj):
        """ 
        Compute the model trace, given the peak positions
        ------------------
        Parameters:
        - xj: list
            Indices of peak positions
        -----------------
        Returns:
        - peak_in: list of fit.Peak objects
            Model peaks
        """
        warnings.simplefilter('ignore')
        # Estimate peak fwhms, in ppm
        widths, *_ = peak_widths(s, xj)
        # Convert them to Hz
        fwhms = misc.ppm2freq(widths * misc.calcres(ppm), SFO1)
        # Make a dummy 1Hz fwhm for non detected onmes
        for k, x in enumerate(fwhms):
            if x == 0:
                fwhms[k] = 1
        # Estimate the integrals of the peaks
        As = []
        for k, u in enumerate(xj):
            lims=(freq[u]-IW*fwhms[k]/2, freq[u]+IW*fwhms[k]/2)
            try:
                As.append( processing.integrate(s, freq, lims=lims) / (0.5 * SW) )
            except:
                As.append(1)
                print(lims)

        # Make the fit.Peak objects with the estimated parameters
        q = [(fit.Peak(acqus,
                          ppm[x], 
                          fwhms[j],
                          k=As[j],
                          )) for j, x in enumerate(xj)]
        # Select only the peaks that are within the window
        peak_in = [peak for peak in q if is_in(peak.u, ax.get_xlim())]
        # Make the total model trace
        if len(peak_in):    # only if there are peaks inside, to avoid errors
            total = np.sum([p() for p in peak_in], axis=0)
        else:   # if there are no peaks, set zero
            total = np.zeros_like(s)
        # Update the figure
        model.set_ydata(total)
        plt.draw()
        return peak_in

    ##  Initialize variables
    # Write the info on the file
    with open(f'{filename}.ivf', 'a', buffering=1) as f:
        now = datetime.now()
        date_and_time = now.strftime("%d/%m/%Y at %H:%M:%S")
        f.write('! Initial guess computed by {} on {}\n\n'.format(os.getlogin(), date_and_time))

    prev = 0
    # Dwell time
    dw = 1/SW
    # Acquisition points
    TD = data.shape[-1]
    # Acquisition timescale
    t_aq = np.linspace(0, TD*dw, TD)
    # acquisition parameters for the fit.Peak class
    acqus = {'t1':t_aq, 'SFO1':SFO1, 'o1p':o1p, 'N':TD}

    # Frequency scale
    freq = misc.ppm2freq(ppm, SFO1, o1p)
    # Real part of the spectrum
    s = np.copy(data.real)

    # Threshold for height detection
    H = np.max(s) / 4
    _H = np.max(s) / 4
    # Prominence value
    P = np.max(s) / 100
    _P = np.max(s) / 100
    # Integration window = IW * fwhm of the peak
    IW = 2
    _IW = 2

    # Sensitivity for H
    sensH = round(np.max(s) / 50, 2)
    _sensH = round(np.max(s) / 50, 2)
    # Sensitivity for P
    sensP = round(np.max(s) / 1000, 2)
    _sensP = round(np.max(s) / 1000, 2)
    # Sensitivity for IW
    sensIW = 0.25
    _sensIW = 0.25

    # Snap flag
    snap = True

    # Names of the radiobutton entries
    radio_labels = 'Height', 'Prominence', 'Int. Window'

    # Get the positions of the peaks automatically according to the values of H and P
    xj = get_pos(ppm, s, H, P)
    # Placeholder: manually added peaks
    xi = []
    # Placeholder: blacklist. Peak positions that are in here are ignored
    x_blacklist = []
    # Number of peaks
    n_p = len(xj) + len(xi)


    ## SLOTS

    def up_sens(event):
        """
        Doubles the sensitivity of the active value
        """
        sens_fork(2)

    def down_sens(event):
        """
        Halves the sensitivity of the active value
        """
        sens_fork(0.5)

    def sens_fork(val):
        """ 
        Routes the sensitivity modifications to the correct value. 
        Val is 2 if up_sens, 1/2 if down_sens
        """
        if radio.value_selected == 'Height':
            nonlocal sensH
            sensH *= val 
        elif radio.value_selected == 'Prominence':
            nonlocal sensP
            sensP *= val
        elif radio.value_selected == 'Int. Window':
            nonlocal sensIW
            sensIW *= val
        redraw_scales()
        update_text()

    def update_text(null=0):
        """ Updates the values texts """
        Htext.set_text(f'{H:10.3g}'+r' $\pm$ '+f'{sensH:.3g}')
        Ptext.set_text(f'{P:10.3g}'+r' $\pm$ '+f'{sensP:.3g}')
        Wtext.set_text(f'{IW:5.3f}'+r'$\Gamma\,\pm$ '+f'{sensIW:.3g}')
        npeaks.set_text(f'Number of peaks detected: {n_p:5.0f}')
        plt.draw()

    def update(val):
        """ 
        Update the values according to mouse scrolls, compute the trace, updates the figure.
        val = +1 if scroll up, = -1 if scroll down, 0 in all other cases
        """
        nonlocal xj, n_p
        # Update the selected value in the radiobuttons
        if radio.value_selected == 'Height':
            nonlocal H
            H += val * sensH
        elif radio.value_selected == 'Prominence':
            nonlocal P
            P += val * sensP
        elif radio.value_selected == 'Int. Window':
            nonlocal IW
            IW += val * sensIW
        # Check if the values are meaningful
        check_values()
        # Update the green horizontal line
        thr.set_ydata((H,))

        # Compute peak positions
        xj_tmp = get_pos(ppm, s, H, P)
        # Select the peak positions that are inside the selected window and are not in blacklist
        xj = [x for x in xj_tmp if is_in(ppm[x], ax.get_xlim()) and x not in x_blacklist]
        # Compute number of peaks
        n_p = len(xj)
        # Draw the crosses for automatically detected peaks
        crosses.set_data(ppm[xj], s[xj])
        # Draw the plusses for manually added peaks
        squares.set_data(ppm[xi], s[xi])
        # Make xj to include all peaks
        xj.extend([x for x in xi])
        # Remove duplicates from xj
        xj = list(set(xj))
        # Compute the total trace
        maketotal(xj)
        # Make the scales
        redraw_scales()
        # Update the figure
        update_text()
        plt.draw()
        
    def on_scroll(event):
        """ Handles scroll events """
        if event.button == 'up':
            update(+1)
        if event.button == 'down':
            update(-1)
        update_text()

    def redraw_scales(null=0):
        """ Recompute the scales to fit the active zoom """
        misc.pretty_scale(ax, ax.get_xlim(), 'x')
        misc.pretty_scale(ax, ax.get_ylim(), 'y')
        misc.mathformat(ax)
        plt.draw()

    def clear_blacklist(event):
        """ Removes all excluded automatic peaks from the blacklist """
        nonlocal x_blacklist
        x_blacklist = []
        update(0)

    def tracker(event):
        """ Draws the crosshair """
        if event.inaxes:    # Make the crosshair visible
            h_track.set_visible(True)
            v_track.set_visible(True)
            o_track.set_visible(True)
        else:               # hide it
            h_track.set_visible(False)
            v_track.set_visible(False)
            o_track.set_visible(False)
            return
        # Find index of x position on ppm
        i = misc.ppmfind(ppm, event.xdata)[0]
        # x coordinate is ppm[i]
        x = ppm[i]
        if snap:    # y snaps to the spectrum
            y = s[i]
        else:       # y is just the y position of the cursor
            y = event.ydata
        # Update the crosshair
        h_track.set_ydata((y,))
        v_track.set_xdata((x,))
        o_track.set_data((x,), (y,))
        plt.draw()


    def keybindings(event):
        """ Handles key press """
        key = event.key
        if key == 'z':      # switches snap between True and False
            nonlocal snap
            snap = not(snap)
        if key == '<':      # halves sensitivity
            sens_fork(0.5)
        if key == '>':      # doubles sensitivity
            sens_fork(2)
        if key == 'pageup': # cycle radiobuttons down
            i = radio_labels.index(radio.value_selected)
            j = i - 1
            if j < 0:   # e.g. j=-1
                j = len(radio.labels) + j
            radio.set_active(j)
        if key == 'pagedown':   # cycle radiobuttons up
            i = radio_labels.index(radio.value_selected)
            j = i + 1
            if j > len(radio.labels) - 1: # e.g. j = 3, len(radio.labels) = 3
                j = j - len(radio.labels)
            radio.set_active(j)
        if key == 'up':     # as scrolling up
            update(+1)
        if key == 'down':   # as scrolling down
            update(-1)
        redraw_scales()

    def mouseclick(event):
        """ Manually adds/removes peaks """
        if not event.inaxes:    # do nothing
            return
        # Find the position of the mouse 
        x = misc.ppmfind(ppm, event.xdata)[0]
        if event.button == 1 and event.dblclick:    # Left double click
            nonlocal xi, x_blacklist
            xi.append(x)    # Append the value to the manual peak list
            if x in x_blacklist:    # Remove it
                i = x_blacklist.index(x)    # First find the index of x in the blacklist
                x_blacklist.pop(i)          # Then, remove it
            # Redraw everything
            update(0)
        if event.button == 3:   # Right single click
            # Find the closest peak to the point you clicked
            if len(xi): # first in the manual list
                closest_i = misc.find_nearest(xi, x)
            else:   # if it is empty, set None
                closest_i = None
            if len(xj): # then in the automatic list
                closest_j = misc.find_nearest(xj, x)
            else:   # if it is empty, set None
                closest_j = None

            if closest_i is not None:   # If there are any peaks in the manual list:
                # do things only if the closest manual peak is within 10 points to where you actually clicked
                if np.abs(x - closest_i) < 10:  
                    # Find the position of such peak
                    i = xi.index(closest_i)
                    # Remove it from the list
                    xi.pop(i)
                    # Redraw everything, then stop
                    update(0)
                    return

            # It gets here only if the closest peak is not manually added and it is not within 10 points to where you actually clicked

            if closest_j is not None:
                # do things only if the closest automatic peak is within 10 points to where you actually clicked
                if np.abs(x - closest_j) < 10:
                    # Add this point to the blacklist
                    x_blacklist.append(closest_j)
                    # Redraw everything
                    update(0)
                    return

    def save(event):
        """ Write a section in the output file """
        nonlocal prev
        # Adjust the intensities

        peak_in = maketotal(xj)
        keys = np.arange(prev+1, prev+len(peak_in)+1, 1)
        peaks = {key: peak_in[k] for k, key in enumerate(keys)}
        # Use 1 as A because the relative intensities are calculated inside write_wf
        fit.write_vf(f'{filename}.ivf', peaks, ax.get_xlim(), 1, prev)
        prev += len(peak_in)
        
        # Mark a region as "fitted" with a green box
        ax.axvspan(*ax.get_xlim(), color='tab:green', alpha=0.1)
        # Call reset to return at the initial situation
        reset(event)

    def reset(event):
        # setta H e P come all'inizio
        nonlocal H, P, IW, sensH, sensP, sensIW
        H = _H
        P = _P
        IW = _IW    
        sensH = _sensH
        sensP = _sensP
        sensIW = _sensIW
        # resetta i limiti
        ax.set_xlim(max(ppm), min(ppm))
        update(0)
        plt.draw()


    #-------------------------------------------------------------------------------------------------------------

    # Make the figure panel
    fig = plt.figure('Automatic Computation of Initial Guess')
    fig.set_size_inches(figures.figsize_large)
    ax = fig.add_subplot(1,1,1)
    plt.subplots_adjust(left=0.1, right=0.8, top=0.95, bottom=0.1)

    # Boxes where to place the widgets
    #    radiobuttons
    box_radio = plt.axes([0.825, 0.6, 0.15, 0.2])
    #   up_sens button
    box_up = plt.axes([0.825, 0.85, 0.075, 0.05])
    #   down_sens button
    box_down = plt.axes([0.900, 0.85, 0.075, 0.05])
    #   clear_blacklist button
    box_cbl = plt.axes([0.900, 0.10, 0.075, 0.05])
    #   save button
    box_save = plt.axes([0.825, 0.10, 0.075, 0.05])

    # Make the widgets
    #   radiobuttons
    radio = RadioButtons(box_radio, radio_labels)
    #   up_sens button
    up_button = Button(box_up, r'$\uparrow$', hovercolor='0.975')
    #   down_sens button
    down_button = Button(box_down, r'$\downarrow$', hovercolor='0.975')
    #   clear_blacklist button
    cbl_button = Button(box_cbl, r'Clear Blacklist', hovercolor='0.975')
    #   save button
    save_button = Button(box_save, r'SAVE', hovercolor='0.975')
    #   mouse position
    cursor = Cursor(ax, horizOn=False, vertOn=False, useblit=True, lw=0.2, color='tab:green')

    # Draw the spectrum in blue
    ax.plot(ppm, s, c='tab:blue', lw=0.8)
    # Draw the automatic peak positions as crosses
    crosses, = ax.plot(ppm[xj], s[xj], 'x', c='tab:orange')
    # Draw the manual peak positions as plusses
    squares, = ax.plot(ppm[xi], s[xi], '+', c='tab:orange')
    # Draw a placeholder for the total model trace
    model, = ax.plot(ppm, np.zeros_like(ppm), c='tab:red', lw=0.8)
    # Draw a line for the threshold 
    thr = ax.axhline(H, c='tab:green', lw=0.5, ls='--')
    # Draw the crosshair, but invisible
    #   Horizontal line
    h_track = ax.axhline(0, c='g', lw=0.2, visible=False)
    #   Vertical line
    v_track = ax.axvline(0, c='g', lw=0.2, visible=False)
    #   Cross point
    o_track, = ax.plot((0,), (0,), c='g', marker='.', markersize=5, visible=False)
    # Compute the model
    maketotal(xj)

    # Make the text values to be updated 
    #   Compute the positions of the text: in the middle between two subsequent radio labels
    radiolabel_pos = [label.get_position() for label in radio.labels]
    yshift = (radiolabel_pos[1][1] - radiolabel_pos[0][1]) / 2
    #   H value text
    Htext = box_radio.text(0.995, radiolabel_pos[0][1] + yshift, '', ha='right', va='center', transform=box_radio.transAxes)
    #   P value text
    Ptext = box_radio.text(0.995, radiolabel_pos[1][1] + yshift, '', ha='right', va='center', transform=box_radio.transAxes)
    #   IW value text
    Wtext = box_radio.text(0.995, radiolabel_pos[2][1] + yshift, '', ha='right', va='center', transform=box_radio.transAxes)
    #   Number of peaks text
    npeaks = plt.text(0.825, 0.5, '', ha='left', va='center', transform=fig.transFigure)
    update_text()

    # Instruction text
    uparrow_text = r'$\uparrow$'
    downarrow_text = r'$\downarrow$'
    keyboard_text = '\n'.join([
        f'{"KEYBOARD SHORTCUTS":^50s}',
        f'{"Key":>5s}: Action',
        f'-'*50,
        f'{"<":>5s}: Decrease sens.',
        f'{">":>5s}: Increase sens.',
        f'{"Pg"+uparrow_text:>5s}: Change parameter, up',
        f'{"Pg"+downarrow_text:>5s}: Change parameter, down',
        f'{uparrow_text:>5s}: Increase value',
        f'{downarrow_text:>5s}: Decrease value',
        f'{"z":>5s}: Toggle snap on cursor',
        f'-'*50,
        ])
    keyboard_print = ax.text(0.86, 0.2, keyboard_text, 
            ha='left', va='bottom', transform=fig.transFigure, fontsize=8, linespacing=1.55)

    # Adjust visual shit
    misc.pretty_scale(ax, (max(ppm), min(ppm)), 'x')
    misc.pretty_scale(ax, ax.get_ylim(), 'y')
    misc.mathformat(ax)

    # Connect widgets to slots
    up_button.on_clicked(up_sens)
    down_button.on_clicked(down_sens)
    cbl_button.on_clicked(clear_blacklist)
    save_button.on_clicked(save)
    fig.canvas.mpl_connect('motion_notify_event', tracker)
    fig.canvas.mpl_connect('scroll_event', on_scroll)
    fig.canvas.mpl_connect('button_press_event', mouseclick)
    fig.canvas.mpl_connect('key_press_event', keybindings)

    # Start event loop
    plt.show()

    # Re-enable warnings
    warnings.simplefilter('default')


# --------------------------------------------------------------------
def write_vf(filename, peaks, lims, I, prev=0, header=False):
    """
    Write a section in a fit report file, which shows the fitting region and the parameters of the peaks to feed into a Voigt lineshape model.
    -----------
    Parameters:
    - filename: str
        Path to the file to be written
    - peaks: dict
        Dictionary of fit.Peak objects
    - lims: tuple
        (left limit /ppm, right limit /ppm)
    - I: float
        Absolute intensity value
    - prev: int
        Number of previous peaks already saved. Increases the peak index
    - header: bool
        If True, adds a "!" starting line to separate fit trials
    """
    # Adjust the intensities
    r_i, I_corr = misc.molfrac([peak.k for _, peak in peaks.items()])

    # Open the file in append mode
    f = open(f'{filename}', 'a', buffering=1)
    # Info on the region to be fitted
    if header:
        now = datetime.now()
        date_and_time = now.strftime("%d/%m/%Y at %H:%M:%S")
        f.write('! Fit performed by {} on {}\n\n'.format(os.getlogin(), date_and_time))
    #   Header
    f.write('{:>16};\t{:>12}\n'.format('Region', 'Intensity'))
    f.write('-'*96+'\n')
    #   Values
    region = '{:-.3f}:{:-.3f}'.format(*lims)   # From the zoom of the figure
    f.write(f'{region:>16};\t{I*I_corr:14.8e}\n\n')

    # Info on the peaks
    #   Header
    f.write('{:>4};\t{:>12};\t{:>12};\t{:>8};\t{:>8};\t{:>8};\t{:>8}\n'.format(
        '#', 'u', 'fwhm', 'Rel. I.', 'Phase', 'Beta', 'Group'))
    f.write('-'*96+'\n')
    #   Values
    for k, key in enumerate(peaks.keys()):
        peak = peaks[key]
        f.write('{:>4.0f};\t{:=12.8f};\t{:12.6f};\t{:8.6f};\t{:-8.3f};\t{:8.5f};\t{:>8.0f}\n'.format(
            k+prev+1, peak.u, peak.fwhm, r_i[k], peak.phi, peak.b, peak.group))
    f.write('-'*96+'\n\n')

    # Add region separator and close the file
    f.write('='*96+'\n\n')
    f.close()

def read_vf(filename, n=-1):
    """
    Reads a .ivf (initial guess) or .fvf (final fit) file, containing the parameters for a lineshape deconvolution fitting procedure.
    The file is separated and unpacked into a list of dictionaries, each of which contains the limits of the fitting window, the total intensity value, and a dictionary for each peak with the characteristic values to compute it with a Voigt line.
    --------------
    Parameters:
    - filename: str
        Path to the filename to be read
    - n: int
        Number of performed fit to be read. Default: last one. The breakpoints are lines that start with "!". For this reason, n=0 returns an empty dictionary, hence the first fit is n=1. 
    -------------
    Returns:
    - regions: list
        List of dictionaries for running the fit.
    """
    def read_region(R):
        """ Creates a dictionary of parameters from a section of the input file.  """
        # Placeholder
        dic_r = {}
        # Separate the lines and remove the empty ones
        R = R.split('\n')
        for k, r in enumerate(R):
            if len(r)==0 or r.isspace():
                _ = R.pop(k)

        n_bp = 0        # Number of breaking points (----)
        k_bp = 0        # Line of the last breaking point detected
        for k, r in enumerate(R):
            if '------' in r:   # Increase breakpoint and store the line number
                n_bp += 1
                k_bp = k
                continue

            if n_bp == 1 and k_bp == k-1:   # First section: region limits and total intensity
                line = r.split(';') # Separate the values
                dic_r['limits'] = eval(line[0].replace(':',', '))   # Get the limits
                dic_r['I'] = eval(line[-1]) # Get the intensity

            if n_bp == 2:       # Second section: peak parameters
                line = r.split(';') # Separate the values
                # Unpack the line
                idx, u, fwhm, k, phi, b, group = [eval(w) for w in line]
                # Put the values in a dictionary
                dic_p = {
                        'u': u,
                        'fwhm': fwhm,
                        'k': k,
                        'b': b,
                        'phi': phi,
                        'group': group
                        }
                # Put the values in the returned dictionary
                dic_r[idx] = dic_p

            if n_bp == 3:   # End of file: stop reading
                break

        return dic_r

    # Read the file
    with open(filename, 'r') as J:
        ff = J.read()
    # Get the actual section from an output file
    f = ff.split('!')[n]
    # Separate the bigger sections
    R = f.split('='*96)
    # Remove the empty lines
    for k, r in enumerate(R):
        if r.isspace():
            _ = R.pop(k)

    regions = []    # Placeholder for return values
    for r in R: # Loop on the big sections to read them
        regions.append(read_region(r))
    return regions
        

def read_par(filename):
    """
    Reads the input file of the fit and returns the values.
    --------
    Parameters:
    - filename: str
        directory and name of the input file to be read
    --------
    Returns:
    - V : 2darray
        matrix (# signals, parameters)
    - C : 1darray or False
        Coefficients of the polynomion to be used as baseline correction. If the 'baseline' checkbox in the interactive figure panel is not checked, C_f is False.
    - limits : tuple or None
        Trim limits for the spectrum (left, right). If None, the whole spectrum is used.
    """
    # Declare empty variables
    V = []
    C = False
    limits = None

    f = open(filename, 'r')
    L = f.readlines()
    V_flag = 0      # Am I reading the signal parameter section?
    B_flag = 0      # Am I reading the baseline parameter section?
    L_flag = 0      # Am I reading the window limits?
    for i, line in enumerate(L):
        if line[0] == '#' or line == '\n':  # Comment or empty line
            continue
        line = line.strip()
        if line == '***{:^60}***'.format('SIGNAL PARAMETERS'):
            V_flag = 1
            continue
        if line == '***{:^60}***'.format('END OF SIGNAL PARAMETERS'):
            V_flag = 0
            continue
        if line == '***{:^60}***'.format('BASELINE PARAMETERS'):
            B_flag = 1
            continue
        if line == '***{:^60}***'.format('WINDOW DELIMITERS /ppm'):
            L_flag = 1
            continue
        if line == '***{:^60}***'.format('END OF FILE'):
            f.close()
            break


        if L_flag:
            v = line.split('\t')
            limits = float(v[-2].replace(' ','')), float(v[-1].replace(' ', ''))
            L_flag = 0

        if V_flag:
            v = line.split('\t')
            V.append(np.array([float(w.replace(' ', '')) for w in v[1:]]))   # [1:] because first column is signal index

        if B_flag:
            v = line.split('\t')
            C = np.array([float(w.replace(' ', '')) for w in v])
            B_flag = 0

    V = np.array(V)
    return V, C, limits


def write_par(V, C, limits, filename='i_guess.inp'):
    """
    Write the parameters of the fit, whether they are input or output.
    --------
    Parameters:
    - V : 2darray
        matrix (# signals, parameters)
    - C : 1darray or False
        Coefficients of the polynomion to be used as baseline correction. If the 'baseline' checkbox in the interactive figure panel is not checked, C_f is False.
    - limits : tuple 
        Trim limits for the spectrum (left, right).
    - filename: str
        directory and name of the file to be written
    """
    if isinstance(filename, str):
        f = open(filename, 'w')
    else:
        f = filename
    f.write('***{:^60}***\n'.format('WINDOW DELIMITERS /ppm'))
    f.write('{:=7.2f}\t{:=7.2f}\n\n'.format(limits[0], limits[1]))

    f.write('***{:^60}***\n'.format('SIGNAL PARAMETERS'))
    f.write('{:<4}\t{:>7}\t{:>5}\t{:>5}\t{:>5}\t{:>5}\t{:<9}\n'.format('#', 'u', 's', 'k', 'b', 'phi', 'A'))
    for i in range(V.shape[0]):
        f.write('{:<4.0f}\t{:=7.2f}\t{:5.0f}\t{:5.3f}\t{:5.2f}\t{: 5.2f}\t{:5.2e}\n'.format( i+1, V[i,0], V[i,1], V[i,2], V[i,3], V[i,4], V[i,5]))
    f.write('***{:^60}***\n'.format('END OF SIGNAL PARAMETERS'))

    if C is not False:      # Write baseline coefficients only if explicitely said
        f.write('\n***{:^60}***\n'.format('BASELINE PARAMETERS'))
        f.write('#\t{:^9}\t{:^9}\t{:^9}\t{:^9}\t{:^9}\n'.format('a', 'b', 'c', 'd', 'e'))
        f.write(' \t{: 5.2e}\t{: 5.2e}\t{: 5.2e}\t{: 5.2e}\t{: 5.2e}\n'.format(C[0], C[1], C[2], C[3], C[4]))
        f.write('\n***{:^60}***\n'.format('END OF BASELINE PARAMETERS'))
    
    if isinstance(filename, str):
        f.write('\n***{:^60}***\n'.format('END OF FILE'))
        f.close()


def print_par(V, C, limits=[None,None]):
    """
    Prints on screen the same thing that write_par writes in a file.
    --------
    Parameters:
    - V : 2darray
        matrix (# signals, parameters)
    - C : 1darray or False
        Coefficients of the polynomion to be used as baseline correction. If the 'baseline' checkbox in the interactive figure panel is not checked, C_f is False.
    - limits : tuple or None
        Trim limits for the spectrum (left, right). If None, the whole spectrum is used.
    """
    print('***{:^60}***'.format('SIGNAL PARAMETERS'))
    print('{:<4}\t{:>7}\t{:>5}\t{:>5}\t{:>5}\t{:>5}\t{:<9}'.format('#', 'u', 's', 'k', 'b', 'phi', 'A'))
    for i in range(V.shape[0]):
        print('{:<4.0f}\t{:=7.2f}\t{:5.0f}\t{:5.3f}\t{:5.2f}\t{: 5.2f}\t{:5.2e}'.format( i+1, V[i,0], V[i,1], V[i,2], V[i,3], V[i,4], V[i,5]))

    if C is not False:
        print('***{:^60}***\n'.format('BASELINE PARAMETERS'))
        print('#\t{:^9}\t{:^9}\t{:^9}\t{:^9}\t{:^9}'.format('a', 'b', 'c', 'd', 'e'))
        print(' \t{: 5.2e}\t{: 5.2e}\t{: 5.2e}\t{: 5.2e}\t{: 5.2e}'.format(C[0], C[1], C[2], C[3], C[4]))
        print('#\tWINDOW DELIMITERS /ppm')
        print('{:=7.2f}\t{:=7.2f}'.format(limits[0], limits[1]))

def dic2mat(dic, peak_names, ns, A=None):
    """
    This is used to make the matrix of the parameters starting from a dictionary like the one produced by l.
    The column of the total intensity is not added, unless the parameter 'A' is passed. In this case, the third column (which is the one with the relative intesities) is corrected using the function molfrac.
    --------
    Parameters:
    - dic : dict
        input dictionary
    - peak_names : list
        list of the parameter entries to be looked for
    - ns : int
        number of signals to unpack
    - A : float or None
        Total intensity.
    -------
    Returns:
    - V : 2darray
        Matrix containing the parameters.
    """
    V = []
    #   u   s   k   b  phi A   
    for i in range(ns):
        V.append([])
        for j in range(len(peak_names)):
            V[i].append(dic[peak_names[j]+str(i+1)])
    V = np.array(V)
    if A is None:
        return V
    else:
        V[:,2], Acorr = misc.molfrac(V[:,2])
        A_arr = Acorr * np.array([A for w in range(ns)])
        V = np.concatenate((V, A_arr.reshape(-1, 1)), axis=-1)
        return V



# --------------------------------------------------------------------


def test_randomsign(data, thresh=1.96):
    """
    Test an array of residuals for the randomness of the sign changes.
    The result it True if the sequence is recognized as random.
    -----------
    Parameters:
    - data: 1darray
        Residuals to test
    - thresh: float
        Significance level. The default is 1.96, which corresponds to 5% significance level.
    -----------
    Returns:
    - test: bool
        True if the signs are random, False otherwise
    """
    # Size of the data
    N = len(data)
    # Signs of the entries in the residual
    signs = np.sign(data)
    # Number of > 0 residuals
    n_p = len([x for x in signs if x > 0])
    # Number of < 0 residuals
    n_n = len([x for x in signs if x < 0])

    # Where the residuals change sign
    signchange = ((np.roll(signs, 1) - signs) != 0).astype(int)
    # Number of adjacent pieces of the residuals with the same sign
    n_runs = np.sum(signchange)

    # The statistical distribution of runs can be approximated with a Gaussian
    # with mean = u and std = sigma
    u = ( 2 * n_p * n_n) / N + 1
    sigma = ( (u - 1) * (u - 2) / (N - 1) )**0.5

    # If z < thresh, the sign distribution is random
    z = np.abs(n_runs - u) / sigma
    return z < thresh

def test_correl(data, subtract_mean=True):
    """
    Tests an array of residuals for their correlation.
    It compares the unit-lag autocorrelation P of the data (see below) with the theoretical value for non-correlated data T_P:
    P = sum_i^(N-1) r_i r_(i+1) ; T_P = (N-1)^(0.5) \sum_i r_i^2
    If P < T_P, the residuals are not correlated, and the result is True.
    ----------
    Parameters:
    - data: 1darray
        Residuals to be test
    - subtract_mean: bool
        If True, subtracts from the residuals their mean.
    ----------
    Returns:
    - test: bool
        True if the residuals are non correlated, False otherwise
    """
    # Shallow copy of the residuals
    r = np.copy(data)
    # Size of the data
    N = len(r)
    if subtract_mean: # Subtract from the residuals their mean
        r -= np.mean(r)

    # Compute the discrete correlation function of the residuals P
    r_roll = np.roll(r, 1)
    P = np.sum( (r * r_roll)[:-1] )
    # Compute threshold for correlation
    T_P = 1 / (N - 1)**0.5 * np.sum(r**2)

    # Residuals are not correlated if P < T_P
    return np.abs(P) < T_P

def test_ks(data, thresh=0.05):
    """
    Performs the Kolmogorov-Smirnov test on the residuals to check if they are drawn from a normal distribution.
    The implementation is scipy.stats.kstest.
    The result is True if the residuals are Gaussian.
    ----------
    Parameters:
    - data: 1darray
        Residuals to test
    - thresh: float
        Significance level for the test. Default is 5%
    ---------
    Returns:
    - test: bool
        True if the residuals are Gaussian, False otherwise
    """
    # Shallow copy
    r = np.copy(data)
    from scipy.stats import kstest
    ksstat = kstest(r, "norm", args=(np.mean(r), np.std(r)))
    # Residuals are gaussian distributed if p_value > thresh
    return ksstat.pvalue > thresh

def test_residuals(res, alpha=0.05):
    """
    Tests an array of residuals for their randomness, correlation, and underlying distribution.
    To do this, it uses the functions "fit.test_randomsign", "fit.test_correl", "fit.test_ks".
    The results of the tests will be print in standard output and returned.
    ------------------
    Parameters:
    - res: ndarray
        Residuals to be tested
    - alpha: float
        Significance level
    ------------------
    Returns:
    - test_random: bool
        Randomness of the residuals (True = random)
    - test_correlation: bool
        Correlation of the residuals (True = non-correlated)
    - test_gaussian: bool
        Normal-distribution of the residuals (True = normally-distributed)
    """
    from scipy.stats import norm
    
    # Get the z-score from the significance level
    z_value = norm.ppf(1 - alpha/2)

    # Convert the residuals to 1D
    r = np.copy(res).flatten()
    
    # Compute the tests
    test_randomness = fit.test_randomsign(r, z_value) 
    test_correlation = fit.test_correl(r, False) 
    test_gaussian = fit.test_ks(r, alpha)

    # Print the results
    print('\n'.join([
        f'{"Random":22s}: {test_randomness}',
        f'{"Non-correlated":22s}: {test_correlation}',
        f'{"Normally distributed":22s}: {test_gaussian}',
        ]))

    return test_randomness, test_correlation, test_gaussian



def write_log(input_file, output_file, limits, V_i, C_i, V_f, C_f, result, runtime, test_res=True, log_file='fit.log'):
    """
    Write a log file with all the information of the fit.
    -------
    Parameters:
    - input_file: str
        Location and filename of the input file
    - output_file: str
        Location and filename of the output file
    - limits: tuple
        Delimiters of the spectral region that was fitted. (left, right)
    - V_i: 2darray
        Initial parameters of the fit
    - C_i: 1darray or False
        Coefficients of the starting polynomion used for baseline correction. If False, it was not used.
    - V_f: 2darray
        Final parameters of the fit
    - C_f: 1darray or False
        Coefficients of the final polynomion used for baseline correction. If False, it was not used.
    - result: lmfit.FitResult Object
        Object returned by lmfit after the fit.
    - runtime: datetime.datetime Object
        Time taken for the fit
    - test_res: bool
        Choose if to test the residual with the fit.test_residual function (True) or not (False)
    - log_file: str
        Filename of the log file to be saved.
    """
    now = datetime.now()
    date_and_time = now.strftime("%d/%m/%Y at %H:%M:%S")
    f = open(log_file, 'w')

    f.write('***{:^60}***\n\n'.format('FIT LOG'))
    f.write('Fit performed by {} on {}\n\n'.format(os.getlogin(), date_and_time))
    f.write('-'*60)
    f.write('\n\n')

    f.write('{:<12}{:>}\n'.format('Input file:', os.path.abspath(input_file))) 
    write_par(V_i, C_i, limits=limits, filename=f)

    f.write('-'*60)
    f.write('\n\n')

    f.write('{:<12}{:>}\n'.format('Output file:', os.path.abspath(output_file))) 
    write_par(V_f, C_f, limits=limits, filename=f)

    f.write('-'*60)
    f.write('\n')

    f.write('{}\nTotal runtime: {}.\nNumber of function evaluations: {:5.0f}\n\n'.format(result.message, runtime, result.nfev))
    
    # Check for the gaussianity of the residual
    if test_res is True:
        R = result.residual
        m_R = np.mean(R)
        SYSDEV, Q_G = test_residuals(R)
        f.write('{:^60}\n'.format('Statistics of the fit'))
        f.write('{:<30} = {:=9.2e} | Optimal : 0\n'.format('Mean of residuals', m_R)) 
        f.write('{:<30} = {:9.6f} | Optimal : 1\n'.format('Systematic deviation', SYSDEV)) 
        f.write('{:<30} = {:9.6f} | Optimal : 1\n'.format('Gaussianity of residuals', Q_G)) 
    f.write('-' * 60)
    f.close()


def gaussian_fit(x, y, s_in=None):
    """
    Fit 'y' with a gaussian function, built using 'x' as independent variable
    -------
    Parameters:
    - x : 1darray
        x-scale
    - y : 1darray
        data to be fitted
    -------
    Returns:
    - u : float
        mean 
    - s : float
        standard deviation
    - A : float
        Integral
    """

    # Make parameter dictionary
    param = l.Parameters()
    param.add('u', value=np.mean(y), min=min(y), max=max(y))
    param.add('s', value=np.std(y), min=0, max=np.inf)
    if s_in:
        param['s'].set(value=s_in)
    param.add('A', value=np.trapz(y, dx=misc.calcres(x)), min=0, max=5*np.trapz(y, dx=misc.calcres(x)))

    def f2min(param, x, y):
        # Cost function
        par = param.valuesdict()
        G = sim.f_gaussian(x, par['u'], par['s'], par['A'])
        return y - G

    minner = l.Minimizer(f2min, param, fcn_args=(x, y))
    result = minner.minimize(method='leastsq', max_nfev=10000, xtol=1e-15, ftol=1e-15)

    # Return the result
    popt = result.params.valuesdict()
    return popt['u'], popt['s'], popt['A']



# -------------------------------------------------------------------------------------------

class Voigt_Fit:
    """
    This class offers an "interface" to fit a 1D NMR spectrum.
    -------
    Attributes:
    - ppm_scale: 1darray
        Self-explanatory
    - S : 1darray
        Spectrum to fit. Only real part
    - t_AQ: 1darray
        acquisition timescale of the spectrum
    - SW: float
        Spectral width /Hz
    - SFO1: float
        Larmor frequency of the nucleus
    - o1p : float
        Pulse carrier frequency
    - filename: str
        Root of the names of the files that will be saved 
    - X_label: str
        Label for the chemical shift axis in the figures
    - i_guess: list
        Initial guess for the fit, read by a .ivf file with fit.read_vf
    - result: list
        Result the fit, read by a .fvf file with fit.read_vf
    """

    def __init__(self, ppm_scale, S, t_AQ, SFO1, o1p, nuc=None, filename='fit'):
        """
        Initialize the class with common values.
        --------
        Parameters:
        - ppm_scale: 1darray
            ppm scale of the spectrum
        - S: 1darray
            Spectrum to be fitted
        - t_AQ: 1darray
            Acquisition timescale
        - SFO1: float
            Larmor frequency of the observed nucleus, in MHz
        - o1p: float
            Carrier position, in ppm
        - nuc: str
            Observed nucleus. Used to customize the x-scale of the figures.
        - filename: str or None
            Root of the name of the files that will be saved
        """
        self.ppm_scale = ppm_scale
        self.S = S
        self.t_AQ = misc.extend_taq(t_AQ, self.S.shape[-1])
        self.SFO1 = SFO1
        self.SW = np.abs( (max(ppm_scale) - min(ppm_scale)) * SFO1 )
        self.o1p = o1p
        self.filename = filename
        if nuc is None:
            self.X_label = r'$\delta\,$ /ppm'
        elif isinstance(nuc, str):
            fnuc = misc.nuc_format(nuc)
            self.X_label = r'$\delta$ ' + fnuc +' /ppm'

    def iguess(self, filename=None, n=-1, ext='ivf', auto=False):
        """
        Reads, or computes, the initial guess for the fit.
        If the file is there already, it just reads it with fit.read_vf. Otherwise, it calls fit.make_iguess to make it.
        --------
        Parameters:
        - filename: str or None
            Path to the input file. If None, "<self.filename>.ivf" is used
        - n: int
            Index of the initial guess to be read (default: last one)
        - ext: str
            Extension of the file to be used
        - auto: bool
            If True, uses the GUI for automatic peak picking, if False, the manual one
        """
        # Set the default filename, if not given
        if filename is None:
            filename = f'{self.filename}'
        # Check if the file exists
        in_file_exist = os.path.exists(f'{filename}.{ext}')

        if in_file_exist is True:       # Read everything you need from the file
            regions = fit.read_vf(f'{filename}.{ext}')
        else:                           # Make the initial guess interactively and save the file.
            if auto:
                fit.make_iguess_auto(self.ppm_scale, self.S, self.SW, self.SFO1, self.o1p, filename=filename)
            else:
                fit.make_iguess(self.S, self.ppm_scale, self.t_AQ, self.SFO1, self.o1p, filename=filename)
            regions = fit.read_vf(f'{filename}.{ext}')
        # Store it
        self.i_guess = regions
        print(f'{filename}.{ext} loaded as input file.')

    def load_fit(self, filename=None, n=-1, ext='fvf'):
        """
        Reads a file with fit.read_vf and stores the result in self.result.
        ---------
        Parameters:
        - filename: str
            Path to the .fvf file to be read. If None, "<self.filename>.fvf" is used.
        - n: int
            Index of the fit to be read (default: last one)
        - ext: str
            Extension of the file to be used
        """
        # Set the default filename, if not given
        if filename is None:
            filename = f'{self.filename}'
        # Check if the file exists
        out_file_exist = os.path.exists(f'{filename}.{ext}')
        if out_file_exist is True:       # Read everything you need from the file
            regions = fit.read_vf(f'{filename}.{ext}', n=n)
        else:
            raise NameError(f'{filename}.{ext} does not exist.')
        # Store
        self.result = regions
        print(f'{filename}.{ext} loaded as fit result file.')

    def dofit(self, indep=True, u_lim=1, f_lim=10, k_lim=(0,3), vary_phase=False, vary_b=True, itermax=10000, fit_tol=1e-8, filename=None, method='leastsq'):
        """
        Perform a lineshape deconvolution fitting.
        The initial guess is read from the attribute self.i_guess.
        The components can be considered to be all independent from one to another by setting "indep" to True: this means that the fit will be done using fit.voigt_fit_indep.
        The indep=False option has not been implemented yet.
        ------------
        Parameters:
        - indep: bool
            True to consider all the components to be independent
        - u_lim: float
            Determines the displacement of the chemical shift (in ppm) from the starting value.
        - f_lim: float
            Determines the displacement of the linewidth (in Hz) from the starting value.
        - k_lim: float or tuple
            If tuple, minimum and maximum allowed values for k during the fit. If float, maximum displacement from the initial guess
        - vary_phase: bool
            Allow the peaks to change phase (True) or not (False)
        - vary_b: bool
            Allow the peaks to change Lorentzian/Gaussian ratio
        - itermax: int
            Maximum number of allowed iterations
        - fit_tol: float
            Value of the target function to be set as x_tol and f_tol
        - filename: str
            Path to the output file. If None, "<self.filename>.fvf" is used
        - method: str
            Method to use for the optimization (see lmfit)
        """

        # Make a shallow copy of the real part of the experimental spectrum
        S = np.copy(self.S.real)
        # Check if the initial guess was loaded correctly
        if not isinstance(self.i_guess, list):
            raise ValueError('Initial guess not correctly loaded')
        # Set the output filename, if not given
        if filename is None:
            filename = f'{self.filename}'

        # Do the fit
        if indep is True:
            fit.voigt_fit_indep(S, self.ppm_scale, self.i_guess, self.t_AQ, self.SFO1, self.o1p, u_lim=u_lim, f_lim=f_lim, k_lim=k_lim, vary_phase=vary_phase, vary_b=vary_b, itermax=itermax, fit_tol=fit_tol, filename=filename, method=method)
        else:
            raise NotImplementedError('More and more exciting adventures in the next release!')
        # Store
        self.result = fit.read_vf(f'{filename}.fvf')


    def plot(self, what='result', show_total=True, show_res=False, res_offset=0, labels=None, filename=None, ext='tiff', dpi=600):
        """
        Plots either the initial guess or the result of the fit, and saves all the figures. Calls fit.plot_fit.
        The figure <filename>_full will show the whole model and the whole spectrum. 
        The figures labelled with _R<k> will depict a detail of the fit in the k-th fitting region.
        Optional labels for the components can be given: in this case, the structure of 'labels' should match the structure of self.result (or self.i_guess). This means that the length of the outer list must be equal to the number of fitting region, and the length of the inner lists must be equal to the number of peaks in that region.
        ------------
        Parameters:
        - what: str
            'iguess' to plot the initial guess, 'result' to plot the fitted data
        - show_total: bool
            Show the total trace (i.e. sum of all the components) or not
        - show_res: bool
            Show the plot of the residuals
        - res_offset: float
            Displacement of the residuals plot from 0, to be given as a fraction of the height of the experimental spectrum. res_offset > 0 will move the residuals BELOW the zero-line!
        - labels: list of list
            Optional labels for the components. The structure of this parameter must match the structure of self.result
        - filename: str
            Root of the name of the figures that will be saved. If None, <self.filename> is used
        - ext: str
            Format of the saved figures
        - dpi: int
            Resolution of the figures, in dots per inches
        """
        # select the correct object to plot
        if what == 'iguess':
            regions = deepcopy(self.i_guess)
        elif what == 'result':
            regions = deepcopy(self.result)
        else:
            raise ValueError('Specify what you want to plot: "iguess" or "result"')
               
        # Set the filename, if not given
        if filename is None:
            filename = f'{self.filename}'

        # Make the figures
        S = np.copy(self.S.real)
        fit.plot_fit(S, self.ppm_scale, regions, self.t_AQ, self.SFO1, self.o1p, show_total=show_total, show_res=show_res, res_offset=res_offset, X_label=self.X_label, labels=labels, filename=filename, ext=ext, dpi=dpi)

    def get_fit_lines(self, what='result'):
        """
        Calculates the components, and the total fit curve used as initial guess, or as fit results..
        The components will be returned as a list, not split by region.
        --------
        Parameters:
        - what: str
            'iguess' or 'result' 
        --------
        Returns:
        - signals: list of 1darray
            Components used for the fit
        - total: 1darray
            Sum of all the signals
        - limits_list: list
            List of region delimiters, in ppm
        """
        # Select the correct object
        if what == 'iguess':
            regions = deepcopy(self.i_guess)
        elif what == 'result':
            regions = deepcopy(self.result)
        else:
            raise ValueError('Specify what you want to plot: "iguess" or "result"')

        # Make the acqus dictionary for the fit.Peak objects
        acqus = { 't1': self.t_AQ, 'SFO1': self.SFO1, 'o1p': self.o1p, }
        # Placeholder
        signals = []
        limits_list = []
        # Loop on the regions
        for region in regions:
            # Remove the limits and the intensity from the region dictionary
            param = deepcopy(region)
            limits = param.pop('limits')
            I = param.pop('I')
            # Make the fit.Peak objects
            peaks = {key : fit.Peak(acqus, N=self.S.shape[-1], **value) for key, value in param.items()}
            # Get the arrays from the dictionary and put them in the list
            signals.extend([p(I) for _, p in peaks.items()])
            limits_list.append(limits)
        # Compute the total trace
        total = np.sum(signals, axis=0)
        return signals, total, limits_list

    def res_histogram(self, what='result', nbins=500, density=True, f_lims=None, xlabel='Residuals', x_symm=True, barcolor='tab:green', fontsize=20, filename=None, ext='tiff', dpi=300):
        """
        Computes the histogram of the residuals and saves it.
        Employs fit.histogram to make the figure.
        --------
        Parameters:
        - what: str
            'iguess' or 'result' 
        - nbins : int
            number of bins to be calculated
        - density : bool
            True for normalize data
        - f_lims : tuple or None
            limits for the x axis of the figure
        - xlabel : str or None
            Text to be displayed under the x axis
        - x_symm : bool
            set it to True to make symmetric x-axis with respect to 0
        - barcolor: str
            Color of the bins
        - fontsize: float
            Biggest fontsize in the figure
        - name : str
            name for the figure to be saved
        - ext: str
            Format of the image
        - dpi: int
            Resolution of the image in dots per inches
        """
        # Filename check
        if filename is None:
            filename = f'{self.filename}'
        filename += '_rhist'

        # Select the correct object
        if what == 'iguess':
            regions = deepcopy(self.i_guess)
        elif what == 'result':
            regions = deepcopy(self.result)
        else:
            raise ValueError('Specify what you want to plot: "iguess" or "result"')

        # Make the acqus dictionary for the fit.Peak objects'
        acqus = { 't1': self.t_AQ, 'SFO1': self.SFO1, 'o1p': self.o1p, }

        # Get the total function and the limits
        _, total, limits_list = self.get_fit_lines(what)
        # Convert the limits in points according to the ppm scale
        limits_pt_list = [ [misc.ppmfind(self.ppm_scale, w)[0] for w in lims] for lims in limits_list ]

        # Placeholders
        exp_trim, total_trim = [], []
        for k, region in enumerate(regions):        # loop on the regions
            # Compute the slice
            lims = slice(min(limits_pt_list[k]), max(limits_pt_list[k]))
            # Trim the experimental data and the total 
            exp_trim.append(self.S[...,lims].real)
            total_trim.append(total[...,lims])
        # Sum on different regions
        exp_trim = np.concatenate(exp_trim, axis=-1)
        total_trim = np.concatenate(total_trim, axis=-1)

        # Compute the residuals 
        residual_arr = exp_trim - total_trim

        fit.histogram(residual_arr, nbins=nbins, density=density, f_lims=f_lims, xlabel=xlabel, x_symm=x_symm, barcolor=barcolor, fontsize=fontsize, name=filename, ext=ext, dpi=dpi)


def gen_iguess(x, experimental, param, model, model_args=[], sens0=1):
    """
    GUI for the interactive setup of a Parameters object to be used in a fitting procedure. 
    Once you initialized the Parameters object with the name of the parameters and a dummy value, you are allowed to set the value, minimum, maximum and vary status through the textboxes given in the right column, and see their effects in real time.
    Upon closure of the figure, the Parameters object with the updated entries is returned.
    
    Keybinding:
    > '>': increase sensitivity
    > '<': decrease sensitivity
    > 'up': increase value
    > 'down': decrease value
    > 'left': change parameter
    > 'right': change parameter
    > 'v': change "vary" status
    > '<': toggle automatic zoom adjustment
    ---------
    Parameters:
    - x: 1darray
        Independent variable
    - experimental: 1darray
        The objective values you are trying to fit
    - param: lmfit.Parameters Object
        Initialized parameters object
    - model: function
        Function to be used for the generation of the fit model. Param must be the first argument.
    - model_args: list
        List of args to be passed to model, after param
    - sens0: float
        Default sensitivity for the change of the parameters with the mouse
    ---------
    Returns:
    - param: lmfit.Parameters Object
        Updated Parameters Object
    """

    class Event:
        """ Custom 'event' to trigger certain reactions from the widgets """
        def __init__(self, event=None, key=None, button=None):
            """ Initialize the parameters as you want """
            self.event = event
            self.key = key
            self.button = button

    nullevent = Event()         # Just a placeholder: event that does nothing

    
    names = [key for key in param]          # Name of the parameters, from the param dictionary
    K = 0                                   # List index for the active parameter
    act = names[K]                          # Name of the active parameter
    zoom_toggle = True                      # Allow automatic zoom adjustment

    sens = {key: sens0 for key in param}    # Initialize the sensitivity dictionary

    # Make the figure
    fig = plt.figure('Computation of General Initial Guess')
    fig.set_size_inches(figures.figsize_large)
    plt.subplots_adjust(top=0.95, bottom=0.05, left=0.05, right=0.8, hspace=0.6, wspace=0.2)
    ax = fig.add_subplot(5,1,(1,4))
    axr = fig.add_subplot(5,1,5)

    # Boxes
    up_box = plt.axes([0.825, 0.875, 0.075, 0.075])             # increase sensitivity
    down_box = plt.axes([0.900, 0.875, 0.075, 0.075])           # decrease sensitivity

    val_box = plt.axes([0.825, 0.575, 0.05, 0.075])             # textbox to insert value
    min_box = plt.axes([0.875, 0.575, 0.05, 0.075])             # textbox to insert min
    max_box = plt.axes([0.925, 0.575, 0.05, 0.075])             # textbox to insert max

    pup_box = plt.axes([0.825, 0.275, 0.075, 0.075])            # increase parameter list index
    pdown_box = plt.axes([0.900, 0.275, 0.075, 0.075])          # decrease parameter list index

    print_box = plt.axes([0.825, 0.05, 0.15, 0.075])            # print the Parameter to stdout

    vary_box = plt.axes([0.95, 0.80, 0.0425, 0.04])             # slider for "vary"

    # Widgets   -   as the boxes
    up_button = Button(up_box, r'$\uparrow$', hovercolor='0.975')           
    down_button = Button(down_box, r'$\downarrow$', hovercolor='0.975')

    sens_text = ax.text(0.825, 0.825, f'Sens: {sens[act]:.5g}', ha='left', va='center', transform=fig.transFigure, fontsize=12)
    value_text = ax.text(0.900, 0.775, f'{act}', ha='center', va='center', transform=fig.transFigure, fontsize=20, color='tab:blue')

    # The texts describe what the textboxes are associated to
    val_tb = TextBox(val_box, '', textalignment='center', initial=f'{param[act].value}')
    ax.text(0.825 + 0.05/2, 0.575+0.075, 'VAL', ha='center', va='bottom', transform=fig.transFigure, fontsize=16)
    min_tb = TextBox(min_box, '', textalignment='center', initial=f'{param[act].min}')
    ax.text(0.875 + 0.05/2, 0.575+0.075, 'MIN', ha='center', va='bottom', transform=fig.transFigure, fontsize=16)
    max_tb = TextBox(max_box, '', textalignment='center', initial=f'{param[act].max}')
    ax.text(0.925 + 0.05/2, 0.575+0.075, 'MAX', ha='center', va='bottom', transform=fig.transFigure, fontsize=16)

    pup_button = Button(pup_box, '\n'.join(['CH. PAR.',r'$\uparrow$']), hovercolor='0.975')
    pdown_button = Button(pdown_box, '\n'.join(['CH. PAR.',r'$\downarrow$']), hovercolor='0.975')
    
    print_button = Button(print_box, 'PRINT PARAMETERS', hovercolor='0.975')

    valinit = [1 if param[act].vary else 0]
    vary_sl= Slider(vary_box, 'Vary', valmin=0, valmax=1, valinit=valinit[0], valstep=1) 

    # ---------------------------------------------------------------------------------------
    # SLOTS
    def up_sens(event):
        """ Double sensitivity of the active parameter """
        nonlocal sens
        sens[act] *= 2
        sens_text.set_text(f'Sens: {sens[act]:.5g}')
        plt.draw()

    def down_sens(event):
        """ Halves sensitivity of the active parameter """
        nonlocal sens
        sens[act] /= 2
        sens_text.set_text(f'Sens: {sens[act]:.5g}')
        plt.draw()

    def update_max(text):
        """ Update the 'max' value of the active parameter """
        def get_val(tb):
            """ Overwrite inf with np.inf otherwise raises error """
            if 'inf' in tb.text:
                return eval(tb.text.replace('inf', 'np.inf'))
            else:
                return eval(tb.text)

        nonlocal param
        param[act].set(max=get_val(max_tb))

    def update_min(text):
        """ Update the 'min' value of the active parameter """
        def get_val(tb):
            """ Overwrite inf with np.inf otherwise raises error """
            if 'inf' in tb.text:
                return eval(tb.text.replace('inf', 'np.inf'))
            else:
                return eval(tb.text)

        nonlocal param
        param[act].set(min=get_val(min_tb))

    def update_val(text):
        """ Update the 'value' of the active parameter """
        nonlocal param
        param[act].set(value=eval(text))
        # Update the plots: we need nullevent to avoid raising errors
        on_scroll(nullevent)

    def cycle():
        """ Redraws the information on values and stuff """
        nonlocal act
        act = names[K]
        value_text.set_text(f'{act}')
        val_tb.set_val(f'{param[act].value:.5g}')
        min_tb.set_val(f'{param[act].min:.5g}')
        max_tb.set_val(f'{param[act].max:.5g}')
        if param[act].vary: # = True
            vary_sl.set_val(1)
        else:               # = False
            vary_sl.set_val(0)
        plt.draw()

    def cycle_up(event):
        """ Increase the parameter list index """
        nonlocal K
        # when reaches top border, starts over
        K = np.mod(K+1, len(names))
        cycle()

    def cycle_down(event):
        """ Decrease the parameter list index """
        nonlocal K
        # when reaches bottom border, starts over
        K = np.mod(K-1, len(names))
        cycle()

    def on_scroll(event):
        """ Updates the value of the active parameter and draws the new model """
        nonlocal param
        if event.button == 'up': 
            param[act].value += sens[act]
        if event.button == 'down':
            param[act].value -= sens[act]
        val_tb.set_val(f'{param[act].value:.5g}')

        # Compute and redraw the model function
        newmodel = model(param, *model_args)
        model_plot.set_ydata(newmodel)
        # Update the residuals
        res_plot.set_ydata(experimental - newmodel)
        # Adjust the zoom interactively
        adjust_zoom(0)
        plt.draw()

    def update_vary(value):
        """ Set the 'vary' attribute according to the slider """
        nonlocal param
        if value == 0:
            param[act].set(vary=False)
        elif value == 1:
            param[act].set(vary=True)

    def adjust_zoom(event):
        """ Adjusts the zoom accordingly to model and exp, and the residuals """
        if not zoom_toggle:
            return
        # Compute new model
        newmodel = model(param, *model_args)
        # Adjust scale of top subplot
        misc.set_ylim(ax, [experimental, newmodel])
        misc.pretty_scale(ax, ax.get_ylim(), 'y')
        # Adjust scale of bottom subplot
        misc.set_ylim(axr, [experimental-newmodel, np.zeros_like(newmodel)]) # concatenate with 0 to keep the horizontal line visible
        misc.pretty_scale(axr, axr.get_ylim(), 'y', 4)
        plt.draw()

    def key_binding(event):
        """ Handles keyboard shortcuts """
        if event.key == '>':
            up_sens(nullevent)
        if event.key == '<':
            down_sens(nullevent)
        if event.key == 'up':
            upevent = Event(button='up')
            on_scroll(upevent)
        if event.key == 'down':
            downevent = Event(button='down')
            on_scroll(downevent)
        if event.key == 'right':
            cycle_up(nullevent)
        if event.key == 'left':
            cycle_down(nullevent)
        if event.key == 'v':
            param[act].set(vary=not(param[act].vary))
            cycle()
        if event.key == 'z':
            nonlocal zoom_toggle
            zoom_toggle = not(zoom_toggle)

    def print_param(event):
        """ Print the Parameters object to stdout """
        param.pretty_print()
        print()

    # ---------------------------------------------------------------------------------------


    #   Plot the data and the model
    ax.plot(x, experimental, '.', markersize=2, c='tab:red', label='Observed data')
    model_plot, = ax.plot(x, model(param, *model_args), c='tab:blue', label='Model')
    # Plot the residuals
    axr.axhline(0, c='k', lw=1.2)
    res_plot, = axr.plot(x, experimental - model(param, *model_args), '.', markersize=1, c='tab:green', label='Residuals')

    # Fancy shit
    misc.pretty_scale(ax, ax.get_xlim(), 'x')
    misc.pretty_scale(ax, ax.get_ylim(), 'y')
    misc.mathformat(ax)
    ax.legend()
    misc.set_fontsizes(ax, 15)
    misc.pretty_scale(axr, axr.get_xlim(), 'x')
    misc.pretty_scale(axr, axr.get_ylim(), 'y', 5)
    misc.mathformat(axr)
    axr.legend()
    misc.set_fontsizes(axr, 15)

    # Connect the widgets to their slots
    up_button.on_clicked(up_sens)
    down_button.on_clicked(down_sens)

    val_tb.on_submit(update_val)
    max_tb.on_submit(update_max)
    min_tb.on_submit(update_min)

    pup_button.on_clicked(cycle_up)
    pdown_button.on_clicked(cycle_down)
    vary_sl.on_changed(update_vary)

    print_button.on_clicked(print_param)

    fig.canvas.mpl_connect('scroll_event', on_scroll)
    fig.canvas.mpl_connect('key_press_event', key_binding)

    # Start event loop
    plt.show()

    return param


def peak_pick(ppm_f1, ppm_f2, data, coord_filename='coord.tmp'):
    """
    Make interactive peak_picking.
    The position of the selected signals are saved in coord_filename.
    If coord_filename already exists, the new signals are appended at its bottom: nothing is overwritten.
    Calls misc.select_traces for the selection.
    -------
    Parameters:
    - ppm_f1: 1darray
        ppm scale for the indirect dimension
    - ppm_f2: 1darray
        ppm scale for the direct dimension
    - data: 2darray
        Spectrum to peak-pick. The dimension should match the scale sizes.
    - coord_filename: str
        Path to the file where to save the peak coordinates
    -------
    Returns:
    - coord: list
        List of (u2, u1) for each peak
    """
    # Check for the existence of coord_filename
    if os.path.exists(coord_filename):
        with open(coord_filename, 'r') as Q:
            # number of already present signals: last linei, first value before tab
            n_C = eval(Q.readlines()[-1].split('\t')[0])   
        C = open(coord_filename, 'a', buffering=1)  
    else:
        C = open(coord_filename, 'w', buffering=1)
        C.write(r'#'+'\t'+f'{"u2":^8s},{"u1":^8s}'+'\n')    # Header line
        n_C = 0 

    # Make peak_picking
    coord = misc.select_traces(ppm_f1, ppm_f2, data)

    # Update the fucking coord file
    for k, obj in enumerate(coord):
        C.write(f'{k+1+n_C}'+'\t'+f'{obj[0]:-8.3f},{obj[1]:-8.3f}'+'\n')
    C.close()

    return coord

def gen_iguess_2D(ppm_f1, ppm_f2, tr1, tr2, u1, u2, acqus, fwhm0=100, procs=None):
    """
    Generate the initial guess for the fit of a 2D signal.
    The employes model is the one of a 2D Voigt signal, acquired with the States-TPPI scheme in the indirect dimension (i.e. sim.t_2DVoigt).
    The program allows for the inclusion of up to 10 components for the signal, in order to improve the fit.
    The acqus dictionary must contain the following keys: 
        > t1: acquisition timescale in the indirect dimension (States)
        > t2: acquisition timescale in the direct dimension
        > SFO1: Larmor frequency of the nucleus in the indirect dimension
        > SFO2: Larmor frequency of the nucleus in the direct dimension
        > o1p: carrier position in the indirect dimension /ppm
        > o2p: carrier position in the direct dimension /ppm
    The signals will be processed according to the values in the procs dictionary, if given; otherwise, they will be just zero-filled up to the data size (i.e. (len(ppm_f1), len(ppm_f2)) ). 
    -------
    Parameters:
    - ppm_f1: 1darray
        ppm scale for the indirect dimension
    - ppm_f2: 1darray
        ppm scale for the direct dimension
    - tr1: 1darray
        Trace of the original 2D peak in the indirect dimension
    - tr2: 1darray
        Trace of the original 2D peak in the direct dimension
    - u1: float
        Chemical shift of the original 2D peak in the indirect dimension /ppm
    - u2: float
        Chemical shift of the original 2D peak in the direct dimension /ppm
    - acqus: dict
        Dictionary of acquisition parameters
    - fwhm0: float
        Initial value for FWHM in both dimensions
    - procs: dict
        Dictionary of processing parameters
    -------
    Returns:
    - final_parameters: 2darray
        Matrix of dimension (# signals, 6) that contains, for each row: v1(Hz), v2(Hz), fwhm1(Hz), fwhm2(Hz), A, b
    - fit_interval: tuple of tuple
        Fitting window. ( (left_f1, right_f1), (left_f2, right_f2) )
    """

    ### FIRST OF ALL, THE FIGURE!
    fig = plt.figure('Manual Computation of Initial Guess - 2D')
    fig.set_size_inches(figures.figsize_large)
    plt.subplots_adjust(left=0.05, right=0.65, top=0.90, bottom=0.10, wspace=0.2)
    ax2, ax1 = [fig.add_subplot(1, 2, w+1) for w in range(2)]

    ### INITIALIZE THE VALUES

    # Values to be returned
    final_parameters = []
    fit_interval = None, None

    #limits of the window
    lim_f1 = u1 + 100/np.abs(acqus['SFO1']), u1 - 100/np.abs(acqus['SFO1'])
    lim_f2 = u2 + 100/np.abs(acqus['SFO2']), u2 - 100/np.abs(acqus['SFO2'])

    V = [{
        'u1': u1,   # ppm
        'u2': u2,   # ppm
        'fwhm1': fwhm0, # Hz
        'fwhm2': fwhm0, # Hz
        'k': 0.1,   # relative intensity
        'b': 0.5, # fraction of gaussianity
        } for w in range(10)]
    I1 = processing.integrate(tr1, x=ppm_f1, lims=lim_f1)
    I2 = processing.integrate(tr2, x=ppm_f2, lims=lim_f2)
    A = (I1 + I2) / (2*np.pi*fwhm0)


    # Sensitivity for mouse
    sens = {
        'u1': 0.25,   # ppm
        'u2': 0.25,   # ppm
        'fwhm1': 10, # Hz
        'fwhm2': 10, # Hz
        'k': 0.01,  # 1%
        'b': 0.1, # 10%
        'A': np.floor(np.log10(A)) - 1      # This goes according to order of magnitudes
        }


    # Copy initial values for reset
    V_in = [dict(q) for q in V]
    A_in = np.copy(A)
    sens_in = dict(sens)

    #conversion of the names from radio labels to dict keys
    conv_r2d = {
            r'$\delta$ /ppm': 'u',
            r'$\Gamma$ /Hz': 'fwhm',
            r'$k$': 'k',
            r'$\beta$': 'b',
            }

    #--------------------------------------------------------------------------
    ### SETUP OF THE INTERACTIVE FIGURE PANEL

    # make boxes for widgets
    tb2_boxes = [   # limits for the direct dimension
            plt.axes([0.050, 0.025, 0.05, 0.03]),   # left
            plt.axes([0.275, 0.025, 0.05, 0.03]),   # right
            ]
    tb1_boxes = [   # limits for the indirect dimension
            plt.axes([0.370, 0.025, 0.05, 0.03]),   # left
            plt.axes([0.600, 0.025, 0.05, 0.03]),   # right
            ]

    slider_box = plt.axes([0.68, 0.10, 0.01, 0.65])     # peak selector
    peak_box = plt.axes([0.72, 0.45, 0.10, 0.3])        # radiobuttons
    su_box = plt.axes([0.815, 0.825, 0.08, 0.075])      # increase sensitivity button
    giu_box = plt.axes([0.894, 0.825, 0.08, 0.075])     # decrease sensitivity button
    save_box = plt.axes([0.7, 0.825, 0.085, 0.04])      # SAVE button
    reset_box = plt.axes([0.7, 0.865, 0.085, 0.04])     # RESET button
    p_or_s_box = plt.axes([0.73, 0.78, 0.04, 0.03])     # F1 or F2 selector
    check_box = plt.axes([0.85, 0.1, 0.1, 0.7])         # Peak checker
    
    # Make widgets
    #   Buttons
    up_button = Button(su_box, r'$\uparrow$', hovercolor = '0.975')         # increase sensitivity button
    down_button = Button(giu_box, r'\downarrow$', hovercolor = '0.975')    # decrease sensitivity button
    save_button = Button(save_box, 'SAVE', hovercolor = '0.975')            # SAVE button
    reset_button = Button(reset_box, 'RESET', hovercolor = '0.975')         # RESET button
    
    #   textboxes
    TB1 = [TextBox(box, '', initial=f'{value:.2f}', textalignment='center') for box, value in zip(tb1_boxes, lim_f1)]   # set limits for F1
    TB2 = [TextBox(box, '', initial=f'{value:.2f}', textalignment='center') for box, value in zip(tb2_boxes, lim_f2)]   # set limits for F2

    #   Radiobuttons for parameter selection
    peak_name = [r'$\delta$ /ppm', r'$\Gamma$ /Hz', r'$k$', r'$\beta$', r'$A$']         # Labels for the parameters
    peak_radio = RadioButtons(peak_box, peak_name, active=2, activecolor='tab:blue')      # Actual radiobuttons
    
    #   Sliders
    #       Peak selector
    slider = Slider(ax = slider_box, label = 'Active\nSignal', valmin = 1, valmax = 10, valinit = 1, valstep = 1, orientation='vertical', color='tab:blue')
    #       Ruler for slider
    for i, H in enumerate(np.linspace(0.10, 0.75, 10)):
        plt.text(0.685, H, '$-$', ha='center', va='center', fontsize=20, color=COLORS[i], transform=fig.transFigure)
    #       Dimension selector
    f1_or_f2 = Slider(p_or_s_box, 'F', valmin=1, valmax=2, valinit=2, valstep=1, track_color='tab:blue', color='tab:orange')

    #   Checkbox: peak checker
    check_name = [str(w+1) for w in range(10)]      # 1, 2, 3...
    check_status = [False if w else True for w in range(10)]    # Only the first
    check = CheckButtons(check_box, check_name, check_status)   # Make the checkbutton

    #       Customize checkbox appearance
    #       ... make boxes more squared
    HBOX = check_box.dataLim.bounds[-1]
    misc.edit_checkboxes(check, xadj=0, yadj=0.001, dim=100, color=COLORS)

    # Text that shows the current values of the parameters
    head_print = plt.text(0.725, 0.4,
            '{:9s}:'.format(r'$\delta$ F2') + f'{V[0]["u2"]:-9.2f}\n'+
            '{:9s}:'.format(r'$\delta$ F1') + f'{V[0]["u1"]:-9.2f}\n'+
            '{:9s}:'.format(r'$\Gamma$ F2') + f'{V[0]["fwhm2"]:-9.2f}\n'+
            '{:9s}:'.format(r'$\Gamma$ F1') + f'{V[0]["fwhm1"]:-9.2f}\n'+
            '{:9s}:'.format(r'$k$') + f'{V[0]["k"]:-9.2f}\n'+
            '{:9s}:'.format(r'$\beta$') + f'{V[0]["b"]:-9.2f}\n'+
            '{:9s}:'.format(r'$A$') + f'{A:-9.2e}\n',
            ha='left', va='top', transform=fig.transFigure, fontsize=12, color=COLORS[0])

    #--------------------------------------------------------------------------
    # SLOTS
    def reset(event):
        """ Bring all parameters and sens to the starting values """
        nonlocal V, A, sens
        # Copy initial values for reset
        V = [dict(q) for q in V_in]
        A = np.copy(A_in)
        sens = dict(sens_in)
        [update(q) for q in range(10)]

    def edit_lims(text):
        """ Read the limits from the textboxes and change the scales accordingly.  """
        nonlocal lim_f1, lim_f2
        lim_f1 = [eval(TB.text) for TB in TB1]
        lim_f2 = [eval(TB.text) for TB in TB2]
        misc.pretty_scale(ax2, lim_f2, 'x')
        misc.pretty_scale(ax1, lim_f1, 'x')
        fig.canvas.draw()

    def set_visibility(event):
        """
        Set the signals visible or invisible according to their status in the checkbutton.
        Recomputes the total function considering only the active signals.
        """
        slider.set_val(eval(event))     # Moves the selector to the "new" peak for easier handling
        for k, stat in enumerate(check.get_status()):
            s1_plot[k].set_visible(stat)
            s2_plot[k].set_visible(stat)

        total_1 = np.sum([sgn_1[k] for k in range(len(V)) if check.get_status()[k]], axis=0)
        t1_plot.set_ydata(total_1)
        total_2 = np.sum([sgn_2[k] for k in range(len(V)) if check.get_status()[k]], axis=0)
        t2_plot.set_ydata(total_2)
        fig.canvas.draw()


    def make_sgn_2D(values, acqus, N=None, procs=None):
        """
        Create a 2D signal according to the final parameters returned by make_iguess_2D.
        -------
        Parameters:
        - final_parameters: list or 2darray
            sequence of the parameters: u1, u2, fwhm1, fwhm2, I, b
        - acqus: dict
            2D-like acqus dictionary containing the acquisition timescales (keys t1 and t2)
        - N: tuple of int
            Zero-filling values (F1, F2). Not read if procs is not None
        - procs: dict
            2D-like procs dictionary.
        -------
        Returns:
        - peaks: list of 2darray
            rr part of the generated signals
        """
        # Shallow copy of acquisition timescales
        t1 = np.copy(acqus['t1'])
        t2 = np.copy(acqus['t2'])
        # Organize the parameters
        to_pass = [
                misc.ppm2freq(values[0], acqus['SFO1'], acqus['o1p']),  # u1 from ppm to Hz
                misc.ppm2freq(values[1], acqus['SFO2'], acqus['o2p']),  # u2 from ppm to Hz
                values[2] * 2 * np.pi,    # fwhm1 from Hz to radians
                values[3] * 2 * np.pi,    # fwhm2 from Hz to radians
                values[4],  # Intensity
                values[5],   # b
                ]
        signal = sim.t_2Dvoigt(t1, t2, *to_pass)    # Make the 2D signal

        if procs is not None:   # Processing according to procs
            peak, *_ = processing.xfb(signal, wf=procs['wf'], zf=procs['zf'])
        else:   # just zero-fill before FT
            peak, *_ = processing.xfb(signal, zf=N)     # Just zero-fill

        # Extract the traces
        tr_f1 = misc.get_trace(peak, ppm_f2, ppm_f1, a=values[1], column=True)  # F2 @ u1 ppm
        tr_f2 = misc.get_trace(peak, ppm_f2, ppm_f1, a=values[0], column=False) # F1 @ u2 ppm

        return tr_f1, tr_f2

    def get_key2edit():
        """ Makes the conversion between the radiobutton labels and the keys of V/ A """
        F = f'{f1_or_f2.val}'   # F1 or F2
        label = peak_radio.value_selected    # active parameter
        if label in conv_r2d.keys():    # i.e. it is not A
            key2edit = f'{conv_r2d[label]}'
            if 'k' not in key2edit and 'b' not in key2edit:   # i.e. it is u or fwhm
                key2edit += F   # add 1 for f1 or 2 for f2
        else:
            key2edit = 'A'
        return key2edit

    def update(s_idx):
        """ 
        Computes the s_idx-th 2D signal, extract the traces in F1 and F2, then redraws them.
        Updates the total functions with the sum of the active signals.
        """
        nonlocal sgn_1, sgn_2
        # Organize the parameters
        values = [V[s_idx][f'{key}'] for key in ('u1', 'u2', 'fwhm1', 'fwhm2', 'k', 'b')]
        values[-2] *= A
        # Compute the 2D signal and extract the traces
        sgn_1[s_idx], sgn_2[s_idx] = make_sgn_2D(values, acqus, N=(N1,N2), procs=procs)
        
        # Update the plots:
        #   F1
        s1_plot[s_idx].set_ydata(sgn_1[s_idx])  # update plot
        total_1 = np.sum([sgn_1[k] for k in range(len(V)) if check.get_status()[k]], axis=0)
        t1_plot.set_ydata(total_1.real)
        #   F2
        s2_plot[s_idx].set_ydata(sgn_2[s_idx])
        total_2 = np.sum([sgn_2[k] for k in range(len(V)) if check.get_status()[k]], axis=0)
        t2_plot.set_ydata(total_2.real)

        redraw_text(0)      # Update the text with the new values
        fig.canvas.draw()

    def roll(event):
        """ Slot for the mouse wheel """
        nonlocal V, A
        s_idx = slider.val - 1 # active signal
        key2edit = get_key2edit()   # active parameter
        if key2edit == 'A':     # move A
            if event.button == 'up':
                A += 10**sens['A']
            elif event.button == 'down':
                A -= 10**sens['A']
        else:   # Move the selected parameter
            if event.button == 'up':
                V[s_idx][key2edit] += sens[key2edit]
            elif event.button == 'down':
                V[s_idx][key2edit] -= sens[key2edit]
        # Safety check for FWHM
        if V[s_idx]['fwhm1'] <= 1:
            V[s_idx]['fwhm1'] = 1
        if V[s_idx]['fwhm2'] <= 1:
            V[s_idx]['fwhm2'] = 1
        # Safety check for b
        if V[s_idx]['b'] <= 0:
            V[s_idx]['b'] = 0
        elif V[s_idx]['b'] >= 1:
            V[s_idx]['b'] = 1
        update(s_idx)   # Redraw everything and update text

    def up_sens(event):
        """ Slot for the up-arrow button"""
        nonlocal sens
        key2edit = get_key2edit()
        if key2edit == 'A': # increase it by one order of magnitude
            sens['A'] += 1
        else:    # double
            sens[key2edit] *= 2

    def down_sens(event):
        """ Slot for the down-arrow button"""
        nonlocal sens
        key2edit = get_key2edit()
        if key2edit == 'A':
            sens['A'] -= 1  # decrease it by one order of magnitude
        else:   # halve
            sens[key2edit] /= 2

    def redraw_text(event):
        """ Updates the text according to the current values. Also changes its color. """
        s_idx = slider.val - 1  # python numbering
        value_string = '{:9s}:'.format(r'$\delta$ F2') + f'{V[s_idx]["u2"]:-9.2f}\n'+ '{:9s}:'.format(r'$\delta$ F1') + f'{V[s_idx]["u1"]:-9.2f}\n'+ '{:9s}:'.format(r'$\Gamma$ F2') + f'{V[s_idx]["fwhm2"]:-9.2f}\n'+ '{:9s}:'.format(r'$\Gamma$ F1') + f'{V[s_idx]["fwhm1"]:-9.2f}\n'+ '{:9s}:'.format(r'$k$') + f'{V[s_idx]["k"]:-9.2f}\n'+ '{:9s}:'.format(r'$\beta$') + f'{V[s_idx]["b"]:-9.2f}\n'+ '{:9s}:'.format(r'$A$') + f'{A:-9.2e}\n'
        head_print.set_text(value_string)
        head_print.set_color(COLORS[s_idx]) # color of the active signal
        fig.canvas.draw()

    def save(event):
        """ Slot for the save button: store the current values into the final variables """
        nonlocal final_parameters, fit_interval
        final_parameters = [[   # u1, u2, fwhm1, fwhm2, k*A, b
                    misc.ppm2freq(V[x]['u1'], acqus['SFO1'], acqus['o1p']),
                    misc.ppm2freq(V[x]['u2'], acqus['SFO2'], acqus['o2p']),
                    V[x]['fwhm1'],
                    V[x]['fwhm2'],
                    V[x]['k'] * A,
                    V[x]['b'],
                    ] for x in range(len(V)) if check.get_status()[x]]
        # ( (L_F1, R_F1), (L_F2, R_F2) )
        fit_interval = tuple([eval(x.text) for x in TB1]), tuple([eval(x.text) for x in TB2])

    #--------------------------------------------------------------------------

    N1, N2 = ppm_f1.shape[-1], ppm_f2.shape[-1]         # Zero-filling dimension
    if procs is None:   # I do not care
        proc1s, proc2s = None, None
    else:   # Split it into two 1D-like procs dictionaries
        proc1s, proc2s = misc.split_procs_2D(procs)

    # Figure titles
    ax2.set_title(f'F2 trace @ {u1:.1f} ppm')
    ax1.set_title(f'F1 trace @ {u2:.1f} ppm')
    
    # red dashed line as marker for the initially selected chemical shifts
    ax2.axvline(u2, c='r', lw=0.3, ls='--')
    ax1.axvline(u1, c='r', lw=0.3, ls='--')

    # Draw the experimental spectrum
    ax2.plot(ppm_f2, tr2, c='k', lw=1.0, label='Exp.')
    ax1.plot(ppm_f1, tr1, c='k', lw=1.0, label='Exp.')

    # Initialize the simulated signals with the starting values
    #   F1
    sgn_1, sgn_2 = [], []
    for k, Vline in enumerate(V_in):
        # Organize parameters
        values = [Vline[f'{key}'] for key in ('u1', 'u2', 'fwhm1', 'fwhm2', 'k', 'b')]
        values[-2] *= A
        # Build the 2D signal and extract the traces
        tmp1, tmp2 = make_sgn_2D(values, acqus, N=(N1, N2), procs=procs)
        sgn_1.append(tmp1)
        sgn_2.append(tmp2)

    s1_plot = []        # lines
    for i in range(len(V)):
        temp1, = ax1.plot(ppm_f1, sgn_1[i].real, c=COLORS[i], lw=1.0, ls='--')
        s1_plot.append(temp1)
        s1_plot[i].set_visible(check.get_status()[i])
    total_1 = np.sum([sgn_1[k] for k in range(len(V)) if check.get_status()[k]], axis=0)    # spectrum
    t1_plot, = ax1.plot(ppm_f1, total_1.real, label='Fit', c='blue', lw=1.0)    # line

    s2_plot = []        # lines
    for i in range(len(V)):
        temp2, = ax2.plot(ppm_f2, sgn_2[i].real, c=COLORS[i], lw=1.0, ls='--')
        s2_plot.append(temp2)
        s2_plot[i].set_visible(check.get_status()[i])
    total_2 = np.sum([sgn_2[k] for k in range(len(V)) if check.get_status()[k]], axis=0)    # spectrum
    t2_plot, = ax2.plot(ppm_f2, total_2.real, label='Fit', c='blue', lw=1.0)    # line

    # Fancy shit
    #   x scales
    ax2.set_xlabel(r'$\delta$ '+f'{misc.nuc_format(acqus["nuc2"])}'+r' /ppm')
    ax1.set_xlabel(r'$\delta$ '+f'{misc.nuc_format(acqus["nuc1"])}'+r' /ppm')
    misc.pretty_scale(ax2, lim_f2, 'x')
    misc.pretty_scale(ax1, lim_f1, 'x')
    for ax in (ax2, ax1):
        # y scales
        misc.pretty_scale(ax, ax.get_ylim(), 'y')
        misc.mathformat(ax, 'y')
        # Draw legend
        ax.legend()
        # Bigger fontsizes
        misc.set_fontsizes(ax, 14)

    # Connect widgets to the slots
    #   Textboxes
    [TB.on_submit(edit_lims) for TB in TB1]
    [TB.on_submit(edit_lims) for TB in TB2]
    #   Checkbox
    check.on_clicked(set_visibility)
    #   Slider
    slider.on_changed(redraw_text)
    #   Mouse scroll
    fig.canvas.mpl_connect('scroll_event', roll)

    #   up-down buttons
    up_button.on_clicked(up_sens)
    down_button.on_clicked(down_sens)

    #   reset and save
    reset_button.on_clicked(reset)
    save_button.on_clicked(save)

    # fill the return variables with the starting ones
    save('initial values')

    plt.show()
    plt.close()

    return final_parameters, fit_interval


def build_2D_sgn(parameters, acqus, N=None, procs=None):
    """
    Create a 2D signal according to the final parameters returned by make_iguess_2D.
    Process it according to procs.
    -------
    Parameters:
    - parameters: list or 2darray
        sequence of the parameters: u1, u2, fwhm1, fwhm2, I, b. Multiple components are allowed
    - acqus: dict
        2D-like acqus dictionary containing the acquisition timescales (keys t1 and t2)
    - N: tuple of int
        Zero-filling values (F1, F2). Read only if procs is None
    - procs: dict
        2D-like procs dictionary.
    -------
    Returns:
    - peak: 2darray
        rr part of the generated signal
    """
    parameters = np.array(parameters)
    if len(parameters.shape) == 1:
        parameters = parameters.reshape(1,-1)
    # Get timescales from acqus
    t1 = np.copy(acqus['t1'])
    t2 = np.copy(acqus['t2'])
    signals = []        # Time domain
    for k, values in enumerate(parameters):
        values[0] = misc.ppm2freq(values[0], acqus['SFO1'], acqus['o1p'])
        values[1] = misc.ppm2freq(values[1], acqus['SFO2'], acqus['o2p'])
        values[2] *= 2*np.pi    # fwhm1 from Hz to radians
        values[3] *= 2*np.pi    # fwhm2 from Hz to radians
        sgn = sim.t_2Dvoigt(t1, t2, *values)    # Make the signal in the time domain
        signals.append(sgn)
    signal = np.sum(signals, axis=0)    # Sum the components

    if procs is not None: # Process the data according to procs
        peak, *_ = processing.xfb(signal, wf=procs['wf'], zf=procs['zf'])
    else:
        peak, *_ = processing.xfb(signal, zf=N)     # Just zero-fill

    return peak
#----------------------------------------------------------------------------------------------------------------------

class Voigt_Fit_2D:
    """
    Class that wraps methods for the fit of 2D spectra with a set of 2D Voigtian lines.
    This is work in progress.
    """
    def __init__(self, ppm_f1, ppm_f2, data, acqus, procs=None, label_list=None):
        """
        Initialize the class with ppm scales, experimental spectrum, acqus and procs dictionaries.
        -------
        Parameters:
        - ppm_f1: 1darray
            ppm scale for the indirect dimension
        - ppm_f2: 1darray
            ppm scale for the direct dimension
        - data: 2darray
            Spectrum to fit. The dimension should match the scale sizes.
        - acqus: dict
            Dictionary of acquisition parameters
        - procs: dict
            Dictionary of processing parameters
        - label_list: list
            Labels for the peaks
        """
        self.ppm_f1 = np.copy(ppm_f1)
        self.ppm_f2 = np.copy(ppm_f2)
        self.data = np.copy(data)
        self.acqus = dict(acqus)
        if procs is None:
            self.procs = None
        else:
            self.procs = dict(procs)
        self.label_list = label_list

    def plot(self, name=None, show_exp=True, dpi=600, **kwargs):
        """ 
        Draw a plot of the guessed/fitted peaks.
        -------
        Parameters:
        - name: str or None
            Filename for the figure. If it is None, the figure is shown.
        - show_exp: bool
            Choose if to plot the experimental spectrum or not
        - dpi: int
            Resolution of the saved image
        - kwargs: keyworded arguments
            Additional parameters to be passed to figures.ax2D.
        """

        # Generate the full spectrum to make the plot computationally less expensive
        fitted_data = np.sum(self.peaks, axis=0)

        # Make the figure
        fig = plt.figure('Fit')
        fig.set_size_inches(figures.figsize_large)
        ax = fig.add_subplot(1,1,1)
        if 'cmap' in kwargs.keys():
            kwargs.pop('cmap')
            
        if show_exp:
            if 'lvl' in kwargs.keys():
                l_E = kwargs['lvl']
                kwargs.pop('lvl')
            else:
                l_E = 0.1
            # Plot experimental spectrum
            figures.ax2D(ax, self.ppm_f2, self.ppm_f1, self.data, cmap=CM['Greys_r'], lvl=l_E, **kwargs)
            m_E = np.max(self.data)
            m_C = np.max(fitted_data)
            l_C = l_E * m_E / m_C
        else:
            if 'lvl' in kwargs.keys():
                l_C = kwargs['lvl']
                kwargs.pop('lvl')
            else:
                l_C = 0.1

        # Plot fitted spectrum
        figures.ax2D(ax, self.ppm_f2, self.ppm_f1, fitted_data, cmap=CM['Greens_r'], lvl=l_C, **kwargs)
        # Draw the labels of the fitted peaks according to coord and peak_labels
        if 'fontsize' in kwargs.keys():
            labelsize = kwargs['fontsize'] - 2
        else:
            labelsize = 8
        self.draw_crossmarks(self.coord, ax, markersize=5, labelsize=labelsize, label_list=self.label_list)

        # Visual shit
        ax.set_xlabel(r'$\delta$ '+f'{misc.nuc_format(self.acqus["nuc2"])}'+r' /ppm')
        ax.set_ylabel(r'$\delta$ '+f'{misc.nuc_format(self.acqus["nuc1"])}'+r' /ppm')
        # Save/plot the figure
        if name is None:
            misc.set_fontsizes(ax, 14)
            plt.show()
        else:
            plt.savefig(f'{name}.tiff', dpi=dpi)
        plt.close()

    @staticmethod
    def draw_crossmarks(coord, ax, label_list=None, markersize=5, labelsize=8, markercolor='tab:blue', labelcolor='b'):
        """
        Draw crossmarks and peak labels on a figure.
        -------
        Parameters:
        - ax: matplotlib.Subplot object
            Subplot where to plot the crossmarks and the labels.
        - label_list: list
            Labels for the peaks. If None, they are computed as 1, 2, 3, ...
        - markersize: int
            Dimension of the crossmark
        - labelsize: int
            Fontsize for the labels
        - markercolor: str
            Color of the crossmark
        - labelcolor: str
            Color of the labels
        """

        for k, C in enumerate(coord):
            x, y = C    # position of the crossmark
            ax.plot(x, y, '+', c=markercolor, ms=markersize)    # Plot crossmark
            # Draw the text on the top-right of the crossmark
            if label_list is None:
                ax.text(x, y, f'{k+1}', c=labelcolor, ha='left', va='bottom', fontsize=labelsize)
            else:
                ax.text(x, y, f'{label_list[k]}', c=labelcolor, ha='left', va='bottom', fontsize=labelsize)

    def peak_pick(self, coord_filename='coord.tmp'):
        """
        Performs peak_picking by calling fit.peak_pick.
        Saves the list of peak positions in the attribute coord
        -------
        Parameters:
        - coord_filename: str
            Path to the file where to save the peak coordinates
        """
        fit.peak_pick(self.ppm_f1, self.ppm_f2, self.data, coord_filename)

    def load_coord(self, coord_filename='coord.tmp'):
        """
        Read the values from the coord filename and save them into the attribute "coord".
        --------
        Parameters:
        - coord_filename: str
            Path to the file to be read
        """
        f = open(coord_filename, 'r')
        R = f.readlines()

        coord = []
        label_list = []
        for k, line in enumerate(R):
            if line[0] == '#' or line.isspace():    # Skip comments and empty lines
                continue
            else:   
                x, y = eval(line.split('\t',2)[1].strip('\n'))  # second and third column
                coord.append([x, y])
                if len(line.split('\t',2)) > 2: # If there is the label
                    label = line.split("\t",2)[-1].strip("\n")
                    if not label.isspace():
                        label_list.append(f'{label}')
        # Store coord into the attribute coord
        self.coord = coord
        print(f'Loaded {coord_filename} as coord.')

        # Update label_list, if there are labels in the coord file
        if len(label_list) > 0:
            self.label_list = label_list
        if self.label_list is not None:
            if len(self.label_list) < len(self.coord):
                raise ValueError('The number of provided labels is not enough for the peaks.')

    def draw_coord(self, filename=None, labelsize=8, ext='tiff', dpi=600, **kwargs):
        """
        Makes a figure with the experimental dataset and the peak-picked signals as crosshairs.
        --------
        Parameters:
        - filename: str or None
            Filename for the figure to be saved. If None, it is shown instead.
        - labelsize: float
            Font size for the peak index
        - ext: str
            Format of the image
        - dpi: int
            Resolution of the saved image in dots per inches
        - kwargs: keyworded arguments
            Additional options for figures.ax2D
        """

        fig = plt.figure('Picked Peaks')
        fig.set_size_inches(figures.figsize_large)
        ax = fig.add_subplot(1,1,1)

        figures.ax2D(ax, self.ppm_f2, self.ppm_f1, self.data, **kwargs)
        self.draw_crossmarks(self.coord, ax, label_list=self.label_list, markersize=5, labelsize=labelsize, markercolor='tab:blue', labelcolor='b')
        
        ax.set_xlabel(r'$\delta$ '+f'{misc.nuc_format(self.acqus["nuc2"])}'+ r' /ppm')
        ax.set_ylabel(r'$\delta$ '+f'{misc.nuc_format(self.acqus["nuc1"])}'+ r' /ppm')
        misc.set_fontsizes(ax, 14)
        if filename is None:
            plt.show()
        else:
            plt.savefig(f'{filename}.{ext}', dpi=dpi)
        plt.close()


    @cron
    def make_peaks(self, idx, V):
        """
        Calculate the set of 2D peaks, given the matrix of their parameters and their index.
        The array of indexes is required in order to recognize the different components that contribute to a single peak.
        The attribute peaks of the class will be cleared and updated.
        --------
        Parameters:
        - idx: 1darray
            Array of indexes of the peaks.
        - V: list or 2darray
            List of parameters that describe the peaks.
        """

        self.peaks = []     # Clear peaks attribute
        # Get the number of signals
        if isinstance(V, np.ndarray):
            n_sgn = V.shape[0]
        elif isinstance(V, list):
            n_sgn = len(V)

        print('Calculation of the peaks from the input file.\nThis might take a while...')
        for k in range(n_sgn):
            peak_index = k + 1  # Peak index
            # Read the whole file, append only the parameters labelled with peak_index 
            tmp_par = [V[k] for k in range(n_sgn) if idx[k] == peak_index]
            if len(tmp_par) == 0:   # No peaks found -> skip
                del tmp_par
                continue
            # Compute the signal and put it into peaks
            self.peaks.append(fit.build_2D_sgn(tmp_par, self.acqus, N=(len(self.ppm_f1), len(self.ppm_f2)), procs=self.procs))
            del tmp_par # Clear memory
        print('Done.', end=' ') # remove \n so that the runtime is shown in the same line


    def load_iguess(self, filename='peaks.inp'):
        """
        Reads the initial guess file with the parameters of the peaks, separates the values and stores them into attributes.
        In particular: 
            > idx will contain the peak index (first column of the file), 
            > Vi will contain [u1, u2, fwhm1, fwhm2, Im, b] for each peak,
            > Wi will contain the fitting interval as ( (L_f1, R_f1), (L_f2, R_f2) )
        --------
        Parameters:
        - filename: str
            Path to the input file to be read
        """

        # Safety check: if filename does exist
        if os.path.exists(filename): # open the file and reads the lines,
            f = open(filename, 'r')
            R = f.readlines()
        else:   # raises error
            raise NameError(f'{filename} does not exist.')

        # Initialize empty attributes
        self.idx = []
        self.Vi = []
        self.Wi = []

        for k, line in enumerate(R):    # Loop on the lines of the file
            if line[0] == r'#' or line.isspace():
                continue    # Skip empty lines and comments
            index, values, fit_interval = self._read_par_line(line)   
            self.idx.append(index)
            self.Vi.append(values)
            self.Wi.append(fit_interval)
        f.close()
        self.make_peaks(self.idx, self.Vi)

    def load_fit(self, filename='fit.out'):
        """
        Reads the file with the parameters of the fitted peaks, separates the values and stores them into attributes.
        Then, uses these values to compute the peaks and save them into self.peaks.
        In particular: 
            > idx will contain the peak index (first column of the file), 
            > Vf will contain [u1, u2, fwhm1, fwhm2, Im, b] for each peak,
            > Wf will contain the fitting interval as ( (L_f1, R_f1), (L_f2, R_f2) )
        --------
        Parameters:
        - filename: str
            Path to the input file to be read
        """

        # Safety check: if filename does exist
        if os.path.exists(filename): # open the file and reads the lines,
            f = open(filename, 'r')
            R = f.readlines()
        else:   # raises error
            raise NameError(f'{filename} does not exist.')

        # Initialize empty attributes
        self.idx = []
        self.Vf = []
        self.Wf = []

        for k, line in enumerate(R):    # Loop on the lines of the file
            if line[0] == r'#' or line.isspace():
                continue    # Skip empty lines and comments
            index, values, fit_interval = self._read_par_line(line)   
            self.idx.append(index)
            self.Vf.append(values)
            self.Wf.append(fit_interval)
        f.close()
        self.make_peaks(self.idx, self.Vf)  # Update the peaks attribute

    def iguess(self, filename='peaks.inp', start_index=1, only_edit=None, fwhm0=100, overwrite=False, auto=False):
        """
        Make the initial guess for all the peaks.
        ---------
        Parameters:
        - filename: str
            Path to the file where the peak parameters will be written
        - start_index: int
            Index of the first peak to be guessed. 
        - only_edit: sequence of ints or None
            Index of the peak that have to be guessed interactively. The ones that do not appear here are guessed automatically.
        - fwhm0: float
            Default value for fwhm in both dimension for automatic guess
        - overwrite: bool
            Choose if to overwrite the file or append the new peaks at the bottom
        - auto: bool
            Allow automatic guess for the peaks. To be used in conjunction with only_edit: if auto is False, all the peaks are guessed interactively!
        ----------
        """
        def auto_val(ppm_f1, ppm_f2, tr1, tr2, u1, u2, fwhm0, acqus):
            """ Compute initial guess automatically """
            # Limits
            lim_f1 = u1 + 100/np.abs(acqus['SFO1']), u1 - 100/np.abs(acqus['SFO1'])
            lim_f2 = u2 + 100/np.abs(acqus['SFO2']), u2 - 100/np.abs(acqus['SFO2'])
            interval = lim_f1, lim_f2
            # Parameters
            parameters = [[   # u1, u2, fwhm1, fwhm2, k*A, b
                misc.ppm2freq(u1, acqus['SFO1'], acqus['o1p']), # v1 /Hz
                misc.ppm2freq(u2, acqus['SFO2'], acqus['o2p']), # v2 /Hz
                fwhm0,  # fwhm1 /Hz
                fwhm0,  # fwhm2 /Hz
                1,     # I
                0.5,    # b
                ]] 
            # Integral
            A0 = np.max(self.data) / np.prod(self.data.shape)**0.5
            parameters[0][-2] = A0
            
            return parameters, interval
        #-------------------------------------------------------------------------------

        if os.path.exists(filename) and overwrite is False:    # append next peaks 
            f = open(filename, 'a', buffering=1)
        else:   # create a new file
            f = open(filename, 'w', buffering=1)
            self._write_head_line(f)

        # Make the generator where to loop on peaks
        def extract(coord):
            """ Generator: yields the chemical shifts and the traces onto which to loop """
            for x, y in coord:   # u2, u1
                tr1 = misc.get_trace(self.data, self.ppm_f2, self.ppm_f1, x, column=True)  # F1 @ u2 ppm
                tr2 = misc.get_trace(self.data, self.ppm_f2, self.ppm_f1, y, column=False) # F2 @ u1 ppm
                yield (y, tr1), (x, tr2)    # (u1, f1), (u2, f2)
        peaks_coord = extract(self.coord)   # Call the generator

        # Start looping
        peak_index = 1
        for TR1, TR2 in peaks_coord:
            if peak_index < start_index:    # Do not guess
                peak_index += 1
                continue
            print(f'Preparing iguess for {peak_index:4.0f}/{len(self.coord):4.0f} peak', end='\r')
            # Unpack TR1 and TR2
            u1, tr1 = TR1
            u2, tr2 = TR2

            if auto is True and only_edit is None:  # All automatic
                parameters, interval = auto_val(self.ppm_f1, self.ppm_f2, tr1, tr2, u1, u2, fwhm0, self.acqus)
            elif auto is True and only_edit is not None:    # Interactively guess only the given peaks, all the others automatically
                if peak_index in only_edit:
                    parameters, interval = fit.gen_iguess_2D(self.ppm_f1, self.ppm_f2, tr1, tr2, u1, u2, self.acqus, fwhm0, self.procs)
                else:
                    parameters, interval = auto_val(self.ppm_f1, self.ppm_f2, tr1, tr2, u1, u2, fwhm0, self.acqus)
            else:   # All interactively
                parameters, interval = fit.gen_iguess_2D(self.ppm_f1, self.ppm_f2, tr1, tr2, u1, u2, self.acqus, fwhm0, self.procs)

            for values in parameters:   # Write the parameters
                self._write_par_line(f, self.acqus, peak_index, values, interval_f1=interval[0], interval_f2=interval[1])
            peak_index += 1 # Increment the peak index
        print('')

    def fit(self, filename='fit.out', overwrite=False, start_index=1, **fit_kws):
        """
        Perform the fit of all the 2D peaks, one by one, by reading the starting values from Vi.
        
        """

        # Flag: did you set a logfile?
        if 'logfile' in fit_kws.keys():
            use_logfile = True
        else:
            use_logfile = False

        # Generator: group the parameters with the same index.
        def values_loop(V, idx):
            if isinstance(V, list):
                n_sgn = len(V)
            else:
                n_sgn = V.shape[0]

            for k in range(n_sgn):
                peak_index = k + 1
                if peak_index > max(idx):
                    break
                tmp_par = [V[k] for k in range(n_sgn) if idx[k] == peak_index]
                yield peak_index, tmp_par
        looped_values = values_loop(self.Vi, self.idx)  # Call the generator

        if os.path.exists(filename) and overwrite is False:    # append next peaks 
            f = open(filename, 'a', buffering=1)
        else:   # create a new file
            f = open(filename, 'w', buffering=1)
            self._write_head_line(f)

        # Loop for the fit
        for peak_index, peak_values in looped_values:
            if use_logfile: # Redirect the standard output to the logfile
                sys.stdout = open(fit_kws['logfile'], 'a', buffering=1) 
            print(f'Fitting peak {peak_index:4.0f} / {max(self.idx):4.0f}')
            if peak_index < start_index:    # Skip
                continue
            if len(peak_values) == 0:   # Skip empty parameters
                continue
            lim_f1, lim_f2 = self.Wi[peak_index-1]  # Get the window for the fit
            # Call the fit
            fit_parameters = voigt_fit_2D(self.ppm_f2, self.ppm_f1, self.data, peak_values, lim_f1, lim_f2, self.acqus, N=(len(self.ppm_f1),len(self.ppm_f2)), procs=self.procs, **fit_kws)
            # Write the output in the new file
            for fit_values in fit_parameters:
                self._write_par_line(f, self.acqus, peak_index, fit_values, interval_f1=lim_f1, interval_f2=lim_f2, conv_u=False)

        # Revert standard output to default
        if use_logfile:
            sys.stdout = sys.__stdout__

    ### PRIVATE METHODS
    def _write_head_line(self, f):
        """ 
        Writes the header of the output file.
        -------
        Parameters:
        - f: TextIOWrapper
            writable file generated by open(filename, 'w'/'a')
        """
        f.write(f'{"#":<4s}\t{"clu":>4s}\t{"u1":>8s}\t{"u2":>8s}\t{"fwhm1":>8s}\t{"fwhm2":>8s}\t{"I":>8s}\t{"b":>8s}\t{"Fit. interv.":>20s}\n')

    def _write_par_line(self, f, acqus, index, clu, values, interval_f1=None, interval_f2=None, conv_u=True):
        """ 
        Writes a line of parameters to the output file.
        -------
        Parameters:
        - f: TextIOWrapper
            writable file generated by open(filename, 'w'/'a')
        - acqus: dict
            2D-like acquisition parameters
        - index: int
            Index of the peak
        - clu: int
            Cluster index
        - values: 1darray
            u1, u2, fwhm1, fwhm2, I, b
        - interval_f1: tuple
            left limit F1, right limit F1
        - interval_f2: tuple
            left limit F2, right limit F2
        - conv_u: bool
            Conversion of u1 and u2 from Hz to ppm
        """
        if interval_f1 is None:
            interval_f1 = ('SW', 'SW')  # Whole SW
        else:
            interval_f1 = tuple([eval(f'{x:4.1f}') for x in interval_f1])
        if interval_f2 is None:
            interval_f2 = ('SW', 'SW')  # Whole SW
        else:
            interval_f2 = tuple([eval(f'{x:4.1f}') for x in interval_f2])
        interval = f'{interval_f1},{interval_f2}'

        v1, v2, fwhm1, fwhm2, I, b = values   # Unpack
        if conv_u:
            # Convert u1, u2 from Hz to ppm
            u1, u2 = [misc.freq2ppm(x, acqus[f'SFO{y}'], acqus[f'o{y}p']) for x, y in zip([v1, v2], [1, 2])]
        else:
            u1, u2 = v1, v2

        # Write the line
        f.write(f'{index:4.0f}\t{clu:4.0f}\t{u1:8.2f}\t{u2:8.2f}\t{fwhm1:8.1f}\t{fwhm2:8.1f}\t{I:8.3e}\t{b:8.4f}\t{interval}\n')

    def _read_par_line(self, line):
        """ Splits the line into the three attributes """
        split = line.split('\t')
        index = eval(split[0])
        clu = eval(split[1])
        values = [eval(w) for w in split[2:-1]]
        fit_interval = eval(split[-1])
        return index, clu, values, fit_interval


#-------------------------------------------------------------------------------------


class CostFunc:
    """
    Class that groups several ways to compute the target of the minimization in a fitting procedure. It includes the classic squared sum of the residuals, as well as some other non-quadratic cost functions.
    Let x be the residuals and s the chosen threshold value. Then the objective value R is computed as:
        R = \sum_i f(x_i)
    where f(x) can be chosen between the following options:
    > Quadratic:
        f(x) = x^2
    > Truncated Quadratic:
        f(x) =  x^2         if |x| < s
                s^2         otherwise
    > Huber function:
        f(x) =  x^2         if |x| < s
                2s|x| - s^2 otherwise
    > Asymmetric Truncated Quadratic:
        f(x) =  x^2         if x < s
                s^2         otherwise
    > Asymmetric Huber function:
        f(x) =  x^2         if x < s
                2sx - s^2   otherwise
    ---------
    Attributes:
    - method: function
        Function to be used for the computation of the objective value. It must take as input the array of the residuals and the threshold, no matter if the latter is actually used or not.
    - s: float
        Threshold value
    """
    def __init__(self, method='q', s=None):
        """
        Initialize the method according to your choice, then stores the threshold value in the attribute "s".
        Allowed choices are:
        > "q": Quadratic
        > "tq": Truncated Quadratic
        > "huber": Huber function
        > "atq": Asymmetric Truncated Quadratic
        > "ahuber": Asymmetric Huber function
        ---------
        Parameters:
        - method: str
            Label for the method selection
        - s: float
            Threshold value
        """
        self.method = self.method_selector(method)
        self.s = s

    def method_selector(self, method):
        """
        Performs the selection of the method according to the identifier string.
        -------
        Parameters:
        - method: str
            Method label
        --------
        Returns:
        - f: function
            Selected model
        """
        if method == 'q':
            return self.squared_sum
        elif method == 'tq':
            return self.truncated_quadratic
        elif method == 'huber':
            return self.huber
        elif method == 'atq':
            return self.asymm_truncated_quadratic
        elif method == 'ahuber':
            return self.asymm_huber
        else:
            raise ValueError(f'{method} method not recognized')

    def __call__(self, x):
        """
        Computes the objective value according to the chosen method and the residuals array x.
        ---------
        Parameters:
        - x: 1darray
            Array of the residuals
        ---------
        Returns:
        - R: float
            Computed objective value
        """
        return self.method(x, self.s)

    @staticmethod
    def squared_sum(r, s=0):
        """ Quadratic everywhere """
        return r 

    @staticmethod
    def truncated_quadratic(r, s):
        """ Constant behaviour above s """
        x = np.copy(r)
        for i, x_i in enumerate(x):
            if np.abs(x_i) < s:
                pass
            else:
                x[i] = np.sign(x_i) * s
        return x

    @staticmethod
    def huber(r, s):
        """ Linear behaviour above s """
        x = np.copy(r)
        for i, x_i in enumerate(x):
            if np.abs(x_i) < s:
                pass
            else:
                x[i] = np.sign(x_i) * ( 2*s*np.abs(x_i) - s**2 )**0.5
        return x

    @staticmethod
    def asymm_huber(r, s):
        """ Linear behaviour above s, penalizes negative entries """
        x = np.copy(r)
        for i, x_i in enumerate(x):
            if x_i < s:
                pass
            else:
                x[i] = ( 2*s*x_i - s**2 )**0.5
        return x

    @staticmethod
    def asymm_truncated_quadratic(r, s):
        """ Constant behaviour above s, penalizes negative entries """
        x = np.copy(r)
        for i, x_i in enumerate(x):
            if x_i < s:
                pass
            else:
                x[i] = s
        return x


def lsp(y, x, n=5):
    """
    Linear-System Polynomion
    Make a polynomial fit on the experimental data y by solving the linear system
        y = T c
    where T is the Vandermonde matrix of the x-scale and c is the set of coefficients that minimize the problem in the least-squares sense.
    ----------
    Parameters:
    - y: 1darray
        Experimental data
    - x: 1darray
        Independent variable (better if normalized)
    - n: int
        Order of the polynomion + 1, i.e. number of coefficients
    ----------
    Returns:
    - c: 1darray
        Set of minimized coefficients
    """
    # Make the Vandermonde matrix of the x-scale
    T = np.array(
            [x**k for k in range(n)]
            ).T
    # Pseudo-invert it
    Tpinv = np.linalg.pinv(T)
    # Solve the system
    c = Tpinv @ y
    return c



def polyn_basl(y, n=5, method='huber', s=0.2, c_i=None, itermax=1000):
    """
    Fit the baseline of a spectrum with a low-order polynomion using a non-quadratic objective function.
    Let y be an array of N points. The polynomion is generated on a normalized scale that goes from -1 to 1 in N steps, and the coefficients are initialized either from outside through the parameter c_i or with the ordinary least squares fit.
    Then, the guess is refined using the objective function of choice employing the trust-region reflective least-squares algorithm.
    -----------
    Parameters:
    - y: 1darray
        Experimental data
    - n: int
        Order of the polynomion + 1, i.e. number of coefficients
    - method: str
        Objective function of choice. 'q': quadratic, 'tq': truncated quadratic, 'huber': Huber, 'atq': asymmetric truncated quadratic, 'ahuber': asymmetric huber
    - s: float
        Relative threshold value for the non-quadratic behaviour of the objective function
    - c_i: sequence or None
        Initial guess for the polynomion coefficient. If None, the least-squares fit is used
    - itermax: int
        Number of maximum iterations
    -----------
    Returns:
    - px: 1darray
        Fitted polynomion
    - c: list
        Set of coefficients of the polynomion
    """
    def f2min_real(param, y, x, n, res_f):
        """
        Minimizer function.
        ----------
        Parameters:
        - param: lmfit.Parameters object
            Parameters to be optimized
        - y: 1darray
            Experimental data
        - x: 1darray
            Scale on which to build the model
        - n: int
            Number of coefficients
        - res_f: function
            Returns the objective value to be minimized
        """
        # Unpack the parameters
        par = param.valuesdict()
        # Make a list of coefficients from the dictionary
        c = [par[f'c_{k}'].real for k in range(n)]
        # Make the polynomion
        px = par['I'] * misc.polyn(x, c)
        # Compute the residual
        r = y - px
        return res_f(r)

    def f2min_cplx(param, y, x, n, res_f):
        """
        Minimizer function.
        ----------
        Parameters:
        - param: lmfit.Parameters object
            Parameters to be optimized
        - y: 1darray
            Experimental data
        - x: 1darray
            Scale on which to build the model
        - n: int
            Number of coefficients
        - res_f: function
            Returns the objective value to be minimized
        """
        # Unpack the parameters
        par = param.valuesdict()
        # Make a list of coefficients from the dictionary
        c = [par[f'c_{k}'] + 1j*par[f'c_{k}'] for k in range(n)]
        # Make the polynomion
        px = par['I'] * misc.polyn(x, c)
        # Compute the residual
        r_r = y.real - px.real
        r_i = y.imag - px.imag
        r = np.concatenate((r_r, r_i), axis=-1)
        return res_f(r)

    cplx = np.iscomplexobj(y)

    # Make the normalized scales
    x = np.linspace(-1, 1, y.shape[-1])

    # Make initial guess of the polynomion coefficients
    print('Make initial guess of the polynomion coefficients...')
    if c_i:
        c = np.copy(c_i)
    else:
        c = fit.lsp(y, x, n)
    px_iguess = misc.polyn(x, c)
    print('Done.')

    # Compute an intensity factor to decrease the weight on the fit procedure
    I = fit.fit_int(np.abs(y), np.abs(px_iguess))[0]
    s *= I      # Set absolute threshold values
    c /= I      # Normalize the coefficients to I
    
    # Generate the parameters for the fit
    param = l.Parameters()
    param.add('I', value=I, vary=False) # Just to keep track of it

    for k in range(n):
        param.add(f'c_{k}', value=c[k].real)

    # Get the objective function of choice
    R = fit.CostFunc(method, s)

    print('Optimizing the baseline...')
    # Make the fit
    if cplx:
        f2min = f2min_cplx
    else:
        f2min = f2min_real
    minner = l.Minimizer(f2min, param, fcn_args=(y, x, n, R))
    result = minner.minimize(method='least_squares', max_nfev=int(itermax), gtol=1e-15)
    print(f'The fit has ended. {result.message}.\nNumber of function evaluations: {result.nfev}')

    # Get the fitted parameters
    popt = result.params.valuesdict()

    # Make a list of the fitted coefficients from the dictionary
    if cplx:
        c_opt = [popt['I'] * ( popt[f'c_{k}'] + 1j*popt[f'c_{k}'] ) for k in range(n)]
    else:
        c_opt = [popt['I'] * popt[f'c_{k}'] for k in range(n)]
    # Build the polynomion with them
    px = misc.polyn(x, c_opt)

    return px, c_opt


class SINC_ObjFunc:
    """
    Computes the objective function as explained in M. Sawall et al., Journal of Magnetic Resonance 289 (2018), 132-141.
    The cost function is computed as:
        f(d) = \sum_{i=1}^3  gamma_i g_i(d|e_i)
    where d is the real part of the NMR spectrum.
    ---------
    Attributes:
    - gamma1: float
        Weighting factor for function g1
    - gamma2: float
        Weighting factor for function g2
    - gamma3: float
        Weighting factor for function g3
    - e1: float
        Tolerance value for function g1
    - e2: float
        Tolerance value for function g2
    """
    def __init__(self, gamma1=10, gamma2=0.01, gamma3=0, e1=0, e2=0):
        """
        Initialize the coefficients used to weigh the objective function.
        -------
        Parameters:
        - gamma1: float
            Weighting factor for function g1
        - gamma2: float
            Weighting factor for function g2
        - gamma3: float
            Weighting factor for function g3
        - e1: float
            Tolerance value for function g1
        - e2: float
            Tolerance value for function g2
        """
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.gamma3 = gamma3
        self.e1 = e1
        self.e2 = e2

    def __call__(self, d):
        """ Computes the objective function f as explained in the paper """
        # Normalize d with respect to the sup-norm
        norm = np.max(np.abs(d))
        d_norm = np.copy(d) / norm
        # Compute the three contributions
        g1 = self.g1(d_norm, self.e1)
        g2 = self.g2(d_norm, self.e2)
        g3 = self.g3(d_norm)
        # Weigh the three contribution by the three coefficients
        f = self.gamma1 * g1 + self.gamma2 * g2 + self.gamma3 * g3
        return f


    @staticmethod
    def g1(d, e1=0):
        """ 
        Penalty function for negative entries of the spectrum
        --------
        Parameters:
        - d: 1darray
            Spectrum
        - e1: float
            Tolerance for negative entries
        """

        data = np.copy(d) + e1
        datac = np.array([min(0, x) for x in data])
        g = np.sum(datac**2)
        return g

    @staticmethod
    def g2(d, e2=0):
        """
        Regularization function that favours the smallest integral.
        --------
        Parameters:
        - d: 1darray
            Spectrum
        - e2: float
            Tolerance for ideal baseline
        """

        data = np.abs(np.copy(d)) - e2
        datac = np.array([max(0, x) for x in data])
        g = np.sum(datac**2)
        return g

    @staticmethod
    def g3(d):
        """
        Regularization function for the smoothing.
        --------
        Parameters:
        - d: 1darray
            Spectrum
        """
        diffd = np.diff(d, 2)
        g = np.sum(diffd**2)
        return g


def sinc_phase(data, gamma1=10, gamma2=0.01, gamma3=0, e1=0, e2=0, **fit_kws):
    """
    Perform automatic phase correction according to the SINC algorithm, as described in M. Sawall et. al., Journal of Magnetic Resonance 289 (2018), 132–141.
    The fitting method defaults to "least_squares".
    --------
    Parameters:
    - data: 1darray
        Spectrum to phase-correct
    - gamma1: float
        Weighting factor for function g1: non-negativity constraint
    - gamma2: float
        Weighting factor for function g2: smallest-integral constraint
    - gamma3: float
        Weighting factor for function g3: smoothing constraint
    - e1: float
        Tolerance factor for function g1: adjustment for noise
    - e2: float
        Tolerance factor for function g2: adjustment for non-ideal baseline
    - fit_kws: keyworded arguments
        additional parameters for the fit function. See lmfit.Minimizer.minimize for details. Do not use "leastsq" because the cost function returns a scalar value!
    --------
    Returns:
    - p0: float
        Fitted zero-order phase correction angle, in degrees
    - p1: float
        Fitted first-order phase correction angle, in degrees
    """

    def f2min(param, data, r_func):
        """ Cost function for the fit. Applies the algorithm. """
        # Unpack the parameters
        par = param.valuesdict()
        p0 = par['p0']
        p1 = par['p1']

        # Phase data and take real part
        Rp, *_ = processing.ps(data, p0=p0, p1=p1)
        R = Rp.real
        # Compute the objective function using the phased data
        return r_func(R)

    # Safety checks
    if not np.iscomplexobj(data):
        raise ValueError('Input data is not complex.')

    if 'method' not in fit_kws.keys():
        fit_kws['method'] = 'least_squares'

    # Shallow copy to prevent overwriting
    d = np.copy(data)

    # Create the Parameters object
    param = l.Parameters()
    param.add('p0', value=0, min=-180, max=180)
    param.add('p1', value=0, min=-720, max=720)

    # Create the objective function
    R = fit.SINC_ObjFunc(gamma1, gamma2, gamma3, e1, e2)

    # Minimize using the method of choice. "leastsq" not accepted!
    print('Starting phase correction...')
    minner = l.Minimizer(f2min, param, fcn_args=(d, R))
    result = minner.minimize(**fit_kws)
    print(f'The fit has ended. {result.message}.\nNumber of function evaluations: {result.nfev}')
    popt = result.params.valuesdict()

    return popt['p0'], popt['p1']



def write_vf_P2D(filename, peaks, lims, prev=0):
    """
    Write a section in a fit report file, which shows the fitting region and the parameters of the peaks to feed into a Voigt lineshape model.
    -----------
    Parameters:
    - filename: str
        Path to the file to be written
    - peaks: list of dict
        list of dictionares of fit.Peak objects, one per experiment
    - lims: tuple
        (left limit /ppm, right limit /ppm)
    - prev: int
        Number of previous peaks already saved. Increases the peak index
    """

    # Open the file in append mode
    f = open(f'{filename}', 'a', buffering=1)
    # Info on the region to be fitted
    #   Header
    f.write('{:>16};\n'.format('Region'))
    f.write('-'*96+'\n')
    #   Values
    region = '{:-.3f}:{:-.3f}'.format(*lims)   # From the zoom of the figure
    f.write(f'{region:>16};\n\n')

    # Info on the peaks
    #   Header
    f.write('{:>4};\t{:>8};\t{:>8};\t{:>8};\t{:>8};\t{:>8}\n'.format(
        '#', 'u', 'fwhm', 'Phase', 'Beta', 'Group'))
    f.write('-'*96+'\n')
    #   Values
    for k, key in enumerate(peaks[0].keys()):
        peak = peaks[0][key]
        f.write('{:>4.0f};\t{:=8.3f};\t{:8.3f};\t{:-8.3f};\t{:8.3f};\t{:>8.0f}\n'.format(
            k+prev+1, peak.u, peak.fwhm, peak.phi, peak.b, peak.group))
    f.write('-'*96+'\n\n')

    #   Intensities
    f.write('Intensities\n')
    f.write(f'{"#":>4};\t'+';\t'.join(['{:>12}'.format('Exp'+f'{w+1}') for w in range(len(peaks))])+'\n')
    f.write('-'*96+'\n')

    peak_keys = peaks[0].keys()
    for k, key in enumerate(peak_keys):
        f.write(f'{k+prev+1:>4}')
        for _, dicpeaks in enumerate(peaks):
            f.write(';\t{:12.5e}'.format(dicpeaks[key].k))
        f.write('\n')
    f.write('-'*96+'\n\n')

    # Add region separator and close the file
    f.write('='*96+'\n\n')
    f.close()

def read_vf_P2D(filename, n=-1):
    """
    Reads a .ivf (initial guess) or .fvf (final fit) file, containing the parameters for a lineshape deconvolution fitting procedure.
    The file is separated and unpacked into a list of list of dictionaries, each of which contains the limits of the fitting window, and a dictionary for each peak with the characteristic values to compute it with a Voigt line.
    --------------
    Parameters:
    - filename: str
        Path to the filename to be read
    - n: int
        Number of performed fit to be read. Default: last one. The breakpoints are lines that start with "!". For this reason, n=0 returns an empty dictionary, hence the first fit is n=1.
    -------------
    Returns:
    - regions: list of list of dict
        List of dictionaries for running the fit.
    """
    def read_region(R):
        """ Creates a dictionary of parameters from a section of the input file.  """
        # Placeholder
        dic_r = {}
        # Separate the lines and remove the empty ones
        R = R.split('\n')
        for k, r in enumerate(R):
            if len(r)==0 or r.isspace():
                _ = R.pop(k)

        n_bp = 0        # Number of breaking points (----)
        k_bp = 0        # Line of the last breaking point detected
        flag = True     # To create the final list of dictionaries to be returned only once
        for k, r in enumerate(R):
            if '------' in r:   # Increase breakpoint and store the line number
                n_bp += 1
                k_bp = k
                continue

            if n_bp == 1 and k_bp == k-1:   # First section: region limits and total intensity
                line = r.split(';') # Separate the values
                dic_r['limits'] = eval(line[0].replace(':',', '))   # Get the limits

            if n_bp == 2:       # Second section: peak parameters
                line = r.split(';') # Separate the values
                # Unpack the line
                idx, u, fwhm, phi, b, group = [eval(w) for w in line]
                # Put the values in a dictionary
                dic_p = {
                        'u': u,
                        'fwhm': fwhm,
                        'k': 0,     # they will be added later, in section 3
                        'b': b,
                        'phi': phi,
                        'group': group
                        }
                # Put the values in the returned dictionary
                dic_r[idx] = dic_p

            # Skip n_bp == 3 because it is the end of section 2

            if n_bp == 4:       # Third section: intensity values for all the peaks for each experiment
                line = r.split(';') # Separate the values
                if flag:    # Create the final dictionary to be returned and turn off the flag
                    dic_rr = [deepcopy(dic_r) for q in range(len(line)-1)]
                    flag = False

                # Unpack the line
                eval_line = [eval(w) for w in line]
                idx = int(eval_line[0])     # index of the peak
                Ks = eval_line[1:]          # Intensity, each for each experiment
                for q, K in enumerate(Ks):
                    dic_rr[q][idx]['k'] = K # Overwrite the intensities

            if n_bp == 5:   # End of file: stop reading
                break

        return dic_rr

    # Read the file
    with open(filename, 'r') as J:
        ff = J.read()
    # Get the actual section from an output file
    f = ff.split('!')[n]
    # Separate the bigger sections
    R = f.split('='*96)
    # Remove the empty lines
    for k, r in enumerate(R):
        if r.isspace():
            _ = R.pop(k)

    regions = []    # Placeholder for return values
    for r in R: # Loop on the big sections to read them
        regions.append(read_region(r))
    return regions



def make_iguess_P2D(S_in, ppm_scale, expno, t_AQ, SFO1=701.125, o1p=0, filename='i_guess'):
    """
    Creates the initial guess for a lineshape deconvolution fitting procedure of a pseudo-2D experiment, using a dedicated GUI.
    It will be donw on only one experiment of the whole pseudo-2D.
    The GUI displays the experimental spectrum in black and the total function in blue.
    First, select the region of the spectrum you want to fit by focusing the zoom on it using the lens button.
    Then, use the "+" button to add components to the spectrum. The black column of text under the textbox will be colored with the same color of the active peak.
    Use the mouse scroll to adjust the parameters of the active peak. Write a number in the "Group" textbox to mark the components of the same multiplet.
    Group 0 identifies independent peaks, not part of a multiplet (default).
    The sensitivity of the mouse scroll can be regulated using the "up arrow" and "down arrow" buttons. 
    The active peak can be changed in any moment using the slider.

    When you are satisfied with your fit, press "SAVE" to write the information in the output file. Then, the GUI is brought back to the initial situation, and the region you were working on will be marked with a green rectangle. You can repeat the procedure as many times as you wish, to prepare the guess on multiple spectral windows.

    Keyboard shortcuts:
    > "increase sensitivity" : '>'
    > "decrease sensitivity" : '<'
    > mouse scroll up: 'up arrow key'
    > mouse scroll down: 'down arrow key'
    > "add a component": '+'
    > "remove the active component": '-'
    > "change component, forward": 'page up'
    > "change component, backward": 'page down'

    ------------
    Parameters:
    - S_in: 1darray
        Experimental spectrum
    - ppm_scale: 1darray
        PPM scale of the spectrum
    - expno: int
        Index of experiment of the pseudo 2D on which to compute the initial guess, in python numbering
    - t_AQ: 1darray
        Acquisition timescale
    - SFO1: float
        Nucleus Larmor frequency /MHz
    - o1p: float
        Carrier frequency /ppm
    - filename: str
        Path to the filename where to save the information. The '.ivf' extension is added automatically.
    """

    #-----------------------------------------------------------------------
    ## USEFUL STRUCTURES
    def rename_dic(dic, Np):
        """
        Change the keys of a dictionary with a sequence of increasing numbers, starting from 1.
        ----------
        Parameters:
        - dic: dict
            Dictionary to edit
        - Np: int
            Number of peaks, i.e. the sequence goes from 1 to Np
        ----------
        Returns:
        - new_dic: dict
            Dictionary with the changed keys
        """
        old_keys = list(dic.keys())         # Get the old keys
        new_keys = [int(i+1) for i in np.arange(Np)]    # Make the new keys
        new_dic = {}        # Create an empty dictionary
        # Copy the old element in the new dictionary at the correspondant point
        for old_key, new_key in zip(old_keys, new_keys):
            new_dic[new_key] = dic[old_key]
        del dic
        return new_dic

    def calc_total(peaks):
        """
        Calculate the sum trace from a collection of peaks stored in a dictionary.
        If the dictionary is empty, returns an array of zeros.
        ---------
        Parameters:
        - peaks: dict
            Components
        --------
        Returns:
        - total: 1darray
            Sum spectrum
        """
        # Get the arrays from the dictionary
        T = [p(A) for _, p in peaks.items()]
        if len(T) > 0:  # Check for any peaks
            total = np.sum(T, axis=0)
            return total
        else:
            return np.zeros_like(ppm_scale)

    #-------------------------------------------------------------------------------

    # Initial figure
    fig = plt.figure('Manual Computation of Initial Guess - Pseudo2D')
    fig.set_size_inches(15,8)
    plt.subplots_adjust(bottom=0.10, top=0.90, left=0.05, right=0.65)
    ax = fig.add_subplot(1,1,1)

    # Write the info on the file
    with open(f'{filename}.ivf', 'a', buffering=1) as f:
        now = datetime.now()
        date_and_time = now.strftime("%d/%m/%Y at %H:%M:%S")
        f.write('! Initial guess computed by {} on {}\n\n'.format(os.getlogin(), date_and_time))

    # Remove the imaginary part from the experimental data and make a shallow copy
    if np.iscomplexobj(S_in):
        S = np.copy(S_in).real[expno]
    else:
        S = np.copy(S_in)[expno]

    n_exp = S_in.shape[0]   # Number of experiments
    N = S.shape[-1]         # Number of points
    Np = 0                  # Number of peaks
    lastgroup = 0           # Placeholder for last group added
    prev = 0                # Number of previous peaks

    # Make an acqus dictionary based on the input parameters.
    acqus = {'t1': t_AQ, 'SFO1': SFO1, 'o1p': o1p}

    # Set limits
    limits = [max(ppm_scale), min(ppm_scale)]
    
    # Get point indices for the limits
    lim1 = misc.ppmfind(ppm_scale, limits[0])[0]
    lim2 = misc.ppmfind(ppm_scale, limits[1])[0]
    # Calculate the absolute intensity (or something that resembles it)
    A = processing.integrate(S, x=ppm_scale*SFO1, lims=[w*SFO1 for w in limits])*2*misc.calcres(acqus['t1'])
    _A = 1 * A
    # Make a sensitivity dictionary
    sens = {
            'u': np.abs(limits[0] - limits[1]) / 50,    # 1/50 of the SW
            'fwhm': 2.5,
            'k': 0.05,
            'b': 0.1,
            'phi': 10,
            'A': 10**(np.floor(np.log10(A)-1))    # approximately
            }
    _sens = dict(sens)                          # RESET value
    # Peaks dictionary
    peaks = {}

    # make boxes for widgets
    slider_box = plt.axes([0.68, 0.10, 0.01, 0.65])     # Peak selector slider
    peak_box = plt.axes([0.72, 0.45, 0.10, 0.30])       # Radiobuttons
    up_box = plt.axes([0.815, 0.825, 0.08, 0.075])      # Increase sensitivity button
    down_box = plt.axes([0.894, 0.825, 0.08, 0.075])    # Decrease sensitivity button
    save_box = plt.axes([0.7, 0.825, 0.085, 0.04])      # Save button
    reset_box = plt.axes([0.7, 0.865, 0.085, 0.04])     # Reset button
    group_box = plt.axes([0.76, 0.40, 0.06, 0.04])      # Textbox for the group selection
    plus_box = plt.axes([0.894, 0.65, 0.08, 0.075])     # Add button
    minus_box = plt.axes([0.894, 0.55, 0.08, 0.075])    # Minus button
    
    # Make widgets
    #   Buttons
    up_button = Button(up_box, r'$\uparrow$', hovercolor = '0.975')    
    down_button = Button(down_box, r'$\downarrow$', hovercolor = '0.975')
    save_button = Button(save_box, 'SAVE', hovercolor = '0.975')
    reset_button = Button(reset_box, 'RESET', hovercolor = '0.975')
    plus_button = Button(plus_box, '$+$', hovercolor='0.975')
    minus_button = Button(minus_box, '$-$', hovercolor='0.975')

    #   Textbox
    group_tb = TextBox(group_box, 'Group', textalignment='center')
    
    #   Radiobuttons
    peak_name = [r'$\delta$ /ppm', r'$\Gamma$ /Hz', '$k$', '$x_{g}$', r'$\phi$', '$A$']
    peak_radio = RadioButtons(peak_box, peak_name, activecolor='tab:blue')      # Signal parameters
    
    #   Slider
    slider = Slider(ax=slider_box, label='Active\nSignal', valmin=0, valmax=1-1e-3, valinit=0, valstep=1e-10, orientation='vertical', color='tab:blue')


    #-------------------------------------------------------------------------------
    ## SLOTS

    def redraw():
        misc.pretty_scale(ax, ax.get_xlim(), 'x')
        plt.draw()

    def radio_changed(event):
        """ Change the printed value of sens when the radio changes """
        active = peak_name.index(peak_radio.value_selected)
        param = list(sens.keys())[active]
        write_sens(param)

    def up_sens(event):
        """ Doubles sensitivity of active parameter """
        nonlocal sens
        active = peak_name.index(peak_radio.value_selected)
        param = list(sens.keys())[active]
        sens[param] *= 2
        write_sens(param)

    def down_sens(event):
        """ Halves sensitivity of active parameter """
        nonlocal sens
        active = peak_name.index(peak_radio.value_selected)
        param = list(sens.keys())[active]
        sens[param] /= 2
        write_sens(param)

    def up_value(param, idx):
        """ Increase the value of param of idx-th peak """
        if param == 'A':        # It is outside the peaks dictionary!
            nonlocal A
            A += sens['A']
        else:
            nonlocal peaks
            peaks[idx].__dict__[param] += sens[param]
            # Make safety check for b
            if peaks[idx].b > 1:
                peaks[idx].b = 1

    def down_value(param, idx):
        """ Decrease the value of param of idx-th peak """
        if param == 'A':    # It is outside the peaks dictionary!
            nonlocal A
            A -= sens['A']
        else:
            nonlocal peaks
            peaks[idx].__dict__[param] -= sens[param]
            # Safety check for fwhm
            if peaks[idx].fwhm < 0:
                peaks[idx].fwhm = 0
            # Safety check for b
            if peaks[idx].b < 0:
                peaks[idx].b = 0

    def scroll(event):
        """ Connection to mouse scroll """
        if Np == 0: # No peaks!
            return
        # Get the active parameter and convert it into Peak's attribute
        active = peak_name.index(peak_radio.value_selected)
        param = list(sens.keys())[active]
        # Get the active peak
        idx = int(np.floor(slider.val * Np) + 1)
        # Fork for up/down
        if event.button == 'up':
            up_value(param, idx)
        if event.button == 'down':
            down_value(param, idx)

        # Recompute the components
        for k, _ in enumerate(peaks):
            p_sgn[k+1].set_ydata(peaks[k+1](A)[lim1:lim2])
        # Recompute the total trace
        p_fit.set_ydata(calc_total(peaks)[lim1:lim2])
        # Update the text
        write_par(idx)
        redraw()

    def write_par(idx):
        """ Write the text to keep track of your amounts """
        if idx:     # Write the things
            dic = dict(peaks[idx].par())
            dic['A'] = A
            # Update the text
            values_print.set_text('{u:+7.3f}\n{fwhm:5.3f}\n{k:5.3f}\n{b:5.3f}\n{phi:+07.3f}\n{A:5.2e}\n{group:5.0f}'.format(**dic))
            # Color the heading line of the same color of the trace
            head_print.set_color(p_sgn[idx].get_color())
        else:   # Clear the text and set the header to be black
            values_print.set_text('')
            head_print.set_color('k')

    def write_sens(param):
        """ Updates the current sensitivity value in the text """
        # Discriminate between total intensity and other parameters
        if param == 'A':
            text = f'Sensitivity: $\\pm${sens[param]:10.4g}'
        else:
            text = f'Sensitivity: $\\pm${sens[param]:10.4g}'
        # Update the text
        sens_print.set_text(text)
        # Redraw the figure
        plt.draw()

    def set_group(text):
        """ Set the attribute 'group' of the active signal according to the textbox """
        nonlocal peaks
        if not Np:  # Clear the textbox and do nothing more
            group_tb.text_disp.set_text('')
            plt.draw()
            return
        # Get active peak
        idx = int(np.floor(slider.val * Np) + 1)
        try:
            group = int(eval(text))
        except:
            group = peaks[idx].group
        group_tb.text_disp.set_text('')
        peaks[idx].group = group
        write_par(idx)
        redraw()

    def selector(event):
        """ Update the text when you move the slider """
        idx = int(np.floor(slider.val * Np) + 1)
        if Np:
            write_par(idx)
        redraw()

    def key_binding(event):
        """ Keyboard """
        key = event.key
        if key == '<':
            down_sens(0)
        if key == '>':
            up_sens(0)
        if key == '+':
            add_peak(0)
        if key == '-':
            remove_peak(0)
        if key == 'pagedown':
            if slider.val - slider.valstep >= 0:
                slider.set_val(slider.val - slider.valstep)
            selector(0)
        if key == 'pageup':
            if slider.val + slider.valstep < 1:
                slider.set_val(slider.val + slider.valstep)
            selector(0)
        if key == 'up' or key == 'down':
            event.button = key
            scroll(event)

    def reset(event):
        """ Return everything to default """
        nonlocal Np, peaks, p_sgn, A, sens
        Q = Np
        for k in range(Q):
            remove_peak(event)
        A = _A
        sens = dict(_sens)
        ax.set_xlim(*_xlim)
        ax.set_ylim(*_ylim)
        redraw()

    def add_peak(event):
        """ Add a component """
        nonlocal Np, peaks, p_sgn
        # Increase the number of peaks
        Np += 1 
        # Add an entry to the dictionary labelled as last
        peaks[Np] = fit.Peak(acqus, u=np.mean(ax.get_xlim()), N=N, group=lastgroup)
        # Plot it and add the trace to the plot dictionary
        p_sgn[Np] = ax.plot(ppm_scale[lim1:lim2], peaks[Np](A)[lim1:lim2], lw=0.8)[-1]
        # Move the slider to the position of the new peak
        slider.set_val( (Np - 1) / Np )
        # Recompute the step of the slider
        slider.valstep = 1 / Np
        # Calculate the total trace with the new peak
        total = calc_total(peaks)
        # Update the total trace plot
        p_fit.set_ydata(total[lim1:lim2])
        # Update the text
        write_par(Np)
        redraw()

    def remove_peak(event):
        """ Remove the active component """
        nonlocal Np, peaks, p_sgn
        if Np == 0:
            return
        # Get the active peak
        idx = int(np.floor(slider.val * Np) + 1)
        # Decrease Np of 1
        Np -= 1
        # Delete the entry from the peaks dictionary
        _ = peaks.pop(idx)
        # Remove the correspondant line from the plot dictionary
        del_p = p_sgn.pop(idx)
        # Set it invisible because I cannot truly delete it
        del_p.set_visible(False)
        del del_p   # ...at least clear some memory
        # Change the labels to the dictionary
        peaks = rename_dic(peaks, Np)
        p_sgn = rename_dic(p_sgn, Np)
        # Calculate the total trace without that peak
        total = calc_total(peaks)
        # Update the total trace plot
        p_fit.set_ydata(total[lim1:lim2])
        # Change the slider position
        if Np == 0: # to zero and do not make it move
            slider.set_val(0)
            slider.valstep = 1e-10
            write_par(0)
        elif Np == 1:   # To zero and that's it
            slider.set_val(0)
            slider.valstep = 1 / Np
            write_par(1)
        else:   # To the previous point
            if idx == 1:
                slider.set_val(0)
            else:
                slider.set_val( (idx - 2) / Np)     # (idx - 1) -1
            slider.valstep = 1 / Np
            write_par(int(np.floor(slider.val * Np) + 1))
        redraw()

    def save(event):
        """ Write a section in the output file """
        nonlocal prev
        # Adjust the intensities
        for _, peak in peaks.items():
            peak.k *= A
        write_vf_P2D(f'{filename}.ivf', [peaks for w in range(n_exp)], ax.get_xlim(), prev)
        prev += len(peaks)
        
        # Mark a region as "fitted" with a green box
        ax.axvspan(*ax.get_xlim(), color='tab:green', alpha=0.1)
        # Call reset to return at the initial situation
        reset(event)

    #-------------------------------------------------------------------------------


    ax.plot(ppm_scale[lim1:lim2], S[lim1:lim2], label='Experimental', lw=1.0, c='k')  # experimental
    p_fit = ax.plot(ppm_scale[lim1:lim2], np.zeros_like(S)[lim1:lim2], label='Fit', lw=0.9, c='b')[-1]  # Total trace
    p_sgn = {}  # Components
    
    # Header for current values print
    head_print = ax.text(0.75, 0.35, 
            '{:>7s}\n{:>5}\n{:>5}\n{:>5}\n{:>7}\n{:>7}\n{:>7}'.format(
                r'$\delta$', r'$\Gamma$', '$k$', r'$\beta$', 'Phase', '$A$', 'Group'),
            ha='right', va='top', transform=fig.transFigure, fontsize=14, linespacing=1.5)
    # Text placeholder for the values - linspacing is different to align with the header
    values_print = ax.text(0.85, 0.35, '',
            ha='right', va='top', transform=fig.transFigure, fontsize=14, linespacing=1.55)
    # Text to display the active sensitivity values
    sens_print = ax.text(0.875, 0.775, f'Sensitivity: $\\pm${sens["u"]:10.4g}',
            ha='center', va='bottom', transform=fig.transFigure, fontsize=14)
    # Text to remind keyboard shortcuts
    t_uparrow = r'$\uparrow$'
    t_downarrow = r'$\downarrow$'
    keyboard_text = '\n'.join([
        f'{"KEYBOARD SHORTCUTS":^50s}',
        f'{"Key":>5s}: Action',
        f'-'*50,
        f'{"<":>5s}: Decrease sens.',
        f'{">":>5s}: Increase sens.',
        f'{"+":>5s}: Add component',
        f'{"-":>5s}: Remove component',
        f'{"Pg"+t_uparrow:>5s}: Change component, up',
        f'{"Pg"+t_downarrow:>5s}: Change component, down',
        f'{t_uparrow:>5s}: Increase value',
        f'{t_downarrow:>5s}: Decrease value',
        f'-'*50,
        ])
    keyboard_print = ax.text(0.86, 0.025, keyboard_text, 
            ha='left', va='bottom', transform=fig.transFigure, fontsize=8, linespacing=1.55)

    # make pretty scales
    ax.set_xlim(max(limits[0],limits[1]),min(limits[0],limits[1]))
    misc.pretty_scale(ax, ax.get_xlim(), axis='x', n_major_ticks=10)
    misc.pretty_scale(ax, ax.get_ylim(), axis='y', n_major_ticks=10)
    misc.mathformat(ax)

    # RESET values for xlim and ylim
    _xlim = ax.get_xlim()
    _ylim = ax.get_ylim()

    # Connect the widgets to their slots
    plus_button.on_clicked(add_peak)
    minus_button.on_clicked(remove_peak)
    up_button.on_clicked(up_sens)
    down_button.on_clicked(down_sens)
    slider.on_changed(selector)
    group_tb.on_submit(set_group)
    reset_button.on_clicked(reset)
    save_button.on_clicked(save)
    peak_radio.on_clicked(radio_changed)
    fig.canvas.mpl_connect('scroll_event', scroll)
    fig.canvas.mpl_connect('key_press_event', key_binding)

    plt.show()  # Start event loop
    plt.close()


def plot_fit_P2D(S, ppm_scale, regions, t_AQ, SFO1, o1p, show_total=False, show_res=False, res_offset=0, X_label=r'$\delta$ /ppm', labels=None, filename='fit', ext='tiff', dpi=600):
    """
    Plots either the initial guess or the result of the fit, and saves all the figures. 
    A new folder named <filename>_fit will be created.
    The figure <filename>_full will show the whole model and the whole spectrum. 
    The figures labelled with _R<k> will depict a detail of the fit in the k-th fitting region.
    Optional labels for the components can be given: in this case, the structure of 'labels' should match the structure of 'regions'. This means that the length of the outer list must be equal to the number of fitting region, and the length of the inner lists must be equal to the number of peaks in that region.
    ------------
    Parameters:
    - S: 2darray
        Spectrum to be fitted
    - ppm_scale: 1darray
        ppm scale of the spectrum
    - regions: list of dict
        Generated by fit.read_vf_P2D
    - t_AQ: 1darray
        Acquisition timescale
    - SFO1: float
        Larmor frequency of the observed nucleus, in MHz
    - o1p: float
        Carrier position, in ppm
    - nuc: str
        Observed nucleus. Used to customize the x-scale of the figures.
    - show_total: bool
        Show the total trace (i.e. sum of all the components) or not
    - show_res: bool
        Show the plot of the residuals
    - res_offset: float
        Displacement of the residuals plot from 0, to be given as a fraction of the height of the experimental spectrum. res_offset > 0 will move the residuals BELOW the zero-line!
    - X_label: str
        Text to show as label for the chemical shift axis
    - labels: list of list
        Optional labels for the components. The structure of this parameter must match the structure of self.result
    - filename: str
        Root of the name of the figures that will be saved.
    - ext: str
        Format of the saved figures
    - dpi: int
        Resolution of the figures, in dots per inches
    """

    def calc_total(peaks, A=1):
        """
        Calculates the sum trace from a collection of peaks stored in a dictionary.
        If the dictionary is empty, returns an array of zeros.
        ---------
        Parameters:
        - peaks: dict
            Components
        - A: float
            Absolute intensity
        --------
        Returns:
        - total: 1darray
            Sum spectrum
        """

        # Get the arrays from the dictionary
        T = [p(A) for _, p in peaks.items()]
        if len(T) > 0:  # Check for any peaks
            total = np.sum(T, axis=0)
            return total.real
        else:
            return np.zeros_like(ppm_scale)

    # Try to create the new directory for the figures
    try:
        os.mkdir(f'{filename}_fit')
    except:
        pass
    finally:
        # Update the filename for the figures by including the new directory
        filename = os.path.join(filename+f'_fit', filename)
    print('Saving figures...')
    # Shallow copy of the real part of the experimental spectrum
    S_r = np.copy(S.real)
    N = S_r.shape[-1]       # For (eventual) zero-filling
    # Make the acqus dictionary to be fed into the fit.Peak objects
    acqus = { 't1': t_AQ, 'SFO1': SFO1, 'o1p': o1p, }


    ## Single regions
    # Loop on the regions
    for k, region in enumerate(regions):
        # Get limits from the dictionary
        peaklist = deepcopy(region)
        for peaks in peaklist:
            if 'limits' in list(peaks.keys()):
                limits = peaks.pop('limits')

        # Convert the limits from ppm to points
        limits_pt = misc.ppmfind(ppm_scale, limits[0])[0], misc.ppmfind(ppm_scale, limits[1])[0]
        #   Make the slice
        lims = slice(min(limits_pt), max(limits_pt))

        # Make the calculated spectrum
        fit_peaks = [{} for w in range(S.shape[0])]
        list_signals = []
        signals = []
        for j, peaks in enumerate(peaklist):        # j runs on the experiments
            for key, peakval in peaks.items():
                fit_peaks[j][key] = fit.Peak(acqus, N=S.shape[-1], **peakval)
            # Get the arrays from the dictionary and put them in the list
            list_signals.append([p() for _, p in fit_peaks[j].items()])
        signals.extend(list_signals) # Dimensions (n. experiments, n.peaks per experiment, n.points per experiment)

        # Compute the total trace
        total = np.sum(signals, axis=1) # sum the peaks 

        # Trim the ppm scale according to the fitting region
        t_ppm = ppm_scale[lims]

        # One figure per experiment
        for i, _ in enumerate(S):
            # Make the figure
            fig = plt.figure(f'Fit {i+1}')
            fig.set_size_inches(figures.figsize_large)
            ax = fig.add_subplot()
            plt.subplots_adjust(left=0.10, bottom=0.10, top=0.90, right=0.95)

            # Plot the experimental dataset
            ax.plot(t_ppm, S_r[i, lims], c='k', lw=1, label='Experimental')

            if show_total is True:  # Plot the total trace in blue
                ax.plot(t_ppm, total[i, lims], c='b', lw=0.5, label='Fit')

            for key, peak in zip(fit_peaks[i].keys(), signals[i]): # Plot the components
                p_sgn, = ax.plot(t_ppm, peak[lims], lw=0.6, label=f'{key}')
                if labels is not None:  # Set the custom label
                    p_sgn.set_label(labels[k][key-1])

            if show_res is True:    # Plot the residuals
                # Compute the absolute value of the offset
                r_off = min(S_r[i,lims]) + res_offset * (max(S_r[i,lims])-min(S_r[i,lims]))
                ax.plot(t_ppm, (S_r - total)[i,lims] - r_off, c='g', ls=':', lw=0.6, label='Residuals')

            # Visual adjustments
            ax.set_xlabel(X_label)
            ax.set_ylabel('Intensity /a.u.')
            misc.pretty_scale(ax, (max(t_ppm), min(t_ppm)), axis='x')
            misc.pretty_scale(ax, ax.get_ylim(), axis='y')
            misc.mathformat(ax)
            ax.legend()
            misc.set_fontsizes(ax, 20)
            # Save the figure
            plt.savefig(f'{filename}_R{k+1}_E{i+1}.{ext}', dpi=dpi)
            plt.close()
            continue

    ## Total
    # One figure per experiment
    for i, _ in enumerate(S):
        # Make the figure
        fig = plt.figure(f'Fit {i+1}')
        fig.set_size_inches(figures.figsize_large)
        ax = fig.add_subplot()
        plt.subplots_adjust(left=0.10, bottom=0.10, top=0.90, right=0.95)
        # Residuals
        R = S_r - total
        # Plot the experimental dataset
        ax.plot(ppm_scale, S_r[i], c='k', lw=1, label='Experimental')
        
        if show_total is True:  # Plot the total trace
            ax.plot(ppm_scale, total[i], c='b', lw=0.5, label='Fit', zorder=2)

        for key, peak in zip(fit_peaks[i].keys(), signals[i]): # Plot the components
            p_sgn, = ax.plot(ppm_scale, peak, lw=0.6, label=f'{key}')
            if labels is not None:  # Set the custom label
                p_sgn.set_label(labels[k][key-1])

        # Visual adjustments
        ax.set_xlabel(X_label)
        ax.set_ylabel('Intensity /a.u.')
        misc.pretty_scale(ax, (max(ppm_scale), min(ppm_scale)), axis='x')
        misc.pretty_scale(ax, ax.get_ylim(), axis='y')
        misc.mathformat(ax)
        ax.legend()
        misc.set_fontsizes(ax, 20)
        # Save the figure
        plt.savefig(f'{filename}_full_E{i+1}.{ext}', dpi=dpi)
        plt.close()
    print('Done.')






def voigt_fit_P2D(S, ppm_scale, regions, t_AQ, SFO1, o1p, u_tol=1, f_tol=10, vary_phase=False, vary_b=False, itermax=10000, filename='fit'):
    """
    Performs a lineshape deconvolution fit on a pseudo-2D experiment using a Voigt model.
    The initial guess must be read from a .ivf file. All components are treated as independent, regardless from the value of the "group" attribute.
    The fitting procedure operates iteratively one window at the time.
    During the fit routine, the peak positions and lineshapes will be varied consistently on all the experiments; only the intensities are allowed to change in a different way.
    ------------
    Parameters:
    - S: 2darray
        Experimental spectrum
    - ppm_scale: 1darray
        PPM scale of the spectrum
    - regions: dict
        Generated by fit.read_vf_P2D
    - t_AQ: 1darray
        Acquisition timescale
    - SFO1: float
        Nucleus Larmor frequency /MHz
    - o1p: float
        Carrier frequency /ppm
    - u_tol: float
        Maximum allowed displacement of the chemical shift from the initial value /ppm
    - f_tol: float
        Maximum allowed displacement of the linewidth from the initial value /ppm
    - vary_phase: bool
        Allow the peaks to change phase
    - vary_b: bool
        Allow the peaks to change Lorentzian/Gaussian ratio
    - itermax: int
        Maximum number of allowed iterations
    - filename: str
        Name of the file where the fitted values will be saved. The .fvf extension is added automatically
    """

    ## USED FUNCTIONS

    def calc_total(list_peaks, A=1):
        """
        Calculates the sum trace from a collection of peaks stored in a dictionary.
        If the dictionary is empty, returns an array of zeros.
        ---------
        Parameters:
        - peaks: dict
            Components
        - A: float
            Absolute intensity
        --------
        Returns:
        - total: 1darray
            Sum spectrum
        """
        # Get the arrays from the dictionary
        total_arr = []
        for q, peaks in enumerate(list_peaks):
            T = [p(A) for _, p in peaks.items()]
            if len(T) > 0:  # Check for any peaks
                total = np.sum(T, axis=0)
            else:
                total = np.zeros_like(ppm_scale)
            total_arr.append(total)
        return np.array(total_arr)

    def peaks_frompar(peaks, par, e_idx):
        """
        Replaces the values of a "peaks" dictionary, which contains a fit.Peak object for each key "idx", with the values contained in the "par" dictionary.
        The par dictionary keys must have keys of the form <parameter>_<idx>, where <parameter> is in [u, fwhm, k, 'b', 'phi'], and <idx> are the keys of the peaks dictionary.
        The intensity values of the peaks are stored as k_<idx>_<experiment>, where <experiment> is the associated trace of the pseudo-2D, starting from 1.
        -----------
        Parameters:
        - peaks: dict
            Collection of fit.Peak objects
        - par: dict
            New values for the peaks
        - e_idx: int
            Number of experiment of which to change the intensity of the peak
        ----------
        Returns:
        - peaks: dict
            Updated peaks dictionary with the new values
        """
        for idx, peak in peaks.items():
            peak.u = par[f'u_{idx}']
            peak.fwhm = par[f'fwhm_{idx}']
            peak.k = par[f'k_{idx}_{e_idx}']
            peak.b = par[f'b_{idx}']
            peak.phi = par[f'phi_{idx}']
        return peaks

    def f2min(param, S, fit_peaks, I, lims):
        """
        Function that calculates the residual to be minimized in the least squares sense.
        This function requires a set of pre-built fit.Peak objects, stored in a dictionary. The parameters of the peaks are replaced on this dictionary according to the values in the lmfit.Parameter object. At this point, the total trace is computed and the residual is returned as the difference between the experimental spectrum and the total trace, only in the region delimited by the "lims" tuple.
        ------------
        Parameters:
        - param: lmfit.Parameters object
            Usual lmfit stuff
        - S: 2darray
            Experimental spectrum
        - fit_peaks: list of dict
            Collection of fit.Peak objects
        - I: list or 1darray
            Absolute intensity values for all experiments
        - lims: slice
            Trimming region corresponding to the fitting window, in points
        -----------
        Returns:
        - residual: 1darray
            Experimental - calculated, in the fitting window, concatenated through all the experiments
        """
        param['count'].value += 1
        # Unpack the lmfit.Parameters object
        par = param.valuesdict()
        # create a shallow copy of the fit_peaks dictionary to prevent overwriting
        fit_peaks_in = deepcopy(fit_peaks)
        fit_peaks_up = []   # placeholder
        for j, peaks in enumerate(fit_peaks_in):
            # Update the peaks dictionary according to how lmfit is moving the fit parameters
            peaks_up = peaks_frompar(peaks, par, j+1)
            fit_peaks_up.append(peaks_up)
        # Compute the total trace and the residuals
        total = calc_total(fit_peaks_up, 1)
        residual = np.concatenate([S[j,lims] / I[j] - total[j,lims] for j, _ in enumerate(fit_peaks_in)])

        print(f'Step: {par["count"]:6.0f} | Target: {np.sum(residual**2):10.5e}', end='\r')
        return residual

    def gen_reg(regions):
        """
        Generator function that loops on the regions and extracts the limits of the fitting window, the limits, and the dictionary of peaks.
        """
        for k, region in enumerate(regions):
            # Get limits and total intensity from the dictionary of the first region
            limits = region[0]['limits']
            if 1:   # Switch: turn this print on and off
                print(f'Fitting of region {k+1}/{Nr}. [{limits[0]:.3f}:{limits[1]:.3f}] ppm')
            # Make a copy of the region dictionary and remove what is not a peak
            peaklist = deepcopy(region)
            for peaks in peaklist:
                if 'limits' in list(peaks.keys()):
                    peaks.pop('limits')
            yield limits, peaklist


    # -----------------------------------------------------------------------------
    # Make the acqus dictionary to be fed into the fit.Peak objects
    acqus = { 't1': t_AQ, 'SFO1': SFO1, 'o1p': o1p, }

    N = S.shape[-1]     # Number of points of the spectrum
    Nr = len(regions)   # Number of regions to be fitted

    # Write info on the fit in the output file
    with open(f'{filename}.fvf', 'a', buffering=1) as f:
        now = datetime.now()
        date_and_time = now.strftime("%d/%m/%Y at %H:%M:%S")
        f.write('! Fit performed by {} on {}\n\n'.format(os.getlogin(), date_and_time))

    # Generate the values from the regions dictionary with the gen_reg generator
    Q = gen_reg(regions)

    # Start fitting loop
    prev = 0
    for q in Q:
        limits, peaklist = q    # Unpack
        Np = len(peaklist[0].keys())  # Number of Peaks

        # Convert the limits from ppm to points and make the slice
        limits_pt = misc.ppmfind(ppm_scale, limits[0])[0], misc.ppmfind(ppm_scale, limits[1])[0]
        lims = slice(min(limits_pt), max(limits_pt))

        # Create a list of dictionaries which contains Peak objects
        fit_peaks = [{} for w in range(S.shape[0])]
        # Fill them up
        for j, peaks in enumerate(peaklist):    # j is the index of the experiment
            for key, peakval in peaks.items():  # loop on the items of the single experiment
                # Make the fit.Peak objects
                fit_peaks[j][key] = fit.Peak(acqus, N=N, **peakval)
            # Convert the intensities into smaller values
            Ks = [fit_peaks[j][key].k for key in fit_peaks[j].keys()]
            Ks_norm, _ = misc.molfrac(Ks)
            # Replace with the new ones
            for K, key in zip(Ks_norm, peaks.keys()):
                fit_peaks[j][key].k = K

        # Compute the initial guess array of the fit
        fit_peaks_iguess = deepcopy(fit_peaks)
        i_guess = calc_total(fit_peaks_iguess, 1)
        # Compute the matching-intensity factors
        I_arr = []
        for j in range(S.shape[0]):
            I_arr.append(fit.fit_int(S[j,lims], i_guess[j,lims])[0])
        I_arr = np.array(I_arr)


        # Make the lmfit.Parameters object
        param = l.Parameters()
        # Add the peaks' parameters to a lmfit Parameters object
        peak_keys = ['u', 'fwhm', 'k', 'b', 'phi']

        for j, peaks in enumerate(fit_peaks):       # j is the index of the experiment
            for idx, peak in peaks.items():
                # Name of the object: <parameter>_<index>
                p_key = f'_{idx}'

                # Fill the Parameters object
                for key in peak_keys:
                    if 'k' in key:  # add also the experiment index
                        par_key = f'{key}{p_key}_{j+1}' 
                    else:           # add only the parameter
                        par_key = f'{key}{p_key}'   # Add the parameter to the label
                    val = peak.par()[key]       # Get the value from the input dictionary
                    param.add(par_key, value=val)   # Make the Parameter object

                    # Set the limits for each parameter, and fix the ones that have not to be varied during the fit
                    if 'u' in key:  # u: [u-u_tol, u+u_tol]
                        param[par_key].set(min=val-u_tol, max=val+u_tol)
                    elif 'fwhm' in key: # fwhm: [max(0, fwhm-f_tol), fwhm+f_tol] (avoid negative fwhm)
                        param[par_key].set(min=max(0, val-f_tol), max=val+f_tol)
                    elif 'k' in key:    # k: [0, 2]
                        param[par_key].set(min=0, max=2)
                    elif 'phi' in key:  # phi: [-180°, +180°]
                        param[par_key].set(min=-180, max=180, vary=vary_phase)
                    elif 'b' in key:  # b: [0, 1]
                        param[par_key].set(min=0, max=1, vary=vary_b)

        # Wrap the fitting routine in a function in order to use @cron for measuring the runtime of the fit
        @cron
        def start_fit():
            param.add('count', value=0, vary=False)
            minner = l.Minimizer(f2min, param, fcn_args=(S, fit_peaks, I_arr, lims))
            result = minner.minimize(method='leastsq', max_nfev=int(itermax), xtol=1e-8, ftol=1e-8)
            print(f'{result.message} Number of function evaluations: {result.nfev}.')
            return result
        # Do the fit
        result = start_fit()
        # Unpack the fitted values
        popt = result.params.valuesdict()

        # Replace the initial values with the fitted ones
        fit_peaks_opt = []
        for j, peaks in enumerate(fit_peaks):
            fit_peaks_opt.append(peaks_frompar(peaks, popt, j+1))

        # Correct the intensities
        for j, peaks in enumerate(fit_peaks_opt):
            for _, idx in enumerate(peaks.keys()):
                peaks[idx].k *= I_arr[j]

        # Write a section of the output file
        write_vf_P2D(f'{filename}.fvf', fit_peaks_opt, limits, prev)
        prev += Np



class Voigt_Fit_P2D:
    """
    This class offers an "interface" to fit a pseudo 2D NMR spectrum.
    -------
    Attributes:
    - ppm_scale: 1darray
        Self-explanatory
    - S : 2darray
        Spectrum to fit. Only real part
    - t_AQ: 1darray
        acquisition timescale of the spectrum
    - SFO1: float
        Larmor frequency of the nucleus
    - o1p : float
        Pulse carrier frequency
    - filename: str
        Root of the names of the files that will be saved 
    - X_label: str
        Label for the chemical shift axis in the figures
    - i_guess: list
        Initial guess for the fit, read by a .ivf file with fit.read_vf_P2D
    - result: list
        Result the fit, read by a .fvf file with fit.read_vf_P2D
    """

    def __init__(self, ppm_scale, S, t_AQ, SFO1, o1p, nuc=None, filename='fit'):
        """
        Initialize the class with common values.
        --------
        Parameters:
        - ppm_scale: 1darray
            ppm scale of the spectrum
        - S: 2darray
            Spectrum to be fitted
        - t_AQ: 1darray
            Acquisition timescale
        - SFO1: float
            Larmor frequency of the observed nucleus, in MHz
        - o1p: float
            Carrier position, in ppm
        - nuc: str
            Observed nucleus. Used to customize the x-scale of the figures.
        - filename: str or None
            Root of the name of the files that will be saved
        """
        self.ppm_scale = ppm_scale
        self.S = S
        self.t_AQ = t_AQ
        self.SFO1 = SFO1
        self.o1p = o1p
        self.filename = filename
        if nuc is None:
            self.X_label = r'$\delta\,$ /ppm'
        elif isinstance(nuc, str):
            fnuc = misc.nuc_format(nuc)
            self.X_label = r'$\delta$ ' + fnuc +' /ppm'

    def iguess(self, input_file=None, expno=0, n=-1,):
        """
        Reads, or computes, the initial guess for the fit.
        If the file is there already, it just reads it with fit.read_vf. Otherwise, it calls fit.make_iguess to make it.
        --------
        Parameters:
        - input_file: str or None
            Path to the input file. If None, "<self.filename>.ivf" is used
        - expno: int
            Number of the experiment on which to compute the initial guess, in python numbering
        - n: int
            Index of the initial guess to be read (default: last one)
        """
        # Set the default filename, if not given
        if input_file is None:
            input_file = f'{self.filename}'
        # Check if the file exists
        in_file_exist = os.path.exists(f'{input_file}.ivf')

        if in_file_exist is True:       # Read everything you need from the file
            regions = fit.read_vf_P2D(f'{input_file}.ivf')
        else:                           # Make the initial guess interactively and save the file.
            fit.make_iguess_P2D(self.S, self.ppm_scale, expno, self.t_AQ, self.SFO1, self.o1p, filename=input_file)
            regions = fit.read_vf_P2D(f'{input_file}.ivf')
        # Store it
        self.i_guess = regions
        print(f'{input_file}.ivf loaded as input file.')

    def load_fit(self, output_file=None, n=-1):
        """
        Reads a file with fit.read_vf_P2D and stores the result in self.result.
        ---------
        Parameters:
        - output_file: str
            Path to the .fvf file to be read. If None, "<self.filename>.fvf" is used.
        - n: int
            Index of the fit to be read (default: last one)
        """
        # Set the default filename, if not given
        if output_file is None:
            output_file = f'{self.filename}'
        # Check if the file exists
        out_file_exist = os.path.exists(f'{output_file}.fvf')
        if out_file_exist is True:       # Read everything you need from the file
            regions = fit.read_vf_P2D(f'{output_file}.fvf', n=n)
        else:
            raise NameError(f'{output_file}.fvf does not exist.')
        # Store
        self.result = regions
        print(f'{output_file}.fvf loaded as fit result file.')

    def dofit(self, u_tol=1, f_tol=10, vary_phase=False, vary_b=True, itermax=10000, filename=None):
        """
        Perform a lineshape deconvolution fitting by calling fit.voigt_fit_P2D.
        The initial guess is read from the attribute self.i\_guess.
        ------------
        Parameters:
        - u_tol: float
            Determines the displacement of the chemical shift (in ppm) from the starting value.
        - f_tol: float
            Determines the displacement of the linewidth (in Hz) from the starting value.
        - vary_phase: bool
            Allow the peaks to change phase (True) or not (False)
        - vary_b: bool
            Allow the peaks to change Lorentzian/Gaussian ratio
        - itermax: int
            Maximum number of allowed iterations
        - filename: str
            Path to the output file. If None, "<self.filename>.fvf" is used
        """

        # Make a shallow copy of the real part of the experimental spectrum
        S = np.copy(self.S.real)
        # Check if the initial guess was loaded correctly
        if not isinstance(self.i_guess, list):
            raise ValueError('Initial guess not correctly loaded')
        # Set the output filename, if not given
        if filename is None:
            filename = f'{self.filename}'

        # Do the fit
        fit.voigt_fit_P2D(S, self.ppm_scale, self.i_guess, self.t_AQ, self.SFO1, self.o1p, u_tol=u_tol, f_tol=f_tol, vary_phase=vary_phase, vary_b=vary_b, itermax=itermax, filename=filename)
        # Store
        self.result = fit.read_vf_P2D(f'{filename}.fvf')


    def plot(self, what='result', show_total=True, show_res=False, res_offset=0, labels=None, filename=None, ext='tiff', dpi=600):
        """
        Plots either the initial guess or the result of the fit, and saves all the figures. Calls fit.plot_fit_P2D.
        The figures <filename>_full will show the whole model and the whole spectrum. 
        The figures labelled with _R<k> will depict a detail of the fit in the k-th fitting region.
        Optional labels for the components can be given: in this case, the structure of 'labels' should match the structure of self.result (or self.i_guess). This means that the length of the outer list must be equal to the number of fitting region, and the length of the inner lists must be equal to the number of peaks in that region.
        ------------
        Parameters:
        - what: str
            'iguess' to plot the initial guess, 'result' to plot the fitted data
        - show_total: bool
            Show the total trace (i.e. sum of all the components) or not
        - show_res: bool
            Show the plot of the residuals
        - res_offset: float
            Displacement of the residuals plot from 0, to be given as a fraction of the height of the experimental spectrum. res_offset > 0 will move the residuals BELOW the zero-line!
        - labels: list of list
            Optional labels for the components. The structure of this parameter must match the structure of self.result
        - filename: str
            Root of the name of the figures that will be saved. If None, <self.filename> is used
        - ext: str
            Format of the saved figures
        - dpi: int
            Resolution of the figures, in dots per inches
        """
        # select the correct object to plot
        if what == 'iguess':
            regions = deepcopy(self.i_guess)
        elif what == 'result':
            regions = deepcopy(self.result)
        else:
            raise ValueError('Specify what you want to plot: "iguess" or "result"')
               
        # Set the filename, if not given
        if filename is None:
            filename = f'{self.filename}'

        # Make the figures
        S = np.copy(self.S.real)
        fit.plot_fit_P2D(S, self.ppm_scale, regions, self.t_AQ, self.SFO1, self.o1p, show_total=show_total, show_res=show_res, res_offset=res_offset, X_label=self.X_label, labels=labels, filename=filename, ext=ext, dpi=dpi)

    def get_fit_lines(self, what='result'):
        """
        Calculates the components, and the total fit curve used as initial guess, or as fit results..
        The components will be returned as a list, not split by region.
        --------
        Parameters:
        - what: str
            'iguess' or 'result' 
        --------
        Returns:
        - signals: list of list of 1darray
            Components used for the fit
        - total: 2darray
            Sum of all the signals
        - limits_list: list
            List of the region delimiters, in ppm
        """
        # Select the correct object
        if what == 'iguess':
            regions = deepcopy(self.i_guess)
        elif what == 'result':
            regions = deepcopy(self.result)
        else:
            raise ValueError('Specify what you want to plot: "iguess" or "result"')

        # Make the acqus dictionary for the fit.Peak objects
        acqus = { 't1': self.t_AQ, 'SFO1': self.SFO1, 'o1p': self.o1p, }
        # Placeholder
        signals = []
        limits_list = []

        # Loop on the regions
        for region in regions:
            # Get limits from the dictionary
            peaklist = list(region)
            for peaks in peaklist:
                if 'limits' in list(peaks.keys()):
                    limits = peaks.pop('limits')
            limits_list.append(limits)

            # Placeholder
            fit_peaks = [{} for w in range(self.S.shape[0])]
            list_signals = []
            for j, peaks in enumerate(peaklist):        # j runs on the experiments
                for key, peakval in peaks.items():
                    # Create the fit.Peak objects
                    fit_peaks[j][key] = fit.Peak(acqus, N=self.S.shape[-1], **peakval)
                # Get the arrays from the dictionary and put them in the list
                list_signals.append([p() for _, p in fit_peaks[j].items()])
            signals.extend(list_signals) # Dimensions (n. experiments, n.peaks per experiment, n.points per experiment)

        # Compute the total trace
        total = np.sum(signals, axis=1) # sum the peaks 
        return signals, total, limits_list


    def res_histogram(self, what='result', nbins=500, density=True, f_lims=None, xlabel='Residuals', x_symm=True, barcolor='tab:green', fontsize=20, filename=None, ext='tiff', dpi=300):
        """
        Computes the histogram of the residuals and saves it in the same folder of the fit figures.
        Employs fit.histogram to make the figure.
        --------
        Parameters:
        - what: str
            'iguess' or 'result' 
        - nbins : int
            number of bins to be calculated
        - density : bool
            True for normalize data
        - f_lims : tuple or None
            limits for the x axis of the figure
        - xlabel : str or None
            Text to be displayed under the x axis
        - x_symm : bool
            set it to True to make symmetric x-axis with respect to 0
        - barcolor: str
            Color of the bins
        - fontsize: float
            Biggest fontsize in the figure
        - name : str
            name for the figure to be saved
        - ext: str
            Format of the image
        - dpi: int
            Resolution of the image in dots per inches
        """
        # Filename check
        if filename is None:
            filename = f'{self.filename}'
        try:
            os.mkdir(f'{filename}_fit')
        except:
            pass
        finally:
            # Update the filename for the figures by including the new directory
            filename = os.path.join(filename+f'_fit', f'{filename}_rhist')

        # Select the correct object
        if what == 'iguess':
            regions = deepcopy(self.i_guess)
        elif what == 'result':
            regions = deepcopy(self.result)
        else:
            raise ValueError('Specify what you want to plot: "iguess" or "result"')

        # Make the acqus dictionary for the fit.Peak objects'
        acqus = { 't1': self.t_AQ, 'SFO1': self.SFO1, 'o1p': self.o1p, }

        # Get the total function and the limits
        _, total, limits_list = self.get_fit_lines(what)
        # Convert the limits in points according to the ppm scale
        limits_pt_list = [ [misc.ppmfind(self.ppm_scale, w)[0] for w in lims] for lims in limits_list ]

        # Placeholders
        exp_trim, total_trim = [], []
        for k, region in enumerate(regions):        # loop on the regions
            # Compute the slice
            lims = slice(min(limits_pt_list[k]), max(limits_pt_list[k]))
            # Trim the experimental data and the total 
            exp_trim.append(self.S[...,lims].real)
            total_trim.append(total[...,lims])
        # Sum on different regions
        exp_trim = np.sum(exp_trim, axis=0)
        total_trim = np.sum(total_trim, axis=0)

        # Compute the residuals and concatenate them
        residual = exp_trim - total_trim
        residual_arr = np.concatenate([r for r in residual], axis=-1)

        fit.histogram(residual_arr, nbins=nbins, density=density, f_lims=f_lims, xlabel=xlabel, x_symm=x_symm, barcolor=barcolor, fontsize=fontsize, name=filename, ext=ext, dpi=dpi)



