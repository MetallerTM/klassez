#! /usr/bin/env python3

import os
import io
import sys
import numpy as np
from pathlib import Path
from csaps import csaps
import matplotlib.pyplot as plt
import lmfit
from datetime import datetime
from copy import deepcopy
import getpass

from . import fit, misc, sim, figures, processing, anal, gui
from .config import CM, COLORS, cron, safe_kws, cprint

print = cprint

"""
Functions for performing fits.
"""

s_colors = ['tab:cyan', 'tab:red', 'tab:green', 'tab:purple', 'tab:pink', 'tab:gray', 'tab:brown', 'tab:olive', 'salmon', 'indigo']


def histogram(data, nbins=100, density=True, f_lims=None, xlabel=None, x_symm=False, fitG=True, barcolor='tab:blue', fontsize=10, filename=None, ext='svg', dpi=600):
    """
    Computes an histogram of ``data`` and tries to fit it with a gaussian lineshape.
    The parameters of the gaussian function are calculated analytically directly from ``data`` using ``scipy.stats.norm``

    Parameters
    ----------
    data : ndarray
        the data to be binned
    nbins : int
        number of bins to be calculated
    density : bool
        True for normalize data
    f_lims : tuple or None
        limits for the x axis of the figure
    xlabel : str or None
        Text to be displayed under the x axis
    x_symm : bool
        set it to True to make symmetric x-axis with respect to 0
    fitG : bool
        Shows the gaussian approximation
    barcolor : str
        Color of the bins
    fontsize : float
        Biggest fontsize in the figure
    filename : str or Path
        name for the figure to be saved
    ext : str
        Format of the image
    dpi : int
        Resolution of the image in dots per inches

    Returns
    -------
    m : float
        Mean of data
    s : float
        Standard deviation of data.

    .. seealso::

        :func:`klassez.fit.ax_histogram`
    """
    filename = Path(filename)

    fig = plt.figure('Histogram')
    ax = fig.add_subplot()
    plt.subplots_adjust(left=0.10, bottom=0.10, top=0.90, right=0.95)
    fig.set_size_inches(figures.figsize_large)

    m, s = fit.ax_histogram(ax, data, nbins=nbins, density=density, f_lims=f_lims, xlabel=xlabel, x_symm=x_symm, fitG=fitG, barcolor=barcolor, fontsize=fontsize)

    if filename:
        print(f'Saving {filename}.{ext}...', c='tab:cyan')
        plt.savefig(filename.with_suffix(f'.{ext}'), dpi=dpi)
    else:
        plt.show()
    plt.close()
    print('Done.', c='tab:cyan')

    return m, s


def ax_histogram(ax, data0, nbins=100, density=True, f_lims=None, xlabel=None, x_symm=False, fitG=True, barcolor='tab:blue', fontsize=10):
    """
    Computes an histogram of ``data`` and tries to fit it with a gaussian lineshape.
    The parameters of the gaussian function are calculated analytically directly from ``data`` using ``scipy.stats.norm``

    Parameters
    ----------
    ax : matplotlib.subplot Object
        panel of the figure where to put the histogram
    data0 : ndarray
        the data to be binned
    nbins : int
        number of bins to be calculated
    density : bool
        True for normalize data
    f_lims : tuple or None
        limits for the x axis of the figure
    xlabel : str or None
        Text to be displayed under the x axis
    x_symm : bool
        set it to True to make symmetric x-axis with respect to 0
    fitG : bool
        Shows the gaussian approximation
    barcolor : str
        Color of the bins
    fontsize : float
        Biggest fontsize in the figure

    Returns
    -------
    m : float
        Mean of data
    s : float
        Standard deviation of data.
    """

    if len(data0.shape) > 1:
        data = data0.real.flatten()
    else:
        data = np.copy(data0.real)

    if x_symm:
        lims = (-max(np.abs(data)), max(np.abs(data)))
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

    misc.mathformat(ax, axis='both', limits=(-3, 3))

    misc.set_fontsizes(ax, fontsize)

    return m, s


def bin_data(data0, nbins=100, density=True, x_symm=False):
    """
    Computes the histogram of data, sampling it into nbins bins.

    Parameters
    ----------
    data : ndarray
        the data to be binned
    nbins : int
        number of bins to be calculated
    density : bool
        True for normalize data
    x_symm : bool
        set it to True to make symmetric x-axis with respect to 0

    Returns
    -------
    hist : 1darray
        The bin intensity
    bin_scale : 1darray
        Scale built with the mean value of the bin widths.
    """
    if len(data0.shape) > 1:
        data = data0.flatten()
    else:
        data = np.copy(data0)

    if x_symm:
        lims = (- max(np.abs(data)), max(np.abs(data)))
    else:
        lims = (min(data), max(data))

    hist, bin_edges = np.histogram(data, bins=nbins, range=lims, density=density)   # Computes the bins for the histogram
    bin_scale = np.array([np.mean((bin_edges[k], bin_edges[k+1])) for k in range(len(bin_edges) - 1)])
    return hist, bin_scale


def lr(y, x=None, force_intercept=False, w=None):
    r"""
    Performs a linear regression of ``y`` with a model :math:`y_c = mx + q`.

    If ``w=None`` then ``w = np.ones_like(x)``.

    If ``force_intercept=False``:

    .. math::

        m = \frac{\sum w x y}{\sum w x^2}, \quad q = 0

    else, two more parameters are defined:

    .. math::

        x_w = \frac{ \sum w x }{ \sum w }, \quad y_w = \frac{ \sum w y }{ \sum w }

    .. math::

        m = \frac{ \sum w (x-x_w) (y-y_w)}{\sum w (x - x_w)^2}, \quad
        q = y_w - m\,x_w



    Parameters
    ----------
    y : 1darray
        Data to be fitted
    x : 1darray
        Independent variable. If None, the point indexes are used.
    force_intercept : bool
        If True, forces the intercept to be zero.
    w : 1darray or None
        Weights to be used for the linear regression.

    Returns
    -------
    y_c : 1darray
        Fitted trend
    values : tuple
        ``(m, q)``
    """
    # Make the scale of points, if not given
    if x is None:
        x = np.arange(y.shape[-1])
    if w is None:
        w = np.ones(y.shape[-1])

    if force_intercept:     # It is x
        m = np.sum(w * x * y) / np.sum(w * x**2)
        q = 0
    else:                   # it is [1, x]
        xw = np.sum(w * x) / np.sum(w)
        yw = np.sum(w * y) / np.sum(w)
        m = np.sum(w * (x - xw) * (y - yw)) / np.sum(w * (x - xw)**2)
        q = yw - m * xw

    # Compute the model
    y_c = m * x + q
    return y_c, (m, q)


def calc_R2(y, y_c):
    r"""
    Computes the R-squared coefficient of a linear regression as:

    .. math::

        R^2 = 1 - \frac{ \sum (y - <y>)^2  }{ \sum (y - y_c)^2 }


    Parameters
    ----------
    y : 1darray
        Experimental data
    y_c : 1darray
        Calculated data

    Returns
    -------
    R2 : float
        R-squared coefficient
    """
    sst = np.sum((y - np.mean(y))**2)
    sse = np.sum((y - y_c)**2)
    R2 = 1 - sse/sst
    return R2


def fit_int(y, y_c, q=True):
    r"""
    Computes the optimal intensity and intercept of a linear model in the least squares sense.
    Let :math:`y` be the experimental data and :math:`y_c` the model, and let :math:`<w>` the mean of variable :math:`w`.
    Then, if ``q=False``:

    .. math::

        A = \frac{ <y_c y> }{ <y_c^2> }, \qquad
        q = 0

    else:

    .. math::

        A = \frac{ <y_c y> - <y_c><y> }{ <y_c^2> - <y_c>^2 }, \qquad
        q = \frac{ <y_c>^2<y> - <y_c><y_c y> }{ <y_c^2> - <y_c>^2 }


    Parameters
    ----------
    y : 1darray
        Experimental data
    y_c : 1darray
        Model data
    q : bool
        If True, includes the offset in the calculation. If False, only the intensity factor is computed.

    Returns
    -------
    A : float
        Optimized intensity
    q : float
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


def make_signal(t, u, s, k, b, phi, A, SFO1=701.125, o1p=0, N=None):
    """
    Generates a voigt signal on the basis of the passed parameters in the time domain. Then, makes the Fourier transform and returns it.

    Parameters
    ----------
    t : ndarray
        acquisition timescale
    u : float
        chemical shift /ppm
    s : float
        full-width at half-maximum /Hz
    k : float
        relative intensity
    b : float
        fraction of gaussianity (0 = Lorentzian, 1 = Gaussian)
    phi : float
        phase of the signal, in degrees
    A : float
        total intensity
    SFO1 : float
        Larmor frequency /MHz
    o1p : float
        pulse carrier frequency /ppm
    N : int or None
        length of the final signal. If None, the signal is not zero-filled before to be transformed.

    Returns
    -------
    sgn : 1darray
        generated signal in the frequency domain
    """
    U = misc.ppm2freq(u, SFO1, o1p)         # conversion to frequency units
    S = s * 2 * np.pi                       # conversion to radians
    phi = phi * np.pi / 180                 # conversion to radians
    sgn = sim.t_voigt(t, U, S, A=A*k, phi=phi, b=b)  # make the signal
    if isinstance(N, int):
        sgn = processing.zf(sgn, N)         # zero-fill it
    sgn = processing.ft(sgn)                # transform it
    return sgn


def plot_fit(S, ppm_scale, regions, t_AQ, SFO1, o1p, show_total=False,
             show_res=False, res_offset=0, show_basl=False, X_label=r'$\delta$ /ppm',
             labels=None, filename='fit', ext='svg', dpi=600, dim=None):
    """
    Plots either the initial guess or the result of the fit, and saves all the figures.
    The figure `<filename>_full` will show the whole model and the whole spectrum.
    The figures labelled with `_R<k>` will depict a detail of the fit in the k-th fitting region.
    Optional labels for the components can be given: in this case, the structure of ``labels`` should match the structure of ``regions``.
    This means that the length of the outer list must be equal to the number of fitting region, and the length of the inner lists must be equal to the number of peaks in that region.

    Parameters
    ----------
    S : 1darray
        Spectrum to be fitted
    ppm_scale : 1darray
        ppm scale of the spectrum
    regions : dict
        Generated by :func:`klassez.fit.read_vf`
    t_AQ : 1darray
        Acquisition timescale
    SFO1 : float
        Larmor frequency of the observed nucleus, in MHz
    o1p : float
        Carrier position, in ppm
    show_total : bool
        Show the total trace (i.e. sum of all the components) or not
    show_res : bool
        Show the plot of the residuals
    res_offset : float
        Displacement of the residuals plot from 0, to be given as a fraction of the height of the experimental spectrum. ``res_offset`` > 0 will move the residuals BELOW the zero-line!
    show_basl : bool
        If True, displays the baseline on the spectrum and uses it to compute the total trace.
    X_label : str
        Text to show as label for the chemical shift axis
    labels : list of list
        Optional labels for the components. The structure of this parameter must match the structure of self.result
    filename : str or Path
        Root of the name of the figures that will be saved.
    ext : str
        Format of the saved figures
    dpi : int
        Resolution of the figures, in dots per inches
    dim : tuple
        Size of the figure in inches (length, height)
    """
    def calc_total(peaks, A=1):
        """
        Calculates the sum trace from a collection of peaks stored in a dictionary.
        If the dictionary is empty, returns an array of zeros.

        Parameters
        ----------
        peaks: dict
            Components
        A: float
            Absolute intensity

        Returns
        -------
        total: 1darray
            Sum spectrum
        """
        # Get the arrays from the dictionary
        T = [p(A) for _, p in peaks.items()]
        if len(T) > 0:  # Check for any peaks
            total = np.sum(T, axis=0)
            return total.real
        else:
            return np.zeros_like(ppm_scale)

    print('Saving figures...', c='tab:cyan')
    # Shallow copy of the real part of the experimental spectrum
    S_r = np.copy(S.real)
    N = S_r.shape[-1]       # For (eventual) zero-filling
    # Make the acqus dictionary to be fed into the fit.Peak objects
    acqus = {'t1': t_AQ, 'SFO1': SFO1, 'o1p': o1p, }

    # Single regions
    whole_basl = np.zeros_like(ppm_scale)
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
        Int = in_region.pop('I')
        if 'bas_c' in region.keys():
            bas_c = Int * region['bas_c']
            in_region.pop('bas_c')
        else:
            bas_c = np.zeros(5)

        # Create a dictionary of fit.Peak objects with the same structure of in_region
        peaks = {key: fit.Peak(acqus, N=N, **peakval) for key, peakval in in_region.items()}
        # Get the total trace
        total = calc_total(peaks, Int)

        # Trim the ppm scale according to the fitting region
        t_ppm = ppm_scale[lims]

        # Baseline computation
        x = np.linspace(0, 1, len(t_ppm))
        basl = misc.polyn(x, bas_c)
        whole_basl = misc.sum_overlay(whole_basl, basl, max(limits), ppm_scale)

        # Make the figure
        fig = plt.figure('Fit')
        if dim is None:
            fig.set_size_inches(figures.figsize_large)
        else:
            fig.set_size_inches(dim)
        ax = fig.add_subplot()
        plt.subplots_adjust(left=0.10, bottom=0.10, top=0.90, right=0.95)

        # Plot the experimental dataset
        ax.plot(t_ppm, S_r[lims], c='k', lw=1, label='Experimental')

        if show_total is True:  # Plot the total trace in blue
            ax.plot(t_ppm, total[lims]+basl, c='b', lw=0.5, label='Fit')
        if show_basl is True:
            ax.plot(t_ppm, basl, c='mediumorchid', lw=0.5, label='Baseline', zorder=3)

        for key, peak in peaks.items():  # Plot the components
            p_sgn, = ax.plot(t_ppm, peak(Int)[lims], lw=0.6, label=f'{key}')
            if labels is not None:  # Set the custom label
                p_sgn.set_label(labels[k][key-1])

        if show_res is True:    # Plot the residuals
            # Compute the absolute value of the offset
            r_off = res_offset * (max(S_r[lims])-min(S_r[lims]))
            ax.plot(t_ppm, (S_r - total)[lims] - basl - r_off, c='g', ls=':', lw=0.6, label='Residuals')

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

    # Total
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
        Int = in_region.pop('I')
        if 'bas_c' in region.keys():
            in_region.pop('bas_c')
        else:
            pass
        # Make the dictionary of fit.Peak objects
        peaks = {key: fit.Peak(acqus, N=N, **peakval) for key, peakval in in_region.items()}
        # Plot the components
        for idx, peak in peaks.items():
            p_sgn, = ax.plot(ppm_scale, peak(Int), lw=0.6, label=f'Win. {k+1}, Comp. {idx}', zorder=10)
            if labels is not None:  # Set custom label
                p_sgn.set_label(labels[k][idx-1])

        # Add these contributions to the total trace
        total += calc_total(peaks, Int)

    # Residuals
    if show_total is True:  # Plot the total trace
        ax.plot(ppm_scale, total+whole_basl, c='b', lw=0.5, label='Fit', zorder=2)
    if show_basl is True:
        ax.plot(ppm_scale, whole_basl, c='mediumorchid', lw=0.5, label='Baseline', zorder=3)

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
    print('Done.', c='tab:cyan')


def voigt_fit_indep(S, ppm_scale, regions, t_AQ, SFO1, o1p,
                    u_lim=1, f_lim=10, k_lim=(0, 3), ph_lim=(-180, 180),
                    vary_phase=False, vary_b=True,
                    itermax=10000, fit_tol=1e-8, filename='fit', method='leastsq', basl_fit='no'):
    """
    Performs a lineshape deconvolution fit using a Voigt model.
    The initial guess must be read from a `.ivf file`. All components are treated as independent, regardless from the value of the ``group`` attribute.
    The fitting procedure operates iteratively one window at the time.

    Parameters
    ----------
    S : 1darray
        Experimental spectrum
    ppm_scale : 1darray
        PPM scale of the spectrum
    regions : dict
        Generated by :func:`klassez.fit.read_vf`
    t_AQ : 1darray
        Acquisition timescale
    SFO1 : float
        Nucleus Larmor frequency /MHz
    o1p : float
        Carrier frequency /ppm
    u_lim : float
        Maximum allowed displacement of the chemical shift from the initial value /ppm
    f_lim : float
        Maximum allowed displacement of the linewidth from the initial value /ppm
    k_lim : float or tuple
        If tuple, minimum and maximum allowed values for ``k`` during the fit. If float, maximum displacement from the initial guess
    ph_lim : tuple
        Minimum and maximum allowed values for the phases, in degrees
    vary_phase : bool
        Allow the peaks to change phase
    vary_b : bool
        Allow the peaks to change Lorentzian/Gaussian ratio
    itermax : int
        Maximum number of allowed iterations
    fit_tol : float
        Target value to be set for ``x_tol`` and ``f_tol``
    filename : str or Path
        Name of the file where the fitted values will be saved. The `.fvf` extension is added automatically
    method : str or list of str
        Method to be used for the optimization. See ``lmfit`` for details. There is the option to run multiple optimizations in series.
    basl_fit : str
        How to address the baseline fit. The options are:

        * ``"no"``: Do not use baseline (default)
        * ``"fixed"``: The baseline is computed once and kept fixed during the optimization
        * ``"fit"``: The baseline coefficients enter as fit parameters during the nonlinear optimization
        * ``"calc"``: The baseline coefficients are calculated during the optimization via linear least-squares optimization

    Returns
    -------
    lmfit_results: list of lmfit.Minimizer.MinimizerResult
        Sequence of the fit results, ordered as the regions dictionary


    .. seealso::

        :class:`lmfit.Minimizer`

        :func:`lmfit.Minimizer.minimize`

        :class:`lmfit.Minimizer.MinimizerResult`
    """

    # USED FUNCTIONS

    def calc_total(peaks, A=1):
        """
        Calculates the sum trace from a collection of peaks stored in a dictionary.
        If the dictionary is empty, returns an array of zeros.

        Parameters
        ----------
        peaks : dict
            Components
        A : float
            Absolute intensity

        Returns
        -------
        total : 1darray
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
        Replaces the values of a "peaks" dictionary, which contains a ``fit.Peak`` object for each key ``idx``, with the values contained in the ``par`` dictionary.
        The par dictionary keys must have keys of the form ``<parameter>_<idx>``, where ``<parameter>`` is in ``[u, fwhm, k, 'b', 'phi']``, and ``<idx>`` are the keys of the peaks dictionary.

        Parameters
        ----------
        peaks : dict
            Collection of fit.Peak objects
        par : dict
            New values for the peaks

        Returns
        -------
        peaks : dict
            Updated peaks dictionary with the new values
        """
        for idx, peak in peaks.items():
            peak.u = par[f'u_{idx}']
            peak.fwhm = par[f'fwhm_{idx}']
            peak.k = par[f'k_{idx}']
            peak.b = par[f'b_{idx}']
            peak.phi = par[f'phi_{idx}']
        return peaks

    def f2min(param, S, fit_peaks, Int, lims, x, first_residual=1, basl_fit='no'):
        """
        Function that calculates the residual to be minimized in the least squares sense.
        This function requires a set of pre-built ``fit.Peak`` objects, stored in a dictionary.
        The parameters of the peaks are replaced on this dictionary according to the values in the ``lmfit.Parameter object``.
        At this point, the total trace is computed and the residual is returned as the difference between the experimental spectrum and the total trace,
        only in the region delimited by the "lims" tuple.

        Parameters
        ----------
        param : lmfit.Parameters object
            Usual lmfit stuff
        S : 1darray
            Experimental spectrum
        fit_peaks : dict
            Collection of ``fit.Peak`` objects
        I : float
            Absolute intensity.
        lims : slice
            Trimming region corresponding to the fitting window, in points
        x : 1darray
            Baseline scale
        basl_fit : str
            Method for computation of the baseline
        first_residual : float
            Target value at the first call of this function. Used to compute the relative target function.

        Returns
        -------
        residual : 1darray
            ``experimental - calculated``, in the fitting window
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
        # compute the baseline
        if basl_fit != 'calc':
            bas_c = np.array([par[f'c{w}'] for w in range(5)])
            basl = misc.polyn(x, bas_c)
            calc += basl
            correction_factor, _ = fit.fit_int(exp, calc, q=False)
        else:
            correction_factor = 1
        residual = exp - calc * correction_factor
        if basl_fit == 'calc':
            bas_c = fit.lsp(residual, x, 5)
            basl = misc.polyn(x, bas_c)
            for k, c in enumerate(bas_c):
                param[f'c{k}'].set(value=c)
            residual -= basl
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
            Int = region['I']
            if 'bas_c' in region.keys():
                bas_c = region['bas_c']
            else:
                bas_c = np.zeros(5)
            if 1:   # Switch: turn this print on and off
                print(f'Fitting of region {k+1}/{Nr}. [{limits[0]:.3f}:{limits[1]:.3f}] ppm', c='tab:orange')
            # Make a copy of the region dictionary and remove what is not a peak
            peaks = deepcopy(region)
            peaks.pop('limits')
            peaks.pop('I')
            if 'bas_c' in region.keys():
                peaks.pop('bas_c')
            yield limits, Int, peaks, bas_c

    # -----------------------------------------------------------------------------
    # Make the acqus dictionary to be fed into the fit.Peak objects
    acqus = {'t1': t_AQ, 'SFO1': SFO1, 'o1p': o1p, }

    N = S.shape[-1]     # Number of points of the spectrum
    Nr = len(regions)   # Number of regions to be fitted

    # Write info on the fit in the output file
    filename = Path(filename)
    with filename.with_suffix('.fvf').open('a', buffering=1) as f:
        now = datetime.now()
        date_and_time = now.strftime("%d/%m/%Y at %H:%M:%S")
        f.write('! Fit performed by {} on {}\n\n'.format(getpass.getuser(), date_and_time))

    # Generate the values from the regions dictionary with the gen_reg generator
    Q = gen_reg(regions)
    with open('cnvg', 'w') as f:
        pass

    # Start fitting loop
    prev = 0
    lmfit_results = []      # placeholder
    for q in Q:
        limits, I, peaks, bas_c = q    # Unpack
        Np = len(peaks.keys())  # Number of Peaks

        # Create a dictionary which contains Peak objects
        fit_peaks = {}
        for key, peakval in peaks.items():
            # Same keys of the input dictionary
            fit_peaks[key] = fit.Peak(acqus, N=N, **peakval)

        # Add the peaks' parameters to a lmfit Parameters object
        peak_keys = ['u', 'fwhm', 'k', 'b', 'phi']
        param = lmfit.Parameters()
        # Add baseline coefficients to the Parameters object
        for n in range(5):
            if basl_fit == 'no':
                v = 0
                vary = False
            elif basl_fit == 'fixed' or basl_fit == 'calc':
                v = bas_c[n]
                vary = False
            elif basl_fit == 'fit':
                v = bas_c[n]
                vary = True
            param.add(f'c{n}', value=v, vary=vary)

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
                elif 'fwhm' in key:  # fwhm: [max(0, fwhm-f_tol), fwhm+f_tol] (avoid negative fwhm)
                    param[par_key].set(min=max(0, val-f_lim), max=val+f_lim)
                elif 'k' in key:     # k: [0, 3]
                    if isinstance(k_lim, float):
                        param[par_key].set(min=param[par_key].value-k_lim, max=param[par_key].value+k_lim)
                    else:
                        param[par_key].set(min=min(k_lim), max=max(k_lim))
                elif 'phi' in key:  # phi: ph_lim ([-180°, +180°])
                    param[par_key].set(min=min(ph_lim), max=max(ph_lim), vary=vary_phase)
                elif 'b' in key:  # b: [0, 1]
                    param[par_key].set(min=0, max=1, vary=vary_b)

        # Convert the limits from ppm to points and make the slice
        limits_pt = misc.ppmfind(ppm_scale, limits[0])[0], misc.ppmfind(ppm_scale, limits[1])[0]
        lims = slice(min(limits_pt), max(limits_pt))
        # Baseline x-scale
        x = np.linspace(0, 1, int(np.abs(limits_pt[1] - limits_pt[0])))

        # Wrap the fitting routine in a function in order to use @cron for measuring the runtime of the fit
        @cron
        def start_fit(method):
            # Initialize the nonused fit parameters
            param.add('count', value=0, vary=False)
            param.add('correction_factor', value=1, vary=False)
            # Compute the first residual as reference, suppress the output
            with open(os.devnull, 'w') as sys.stdout:
                first_residual = np.sum(f2min(param, S, fit_peaks, I, lims, x, 1, basl_fit)**2)
                # Reset the iteration counter
                param['count'].set(value=0)
            # Redirect output to stdout
            sys.stdout = sys.__stdout__

            # Initialize the fit
            minner = lmfit.Minimizer(f2min, param, fcn_args=(S, fit_peaks, I, lims, x, first_residual, basl_fit))
            if isinstance(method, str):
                method = [method]
            elif isinstance(method, (list, tuple)):
                pass
            else:
                raise ValueError('The "method" flag must be a string or a list of strings.')
            result = None
            for k, mthd in enumerate(method):
                if k == 0:
                    params = None
                else:
                    params = result.params

                print(f'Optimization {k+1:4.0f}/{len(method)}, method = {mthd}', c='tab:orange')
                if mthd == 'leastsq' or mthd == 'least_squares':
                    result = minner.minimize(method='leastsq', max_nfev=int(itermax), ftol=fit_tol, params=params)
                else:
                    result = minner.minimize(method=mthd, max_nfev=int(itermax), tol=fit_tol, params=params)
                print(f'{result.message} Number of function evaluations: {result.nfev}.\n')
            return result
        # Do the fit
        result = start_fit(method)
        # Unpack the fitted values
        popt = result.params.valuesdict()

        # Replace the initial values with the fitted ones
        fit_peaks = peaks_frompar(fit_peaks, popt)
        # The baseline coefficients must be normalized to the I at the end
        # Hence, first multiply them, and then divide them afterwards for the corrected I
        bas_c_opt = np.array([popt[f'c{k}'] for k in range(5)]) * I
        # Correct the intensities
        #   Get the correct ones
        r_i, I_corr = misc.molfrac([peak.k for _, peak in fit_peaks.items()])
        I *= I_corr * popt['correction_factor']
        # Here correct the baseline coefficients
        bas_c_opt /= I
        #   Replace them
        for k, idx in enumerate(fit_peaks.keys()):
            fit_peaks[idx].k = r_i[k]

        # Write a section of the output file
        fit.write_vf(filename.with_suffix('.fvf'), fit_peaks, limits, I, prev, bas_c=bas_c_opt)
        prev += Np
        lmfit_results.append(result)
    return lmfit_results


@cron
def voigt_fit_2D(x_scale, y_scale, data, parameters, lim_f1, lim_f2, acqus,
                 N=None, procs=None, utol=(1, 1), s1tol=(0, 500), s2tol=(0, 500), vary_b=False, logfile=None):
    """
    Function that performs the fit of a 2D peak using multiple components.
    The program reads a parameter matrix, that contains:
    .. code-block:: bash

        u1 /ppm, u2 /ppm, fwhm1 /Hz, fwhm2 /Hz, I /a.u., b

    in each row. The number of rows corresponds to the number of components used for the computation of the final signal.
    The function returns the analogue version of the parameters matrix, but with the optimized values.

    .. warning::

        Work in progress! Does not work right now.

    Parameters
    ----------
    x_scale : 1darray
        ppm_f2 of the spectrum, full
    y_scale : 1darray
        ppm_f1 of the spectrum, full
    data : 2darray
        spectrum, full
    parameters : 1darray or 2darray
        Matrix (# signals, 6). Read main caption.
    lim_f2 : tuple
        Trimming limits for x_scale
    lim_f1 : tuple
        Trimming limits for y_scale
    acqus : dict
        Dictionary of acquisition parameters.
    N : tuple of ints
        len(y_scale), len(x_scale). Used only if procs is None
    procs : dict
        Dictionary of processing parameters.
    utol : tuple of floats
        Tolerance for the chemical shifts (utol_f1, utol_f2). Values will be set to u1 +/- utol_f1, u2 +/- utol_f2
    s1tol : tuple of floats
        Range of variations for the fwhm in f1, in Hz
    s2tol : tuple of floats
        Range of variations for the fwhm in f2, in Hz
    vary_b : bool
        Choose if to fix the b value or not
    logfile : str or None
        Path to a file where to write the fit information. If it is None, they will be printed into standard output.

    Returns
    -------
    out_parameters : 2darray
        parameters, but with the optimized values.
    """

    def f2min(param, n_sgn, x_scale, y_scale, data_exp, lim_f2, lim_f1):
        """
        Cost function.

        Parameters
        ----------
        param : lmfit.Parameters object
            Fit parameters. See fit_2D caption.
        n_sgn : int
            Number of signals
        x_scale : 1darray
            ppm_f2 of the spectrum, full
        y_scale : 1darray
            ppm_f1 of the spectrum, full
        data_exp : 2darray
            spectrum trimmed around the peak of interest
        lim_f2 : tuple
            Trimming limits for x_scale
        lim_f1 : tuple
            Trimming limits for y_scale

        Returns
        -------
        res : 2darray
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

    # ---------------------------------------------------------------------------------------------

    # Redirect the output to logfile, if logfile is given
    if isinstance(logfile, str):    # Open the file in "append" mode
        sys.stdout = open(logfile, 'a', buffering=1)
    elif isinstance(logfile, io.TextIOWrapper):  # Just redirect
        sys.stdout = logfile

    # Trim the spectrum according to the given limits
    data_exp = misc.trim_data_2D(x_scale, y_scale, data, lim_f2, lim_f1)[-1]

    # Organize parameters
    parameters = np.array(parameters)
    if len(parameters.shape) == 1:  # it means it is only one signal
        parameters = parameters.reshape(1, -1)   # therefore transform in 1 x n matrix
    n_sgn = parameters.shape[0]     # Number of signals: number of rows of parameters

    # Express relative intensities in "molar fractions" and adjust the absolute intensity accordingly
    k_values, A = misc.molfrac(parameters[..., 4])

    # Initialize the Parameters object
    param = lmfit.Parameters()

    param.add('A', value=A, vary=False)     # Absolute intensity
    for i in range(n_sgn):
        param.add(f'u1_{i+1}', value=parameters[i, 0])   # chemical shift f1 /ppm
        param.add(f'u2_{i+1}', value=parameters[i, 1])   # chemical shift f2 /ppm
        param.add(f's1_{i+1}', value=parameters[i, 2])   # fwhm f1 /Hz
        param.add(f's2_{i+1}', value=parameters[i, 3])   # fwhm f2 /Hz
        param.add(f'k_{i+1}', value=k_values[i])        # relative intensity
        param.add(f'b_{i+1}', value=parameters[i, 5], min=0-1e-5, max=1+1e-5)   # Fraction of gaussianity

    # Set limits to u and s
    u1tol, u2tol = utol  # Unpack tolerances for chemical shifts
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
    minner = lmfit.Minimizer(f2min, param, fcn_args=(n_sgn, x_scale, y_scale, data_exp, lim_f2, lim_f1))
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
                except Exception as E:   # Therefore append None instead of NaN
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
    Uses the package ``csaps``.

    Parameters
    ----------
    x : 1darray
        Location of the experimental points
    y : 1darray
        Input data to be fitted
    s_f : float
        Smoothing factor of the spline. 0=best straight line, 1=native spline.
    size : int
        Size of the spline. If ``size=0``, the same dimension as ``y`` is chosen.

    Returns
    -------
    x_s : 1darray
        Location of the spline data points.
    y_s : 1darray
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


class Peak:
    """
    Class to represent the characteristic parameters of an NMR peak, and to compute it.

    Attributes
    ----------
    t: 1darray
        Timescale for the FID
    SFO1: float
        Nucleus Larmor frequency
    o1p: float
        Carrier position
    N: int
        Number of points of the spectrum, i.e. after eventual zero-filling
    u: float
        Chemical shift /ppm
    fwhm: float
        Linewidth /Hz
    k: float
        Intensity, relative
    b: float
        Fraction of gaussianity (``b=0`` equals pure lorentzian)
    phi: float
        Phase /degrees
    group: int
        Identifier for the component of a multiplet
    """
    def __init__(self, acqus, u=None, fwhm=5, k=1, b=0, phi=0, N=None, group=0):
        """
        Initialize the class with the configuration parameters, and with defauls values, if not given.

        Parameters
        ----------
        acqus: dict
            It should contain "t", "SFO1", "o1p", and "N"
        u: float
            Chemical shift /ppm
        fwhm: float
            Linewidth /Hz
        k: float
            Intensity, relative
        b: float
            Fraction of gaussianity (``b=0`` equals pure lorentzian)
        phi: float
            Phase /degrees
        N: int
            Number of points of the spectrum, i.e. after eventual zero-filling. None means to not zero-fill
        group: int
            Identifier for the component of a multiplet
        """
        # Unpack the acqus dictionary
        self.t = processing.extend_taq(acqus['t1'], N)
        self.SFO1 = acqus['SFO1']
        self.o1p = acqus['o1p']
        self.N = N

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

        Parameters
        ----------
        A : float
            Absolute intensity value
        cplx : bool
            Returns the complex (True) or only the real part (False) of the signal
        get_fid : bool
            If True, returns the FID instead of the transformed signal. Always complex!

        Returns
        -------
        sgn : 1darray
            generated signal according to ``get_fid`` and ``cplx``
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

        Parameters
        ----------
        A : float
            Absolute intensity value

        Returns
        -------
        sgn : 1darray
            generated signal in the time domain
        """
        v = misc.ppm2freq(self.u, self.SFO1, self.o1p)         # conversion to frequency units
        fwhm = self.fwhm * 2 * np.pi                    # conversion to radians
        phi = self.phi * np.pi / 180                    # conversion to radians
        sgn = sim.t_voigt(self.t, v, fwhm, A=A*self.k, phi=phi, b=self.b)    # make the signal
        return sgn

    def par(self):
        """
        Creates a dictionary with the currently stored attributes and returns it.

        Returns
        -------
        dic: dict
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


# --------------------------------------------------------------------
def write_vf(filename, peaks, lims, Int, prev=0, header=False, bas_c=None):
    """
    Write a section in a fit report file, which shows the fitting region and the parameters of the peaks to feed into a Voigt lineshape model.

    Parameters
    ----------
    filename : str or Path
        Path to the file to be written
    peaks : dict
        Dictionary of ``fit.Peak`` objects
    lims : tuple
        (left limit /ppm, right limit /ppm)
    I : float
        Absolute intensity value
    prev : int
        Number of previous peaks already saved. Increases the peak index
    header : bool
        If True, adds a "!" starting line to separate fit trials
    bas_c : None or 1darray
        Baseline coefficients

    Returns
    -------
    None

    .. seealso::

        :func:`klassez.fit.read_vf`

        :func:`klassez.gui.make_iguess`
    """
    # Adjust the intensities
    r_i, I_corr = misc.molfrac([peak.k for _, peak in peaks.items()])

    # Open the file in append mode
    f = Path(filename).open('a', buffering=1)
    # Info on the region to be fitted
    if header:
        now = datetime.now()
        date_and_time = now.strftime("%d/%m/%Y at %H:%M:%S")
        f.write('! Fit performed by {} on {}\n\n'.format(getpass.getuser(), date_and_time))
    #   Header
    f.write('{:>16};\t{:>12}\n'.format('Region', 'Intensity'))
    f.write('-'*96+'\n')
    #   Values
    region = '{:-.3f}:{:-.3f}'.format(*lims)   # From the zoom of the figure
    f.write(f'{region:>16};\t{Int*I_corr:14.8e}\n\n')

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

    if bas_c is not None:
        f.write('BASLC:\t'+'; '.join([f'{w/(I_corr):+12.8f}' for w in bas_c])+'\n\n')

    # Add region separator and close the file
    f.write('='*96+'\n\n')
    f.close()


def read_vf(filename, n=-1):
    """
    Reads a `.ivf` (initial guess) or `.fvf` (final fit) file, containing the parameters for a lineshape deconvolution fitting procedure.
    The file is separated and unpacked into a list of dictionaries, each of which contains the limits of the fitting window,
    the total intensity value, and a dictionary for each peak with the characteristic values to compute it with a Voigt line.

    Parameters
    ----------
    filename : str or Path
        Path to the filename to be read
    n : int
        Number of performed fit to be read. Default: last one. The breakpoints are lines that start with "!". For this reason, ``n=0`` returns an empty dictionary, hence the first fit is ``n=1``.

    Returns
    -------
    regions: list
        List of dictionaries for running the fit.

    .. seealso::

        :func:`klassez.fit.write_vf`
    """
    def read_region(R):
        """ Creates a dictionary of parameters from a section of the input file.  """
        # Placeholder
        dic_r = {}
        # Separate the lines and remove the empty ones
        R = R.split('\n')
        for k, r in enumerate(R):
            if len(r) == 0 or r.isspace():
                _ = R.pop(k)

        n_bp = 0        # Number of breaking points (----)
        k_bp = 0        # Line of the last breaking point detected
        for k, r in enumerate(R):
            if '------' in r:   # Increase breakpoint and store the line number
                n_bp += 1
                k_bp = k
                continue

            if n_bp == 1 and k_bp == k-1:   # First section: region limits and total intensity
                line = r.split(';')  # Separate the values
                dic_r['limits'] = eval(line[0].replace(':', ', '))   # Get the limits
                dic_r['I'] = eval(line[-1])     # Get the intensity

            if n_bp == 2:       # Second section: peak parameters
                line = r.split(';')  # Separate the values
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
                if 'BASLC' in r:
                    line = r.split(':', 1)[-1]
                    bas_c = np.array(eval(line.replace(';', ',')))
                    dic_r['bas_c'] = bas_c
                break

        return dic_r

    # Read the file
    ff = Path(filename).read_text()
    # Get the actual section from an output file
    f = ff.split('!')[n]
    # Separate the bigger sections
    R = f.split('='*96)
    # Remove the empty lines
    for k, r in enumerate(R):
        if r.isspace():
            _ = R.pop(k)

    regions = []    # Placeholder for return values
    for r in R:  # Loop on the big sections to read them
        regions.append(read_region(r))
    return regions


def write_dy(filename, diff_c, diff_f, diff_e, label, intensity, offset, header=False):
    """
    Write a section in a fit report file, which shows the fitting region identifier and the parameters to feed into the DOSY model fitting.

    Parameters
    ----------
    filename : str or Path
        Path to the file to be written
    diff_c : list of float
        Diffusion coefficients in m^2/s
    diff_f : list of float
        Fractions of the various components
    diff_e : list of float or list of None
        Fit errors. Initial guess will have ``None`` for each entry
    label : str
        Region identifier for the fit
    intensity : float
        Intensity factor to match the model to the experimental data
    offset : float
        Offset factor to match the model to the experimental data

    Returns
    -------
    None

    .. seealso::

        :func:`klassez.fit.read_dy`

        :func:`klassez.fit.make_iguess_dosy`
    """
    # Adjust the intensities
    diff_f, I_corr = misc.molfrac(diff_f)

    # Open the file in append mode
    f = Path(filename).open('a', buffering=1)
    # Info on the region to be fitted
    if header:
        now = datetime.now()
        date_and_time = now.strftime("%d/%m/%Y at %H:%M:%S")
        f.write('! DOSY fit performed by {} on {}\n\n'.format(getpass.getuser(), date_and_time))
    #   Header
    f.write('{:>24}; {:>14}; {:>14}\n'.format('Region', 'Intensity', 'Offset'))
    f.write('-'*96+'\n')
    #   Values
    f.write(f'{label:>24}; {intensity*I_corr:14.8e}; {offset*I_corr:14.8e}\n\n')

    # Info on the components
    #   Header
    f.write('{:>16}; {:>16}; {:>16}\n'.format(
        'Dosy coeff.', 'Fraction', 'Error'))
    f.write('-'*96+'\n')
    #   Values
    for diffc, difff, diffe in zip(diff_c, diff_f, diff_e):
        f.write('{:=16.8e}; {:16.8f}; '.format(diffc, difff))
        if diffe is None:
            f.write(f'{"None":>16s}\n')
        else:
            f.write(f'{diffe:16.8e}\n')
    f.write('-'*96+'\n\n')

    # Add region separator and close the file
    f.write('='*96+'\n\n')
    f.close()


def read_dy(filename, n=-1):
    """
    Reads a `.idy` (initial guess) or `.fdy` (final fit) file, containing the parameters for a DOSY fitting procedure.
    The file is separated and unpacked into a list of dictionaries, each of which contains the region identifier,
    the total intensity value and the offset, the diffusion coefficients, relative weight of the components, and the fit errors.

    Parameters
    ----------
    filename : str or Path
        Path to the filename to be read
    n : int
        Number of performed fit to be read. Default: last one. The breakpoints are lines that start with "!". For this reason, ``n=0`` returns an empty dictionary, hence the first fit is ``n=1``.

    Returns
    -------
    regions: list
        List of dictionaries for running the fit.

    .. seealso::

        :func:`klassez.fit.write_dy`
    """
    def read_region(R):
        """ Creates a dictionary of parameters from a section of the input file.  """
        # Placeholder
        dic_r = {}
        # Separate the lines and remove the empty ones
        R = R.split('\n')
        for k, r in enumerate(R):
            if len(r) == 0 or r.isspace():
                _ = R.pop(k)

        n_bp = 0        # Number of breaking points (----)
        k_bp = 0        # Line of the last breaking point detected
        for k, r in enumerate(R):
            if '------' in r:   # Increase breakpoint and store the line number
                n_bp += 1
                k_bp = k
                continue

            if n_bp == 1 and k_bp == k-1:   # First section: region identifiere, intensity and offset
                line = r.split(';')  # Separate the values
                dic_r['label'] = line[0].replace(' ', '')   # Get the label
                dic_r['I'] = eval(line[1])     # Get the intensity
                dic_r['q'] = eval(line[2])     # Get the offset
                # Create placeholders for the values
                dic_r['diff_c'] = []
                dic_r['diff_f'] = []
                dic_r['diff_e'] = []

            if n_bp == 2:       # Second section: peak parameters
                line = r.split(';')  # Separate the values
                # Unpack the line
                diffc, difff, diffe = [eval(w) for w in line]
                # Put the values in the dictionary
                dic_r['diff_c'].append(diffc)
                dic_r['diff_f'].append(difff)
                dic_r['diff_e'].append(diffe)

            if n_bp == 3:   # End of file: stop reading
                break

        return dic_r

    # Read the file
    ff = Path(filename).read_text()
    # Get the actual section from an output file
    f = ff.split('!')[n]
    # Separate the bigger sections
    R = f.split('='*96)
    # Remove the empty lines
    for k, r in enumerate(R):
        if r.isspace():
            _ = R.pop(k)

    regions = []    # Placeholder for return values
    for r in R:  # Loop on the big sections to read them
        regions.append(read_region(r))
    return regions


# --------------------------------------------------------------------


def test_randomsign(data, thresh=1.96):
    """
    Test an array of residuals for the randomness of the sign changes.
    The result it True if the sequence is recognized as random.

    Parameters
    ----------
    data : 1darray
        Residuals to test
    thresh : float
        Significance level. The default is 1.96, which corresponds to 5% significance level.

    Returns
    -------
    test : bool
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
    u = (2 * n_p * n_n) / N + 1
    sigma = ((u - 1) * (u - 2) / (N - 1))**0.5

    # If z < thresh, the sign distribution is random
    z = np.abs(n_runs - u) / sigma
    return z < thresh


def test_correl(data, subtract_mean=True):
    r"""
    Tests an array of residuals for their correlation.
    It compares the unit-lag autocorrelation `P` of the ``data`` (see below) with the theoretical value for non-correlated data `T_P`:

    .. math::

        P = \sum_k^{N-1} r[k] \, r[k+1] ;\quad T_P = \sqrt{N-1} \sum_k r[k]^2

    If :math:`P < T_P`, the residuals are not correlated, and the result is True.

    Parameters
    ----------
    data: 1darray
        Residuals to be test
    subtract_mean: bool
        If True, subtracts from the residuals their mean.

    Returns
    -------
    test: bool
        True if the residuals are non correlated, False otherwise
    """
    # Shallow copy of the residuals
    r = np.copy(data)
    # Size of the data
    N = len(r)
    if subtract_mean:    # Subtract from the residuals their mean
        r -= np.mean(r)

    # Compute the discrete correlation function of the residuals P
    r_roll = np.roll(r, 1)
    P = np.sum((r * r_roll)[:-1])
    # Compute threshold for correlation
    T_P = 1 / (N - 1)**0.5 * np.sum(r**2)

    # Residuals are not correlated if P < T_P
    return np.abs(P) < T_P


def test_ks(data, thresh=0.05):
    """
    Performs the Kolmogorov-Smirnov test on the residuals to check if they are drawn from a normal distribution.
    The implementation is :func:`scipy.stats.kstest`.
    The result is True if the residuals are Gaussian.

    Parameters
    ----------
    data : 1darray
        Residuals to test
    thresh : float
        Significance level for the test. Default is 5%

    Returns
    -------
    test : bool
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
    To do this, it uses the functions :func:`klassez.fit.test_randomsign`, :func:`klassez.fit.test_correl`, :func:`klassez.fit.test_ks`.
    The results of the tests will be print in standard output and returned.

    Parameters
    ----------
    res : ndarray
        Residuals to be tested
    alpha : float
        Significance level

    Returns
    -------
    test_random : bool
        Randomness of the residuals (True = random)
    test_correlation : bool
        Correlation of the residuals (True = non-correlated)
    test_gaussian : bool
        Normal-distribution of the residuals (True = normally-distributed)

    .. seealso::

        :func:`klassez.fit.test_randomsign`

        :func:`klassez.fit.test_correl`

        :func:`klassez.fit.test_ks`
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
    print('Residual test', c='tab:orange', s='underline')
    print('\n'.join([
        f'{"Random":22s}: {test_randomness}',
        f'{"Non-correlated":22s}: {test_correlation}',
        f'{"Normally distributed":22s}: {test_gaussian}',
        ]))

    return test_randomness, test_correlation, test_gaussian


def gaussian_fit(x, y, s_in=None):
    """
    Fit ``y`` with a gaussian function, built using ``x`` as independent variable

    Parameters
    ----------
    x : 1darray
        x-scale
    y : 1darray
        data to be fitted
    s_in : float or None
        initial guess for the standard deviation of the gaussian. If None, ``np.std(y)`` is used

    Returns
    -------
    u : float
        mean
    s : float
        standard deviation
    A : float
        Integral

    .. seealso::

        :func:`klassez.sim.f_gaussian`
    """

    # Make parameter dictionary
    param = lmfit.Parameters()
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

    minner = lmfit.Minimizer(f2min, param, fcn_args=(x, y))
    result = minner.minimize(method='leastsq', max_nfev=10000, xtol=1e-15, ftol=1e-15)

    # Return the result
    popt = result.params.valuesdict()
    return popt['u'], popt['s'], popt['A']

# -------------------------------------------------------------------------------------------


class Voigt_Fit:
    """
    This class offers an "interface" to fit a 1D NMR spectrum.

    Attributes
    ----------
    ppm_scale : 1darray
        Self-explanatory
    S : 1darray
        Spectrum to fit. Only real part
    t_AQ : 1darray
        acquisition timescale of the spectrum
    SW : float
        Spectral width /Hz
    SFO1 : float
        Larmor frequency of the nucleus
    o1p : float
        Pulse carrier frequency
    filename : str or Path
        Root of the names of the files that will be saved
    X_label : str
        Label for the chemical shift axis in the figures
    i_guess : list of dict
        Initial guess for the fit, read by a `.ivf` file with :func:`klassez.fit.read_vf`
    result : list of dict
        Result the fit, read by a `.fvf` file with :func:`klassez.fit.read_vf`
    """

    def __init__(self, ppm_scale, S, t_AQ, SFO1, o1p, nuc=None, filename='fit'):
        """
        Initialize the class with common values.

        Parameters
        ----------
        ppm_scale : 1darray
            ppm scale of the spectrum
        S : 1darray
            Spectrum to be fitted
        t_AQ : 1darray
            Acquisition timescale
        SFO1 : float
            Larmor frequency of the observed nucleus, in MHz
        o1p : float
            Carrier position, in ppm
        nuc : str
            Observed nucleus. Used to customize the x-scale of the figures.
        filename : str
            Root of the name of the files that will be saved
        """
        self.ppm_scale = ppm_scale
        self.S = S
        self.t_AQ = processing.extend_taq(t_AQ, self.S.shape[-1])
        self.SFO1 = SFO1
        self.SW = np.abs((max(ppm_scale) - min(ppm_scale)) * SFO1)
        self.o1p = o1p
        self.filename = filename
        if nuc is None:
            self.X_label = r'$\delta\,$ /ppm'
        elif isinstance(nuc, str):
            fnuc = misc.nuc_format(nuc)
            self.X_label = r'$\delta$ ' + fnuc + ' /ppm'

    def iguess(self, filename=None, n=-1, ext='ivf', auto=False):
        """
        Reads, or computes, the initial guess for the fit.
        If the file is there already, it just reads it with ``fit.read_vf``. Otherwise, it calls ``gui.make_iguess`` to make it.

        Parameters
        ----------
        filename: str or None
            Path to the input file. If None, `"<self.filename>.ivf"` is used
        n: int
            Index of the initial guess to be read (default: last one)
        ext: str
            Extension of the file to be used
        auto: bool
            If True, uses the GUI for automatic peak picking, if False, the manual one

        Returns
        -------
        None

        .. seealso::

            :func:`klassez.gui.make_iguess`

            :func:`klassez.gui.make_iguess_auto`

            :func:`klassez.fit.read_vf`
        """
        # Set the default filename, if not given
        if filename is None:
            filename = f'{self.filename}'
        filename = Path(filename)
        filename_x = filename.with_suffix(f'.{ext}')
        # Check if the file exists
        in_file_exist = filename_x.exists()

        if in_file_exist is True:       # Read everything you need from the file
            regions = fit.read_vf(filename_x)
        else:                           # Make the initial guess interactively and save the file.
            if auto:
                gui.make_iguess_auto(self.ppm_scale, self.S, self.SW, self.SFO1, self.o1p, filename=filename)
            else:
                gui.make_iguess(self.S, self.ppm_scale, self.t_AQ, self.SFO1, self.o1p, filename=filename)
            regions = fit.read_vf(filename_x)
        # Store it
        self.i_guess = regions
        print(f'{filename_x} loaded as input file.', c='tab:blue')

    def load_fit(self, filename=None, n=-1, ext='fvf'):
        """
        Reads a file with ``fit.read_vf`` and stores the result in ``self.result``.

        Parameters
        ----------
        filename: str or Path
            Path to the .fvf file to be read. If None, "<self.filename>.fvf" is used.
        n: int
            Index of the fit to be read (default: last one)
        ext: str
            Extension of the file to be used

        Returns
        -------
        None

        .. seealso::

            :func:`klassez.gui.make_iguess`

            :func:`klassez.gui.make_iguess_auto`

            :func:`klassez.fit.read_vf`
        """
        # Set the default filename, if not given
        if filename is None:
            filename = f'{self.filename}'
        filename = Path(filename)
        filename_x = filename.with_suffix(f'.{ext}')
        # Check if the file exists
        if filename_x.exists():
            regions = fit.read_vf(filename_x, n=n)
        else:
            raise FileNotFoundError(f'{filename_x} does not exist.')
        # Store
        self.result = regions
        print(f'{filename_x} loaded as fit result file.', c='tab:blue')

    def dofit(self, indep=True, u_lim=1, f_lim=10, k_lim=(0, 3), ph_lim=(-180, 180),
              vary_phase=False, vary_b=True, itermax=10000, fit_tol=1e-8, filename=None,
              method='leastsq', basl_fit='no'):
        """
        Perform a lineshape deconvolution fitting.
        The initial guess is read from the attribute ``self.i_guess``.
        The components can be considered to be all independent from one to another by setting ``indep=True``: this means that the fit will be done using ``fit.voigt_fit_indep``.
        The ``indep=False`` option has not been implemented yet.

        Parameters
        ----------
        indep : bool
            True to consider all the components to be independent
        u_lim : float
            Determines the displacement of the chemical shift (in ppm) from the starting value.
        f_lim : float
            Determines the displacement of the linewidth (in Hz) from the starting value.
        k_lim : float or tuple
            If tuple, minimum and maximum allowed values for k during the fit. If float, maximum displacement from the initial guess
        ph_lim : tuple
            Minimum and maximum allowed values for the phases of the peaks, in degrees
        vary_phase : bool
            Allow the peaks to change phase (True) or not (False)
        vary_b : bool
            Allow the peaks to change Lorentzian/Gaussian ratio
        itermax : int
            Maximum number of allowed iterations
        fit_tol : float
            Value of the target function to be set as x_tol and f_tol
        filename : str or Path
            Path to the output file. If None, "<self.filename>.fvf" is used
        method : str or list of str
            Method to be used for the optimization. See lmfit for details. There is the option to run multiple optimizations in series.
        basl_fit : str
            How to address the baseline fit. The options are:
            * "no" : Do not use baseline (default)
            * "fixed" : The baseline is computed once and kept fixed during the optimization
            * "fit"  : The baseline coefficients enter as fit parameters during the nonlinear optimization
            * "calc" : The baseline coefficients are calculated during the optimization via linear least-squares optimization

        Returns
        -------
        lmfit_results : list of lmfit.minimizer.MinimizerResult
            Sequence of the fit results, ordered as the regions dictionary

        .. seealso::

            :func:`klassez.fit.voigt_fit_indep`
        """

        # Make a shallow copy of the real part of the experimental spectrum
        S = np.copy(self.S.real)
        # Check if the initial guess was loaded correctly
        if not isinstance(self.i_guess, list):
            raise ValueError('Initial guess not correctly loaded')
        # Set the output filename, if not given
        if filename is None:
            filename = f'{self.filename}'
        filename = Path(filename)

        # Do the fit
        if indep is True:
            lmfit_results = fit.voigt_fit_indep(S, self.ppm_scale, self.i_guess, self.t_AQ, self.SFO1, self.o1p,
                                                u_lim=u_lim, f_lim=f_lim, k_lim=k_lim, vary_phase=vary_phase, vary_b=vary_b,
                                                itermax=itermax, fit_tol=fit_tol, filename=filename, method=method, basl_fit=basl_fit)
        else:
            raise NotImplementedError('More and more exciting adventures in the next release!')
        # Store
        self.result = fit.read_vf(filename.with_suffix('.fvf'))
        return lmfit_results

    def edit_iguess(self, filename=None, ext='ivf'):
        """
        Edit the initial guess of the fit, stored in the ``self.i_guess`` attribute, through the use of a dedicated GUI.
        First, compute or load the initial guess by calling ``self.iguess`` with the appropriate filename.
        Then you can call this method to refine.

        Remember to read the documentation of the GUI to understand how to use it!

        Parameters
        ----------
        filename : str or Path
            Filename to be saved and read again after the edit. If ``None``, ``self.filename`` is used.
        ext : str
            ``'ivf'`` or ``'fvf'``

        Returns
        -------
        None

        .. seealso::

            :func:`klassez.gui.edit_vf`
            :func:`klassez.fit.Voigt_Fit.iguess`
        """
        # Default filename if not given
        if filename is None:
            filename = self.filename
        # Backup
        old_regions = deepcopy(self.i_guess)
        # Call the editing GUI
        self._edit_wgui(old_regions, filename=filename, ext=ext)

        # Check if the file exists
        filename_x = Path(filename).with_suffix(f'.{ext}')
        if not filename_x.exists():
            raise NameError(f'{filename_x} does not exist.')
        else:
            self.iguess(filename=filename, ext=ext)

    def edit_result(self, filename=None, ext='fvf'):
        """
        Edit the result of the fit, stored in the ``self.result`` attribute, through the use of a dedicated GUI.
        First, compute or load the initial guess by calling ``self.dofit`` or ``self.load_fit`` with the appropriate filename.
        Then you can call this method to refine.

        Remember to read the documentation of the GUI to understand how to use it!

        Parameters
        ----------
        filename : str or Path
            Filename to be saved and read again after the edit. If ``None``, ``self.filename`` is used.
        ext : str
            ``'ivf'`` or ``'fvf'``

        Returns
        -------
        None

        .. seealso::

            :func:`klassez.gui.edit_vf`
            :func:`klassez.fit.Voigt_Fit.dofit`
            :func:`klassez.fit.Voigt_Fit.load_fit`
        """
        # Default filename if not given
        if filename is None:
            filename = self.filename
        # Backup
        old_regions = deepcopy(self.result)
        # Call the editing GUI
        self._edit_wgui(old_regions, filename=filename, ext=ext)

        # Check if the file exists
        filename_x = Path(filename).with_suffix(f'.{ext}')
        if not filename_x.exists():
            raise NameError(f'{filename_x} does not exist.')
        else:
            self.load_fit(filename=filename, ext=ext)

    def _edit_wgui(self, regions, filename=None, ext='ivf'):
        """
        Wrapper function to call for the editing via GUI of either the initial guess or the fit result.

        Parameters
        ----------
        regions : list of dict
            ``self.i_guess`` or ``self.result``
        filename : str or Path
            Filename to be saved and read again after the edit. If ``None``, ``self.filename`` is used.
        ext : str
            ``'ivf'`` or ``'fvf'``
        """
        gui.edit_vf(self.S, self.ppm_scale, regions, self.t_AQ, self.SFO1, self.o1p, filename=filename, ext=ext)

    def plot(self, what='result', show_total=True, show_res=False, res_offset=0, show_basl=False, labels=None, filename=None, ext='svg', dpi=600, dim=None):
        """
        Plots either the initial guess or the result of the fit, and saves all the figures. Calls :func:`fit.plot_fit`.
        The figure `<filename>_full` will show the whole model and the whole spectrum.
        The figures labelled with `_R<k>` will depict a detail of the fit in the k-th fitting region.
        Optional labels for the components can be given: in this case, the structure of `labels` should match the structure of ``self.result`` (or ``self.i_guess``).
        This means that the length of the outer list must be equal to the number of fitting region, and the length of the inner lists must be equal to the number of peaks in that region.

        Parameters
        ----------
        what : str
            'iguess' to plot the initial guess, 'result' to plot the fitted data
        show_total : bool
            Show the total trace (i.e. sum of all the components) or not
        show_res : bool
            Show the plot of the residuals
        res_offset : float
            Displacement of the residuals plot from 0, to be given as a fraction of the height of the experimental spectrum. ``res_offset`` > 0 will move the residuals BELOW the zero-line!
        show_basl : bool
            If True, displays the baseline on the spectrum and uses it to compute the total trace.
        labels : list of list
            Optional labels for the components. The structure of this parameter must match the structure of ``self.result``
        filename : str or Path
            Root of the name of the figures that will be saved. If None, `<self.filename>` is used
        ext : str
            Format of the saved figures
        dpi : int
            Resolution of the figures, in dots per inches
        dim : tuple
            Dimension of the figure in inches

        Returns
        -------
        None

        .. seealso::

            :func:`klassez.fit.plot_fit`
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
        filename = Path(filename)

        # Make the figures
        S = np.copy(self.S.real)
        fit.plot_fit(S, self.ppm_scale, regions, self.t_AQ, self.SFO1, self.o1p, show_total=show_total,
                     show_res=show_res, res_offset=res_offset, show_basl=show_basl, X_label=self.X_label,
                     labels=labels, filename=filename, ext=ext, dpi=dpi, dim=dim)

    def get_fit_lines(self, what='result', cplx=False, fid=False):
        """
        Calculates the components, and the total fit curve used as initial guess, or as fit results.
        The components will be returned as a list, not split by region.

        Parameters
        ----------
        what : str
            'iguess' or 'result'

        Returns
        -------
        signals : list of 1darray
            Components used for the fit
        total : 1darray
            Sum of all the signals
        limits_list : list
            List of region delimiters, in ppm
        whole_basl : 1darray
            Computed baseline
        """
        # Select the correct object
        if what == 'iguess':
            regions = deepcopy(self.i_guess)
        elif what == 'result':
            regions = deepcopy(self.result)
        else:
            raise ValueError('Specify what you want to plot: "iguess" or "result"')

        # Make the acqus dictionary for the fit.Peak objects
        acqus = {'t1': self.t_AQ, 'SFO1': self.SFO1, 'o1p': self.o1p, }
        # Placeholders
        signals = []
        limits_list = []
        whole_basl = np.zeros_like(self.ppm_scale)
        # Loop on the regions
        for region in regions:
            # Remove the limits and the intensity from the region dictionary
            param = deepcopy(region)
            limits = param.pop('limits')
            Int = param.pop('I')
            if 'bas_c' in region.keys():
                bas_c = Int * region['bas_c']
                param.pop('bas_c')
            else:
                bas_c = np.zeros(5)
            # Convert the limits from ppm to points and make the slice
            limits_pt = misc.ppmfind(self.ppm_scale, limits[0])[0], misc.ppmfind(self.ppm_scale, limits[1])[0]
            # Baseline x-scale
            x = np.linspace(0, 1, int(np.abs(limits_pt[1]-limits_pt[0])))
            # Compute baseline
            basl = misc.polyn(x, bas_c)
            whole_basl = misc.sum_overlay(whole_basl, basl, max(limits), self.ppm_scale)
            # Make the fit.Peak objects
            peaks = {key: fit.Peak(acqus, N=self.S.shape[-1], **value) for key, value in param.items()}
            # Get the arrays from the dictionary and put them in the list
            signals.extend([p(Int, cplx=cplx, get_fid=fid) for _, p in peaks.items()])
            limits_list.append(limits)
        # Compute the total trace
        total = np.sum(signals, axis=0)
        return signals, total, limits_list, whole_basl

    def res_histogram(self, what='result', nbins=500, density=True, f_lims=None, xlabel='Residuals', x_symm=True, barcolor='tab:green', fontsize=20, filename=None, ext='svg', dpi=300):
        """
        Computes the histogram of the residuals and saves it.
        Employs :func:`klassez.fit.histogram` to make the figure.

        Parameters
        ----------
        what : str
            'iguess' or 'result'
        nbins  : int
            number of bins to be calculated
        density  : bool
            True for normalize data
        f_lims  : tuple or None
            limits for the x axis of the figure
        xlabel  : str or None
            Text to be displayed under the x axis
        x_symm  : bool
            set it to True to make symmetric x-axis with respect to 0
        barcolor : str
            Color of the bins
        fontsize : float
            Biggest fontsize in the figure
        filename  : str or Path
            name for the figure to be saved
        ext : str
            Format of the image
        dpi : int
            Resolution of the image in dots per inches

        Returns
        -------
        None

        .. seealso::

            :func:`klassez.fit.histogram`
        """
        # Filename check
        if filename is None:
            filename = f'{self.filename}'
        filename = Path(filename)
        filename = Path(filename).with_name(filename.stem + '_rhist')

        # Select the correct object
        if what == 'iguess':
            regions = deepcopy(self.i_guess)
        elif what == 'result':
            regions = deepcopy(self.result)
        else:
            raise ValueError('Specify what you want to plot: "iguess" or "result"')

        # Get the total function and the limits
        _, total, limits_list, whole_basl = self.get_fit_lines(what)
        # Convert the limits in points according to the ppm scale
        limits_pt_list = [[misc.ppmfind(self.ppm_scale, w)[0] for w in lims]
                          for lims in limits_list]

        # Placeholders
        exp_trim, total_trim = [], []
        for k, region in enumerate(regions):        # loop on the regions
            # Compute the slice
            lims = slice(min(limits_pt_list[k]), max(limits_pt_list[k]))
            # Trim the experimental data and the total
            exp_trim.append(self.S[..., lims].real)
            total_trim.append((total + whole_basl)[..., lims])
        # Sum on different regions
        exp_trim = np.concatenate(exp_trim, axis=-1)
        total_trim = np.concatenate(total_trim, axis=-1)

        # Compute the residuals
        residual_arr = exp_trim - total_trim

        fit.histogram(residual_arr, nbins=nbins, density=density, f_lims=f_lims,
                      xlabel=xlabel, x_symm=x_symm, barcolor=barcolor,
                      fontsize=fontsize, filename=filename, ext=ext, dpi=dpi)

    def to_tragico(self, which='iguess', filename=None):
        """
        Writes input 1 and input 2 for a TrAGICo run, on the basis of either the initial guess or the results of a fit.
        The files will be named `'<filename>_inp1'` and `'<filename>_inp2'`, respectively.

        Parameters
        ----------
        which : str
            'iguess' or 'result'
        filename : str or Path
            Name of the file that will be saved. If None, the file will be saved in the spectrum directory
        """
        def write_inp1(reg, filename):
            """
            Write the input 1 file for a tragico run, on the basis of ``regions``.
            The file name will be `<filename>_inp1`.

            Parameters
            ----------
            reg : list of dict
                self.i_guess or self.result
            filename : Path
                Name of the file that will be saved
            """

            # Shallow copy to prevent overwriting of the original "regions"
            regions = deepcopy(reg)

            # Add "inp1" to the filename
            fname = filename.with_stem(f'{filename.stem}_inp1')
            # Open the file and write the header
            f = fname.open('w')
            header = 'name\tppm1\tppm2\tv\tmult\n'
            f.write(header)

            # Loop on the regions
            for region in regions:
                # Get the region limits
                ppm1, ppm2 = region.pop('limits')
                # Remove intensity and baseline coefficients, in inp1 they are not needed
                region.pop('I')
                region.pop('bas_c')

                # Sort the indices of the peaks
                idxs = sorted(list(region.keys()))

                # Loop on the peaks
                for idx in idxs:
                    # Get the chemical shift and the group identifier
                    v = region[idx]['u']
                    mult = region[idx]['group']
                    # Write the line in the input file
                    line = f'{"true"}\t{ppm1:.5f}\t{ppm2:.5f}\t{v:.5f}\t{mult}\n'
                    f.write(line)
            f.close()

            print(f'Input 1 for TrAGICo written in {fname}.', c='tab:blue')

        def write_inp2(reg, ppm, spectrum, filename):
            """
            Write the input 2 file for a tragico run, on the basis of ``regions``.
            The file name will be `<filename>_inp1`.

            Parameters
            ----------
            reg : list of dict
                ``self.i_guess`` or ``self.result``
            filename : Path
                Name of the file that will be saved
            """
            def xbaslfact(ppm, lims, bas_c):
                """
                Conversion of the klassez baseline to the tragico baseline
                """
                # Make a shallow copy of the ppm scale
                x = deepcopy(ppm)
                # Cut it in the interested region
                ppm_trimmed, _ = misc.trim_data(x, x, lims)
                # Tragico scale goes from w to 0
                w = max(ppm_trimmed) - min(ppm_trimmed)
                # In tragico, the baseline is computed on a scale that goes from 0 to w, and it is inverted
                # with respect to the one used by klassez. To make the conversion, one must consider that
                #   x_k = 1 - (x_t / w)
                # and the two polynomia should be equal. Solving this gives the given solution for new_coeff
                new_coeff = np.array(
                        [
                            np.sum(bas_c),
                            np.sum([(j+1)*bas_c[j+1] for j in range(4)]) / w,
                            (bas_c[2] + 3 * bas_c[3] + 6 * bas_c[4]) / w**2,
                            (bas_c[3] + 4 * bas_c[4]) / w**3,
                            bas_c[4] / w**4,
                        ])
                return new_coeff

            # Conversion factor used by tragico
            maxr = np.max(spectrum)

            # Make a shallow copy to prevent overwriting of the original "regions"
            regions = deepcopy(reg)

            # Add "inp2" to the filename
            fname = filename.with_stem(f'{filename.stem}_inp2')
            # Open the file and write the header
            f = fname.open('w')
            header = 'i\tppm1\tppm2\tk\tfwhm\tphi\txg\tA\tB\tC\tD\tE\t\n'
            f.write(header)

            # Loop inside the regions
            for region in regions:
                # Get the region limits
                ppm1, ppm2 = region.pop('limits')
                # Get the intensity factor
                Int = region.pop('I')
                # Get the baseline coefficients and return them to their absolute value
                bas_c = region.pop('bas_c') * Int
                # Convert the baseline coefficients to tragico
                bas_c = xbaslfact(ppm, (ppm1, ppm2), bas_c) / maxr

                # At this point, only the peaks indices are left in region
                # Sort the indices of the peaks
                idxs = sorted(list(region.keys()))

                # Loop on the peaks
                for idx in idxs:
                    # Convert the intensity
                    k = region[idx]['k'] * Int / maxr
                    # The fwhm in tragico is in ppm
                    fwhm = misc.freq2ppm(region[idx]['fwhm'], self.SFO1)
                    # The phase in tragico is in radians
                    phi = region[idx]['phi'] * np.pi / 180
                    # Get the fraction of gaussianity
                    xg = region[idx]['b']

                    # Compute the line to write and write it
                    line = f'{idx}\t{ppm1:.1f}\t{ppm2:.1f}\t' + '\t'.join([f'{w:.5e}' for w in [k, fwhm, phi, xg, *bas_c]]) + '\n'
                    f.write(line)
            f.close()

            print(f'Input 2 for TrAGICo written in {fname}.', c='tab:blue')

        # Discriminate who do you want to save
        if which == 'result':
            regions = deepcopy(self.result)
        elif which == 'iguess':
            regions = deepcopy(self.i_guess)
        else:
            raise NameError('which must be "iguess" or "result"')

        if filename is None:
            filename = deepcopy(self.filename)
        filename = Path(filename)

        write_inp1(regions, filename)
        write_inp2(regions, self.ppm_scale, self.S, filename)
        print()


def peak_pick_2D(ppm_f1, ppm_f2, data, coord_filename='coord.tmp'):
    """
    Make interactive peak_picking.
    The position of the selected signals are saved in ``coord_filename``.
    If ``coord_filename`` already exists, the new signals are appended at its bottom: nothing is overwritten.
    Calls :func:`klassez.anal.select_traces` for the selection.

    Parameters
    ----------
    ppm_f1: 1darray
        ppm scale for the indirect dimension
    ppm_f2: 1darray
        ppm scale for the direct dimension
    data: 2darray
        Spectrum to peak-pick. The dimension should match the scale sizes.
    coord_filename: str or Path
        Path to the file where to save the peak coordinates

    Returns
    -------
    coord: list
        List of (u2, u1) for each peak
    """
    # Check for the existence of coord_filename
    coord_filename = Path(coord_filename)
    if coord_filename.exists():
        # number of already present signals: last linei, first value before tab
        n_C = eval(coord_filename.readlines()[-1].split('\t')[0])
        C = coord_filename.open('a', buffering=1)
    else:
        C = coord_filename.open('w', buffering=1)
        C.write(r'#'+'\t'+f'{"u2":^8s},{"u1":^8s}'+'\n')    # Header line
        n_C = 0

    # Make peak_picking
    coord = anal.select_traces(ppm_f1, ppm_f2, data)

    # Update the fucking coord file
    for k, obj in enumerate(coord):
        C.write(f'{k+1+n_C}'+'\t'+f'{obj[0]:-8.3f},{obj[1]:-8.3f}'+'\n')
    C.close()

    return coord


def build_2D_sgn(parameters, acqus, N=None, procs=None):
    """
    Create a 2D signal according to the final parameters returned by :func:`klassez.gui.make_iguess_2D`.
    Process it according to ``procs``.

    Parameters
    ----------
    parameters : list or 2darray
        sequence of the parameters: u1, u2, fwhm1, fwhm2, I, b. Multiple components are allowed
    acqus : dict
        2D-like acqus dictionary containing the acquisition timescales (keys t1 and t2)
    N : tuple of int
        Zero-filling values (F1, F2). Read only if procs is None
    procs : dict
        2D-like procs dictionary.

    Returns
    -------
    peak : 2darray
        rr part of the generated signal
    """
    parameters = np.array(parameters)
    if len(parameters.shape) == 1:
        parameters = parameters.reshape(1, -1)
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

    if procs is not None:    # Process the data according to procs
        peak, *_ = processing.xfb(signal, wf=procs['wf'], zf=procs['zf'])
    else:
        peak, *_ = processing.xfb(signal, zf=N)     # Just zero-fill

    return peak
# ----------------------------------------------------------------------------------------------------------------------


class Voigt_Fit_2D:
    """
    Class that wraps methods for the fit of 2D spectra with a set of 2D Voigtian lines.

    .. warning::

        This is work in progress.
    """
    def __init__(self, ppm_f1, ppm_f2, data, acqus, procs=None, label_list=None):
        """
        Initialize the class with ppm scales, experimental spectrum, acqus and procs dictionaries.

        Parameters
        ----------
        ppm_f1 : 1darray
            ppm scale for the indirect dimension
        ppm_f2 : 1darray
            ppm scale for the direct dimension
        data : 2darray
            Spectrum to fit. The dimension should match the scale sizes.
        acqus : dict
            Dictionary of acquisition parameters
        procs : dict
            Dictionary of processing parameters
        label_list : list
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

    def plot(self, filename=None, show_exp=True, dpi=600, **kwargs):
        """
        Draw a plot of the guessed/fitted peaks.

        Parameters
        ----------
        filename : str or Path or None
            Filename for the figure. If it is None, the figure is shown.
        show_exp : bool
            Choose if to plot the experimental spectrum or not
        dpi : int
            Resolution of the saved image
        kwargs : keyworded arguments
            Additional parameters to be passed to figures.ax2D.
        """

        # Generate the full spectrum to make the plot computationally less expensive
        fitted_data = np.sum(self.peaks, axis=0)

        # Make the figure
        fig = plt.figure('Fit')
        fig.set_size_inches(figures.figsize_large)
        ax = fig.add_subplot()
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
        if filename is None:
            misc.set_fontsizes(ax, 14)
            plt.show()
        else:
            plt.savefig(Path(filename).with_suffix('.svg'), dpi=dpi)
        plt.close()

    @staticmethod
    def draw_crossmarks(coord, ax, label_list=None, markersize=5, labelsize=8, markercolor='tab:blue', labelcolor='b'):
        """
        Draw crossmarks and peak labels on a figure.

        Parameters
        ----------
        ax : matplotlib.Subplot object
            Subplot where to plot the crossmarks and the labels.
        label_list : list
            Labels for the peaks. If None, they are computed as 1, 2, 3, ...
        markersize : int
            Dimension of the crossmark
        labelsize : int
            Fontsize for the labels
        markercolor : str
            Color of the crossmark
        labelcolor : str
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

        Parameters
        ----------
        coord_filename : str
            Path to the file where to save the peak coordinates
        """
        fit.peak_pick_2D(self.ppm_f1, self.ppm_f2, self.data, coord_filename)

    def load_coord(self, coord_filename='coord.tmp'):
        """
        Read the values from the coord filename and save them into the attribute "coord".

        Parameters
        ----------
        coord_filename : str
            Path to the file to be read
        """
        R = Path(coord_filename).readlines()

        coord = []
        label_list = []
        for k, line in enumerate(R):
            if line[0] == '#' or line.isspace():    # Skip comments and empty lines
                continue
            else:
                x, y = eval(line.split('\t', 2)[1].strip('\n'))  # second and third column
                coord.append([x, y])
                if len(line.split('\t', 2)) > 2:  # If there is the label
                    label = line.split("\t", 2)[-1].strip("\n")
                    if not label.isspace():
                        label_list.append(f'{label}')
        # Store coord into the attribute coord
        self.coord = coord
        print(f'Loaded {coord_filename} as coord.', c='tab:blue')

        # Update label_list, if there are labels in the coord file
        if len(label_list) > 0:
            self.label_list = label_list
        if self.label_list is not None:
            if len(self.label_list) < len(self.coord):
                raise ValueError('The number of provided labels is not enough for the peaks.')

    def draw_coord(self, filename=None, labelsize=8, ext='svg', dpi=600, **kwargs):
        """
        Makes a figure with the experimental dataset and the peak-picked signals as crosshairs.

        Parameters
        ----------
        filename : str or Path or None
            Filename for the figure to be saved. If None, it is shown instead.
        labelsize : float
            Font size for the peak index
        ext : str
            Format of the image
        dpi : int
            Resolution of the saved image in dots per inches
        kwargs : keyworded arguments
            Additional options for figures.ax2D
        """

        fig = plt.figure('Picked Peaks')
        fig.set_size_inches(figures.figsize_large)
        ax = fig.add_subplot()

        figures.ax2D(ax, self.ppm_f2, self.ppm_f1, self.data, **kwargs)
        self.draw_crossmarks(self.coord, ax, label_list=self.label_list, markersize=5, labelsize=labelsize, markercolor='tab:blue', labelcolor='b')

        ax.set_xlabel(r'$\delta$ '+f'{misc.nuc_format(self.acqus["nuc2"])}' + r' /ppm')
        ax.set_ylabel(r'$\delta$ '+f'{misc.nuc_format(self.acqus["nuc1"])}' + r' /ppm')
        misc.set_fontsizes(ax, 14)
        if filename is None:
            plt.show()
        else:
            plt.savefig(Path(filename).with_suffix(f'.{ext}'), dpi=dpi)
        plt.close()

    @cron
    def make_peaks(self, idx, V):
        """
        Calculate the set of 2D peaks, given the matrix of their parameters and their index.
        The array of indexes is required in order to recognize the different components that contribute to a single peak.
        The attribute peaks of the class will be cleared and updated.

        Parameters
        ----------
        idx: 1darray
            Array of indexes of the peaks.
        V: list or 2darray
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
            del tmp_par  # Clear memory
        print('Done.', end=' ')  # remove \n so that the runtime is shown in the same line

    def load_iguess(self, filename='peaks.inp'):
        """
        Reads the initial guess file with the parameters of the peaks, separates the values and stores them into attributes.
        In particular:

            * idx will contain the peak index (first column of the file),
            * Vi will contain [u1, u2, fwhm1, fwhm2, Im, b] for each peak,
            * Wi will contain the fitting interval as ( (L_f1, R_f1), (L_f2, R_f2) )

        Parameters
        ----------
        filename : str
            Path to the input file to be read
        """

        # Safety check: if filename does exist
        if Path(filename).exists():    # open the file and reads the lines,
            R = Path(filename).readlines()
        else:   # raises error
            raise FileNotFoundError(f'{filename} does not exist.')

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
        self.make_peaks(self.idx, self.Vi)

    def load_fit(self, filename='fit.out'):
        """
        Reads the file with the parameters of the fitted peaks, separates the values and stores them into attributes.
        Then, uses these values to compute the peaks and save them into self.peaks.
        In particular:

            * idx will contain the peak index (first column of the file),
            * Vf will contain [u1, u2, fwhm1, fwhm2, Im, b] for each peak,
            * Wf will contain the fitting interval as ( (L_f1, R_f1), (L_f2, R_f2) )

        Parameters
        ----------
        filename: str
            Path to the input file to be read
        """

        # Safety check: if filename does exist
        path = Path(filename)
        if path.exists():     # open the file and reads the lines,
            R = path.readlines()
        else:   # raises error
            raise NameError(f'{path} does not exist.')

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
        self.make_peaks(self.idx, self.Vf)  # Update the peaks attribute

    def iguess(self, filename='peaks.inp', start_index=1, only_edit=None, fwhm0=100, overwrite=False, auto=False):
        """
        Make the initial guess for all the peaks.

        Parameters
        ----------
        filename : str or Path
            Path to the file where the peak parameters will be written
        start_index : int
            Index of the first peak to be guessed.
        only_edit : sequence of ints or None
            Index of the peak that have to be guessed interactively. The ones that do not appear here are guessed automatically.
        fwhm0 : float
            Default value for fwhm in both dimension for automatic guess
        overwrite : bool
            Choose if to overwrite the file or append the new peaks at the bottom
        auto : bool
            Allow automatic guess for the peaks. To be used in conjunction with only_edit: if auto is False, all the peaks are guessed interactively!

        """
        def auto_val(ppm_f1, ppm_f2, tr1, tr2, u1, u2, fwhm0, acqus):
            """ Compute initial guess automatically """
            # Limits
            lim_f1 = u1 + 100/np.abs(acqus['SFO1']), u1 - 100/np.abs(acqus['SFO1'])
            lim_f2 = u2 + 100/np.abs(acqus['SFO2']), u2 - 100/np.abs(acqus['SFO2'])
            interval = lim_f1, lim_f2
            # Parameters
            parameters = [[   # u1, u2, fwhm1, fwhm2, k*A, b
                misc.ppm2freq(u1, acqus['SFO1'], acqus['o1p']),  # v1 /Hz
                misc.ppm2freq(u2, acqus['SFO2'], acqus['o2p']),  # v2 /Hz
                fwhm0,  # fwhm1 /Hz
                fwhm0,  # fwhm2 /Hz
                1,     # I
                0.5,    # b
                ]]
            # Integral
            A0 = np.max(self.data) / np.prod(self.data.shape)**0.5
            parameters[0][-2] = A0

            return parameters, interval

        # -------------------------------------------------------------------------------

        path = Path(filename)
        if path.exists() and overwrite is False:    # append next peaks
            f = path.open('a', buffering=1)
        else:   # create a new file
            f = path.open('w', buffering=1)
            self._write_head_line(f)

        # Make the generator where to loop on peaks
        def extract(coord):
            """ Generator: yields the chemical shifts and the traces onto which to loop """
            for x, y in coord:   # u2, u1
                tr1 = anal.get_trace(self.data, self.ppm_f2, self.ppm_f1, x, column=True)   # F1 @ u2 ppm
                tr2 = anal.get_trace(self.data, self.ppm_f2, self.ppm_f1, y, column=False)   # F2 @ u1 ppm
                yield (y, tr1), (x, tr2)    # (u1, f1), (u2, f2)
        peaks_coord = extract(self.coord)   # Call the generator

        # Start looping
        peak_index = 1
        for TR1, TR2 in peaks_coord:
            if peak_index < start_index:    # Do not guess
                peak_index += 1
                continue
            print(f'Preparing iguess for {peak_index:4.0f}/{len(self.coord):4.0f} peak', end='\r', c='tab:orange')
            # Unpack TR1 and TR2
            u1, tr1 = TR1
            u2, tr2 = TR2

            if auto is True and only_edit is None:  # All automatic
                parameters, interval = auto_val(self.ppm_f1, self.ppm_f2, tr1, tr2, u1, u2, fwhm0, self.acqus)
            elif auto is True and only_edit is not None:    # Interactively guess only the given peaks, all the others automatically
                if peak_index in only_edit:
                    parameters, interval = gui.gen_iguess_2D(self.ppm_f1, self.ppm_f2, tr1, tr2, u1, u2, self.acqus, fwhm0, self.procs)
                else:
                    parameters, interval = auto_val(self.ppm_f1, self.ppm_f2, tr1, tr2, u1, u2, fwhm0, self.acqus)
            else:   # All interactively
                parameters, interval = gui.gen_iguess_2D(self.ppm_f1, self.ppm_f2, tr1, tr2, u1, u2, self.acqus, fwhm0, self.procs)

            for values in parameters:   # Write the parameters
                self._write_par_line(f, self.acqus, peak_index, values, interval_f1=interval[0], interval_f2=interval[1])
            peak_index += 1  # Increment the peak index
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

        path = Path(filename)
        if path.exists() and overwrite is False:    # append next peaks
            f = path.open('a', buffering=1)
        else:   # create a new file
            f = path.open('w', buffering=1)
            self._write_head_line(f)

        # Loop for the fit
        for peak_index, peak_values in looped_values:
            if use_logfile:  # Redirect the standard output to the logfile
                sys.stdout = open(fit_kws['logfile'], 'a', buffering=1)
            print(f'Fitting peak {peak_index:4.0f} / {max(self.idx):4.0f}', c='tab:orange')
            if peak_index < start_index:    # Skip
                continue
            if len(peak_values) == 0:   # Skip empty parameters
                continue
            lim_f1, lim_f2 = self.Wi[peak_index-1]  # Get the window for the fit
            # Call the fit
            fit_parameters = voigt_fit_2D(self.ppm_f2, self.ppm_f1, self.data, peak_values,
                                          lim_f1, lim_f2, self.acqus, N=(len(self.ppm_f1), len(self.ppm_f2)),
                                          procs=self.procs, **fit_kws)
            # Write the output in the new file
            for fit_values in fit_parameters:
                self._write_par_line(f, self.acqus, peak_index, fit_values, interval_f1=lim_f1, interval_f2=lim_f2, conv_u=False)

        # Revert standard output to default
        if use_logfile:
            sys.stdout = sys.__stdout__

    # PRIVATE METHODS
    def _write_head_line(self, f):
        """
        Writes the header of the output file.

        Parameters
        ----------
        f : TextIOWrapper
            writable file generated by either ``open(filename, 'w'/'a')`` or  ``filename.open('w'/'a')``
        """
        f.write(f'{"#":<4s}\t{"clu":>4s}\t{"u1":>8s}\t{"u2":>8s}\t{"fwhm1":>8s}\t{"fwhm2":>8s}\t{"I":>8s}\t{"b":>8s}\t{"Fit. interv.":>20s}\n')

    def _write_par_line(self, f, acqus, index, clu, values, interval_f1=None, interval_f2=None, conv_u=True):
        """
        Writes a line of parameters to the output file.

        Parameters
        ----------
        f : TextIOWrapper
            writable file generated by either ``open(filename, 'w'/'a')`` or  ``filename.open('w'/'a')``
        acqus : dict
            2D-like acquisition parameters
        index : int
            Index of the peak
        clu : int
            Cluster index
        values : 1darray
            u1, u2, fwhm1, fwhm2, I, b
        interval_f1 : tuple
            left limit F1, right limit F1
        interval_f2 : tuple
            left limit F2, right limit F2
        conv_u : bool
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


# -------------------------------------------------------------------------------------


class CostFunc:
    r"""
    Class that groups several ways to compute the target of the minimization in a fitting procedure.
    It includes the classic squared sum of the residuals, as well as some other non-quadratic cost functions.
    Let `x` be the residuals and `s` the chosen threshold value. Then the objective value `R` is computed as:

    .. math::

        R = \sum_k f(x[k])

    where :math:`f(x)` can be chosen between the following options:

    * Quadratic:

        .. math::

            f(x) = x^2

    * Truncated Quadratic:

        .. math::

            f(x) = \begin{cases}
            x^2 & \text{if } |x| < s\\
            s^2 & \text{otherwise}\\
            \end{cases}

    * Huber function:

        .. math::

            f(x) = \begin{cases}
            x^2 & \text{if } |x| < s\\
            2s|x| - s^2 & \text{otherwise}\\
            \end{cases}

    * Asymmetric Truncated Quadratic:

        .. math::

            f(x) = \begin{cases}
            x^2 & \text{if } x < s\\
            s^2 & \text{otherwise}\\
            \end{cases}

    * Asymmetric Huber function:

        .. math::

            f(x) = \begin{cases}
            x^2 & \text{if } x < s\\
            2sx - s^2 & \text{otherwise}\\
            \end{cases}

    Attributes
    ----------
    method : function
        Function to be used for the computation of the objective value. It must take as input the array of the residuals and the threshold, no matter if the latter is actually used or not.
    s : float
        Threshold value
    """
    def __init__(self, method='q', s=None):
        """
        Initialize the method according to your choice, then stores the threshold value in the attribute ``s``.
        Allowed choices are:

        * "q": Quadratic
        * "tq": Truncated Quadratic
        * "huber": Huber function
        * "atq": Asymmetric Truncated Quadratic
        * "ahuber": Asymmetric Huber function

        Parameters
        ----------
        method : str
            Label for the method selection
        s : float
            Threshold value
        """
        self.method = self.method_selector(method)
        self.s = s

    def method_selector(self, method):
        """
        Performs the selection of the method according to the identifier string.

        Parameters
        ----------
        method : str
            Method label

        Returns
        -------
        f : function
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
        Computes the objective value according to the chosen method and the residuals array ``x``.

        Parameters
        ----------
        x: 1darray
            Array of the residuals

        Returns
        -------
        R: 1darray
            Computed objective function to be given to a least-squares solver
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
                x[i] = np.sign(x_i) * (2*s*np.abs(x_i) - s**2)**0.5
        return x

    @staticmethod
    def asymm_huber(r, s):
        """ Linear behaviour above s, penalizes negative entries """
        x = np.copy(r)
        for i, x_i in enumerate(x):
            if x_i < s:
                pass
            else:
                x[i] = (2*s*x_i - s**2)**0.5
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


def lsp(y, x, n=5, w=None):
    """
    Linear-System Polynomion
    Make a polynomial fit on the experimental data `y` by solving the linear system

    .. math::

        y = T c

    where `T` is the Vandermonde matrix of the x-scale and `c` is the set of coefficients that minimize the problem in the least-squares sense.
    It is also possible to make it weighted by using an array of weights ``w``.

    Parameters
    ----------
    y : 1darray
        Experimental data
    x : 1darray
        Independent variable (better if normalized)
    n : int
        Order of the polynomion + 1, i.e. number of coefficients
    w : 1darray
        Array of weights for the data. If None, the nonweighted approach is used

    Returns
    -------
    c : 1darray
        Set of minimized coefficients
    """
    # Make the Vandermonde matrix of the x-scale
    T = np.array(
            [x**k for k in range(n)]
            ).T
    if w is None:
        # Pseudo-invert it
        Tpinv = np.linalg.pinv(T)
    else:
        # Equivalent implementation to Tw = np.diag(w) @ T, but faster
        Tw = T * w[:, None]
        Tpinv = np.linalg.pinv(Tw)

    # Solve the system
    c = Tpinv @ y
    return c


def polyn_basl(y, n=5, method='huber', s=0.2, c_i=None, itermax=1000):
    """
    Fit the baseline of a spectrum with a low-order polynomion using a non-quadratic objective function.

    Let ``y`` be an array of ``N`` points. The polynomion is generated on a normalized scale that goes from -1 to 1 in ``N`` steps,
    and the coefficients are initialized either from outside through the parameter ``c_i`` or with the ordinary least squares fit.
    Then, the guess is refined using the objective function of choice employing the trust-region reflective least-squares algorithm.


    Parameters
    ----------
    y : 1darray
        Experimental data
    n : int
        Order of the polynomion + 1, i.e. number of coefficients
    method : str
        Objective function of choice. 'q': quadratic, 'tq': truncated quadratic, 'huber': Huber, 'atq': asymmetric truncated quadratic, 'ahuber': asymmetric huber
    s : float
        Relative threshold value for the non-quadratic behaviour of the objective function
    c_i : sequence or None
        Initial guess for the polynomion coefficient. If None, the least-squares fit is used
    itermax : int
        Number of maximum iterations

    Returns
    -------
    px : 1darray
        Fitted polynomion
    c : list
        Set of coefficients of the polynomion
    """
    def f2min_real(param, y, x, n, res_f):
        """
        Minimizer function.

        Parameters
        ----------
        param : lmfit.Parameters object
            Parameters to be optimized
        y : 1darray
            Experimental data
        x : 1darray
            Scale on which to build the model
        n : int
            Number of coefficients
        res_f : function
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

        Parameters
        ----------
        param : lmfit.Parameters object
            Parameters to be optimized
        y : 1darray
            Experimental data
        x : 1darray
            Scale on which to build the model
        n : int
            Number of coefficients
        res_f : function
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
    print('Make initial guess of the polynomion coefficients...', c='tab:orange')
    if c_i:
        c = np.copy(c_i)
    else:
        c = fit.lsp(y, x, n)
    px_iguess = misc.polyn(x, c)
    print('Done.', c='tab:orange')

    # Compute an intensity factor to decrease the weight on the fit procedure
    Int = fit.fit_int(np.abs(y), np.abs(px_iguess))[0]
    s *= Int      # Set absolute threshold values
    c /= Int      # Normalize the coefficients to I

    # Generate the parameters for the fit
    param = lmfit.Parameters()
    param.add('I', value=Int, vary=False)    # Just to keep track of it

    for k in range(n):
        param.add(f'c_{k}', value=c[k].real)

    # Get the objective function of choice
    R = fit.CostFunc(method, s)

    print('Optimizing the baseline...', c='tab:orange')
    # Make the fit
    if cplx:
        f2min = f2min_cplx
    else:
        f2min = f2min_real
    minner = lmfit.Minimizer(f2min, param, fcn_args=(y, x, n, R))
    result = minner.minimize(method='least_squares', max_nfev=int(itermax), gtol=1e-15)
    print(f'The fit has ended. {result.message}.\nNumber of function evaluations: {result.nfev}')

    # Get the fitted parameters
    popt = result.params.valuesdict()

    # Make a list of the fitted coefficients from the dictionary
    if cplx:
        c_opt = [popt['I'] * (popt[f'c_{k}'] + 1j*popt[f'c_{k}']) for k in range(n)]
    else:
        c_opt = [popt['I'] * popt[f'c_{k}'] for k in range(n)]
    # Build the polynomion with them
    px = misc.polyn(x, c_opt)

    return px, c_opt


class SINC_ObjFunc:
    r"""
    Computes the objective function as explained in M. Sawall et al., Journal of Magnetic Resonance 289 (2018), 132-141.
    The cost function is computed as:

    .. math::

        f(d) = \sum_{i=1}^3  \gamma_i g_i(d|e_i)

    where `d` is the real part of the NMR spectrum.

    Attributes
    ----------
    gamma1 : float
        Weighting factor for function g1
    gamma2 : float
        Weighting factor for function g2
    gamma3 : float
        Weighting factor for function g3
    e1 : float
        Tolerance value for function g1
    e2 : float
        Tolerance value for function g2
    """
    def __init__(self, gamma1=10, gamma2=0.01, gamma3=0, e1=0, e2=0):
        """
        Initialize the coefficients used to weigh the objective function.

        Parameters
        ----------
        gamma1 : float
            Weighting factor for function g1
        gamma2 : float
            Weighting factor for function g2
        gamma3 : float
            Weighting factor for function g3
        e1 : float
            Tolerance value for function g1
        e2 : float
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

        Parameters
        ----------
        d : 1darray
            Spectrum
        e1 : float
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

        Parameters
        ----------
        d : 1darray
            Spectrum
        e2 : float
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

        Parameters
        ----------
        d : 1darray
            Spectrum
        """
        diffd = np.diff(d, 2)
        g = np.sum(diffd**2)
        return g


def sinc_phase(data, gamma1=10, gamma2=0.01, gamma3=0, e1=0, e2=0, **fit_kws):
    """
    Perform automatic phase correction according to the SINC algorithm, as described in M. Sawall et. al., Journal of Magnetic Resonance 289 (2018), 132–141.
    The fitting method defaults to "least_squares".

    Parameters
    ----------
    data : 1darray
        Spectrum to phase-correct
    gamma1 : float
        Weighting factor for function g1: non-negativity constraint
    gamma2 : float
        Weighting factor for function g2: smallest-integral constraint
    gamma3 : float
        Weighting factor for function g3: smoothing constraint
    e1 : float
        Tolerance factor for function g1: adjustment for noise
    e2 : float
        Tolerance factor for function g2: adjustment for non-ideal baseline
    fit_kws : keyworded arguments
        additional parameters for the fit function. See :func:`lmfit.Minimizer.minimize` for details. Do not use "leastsq" because the cost function returns a scalar value!

    Returns
    -------
    p0 : float
        Fitted zero-order phase correction angle, in degrees
    p1 : float
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
    param = lmfit.Parameters()
    param.add('p0', value=0, min=-180, max=180)
    param.add('p1', value=0, min=-720, max=720)

    # Create the objective function
    R = fit.SINC_ObjFunc(gamma1, gamma2, gamma3, e1, e2)

    # Minimize using the method of choice. "leastsq" not accepted!
    print('Starting phase correction...', c='tab:orange')
    minner = lmfit.Minimizer(f2min, param, fcn_args=(d, R))
    result = minner.minimize(**fit_kws)
    print(f'The fit has ended. {result.message}.\nNumber of function evaluations: {result.nfev}')
    popt = result.params.valuesdict()

    return popt['p0'], popt['p1']


def write_vf_P2D(filename, peaks, lims, prev=0):
    """
    Write a section in a fit report file, which shows the fitting region and the parameters of the peaks to feed into a Voigt lineshape model.

    Parameters
    ----------
    filename : str or Path
        Path to the file to be written
    peaks : list of dict
        list of dictionares of :class:`klassez.fit.Peak` objects, one per experiment
    lims : tuple
        (left limit /ppm, right limit /ppm)
    prev : int
        Number of previous peaks already saved. Increases the peak index
    """

    # Open the file in append mode
    f = Path(filename).open('a', buffering=1)
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
    The file is separated and unpacked into a list of list of dictionaries, each of which contains the limits of the fitting window,
    and a dictionary for each peak with the characteristic values to compute it with a Voigt line.

    Parameters
    ----------
    filename : str
        Path to the filename to be read
    n : int
        Number of performed fit to be read. Default: last one. The breakpoints are lines that start with "!". For this reason, n=0 returns an empty dictionary, hence the first fit is n=1.

    Returns
    -------
    regions : list of list of dict
        List of dictionaries for running the fit.
    """
    def read_region(R):
        """ Creates a dictionary of parameters from a section of the input file.  """
        # Placeholder
        dic_r = {}
        # Separate the lines and remove the empty ones
        R = R.split('\n')
        for k, r in enumerate(R):
            if len(r) == 0 or r.isspace():
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
                line = r.split(';')  # Separate the values
                dic_r['limits'] = eval(line[0].replace(':', ', '))   # Get the limits

            if n_bp == 2:       # Second section: peak parameters
                line = r.split(';')  # Separate the values
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
                line = r.split(';')  # Separate the values
                if flag:    # Create the final dictionary to be returned and turn off the flag
                    dic_rr = [deepcopy(dic_r) for q in range(len(line)-1)]
                    flag = False

                # Unpack the line
                eval_line = [eval(w) for w in line]
                idx = int(eval_line[0])     # index of the peak
                Ks = eval_line[1:]          # Intensity, each for each experiment
                for q, K in enumerate(Ks):
                    dic_rr[q][idx]['k'] = K  # Overwrite the intensities

            if n_bp == 5:   # End of file: stop reading
                break

        return dic_rr

    # Read the file
    ff = Path(filename).readlines()
    # Get the actual section from an output file
    f = ff.split('!')[n]
    # Separate the bigger sections
    R = f.split('='*96)
    # Remove the empty lines
    for k, r in enumerate(R):
        if r.isspace():
            _ = R.pop(k)

    regions = []    # Placeholder for return values
    for r in R:  # Loop on the big sections to read them
        regions.append(read_region(r))
    return regions


def make_iguess_dosy(x, labels, data, model, model_args, diff_c_0=1e-10, filename='dosy_fit'):
    """
    Make the initial guess for the fit of a DOSY spectrum by using a GUI to visually adjust the value of
    the diffusion coefficient and the number of components to use.
    Calls :func:`gui.make_iguess_dosy_panel` in a loop. A section of the output file is written at the end
    of each loop.

    Parameters
    ----------
    x : 1darray
        Independent variable for the model (usually the gradient list)
    labels : list of str
        Identifier for the region, typically the integration window or peak number
    data : list of 1darray or 2darray
        Integrated profiles to fit
    model : callable
        Functional model for the DOSY profile. Signature:

        ::

            def model(x, diffc, **model_args):
                return 1darray

    model_args : dict of keyworded arguments
        Additional parameters for ``model``.
    diff_c_0 : float
        Default initial value for the diffusion coefficient, in m^2/s
    filename : str
        The output file of the procedure will be ``<filename>.idy``

    Returns
    -------
    None

    .. seealso::

        :func:`klassez.gui.make_iguess_dosy_panel`

        :func:`klassez.write_dy`
    """
    # Write the header of the idy file as it would in write_dy with header=True
    with open(f'{filename}.idy', 'a', buffering=1) as f:
        now = datetime.now()
        date_and_time = now.strftime("%d/%m/%Y at %H:%M:%S")
        f.write('! DOSY fit performed by {} on {}\n\n'.format(getpass.getuser(), date_and_time))

    # Make a loop: call the GUI for each set of integrals
    for k, (label, y) in enumerate(zip(labels, data)):
        print(f'Region {label} [ # {k+1} of {len(labels)}]', end='\r')
        gui.make_iguess_dosy_panel(x, label, y, model, model_args, diff_c_0, filename)
    print(f'\n{filename}.idy saved.', c='tab:blue')


def plot_fit_P2D(S, ppm_scale, regions, t_AQ, SFO1, o1p, show_total=False, show_res=False, res_offset=0, X_label=r'$\delta$ /ppm', labels=None, filename='fit', ext='svg', dpi=600):
    """
    Plots either the initial guess or the result of the fit, and saves all the figures.
    A new folder named <filename>_fit will be created.
    The figure `<filename>_full` will show the whole model and the whole spectrum.
    The figures labelled with `_R<k>` will depict a detail of the fit in the k-th fitting region.
    Optional labels for the components can be given: in this case, the structure of ``labels`` should match the structure of ``regions``.
    This means that the length of the outer list must be equal to the number of fitting region, and the length of the inner lists must be equal to the number of peaks in that region.

    Parameters
    ----------
    S : 2darray
        Spectrum to be fitted
    ppm_scale : 1darray
        ppm scale of the spectrum
    regions : list of dict
        Generated by fit.read_vf_P2D
    t_AQ : 1darray
        Acquisition timescale
    SFO1 : float
        Larmor frequency of the observed nucleus, in MHz
    o1p : float
        Carrier position, in ppm
    nuc : str
        Observed nucleus. Used to customize the x-scale of the figures.
    show_total : bool
        Show the total trace (i.e. sum of all the components) or not
    show_res : bool
        Show the plot of the residuals
    res_offset : float
        Displacement of the residuals plot from 0, to be given as a fraction of the height of the experimental spectrum. ``res_offset`` > 0 will move the residuals BELOW the zero-line!
    X_label : str
        Text to show as label for the chemical shift axis
    labels : list of list
        Optional labels for the components. The structure of this parameter must match the structure of self.result
    filename : str
        Root of the name of the figures that will be saved.
    ext : str
        Format of the saved figures
    dpi : int
        Resolution of the figures, in dots per inches
    """

    def calc_total(peaks, A=1):
        """
        Calculates the sum trace from a collection of peaks stored in a dictionary.
        If the dictionary is empty, returns an array of zeros.

        Parameters
        ----------
        peaks: dict
            Components
        A: float
            Absolute intensity

        Returns
        -------
        total: 1darray
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
    figdir = Path(f'{filename}_fit')
    figdir.mkdir(exist_ok=True, parents=True)
    # Update the filename for the figures by including the new directory
    filename = figdir / filename
    print('Saving figures...', c='tab:cyan')
    # Shallow copy of the real part of the experimental spectrum
    S_r = np.copy(S.real)
    # Make the acqus dictionary to be fed into the fit.Peak objects
    acqus = {'t1': t_AQ, 'SFO1': SFO1, 'o1p': o1p, }

    # Single regions
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
        signals.extend(list_signals)     # Dimensions (n. experiments, n.peaks per experiment, n.points per experiment)

        # Compute the total trace
        total = np.sum(signals, axis=1)  # sum the peaks

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

            for key, peak in zip(fit_peaks[i].keys(), signals[i]):  # Plot the components
                p_sgn, = ax.plot(t_ppm, peak[lims], lw=0.6, label=f'{key}')
                if labels is not None:  # Set the custom label
                    p_sgn.set_label(labels[k][key-1])

            if show_res is True:    # Plot the residuals
                # Compute the absolute value of the offset
                r_off = min(S_r[i, lims]) + res_offset * (max(S_r[i, lims])-min(S_r[i, lims]))
                ax.plot(t_ppm, (S_r - total)[i, lims] - r_off, c='g', ls=':', lw=0.6, label='Residuals')

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

    # Total
    # One figure per experiment
    for i, _ in enumerate(S):
        # Make the figure
        fig = plt.figure(f'Fit {i+1}')
        fig.set_size_inches(figures.figsize_large)
        ax = fig.add_subplot()
        plt.subplots_adjust(left=0.10, bottom=0.10, top=0.90, right=0.95)
        # Plot the experimental dataset
        ax.plot(ppm_scale, S_r[i], c='k', lw=1, label='Experimental')

        if show_total is True:  # Plot the total trace
            ax.plot(ppm_scale, total[i], c='b', lw=0.5, label='Fit', zorder=2)

        for key, peak in zip(fit_peaks[i].keys(), signals[i]):   # Plot the components
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
    print('Done.', c='tab:cyan')


def voigt_fit_P2D(S, ppm_scale, regions, t_AQ, SFO1, o1p, u_tol=1, f_tol=10, vary_phase=False, vary_b=False, itermax=10000, filename='fit'):
    """
    Performs a lineshape deconvolution fit on a pseudo-2D experiment using a Voigt model.
    The initial guess must be read from a .ivf file. All components are treated as independent, regardless from the value of the "group" attribute.
    The fitting procedure operates iteratively one window at the time.
    During the fit routine, the peak positions and lineshapes will be varied consistently on all the experiments; only the intensities are allowed to change in a different way.

    Parameters
    ----------
    S: 2darray
        Experimental spectrum
    ppm_scale: 1darray
        PPM scale of the spectrum
    regions: dict
        Generated by ``fit.read_vf_P2D``
    t_AQ: 1darray
        Acquisition timescale
    SFO1: float
        Nucleus Larmor frequency /MHz
    o1p: float
        Carrier frequency /ppm
    u_tol: float
        Maximum allowed displacement of the chemical shift from the initial value /ppm
    f_tol: float
        Maximum allowed displacement of the linewidth from the initial value /ppm
    vary_phase: bool
        Allow the peaks to change phase
    vary_b: bool
        Allow the peaks to change Lorentzian/Gaussian ratio
    itermax: int
        Maximum number of allowed iterations
    filename: str
        Name of the file where the fitted values will be saved. The .fvf extension is added automatically
    """

    # USED FUNCTIONS

    def calc_total(list_peaks, A=1):
        """
        Calculates the sum trace from a collection of peaks stored in a dictionary.
        If the dictionary is empty, returns an array of zeros.

        Parameters
        ----------
        peaks : dict
            Components
        A : float
            Absolute intensity

        Returns
        -------
        total : 1darray
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

        Parameters
        ----------
        peaks : dict
            Collection of fit.Peak objects
        par : dict
            New values for the peaks
        e_idx : int
            Number of experiment of which to change the intensity of the peak

        Returns
        -------
        peaks : dict
            Updated peaks dictionary with the new values
        """
        for idx, peak in peaks.items():
            peak.u = par[f'u_{idx}']
            peak.fwhm = par[f'fwhm_{idx}']
            peak.k = par[f'k_{idx}_{e_idx}']
            peak.b = par[f'b_{idx}']
            peak.phi = par[f'phi_{idx}']
        return peaks

    def f2min(param, S, fit_peaks, Int, lims):
        """
        Function that calculates the residual to be minimized in the least squares sense.
        This function requires a set of pre-built fit.Peak objects, stored in a dictionary.
        The parameters of the peaks are replaced on this dictionary according to the values in the lmfit.Parameter object.
        At this point, the total trace is computed and the residual is returned as the difference between the experimental spectrum and the total trace,
        only in the region delimited by the "lims" tuple.

        Parameters
        ----------
        param : lmfit.Parameters object
            Usual lmfit stuff
        S : 2darray
            Experimental spectrum
        fit_peaks : list of dict
            Collection of fit.Peak objects
        I : list or 1darray
            Absolute intensity values for all experiments
        lims : slice
            Trimming region corresponding to the fitting window, in points

        Returns
        -------
        residual : 1darray
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
        residual = np.concatenate([S[j, lims] / Int[j] - total[j, lims] for j, _ in enumerate(fit_peaks_in)])

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
                print(f'Fitting of region {k+1}/{Nr}. [{limits[0]:.3f}:{limits[1]:.3f}] ppm', c='tab:orange')
            # Make a copy of the region dictionary and remove what is not a peak
            peaklist = deepcopy(region)
            for peaks in peaklist:
                if 'limits' in list(peaks.keys()):
                    peaks.pop('limits')
            yield limits, peaklist

    # -----------------------------------------------------------------------------
    # Make the acqus dictionary to be fed into the fit.Peak objects
    acqus = {'t1': t_AQ, 'SFO1': SFO1, 'o1p': o1p, }

    N = S.shape[-1]     # Number of points of the spectrum
    Nr = len(regions)   # Number of regions to be fitted

    # Write info on the fit in the output file
    with open(f'{filename}.fvf', 'a', buffering=1) as f:
        now = datetime.now()
        date_and_time = now.strftime("%d/%m/%Y at %H:%M:%S")
        f.write('! Fit performed by {} on {}\n\n'.format(getpass.getuser(), date_and_time))

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
            I_arr.append(fit.fit_int(S[j, lims], i_guess[j, lims])[0])
        I_arr = np.array(I_arr)

        # Make the lmfit.Parameters object
        param = lmfit.Parameters()
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
                    elif 'fwhm' in key:  # fwhm: [max(0, fwhm-f_tol), fwhm+f_tol] (avoid negative fwhm)
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
            minner = lmfit.Minimizer(f2min, param, fcn_args=(S, fit_peaks, I_arr, lims))
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

    Attributes
    ----------
    ppm_scale : 1darray
        Self-explanatory
    S  : 2darray
        Spectrum to fit. Only real part
    t_AQ : 1darray
        acquisition timescale of the spectrum
    SFO1 : float
        Larmor frequency of the nucleus
    o1p  : float
        Pulse carrier frequency
    filename : str
        Root of the names of the files that will be saved
    X_label : str
        Label for the chemical shift axis in the figures
    i_guess : list
        Initial guess for the fit, read by a .ivf file with fit.read_vf_P2D
    result : list
        Result the fit, read by a .fvf file with fit.read_vf_P2D
    """

    def __init__(self, ppm_scale, S, t_AQ, SFO1, o1p, nuc=None, filename='fit'):
        """
        Initialize the class with common values.

        Parameters
        ----------
        ppm_scale : 1darray
            ppm scale of the spectrum
        S : 2darray
            Spectrum to be fitted
        t_AQ : 1darray
            Acquisition timescale
        SFO1 : float
            Larmor frequency of the observed nucleus, in MHz
        o1p : float
            Carrier position, in ppm
        nuc : str
            Observed nucleus. Used to customize the x-scale of the figures.
        filename : str or None
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
            self.X_label = r'$\delta$ ' + fnuc + ' /ppm'

    def iguess(self, input_file=None, expno=0, n=-1,):
        """
        Reads, or computes, the initial guess for the fit.
        If the file is there already, it just reads it with fit.read_vf. Otherwise, it calls gui.make_iguess to make it.

        Parameters
        ----------
        input_file : str or None
            Path to the input file. If None, "<self.filename>.ivf" is used
        expno : int
            Number of the experiment on which to compute the initial guess, in python numbering
        n : int
            Index of the initial guess to be read (default: last one)
        """
        # Set the default filename, if not given
        if input_file is None:
            input_file = f'{self.filename}'
        path = Path(f'{input_file}.ivf')
        # Check if the file exists
        if path.exists():       # Read everything you need from the file
            regions = fit.read_vf_P2D(path)
        else:                           # Make the initial guess interactively and save the file.
            gui.make_iguess_P2D(self.S, self.ppm_scale, expno, self.t_AQ, self.SFO1, self.o1p, filename=path.parent / path.stem)
            regions = fit.read_vf_P2D(path)
        # Store it
        self.i_guess = regions
        print(f'{path} loaded as input file.', c='tab:blue')

    def load_fit(self, output_file=None, n=-1):
        """
        Reads a file with fit.read_vf_P2D and stores the result in self.result.

        Parameters
        ----------
        output_file : str
            Path to the .fvf file to be read, without the `.fvf` extension. If None, "<self.filename>" is used.
        n : int
            Index of the fit to be read (default: last one)
        """
        # Set the default filename, if not given
        if output_file is None:
            output_file = f'{self.filename}'
        path = Path(f'{output_file}.fvf')
        # Check if the file exists
        if path.exists():       # Read everything you need from the file
            regions = fit.read_vf_P2D(path, n=n)
        else:
            raise NameError(f'{path} does not exist.')
        # Store
        self.result = regions
        print(f'{path} loaded as fit result file.', c='tab:blue')

    def dofit(self, u_tol=1, f_tol=10, vary_phase=False, vary_b=True, itermax=10000, filename=None):
        """
        Perform a lineshape deconvolution fitting by calling ``fit.voigt_fit_P2D``.
        The initial guess is read from the attribute ``self.i_guess``.

        Parameters
        ----------
        u_tol : float
            Determines the displacement of the chemical shift (in ppm) from the starting value.
        f_tol : float
            Determines the displacement of the linewidth (in Hz) from the starting value.
        vary_phase : bool
            Allow the peaks to change phase (True) or not (False)
        vary_b : bool
            Allow the peaks to change Lorentzian/Gaussian ratio
        itermax : int
            Maximum number of allowed iterations
        filename : str
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

    def plot(self, what='result', show_total=True, show_res=False, res_offset=0, labels=None, filename=None, ext='svg', dpi=600):
        """
        Plots either the initial guess or the result of the fit, and saves all the figures. Calls fit.plot_fit_P2D.
        The figures <filename>_full will show the whole model and the whole spectrum.
        The figures labelled with _R<k> will depict a detail of the fit in the k-th fitting region.
        Optional labels for the components can be given: in this case, the structure of 'labels' should match the structure of self.result (or self.i_guess).
        This means that the length of the outer list must be equal to the number of fitting region,
        and the length of the inner lists must be equal to the number of peaks in that region.

        Parameters
        ----------
        what : str
            'iguess' to plot the initial guess, 'result' to plot the fitted data
        show_total : bool
            Show the total trace (i.e. sum of all the components) or not
        show_res : bool
            Show the plot of the residuals
        res_offset : float
            Displacement of the residuals plot from 0, to be given as a fraction of the height of the experimental spectrum. res_offset > 0 will move the residuals BELOW the zero-line!
        labels : list of list
            Optional labels for the components. The structure of this parameter must match the structure of self.result
        filename : str
            Root of the name of the figures that will be saved. If None, <self.filename> is used
        ext : str
            Format of the saved figures
        dpi : int
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
        fit.plot_fit_P2D(S, self.ppm_scale, regions, self.t_AQ, self.SFO1, self.o1p, show_total=show_total,
                         show_res=show_res, res_offset=res_offset, X_label=self.X_label, labels=labels,
                         filename=filename, ext=ext, dpi=dpi)

    def get_fit_lines(self, what='result'):
        """
        Calculates the components, and the total fit curve used as initial guess, or as fit results..
        The components will be returned as a list, not split by region.

        Parameters
        ----------
        what : str
            'iguess' or 'result'

        Returns
        -------
        signals : list of list of 1darray
            Components used for the fit
        total : 2darray
            Sum of all the signals
        limits_list : list
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
        acqus = {'t1': self.t_AQ, 'SFO1': self.SFO1, 'o1p': self.o1p, }
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
            signals.extend(list_signals)     # Dimensions (n. experiments, n.peaks per experiment, n.points per experiment)

        # Compute the total trace
        total = np.sum(signals, axis=1)  # sum the peaks
        return signals, total, limits_list

    def res_histogram(self, what='result', nbins=500, density=True, f_lims=None, xlabel='Residuals', x_symm=True, barcolor='tab:green', fontsize=20, filename=None, ext='svg', dpi=300):
        """
        Computes the histogram of the residuals and saves it in the same folder of the fit figures.
        Employs fit.histogram to make the figure.

        Parameters
        ----------
        what : str
            'iguess' or 'result'
        nbins  : int
            number of bins to be calculated
        density  : bool
            True for normalize data
        f_lims  : tuple or None
            limits for the x axis of the figure
        xlabel  : str or None
            Text to be displayed under the x axis
        x_symm  : bool
            set it to True to make symmetric x-axis with respect to 0
        barcolor : str
            Color of the bins
        fontsize : float
            Biggest fontsize in the figure
        name  : str
            name for the figure to be saved
        ext : str
            Format of the image
        dpi : int
            Resolution of the image in dots per inches
        """
        # Filename check
        if filename is None:
            filename = f'{self.filename}'
        figdir = Path(f'{filename}_fit')
        figdir.mkdir(exist_ok=True, parents=True)
        filename = figdir / filename + '_rhist'

        # Select the correct object
        if what == 'iguess':
            regions = deepcopy(self.i_guess)
        elif what == 'result':
            regions = deepcopy(self.result)
        else:
            raise ValueError('Specify what you want to plot: "iguess" or "result"')

        # Get the total function and the limits
        _, total, limits_list = self.get_fit_lines(what)
        # Convert the limits in points according to the ppm scale
        limits_pt_list = [[misc.ppmfind(self.ppm_scale, w)[0] for w in lims]
                          for lims in limits_list]

        # Placeholders
        exp_trim, total_trim = [], []
        for k, region in enumerate(regions):        # loop on the regions
            # Compute the slice
            lims = slice(min(limits_pt_list[k]), max(limits_pt_list[k]))
            # Trim the experimental data and the total
            exp_trim.append(self.S[..., lims].real)
            total_trim.append(total[..., lims])
        # Sum on different regions
        exp_trim = np.sum(exp_trim, axis=0)
        total_trim = np.sum(total_trim, axis=0)

        # Compute the residuals and concatenate them
        residual = exp_trim - total_trim
        residual_arr = np.concatenate([r for r in residual], axis=-1)

        fit.histogram(residual_arr, nbins=nbins, density=density, f_lims=f_lims, xlabel=xlabel, x_symm=x_symm, barcolor=barcolor, fontsize=fontsize, name=filename, ext=ext, dpi=dpi)


def fit_dosy(x, y, iguess, model, model_args, d_bds=3, f_bds=[0, 3], vary_q=False):
    """
    Perform a fit of a DOSY profile using the specified ``model``.

    Parameters
    ----------
    x : 1darray
        Independent variable, typically the gradient strength in T/m
    y : 1darray
        Experimental data
    iguess : dict
        Initial guess for the fit, as generated by :func:`klassez.gui.make_iguess_dosy_panel`
    model : callable
        Model function. Signature:

    model_args : dict of kwargs
        Additional arguments to model
    d_bds : float or list
        Bounds for the diffusion coefficient.
        If it is a single ``float``, the bounds will be set to ``-d_bds`` and ``+d_bds``
        orders of magnitude with respect to the initial guess. E.g.: if ``diffc = 1e-10`` and
        ``d_bds = 2``, the bounds will be ``[1e-12, 1e-8]``.
        If it is a list of two ``float``s, the bounds will be set to ``-d_bds[0]` and ``+d_bds[1]``
        orders of magnitude with respect to the initial guess. E.g.: if ``diffc = 1e-10`` and
        ``d_bds = [1, 3]``, the bounds will be ``[1e-11, 1e-7]``
    f_bds : float or list
        Bounds for the fraction of component.
        If it is a single ``float``, the bounds will be set to ``-f_bds`` and ``+f_bds``
        with respect to the initial fraction. E.g.: if ``difff = 0.5`` and ``f_bds = 0.3``,
        the bounds will be ``[0.2, 0.8]``
        If it is a list of two ``float``s, the bounds will be set to ``f_bds[0]` and ``f_bds[1]``,
        regardless of what the initial fraction is.
    vary_q : bool
        Include the computation of the offset in the parameters. **Strongly discouraged!**

    Returns
    -------
    dic_result : dict
        Dictionary of optimized parameters, with the same format and shape of ``iguess``.

    .. seealso::

        :func:`klassez.fit.make_iguess_dosy`

        :func:`klassez.gui.make_iguess_dosy_panel`

    """

    def f2min(param, x, y, model, model_args, first_residual=1):
        """ Cost function for the fit """
        # Increase the iteration counter
        param['count'].value += 1
        # Unpack the Parameters object into a normal dictionary
        par = param.valuesdict()

        # Make diffusion coefficient and fractions as lists
        diff_c = [par[key] for key in par.keys() if 'D' in key]
        diff_f = [par[key] for key in par.keys() if 'f' in key]

        # Compute the models
        yc = [difff * model(x, diffc, **model_args) for diffc, difff in zip(diff_c, diff_f)]
        # Sum them to get the total trace
        total = np.sum(yc, axis=0)
        # Compute the residuals
        residual = (y - par['q']) / par['I'] - total
        print(f'Step: {par["count"]:6.0f} | Target: {np.sum(residual**2)/first_residual:10.5e}', end='\r')
        return residual

    # Number of components
    Np = len(iguess['diff_c'])
    # If bounds are single numbers, make them lists in order to do
    # what the docstring tells they do
    if isinstance(d_bds, (int, float)):
        d_bds = [d_bds, d_bds]
    if isinstance(f_bds, (int, float)):
        rel_f = True    # f_bds are relative to the value of f
    else:
        rel_f = False   # or not

    # Initialize the Parameters object
    param = lmfit.Parameters()
    # Common parameters to all components
    #   Intensity factor - adjusted by the single fractions
    param.add('I', value=iguess['I'], vary=False)
    #   Offset (normally 0 and do not move it)
    param.add('q', value=iguess['q'], vary=vary_q)
    #   Iteration counter
    param.add('count', value=0, vary=False)
    # Component-dependent parameters
    for k, (diffc, difff) in enumerate(zip(iguess['diff_c'], iguess['diff_f'])):
        # Bounds for D
        oom = np.log10(diffc)       # order of magnitude of D
        minD = diffc - 10**(oom-d_bds[0])
        maxD = diffc + 10**(oom+d_bds[1])
        # Add diffusion coefficient
        param.add(f'D_{k+1}', value=diffc, min=minD, max=maxD)

        # Bounds for f
        if rel_f:
            minf = difff - f_bds
            maxf = difff + f_bds
        else:
            minf = min(f_bds)
            maxf = max(f_bds)
        # Add fraction
        param.add(f'f_{k+1}', value=difff, min=minf, max=maxf)

    @cron
    def start_fit():
        # We need the first residual to match what lmfit thinks f_tol is
        with open(os.devnull, 'w') as sys.stdout:
            first_residual = np.sum(f2min(param, x, y, model, model_args)**2)
            # Reset the iteration counter
            param['count'].set(value=0)
        # Redirect output to stdout
        sys.stdout = sys.__stdout__

        # Use nelder because normally the guess is very good, hence leastsq might go crazy
        minner = lmfit.Minimizer(f2min, param, fcn_args=(x, y, model, model_args, first_residual))
        result = minner.minimize(method='nelder', max_nfev=10000)
        print(f'\n{result.message}\nNumber of function evaluations: {result.nfev}')
        return result
    result = start_fit()

    # Get the fitted parameters
    popt = result.params
    # Normalize the fractions to make them add up to 1
    diff_f_opt = [popt[f'f_{k+1}'].value for k in range(Np)]
    diff_f_norm, I_corr = misc.molfrac(diff_f_opt)
    # Make the output dictionary
    dic_result = {
            'diff_c': [popt[f'D_{k+1}'].value for k in range(Np)],
            'diff_e': [popt[f'D_{k+1}'].stderr for k in range(Np)],
            'diff_f': [float(difff) for difff in diff_f_norm],
            'I': float(popt['I'].value * I_corr),      # ofc it must take the correction by the fractions into account
            'q': popt['q'].value,
            }
    return dic_result


def fit_dosy_multi(x, y, iguess, model, model_args, d_bds=3, f_bds=[0, 3], vary_I=True, vary_q=False):
    """
    Perform a fit of a DOSY profile using the specified ``model``.

    Parameters
    ----------
    x : 1darray
        Independent variable, typically the gradient strength in T/m
    y : 1darray
        Experimental data
    iguess : dict
        Initial guess for the fit, as generated by :func:`klassez.gui.make_iguess_dosy_panel`
    model : callable
        Model function. Signature:

    model_args : dict of kwargs
        Additional arguments to model
    d_bds : float or list
        Bounds for the diffusion coefficient.
        If it is a single ``float``, the bounds will be set to ``-d_bds`` and ``+d_bds``
        orders of magnitude with respect to the initial guess. E.g.: if ``diffc = 1e-10`` and
        ``d_bds = 2``, the bounds will be ``[1e-12, 1e-8]``.
        If it is a list of two ``float``s, the bounds will be set to ``-d_bds[0]` and ``+d_bds[1]``
        orders of magnitude with respect to the initial guess. E.g.: if ``diffc = 1e-10`` and
        ``d_bds = [1, 3]``, the bounds will be ``[1e-11, 1e-7]``
    f_bds : float or list
        Bounds for the fraction of component.
        If it is a single ``float``, the bounds will be set to ``-f_bds`` and ``+f_bds``
        with respect to the initial fraction. E.g.: if ``difff = 0.5`` and ``f_bds = 0.3``,
        the bounds will be ``[0.2, 0.8]``
        If it is a list of two ``float``s, the bounds will be set to ``f_bds[0]` and ``f_bds[1]``,
        regardless of what the initial fraction is.
    vary_I : bool
        Include the computation of the intensity factor for each interval. If ``False``, all the intensity differences
        will be handled by the fractions parameters.
    vary_q : bool
        Include the computation of the offsets in the parameters. **Strongly discouraged!**
        If ``False``, the initial value read from the initial guess is kept thoughout the whole fitting.

    Returns
    -------
    dic_result : dict
        Dictionary of optimized parameters, with the same format and shape of ``iguess``.

    .. seealso::

        :func:`klassez.fit.make_iguess_dosy`

        :func:`klassez.gui.make_iguess_dosy_panel`

    """
    def f2min(param, x, yy, model, model_args, first_residual=1, calc_I=True, calc_q=False, show_result=False):
        """ Cost function for the fit """
        # Increase the iteration counter
        param['count'].value += 1
        # Unpack the Parameters object into a normal dictionary
        par = param.valuesdict()

        # Make diffusion coefficient and fractions as lists
        diff_c = np.asarray([par[f'D_{k+1}'] for k in range(Nc)])
        diff_f = np.asarray([[par[f'f_{k+1}_p{p}'] for k in range(Nc)] for p in range(Np)])

        # Compute the models
        yyc = np.array([
                [diff_f[p, k] * model(x, diff_c[k], k=p, **model_args)
                 for k in range(Nc)]
                for p in range(Np)
                ])

        # Sum them to get the total trace
        totals = np.sum(yyc, axis=1)
        if calc_I and calc_q:
            # Compute them both analytically
            for p in range(Np):
                I, q = fit.fit_int(yy[p], totals[p], calc_q)
                param[f'I_p{p}'].set(value=float(I))
                param[f'q_p{p}'].set(value=float(q))
        elif calc_I and not calc_q:
            # Compute only the I leaving the q fixed
            for p in range(Np):
                I, _ = fit.fit_int(yy[p]-par[f'q_p{p}'], totals[p], calc_q)
                param[f'I_p{p}'].set(value=float(I))
        else:
            # Both I and q stay fixed
            pass

        # Apply I and q
        for p in range(Np):
            totals[p] *= param[f'I_p{p}'].value
            totals[p] += param[f'q_p{p}'].value

        y = np.concatenate(yy, axis=-1)
        total = np.concatenate(totals, axis=-1)

        # Compute the residuals
        residual = y - total

        print(f'Step: {par["count"]:6.0f} | Target: {np.sum(residual**2)/first_residual:10.5e}', end='\r')
        return residual

    # Number of planes, number of components
    Np, Nc = np.asarray(iguess['diff_f']).shape
    # If bounds are single numbers, make them lists in order to do
    # what the docstring tells they do
    if isinstance(d_bds, (int, float)):
        d_bds = [d_bds, d_bds]
    if isinstance(f_bds, (int, float)):
        rel_f = True    # f_bds are relative to the value of f
    else:
        rel_f = False   # or not

    # Initialize the Parameters object
    param = lmfit.Parameters()
    # Common parameters to all components
    #   Intensity factor - adjusted by the single fractions
    for p in range(Np):
        param.add(f'I_p{p}', value=iguess['I'][p], vary=False)
        #   Offset (normally 0 and do not move it)
        param.add(f'q_p{p}', value=iguess['q'][p], vary=vary_q)

    #   Iteration counter
    param.add('count', value=0, vary=False)
    # Component-dependent parameters
    for k, diffc in enumerate(iguess['diff_c']):
        # Bounds for D
        oom = np.log10(diffc)       # order of magnitude of D
        minD = diffc - 10**(oom-d_bds[0])
        maxD = diffc + 10**(oom+d_bds[1])
        # Add diffusion coefficient
        param.add(f'D_{k+1}', value=diffc, min=minD, max=maxD)

    for p in range(Np):
        for k, difff in enumerate(iguess['diff_f'][p]):
            # Bounds for f
            if rel_f:
                minf = difff - f_bds
                maxf = difff + f_bds
            else:
                minf = min(f_bds)
                maxf = max(f_bds)
            # Add fraction
            param.add(f'f_{k+1}_p{p}', value=difff, min=minf, max=maxf)
            if Nc == 1 and vary_I:
                param[f'f_{k+1}_p{p}'].set(vary=False)

    @cron
    def start_fit():
        # We need the first residual to match what lmfit thinks f_tol is
        with open(os.devnull, 'w') as sys.stdout:
            first_residual = np.sum(f2min(param, x, y, model, model_args, calc_I=vary_I, calc_q=vary_q)**2)
            # Reset the iteration counter
            param['count'].set(value=0)
        # Redirect output to stdout
        sys.stdout = sys.__stdout__

        # Use nelder because normally the guess is very good, hence leastsq might go crazy
        minner = lmfit.Minimizer(f2min, param, fcn_args=(x, y, model, model_args, first_residual, vary_I, vary_q))
        result = minner.minimize(method='nelder', max_nfev=20000)
        print(f'\n{result.message}\nNumber of function evaluations: {result.nfev}')
        return result
    result = start_fit()

    # Get the fitted parameters
    popt = result.params

    dic_result = []     # placeholder
    for p in range(Np):
        # Normalize the fractions to make them add up to 1
        diff_f_opt = [popt[f'f_{k+1}_p{p}'].value for k in range(Nc)]
        diff_f_norm, I_corr = misc.molfrac(diff_f_opt)
        # Make the output dictionary
        dic_result.append({
                'diff_c': [popt[f'D_{k+1}'].value for k in range(Nc)],
                'diff_e': [popt[f'D_{k+1}'].stderr for k in range(Nc)],
                'diff_f': [float(difff) for difff in diff_f_norm],
                'I': float(popt[f'I_p{p}'].value * I_corr),      # ofc it must take the correction by the fractions into account
                'q': popt[f'q_p{p}'].value,
                })
    return dic_result


def plot_fit_dosy(x, label, y, total, yc, region, show_total=True, show_res=False, res_offset=0, filename=None, ext='svg', dpi=600, dim=None):
    """
    Make a plot of a DOSY fit.

    Parameters
    ----------
    x : 1darray
        Independent variable, normally the gradient strength in T/m
    label : str
        This will appear as figure title
    y : 1darray
        Experimental data
    total : 1darray
        Total fit trace
    yc : list of 1darray or 2darray
        Component traces
    region : dict
        Dictionary that contains the fitting values. Used to show the legend entries.
    show_total : bool
        Show the total trace (i.e. sum of all the components) or not
    show_res : bool
        Show the plot of the residuals
    res_offset : float
        Displacement of the residuals plot from 0, to be given as a fraction of the height of the experimental data.
        ``res_offset`` > 0 will move the residuals BELOW the zero-line!
    filename : str
        Filename of the figure that will be saved.
    ext : str
        Format of the saved figures
    dpi : int
        Resolution of the figures, in dots per inches
    dim : tuple
        Size of the figure in inches (length, height)
    """

    # Make the figure panel
    fig = plt.figure()
    if dim is None:
        fig.set_size_inches(figures.figsize_large)
    else:
        fig.set_size_inches(dim)
    plt.subplots_adjust(left=0.10, bottom=0.10, top=0.90, right=0.95)
    ax = fig.add_subplot()

    # Plots
    #   Experimental data
    ax.plot(x, y, '.', ms=10, c='k', label='Experimental')
    #   Total trace
    if show_total is True:
        ax.plot(x, total, c='b', lw=1.5, label='Total Fit')
    #   Components
    for k, y_c in enumerate(yc):
        if region['diff_e'][k] is None:
            error = ''
        else:
            error = r' $\pm$ ' + f'{region["diff_e"][k]:.5e}'
        legend_entry = '\n'.join([
            f'Component {k+1} ({region["diff_f"][k]*100:.2f}%)',
            f'D = {region["diff_c"][k]:.5e}' + error + r' m$^2$/s',
            ])
        ax.plot(x, y_c, '--', lw=1.0, label=legend_entry)
    #   Residuals
    if show_res is True:    # Plot the residuals
        # Compute the absolute value of the offset
        maxy = np.max(np.concatenate([y, total], axis=0))
        miny = np.min(np.concatenate([y, total], axis=0))
        r_off = res_offset * (maxy - miny)
        # actual plot
        ax.plot(x, y - total - r_off, c='g', ls=':', lw=0.8, label='Residuals')

    # Fancy shit
    #   title and axes labels
    ax.set_title(f'{label}')
    ax.set_xlabel(r'Gradient /T m$^{-1}$')
    ax.set_ylabel('Intensity /a.u.')

    #   ticks and co.
    misc.pretty_scale(ax, ax.get_xlim(), 'x')
    misc.pretty_scale(ax, ax.get_ylim(), 'y')
    misc.mathformat(ax)

    #   legend
    ax.legend()
    misc.set_fontsizes(ax, 20)

    # Show/save the figure
    if filename is None:
        plt.show()
    else:
        plt.savefig(f'{filename}.{ext}', dpi=dpi)
    plt.close()


def plot_fit_dosy_multi(x, yy, totals, components, region, bigdeltas=None, colors=None, filename=None, ext='svg', dpi=300, dim=None):
    """
    Makes a cumulative plot of a DOSY fit performed with the :class:`klassez.fit.DosyFit_pp3D` class.

    Parameters
    ----------
    x : 1darray
        Gradients in T/m
    yy : 2darray
        Experimental data. Dimension: ``(plane, integrals)``
    totals : 2darray
        Fitted trends.


        """

    # Default figure dimension
    if dim is None:
        dim = 16, 9
    # Number of planes
    Np = yy.shape[0]

    # Get the diffusion coefficients and their errors from the region dict
    diff_c = list(region['diff_c'])
    diff_e = list(region['diff_e'])

    # Make the figure
    fig = plt.figure()
    fig.set_size_inches(dim)
    plt.subplots_adjust(left=0.05, right=0.975)
    # One subplot per plane, top row
    axs = [fig.add_subplot(2, Np, w+1) for w in range(Np)]
    # Unique subplot that spans the whole bottom row
    axt = fig.add_subplot(2, Np, (Np+1, 2*Np+1))

    # Labels for the subplots that will appear in the legends
    if bigdeltas is None:
        titles = [f'Plane {k+1}' for k in range(Np)]
    else:
        titles = [r'$\Delta = $' + f'{bigdelta*1e3:.4g}' + ' ms' for bigdelta in bigdeltas]

    # Handle the colors
    if colors is None:
        colors = COLORS
    else:
        # Check if you have enough colors for the rendering
        assert len(colors) >= yy.shape[0], f'You need at least {yy.shape[0]} colors!'

    # Bottom panel first (easier)
    for y, yc, c, label in zip(yy, totals, colors, titles):
        # Plot the experimental trends
        yplot, = axt.plot(x, y, '.', c=c)
        # Plot the fit line with the same color
        axt.plot(x, yc, c=c, label=label)

    # Write the value of the diffusion coefficient as the title of the legend
    legend_title = '\n'.join([
        'Diffusion coefficients:',
        *[misc.expformat(diffc) + r'$ \pm $' +
          misc.expformat(diffe) if diffe else '' +
          r' m$^2$ s$^{-1}$'
          for diffc, diffe in zip(diff_c, diff_e)],
        ])

    # Fancy stuff
    axt.set_xlabel(r'Gradients /T m$^{-1}$')
    axt.set_ylabel(r'Intensity /a.u.')
    misc.pretty_scale(axt, axt.get_xlim(), 'x')
    misc.pretty_scale(axt, axt.get_ylim(), 'y')
    misc.mathformat(axt)
    misc.set_fontsizes(axt, 16)
    axt.legend(title=legend_title, fontsize=12, title_fontsize=14)

    # Now the top row
    axs[0].set_ylabel('Intensity /a.u.')    # label of the y axis only for the first subplot

    for ax, y, yc, comps, c, label in zip(axs, yy, totals, components, colors, titles):
        # Experimentals as black dots
        ax.plot(x, y, 'k.')
        # Fit line with the same color they have in the bottom panel
        ax.plot(x, yc, c=c, label=label)
        # Draw the components ONLY if there are more than one
        if comps.shape[0] > 1:
            for comp in comps:
                ax.plot(x, comp, c=c, ls='--', lw=0.8)

        # Fancy stuff
        ax.set_xlabel(r'/T m$^{-1}$')
        ax.legend()
        misc.pretty_scale(ax, ax.get_xlim(), 'x', 4)
        misc.pretty_scale(ax, ax.get_ylim(), 'y', 6)
        misc.mathformat(ax)
        misc.set_fontsizes(ax, 16)

    # Cut the white spaces as much as possible, it is a very big and complicated figure
    fig.tight_layout()
    # save/plot the figure
    if filename is None:
        plt.show()
    else:
        plt.savefig(f'{filename}.{ext}', dpi=dpi)
    plt.close()


class DosyFit:
    """
    Class to fit a DOSY spectrum.

    Attributes
    ----------
    filename : str
        Name for files, figures, and so on
    g : 1darray
        Gradient strength in T/m
    dosy_par : dict
        Dictionary of dosy parameters.
        ::

            dosy_par = {
                    'gamma' : # gyromagnetic ratio in MHz/T,
                    'D'     : # big delta in seconds,
                    'd'     : # little delta in seconds,
                    'tau'   : # tau in seconds,
                    'p90'    : # 90° pulse duration in seconds,
                    }

    keys : list of str
        Identifiers for the profiles to fit
    data : dict of 1darray
        ``data = {key : profile (1darray)  for key in self.keys}``
    i_guess : list of dict
        Dictionary of the initial guess, generated by ``self.iguess``.
        ::

            i_guess = [{
                'I' : # intensity factor (float),
                'q' : # offset (float),
                'diff_c' : # diffusion coefficients (1darray),
                'diff_f' : # fractions (1darray)
                'diff_e' : # fit errors for the diffusion coefficients (2darray),
                } for _ in self.keys]

    result : list of dict
        Dictionary of fit results, generated by either ``self.dofit`` or ``self.read_fit``.
        Same structure and shape of ``self.i_guess``
    """
    def __init__(self, s, pprog='stebp', difflist=None, input_data=None, filename=None):
        """
        Initialize the fitting interface using the DOSY spectrum as input.

        Parameters
        ----------
        s : klassez.DOSY object
            DOSY spectrum. You must have either integrated or fitted it to get the profiles.
        pprog : str
            Pulse sequence used for the acquisition. Modifies the fitting model and the parameters
            to be read on the basis. It can be read from the ``acqus`` dictionary, but in general there
            are two important parameters: if it is single or double stimulated echo (``ste``/``dste``),
            and if it uses bipolar gradients (``bp``) or not.
            Hence, these can be ``ste``, ``stebp``, ``dste``, ``dstebp``. However, something like
            ``xxxdsteyy23ybp47`` also works.
        difflist : 1darray or None
            List of the gradients strength, in Gauss/cm (as the instrument gives them).
            If ``None``, it reads the `difflist` file that should be in ``s.datadir``
        input_data : dict or None
            Dictionary returned by :func:`klassez.anal.integrate_p2D`. If ``None``,
            it will try to read ``s.integrals``.
            The reading from a fit (i.e. Voigt_Fit_p2D) has not been implemented yet.
        filename : str or None
            root filename for all the figures and files that will be generated.
            If ``None``, ``s.filename`` is used.
        """
        # Filename
        if filename is None:
            self.filename = s.filename

        dste, bp = self.parse_pprog(pprog)

        # Reads the parameters from the original spectrum,
        # instances the attributes g and dosy_par
        self.fetch_dosy_par(s, difflist, bp)

        self.select_model(dste, bp)

        # Get the data
        if input_data is not None:
            # Will store self.keys as well
            self.data = input_data
        else:
            # Try to read the integrals from s
            if hasattr(s, 'integrals'):
                if len(s.integrals.keys()) == 0:
                    raise ValueError('No integrals detected')
                self.data = s.integrals
            # Try to read the fit TODO NOTIMPLEMENTED)
            elif hasattr(s.F, 'result'):
                self.data = s.F.result
            else:
                raise ValueError('Neither integrals nor fit were detected in the input DOSY spectrum.')

    @staticmethod
    def parse_pprog(pprog):
        """
        Tries to understand from the name of the pulse program if the experiment was acquired with single or double stimulated echoes, and with or without bipolar gradients.

        Parameters
        ----------
        pprog : str
            Name of the pulse program

        Returns
        -------
        dste : bool
            Double (``True``) or single (``False``) stimulated echo
        bp : bool
            Bipolar gradients (``True``) or without (``False``)
        """
        pprog = pprog.lower()
        if 'dste' in pprog:
            dste = True         # double stimulated echo
        else:
            dste = False        # just stimulated echo
        if 'bp' in pprog:
            bp = True           # bipolar gradients
        else:
            bp = False          # no bipolar gradients
        return dste, bp

    def iguess(self, filename=None, ext='idy', diff_c_0=1e-10):
        """
        Either makes or reads an initial guess file for the fit.
        The resulting dictionary will be saved in ``self.i_guess``

        Parameters
        ----------
        filename : str
            Will try to read ``<filename>.<ext>``. If it does not exist, will write in ``<filename>.idy``
        ext : str
            Extension for the file, either `idy` or `fdy`
        diff_c_0 : float
            Initial default value for the diffusion coefficient.

        Returns
        -------
        None

        .. seealso ::

            :func:`klassez.fit.make_iguess_dosy`

            :func:`klassez.fit.write_dy`

            :func:`klassez.fit.read_dy`
        """
        if filename is None:
            filename = f'{self.filename}'
        path = Path(f'{filename}')
        if not path.with_suffix(f'.{ext}').exists():       # Read everything you need from the file
            ext = 'idy'
            fit.make_iguess_dosy(self.g, labels=self.keys, data=self._data,
                                 model=self.model, model_args=self.dosy_par,
                                 diff_c_0=diff_c_0, filename=path)
        path_x = path.with_suffix(f'.{ext}')
        # Store it
        self.i_guess = fit.read_dy(path_x)
        print(f'{path_x} loaded as input file.\n', c='tab:blue')

    def dofit(self, filename=None, d_bds=3, f_bds=[0, 3], vary_q=False):
        """
        Performs the fit of the profiles. Saves a `.fdy` file and stores the results
        in the attribute ``self.result``.

        Parameters
        ----------
        filename : str or None
            File where to save the results of the fit: ``<filename>.fdy``
        d_bds : float or list
            See :func:`klassez.fit.fit_dosy`
        f_bds : float or list
            See :func:`klassez.fit.fit_dosy`
        vary_q : bool
            Include the offset in the fit **strongly discouraged**

        Returns
        -------
        None

        .. seealso::

            :func:`klassez.fit.fit_dosy`

            :func:`klassez.fit.write_dy`
        """

        if filename is None:
            filename = f'{self.filename}'
        path = Path(filename).with_suffix('.fdy')
        # Check if the file exists
        self.result = []

        f = path.open('a', buffering=1)
        # Info on the region to be fitted
        now = datetime.now()
        date_and_time = now.strftime("%d/%m/%Y at %H:%M:%S")
        f.write('! DOSY fit performed by {} on {}\n\n'.format(getpass.getuser(), date_and_time))

        for k, label in enumerate(self.keys):
            print(f'Fitting region {label} [ # {k+1} of {len(self.keys)}]', c='tab:orange')
            dic_result = fit.fit_dosy(self.g, self._data[k], self.i_guess[k],
                                      self.model, self.dosy_par,
                                      d_bds=d_bds, f_bds=f_bds, vary_q=vary_q)
            fit.write_dy(path, dic_result['diff_c'], dic_result['diff_f'], dic_result['diff_e'],
                         label, dic_result['I'], dic_result['q'])
            dic_result['label'] = label
            self.result.append(dic_result)
        print(f'{path} saved.\n', c='tab:blue')

    def load_fit(self, filename=None, n=-1, ext='fdy'):
        """
        Reads a file with ``fit.read_dy`` and stores the result in ``self.result``.

        Parameters
        ----------
        filename: str
            Path to the .fdy file to be read. If None, "<self.filename>.fdy" is used.
        n: int
            Index of the fit to be read (default: last one)
        ext: str
            Extension of the file to be used

        Returns
        -------
        None

        .. seealso::

            :func:`klassez.fit.make_iguess_dosy`

            :func:`klassez.gui.make_iguess_dosy_panel`

            :func:`klassez.fit.fit_dosy`

            :func:`klassez.fit.read_dy`
        """
        # Set the default filename, if not given
        if filename is None:
            filename = f'{self.filename}'
        # Check if the file exists
        path_x = Path(filename).with_suffix(f'.{ext}')
        if path_x.exists():       # Read everything you need from the file
            regions = fit.read_dy(path_x, n=n)
        else:
            raise FileNotFoundError(f'{path_x} does not exist.')
        # Store
        self.result = regions
        print(f'{path_x} loaded as fit result file.\n', c='tab:blue')

    def plot(self, what='result', show_res=False, res_offset=0, filename=None, ext='svg', dpi=600, dim=None):
        """
        Plots either the initial guess or the result of the fit, and saves all the figures. Calls :func:`fit.plot_fit_dosy`.
        The figures will be saved in the directory `Figures_<filename>/<what>/<label>.svg`.

        Parameters
        ----------
        what : str
            'iguess' to plot the initial guess, 'result' to plot the fitted data
        show_res : bool
            Show the plot of the residuals
        res_offset : float
            Displacement of the residuals plot from 0, to be given as a fraction of the height of the experimental data.
            ``res_offset`` > 0 will move the residuals BELOW the zero-line!
        filename : str
            Determines the name of the directory where the figures will be saved. If None, `<self.filename>` is used
        ext : str
            Format of the saved figures
        dpi : int
            Resolution of the figures, in dots per inches
        dim : tuple
            Dimension of the figure in inches

        Returns
        -------
        None

        .. seealso::

            :func:`klassez.fit.plot_fit_dosy`
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
        filename = Path(filename)

        # Make the directories
        figure_path = filename.with_name('Figures_' + filename.stem) / f'{what}'
        figure_path.mkdir(exist_ok=True, parents=True)

        # Make the figures
        totals, components = self.get_fit_lines(what)
        print(f'Saving figures in {figure_path}...', c='tab:cyan')
        for k, (y, total, yc, region) in enumerate(zip(self._data, totals, components, regions)):
            print(f'{k+1}/{len(totals)}', end='\r')
            label = region['label']
            # Figures will be "Figures_{filename}/{what}/{label}.{ext}
            fit.plot_fit_dosy(self.g, label, y, total, yc, region,
                              show_res=show_res, res_offset=res_offset,
                              filename=figure_path/label, ext=ext, dpi=dpi, dim=dim)
        print('Done.\n', c='tab:cyan')

    def get_fit_lines(self, what='result'):
        """
        Calculates the components, and the total fit curve used as initial guess, or as fit results.

        Parameters
        ----------
        what : str
            ``'iguess'`` or ``'result'``

        Returns
        -------
        totals : list of 1darray
            Sum of all the signals, per region
        components : list of 2darray
            Components fitted for each region.
            Note that regions with only one component will be 2darrays anyways.
        """
        def calc_f(x, diffc, difff, A, q):
            """ Model for a single component """
            # Compute the model and multiply it by its fraction
            f = A * difff * self.model(x, diffc, **self.dosy_par) + q
            return f

        def calc_t(x, diff_c, diff_f, A, q):
            """ Compute all the components """
            # Loop over calc_f using all the values
            yc = [calc_f(x, diffc, difff, A, q) for diffc, difff in zip(diff_c, diff_f)]
            # The total trace is the sum of all the components
            t = np.sum(yc, axis=0)
            return t, yc

        # Discriminate if you want to calculate the initial guess or the result
        if what == 'iguess':
            regions = deepcopy(self.i_guess)
        elif what == 'result':
            regions = deepcopy(self.result)
        else:
            raise ValueError('Specify what you want to plot: "iguess" or "result"')

        # placeholders
        totals, components = [], []
        # Loop inside either iguess or result
        for region in regions:
            # Compute the totals and the components
            t, yc = calc_t(self.g, region['diff_c'], region['diff_f'], region['I'], region['q'])
            # Update the placeholders
            totals.append(t)
            components.append(np.array(yc))

        return totals, components

    @property
    def data(self) -> dict:
        """ Each time you call for it you will get a dictionary with the labels for each profile """
        return {key: self._data[k] for k, key in enumerate(self.keys)}

    @data.setter
    def data(self, input_data: (list, dict)):
        """ Get the data from the integrals (dict) or from the fit (list). ``_data`` is the 2darray! """
        if isinstance(input_data, dict):
            self.keys, self._data = self.data_from_integrals(input_data)
        else:
            self.keys, self._data = self.data_from_vf(input_data)

    @staticmethod
    def data_from_integrals(input_data):
        """ Fetch data from the integrals dictionary """
        keys = list(input_data.keys())
        data = np.array([input_data[key] for key in keys])
        return keys, data

    @staticmethod
    def data_from_vf(input_data):
        """ Fetch data from the fit list """
        raise NotImplementedError('WIP')

    def fetch_dosy_par(self, s, difflist=None, bp=True):
        """
        Reads the acquisition parameters to get the DOSY parameters.

        Creates the ``self.dosy_par`` attribute.

        Parameters
        ----------
        s : DOSY object
            Input spectrum that contains the ``ngdic`` attribute
        difflist : 1darray or None
            Gradient list in Gauss/cm, if existing. If ``None``, it will
            be read as well from ``<s.datadir>/difflist``
        bp : bool
            True if the sequence uses bipolar gradients, False if it does not.
            It is needed because in the former case the little delta is ``2 * p30``
        """
        #   Gradient list
        if difflist is None:
            difflist = np.loadtxt(s.datadir / 'difflist')
        # difflist is in G/cm -> we need T/m
        self.g = difflist * 1e-2

        # Bipolar gradient sequences have littledelta = 2 * p30
        if bp:
            multi_p30 = 2
        else:
            multi_p30 = 1

        # Reading the parameter from acqus
        d20 = s.ngdic['acqus']['D'][20]     # s
        d16 = s.ngdic['acqus']['D'][16]     # s
        p30 = s.ngdic['acqus']['P'][30]     # us
        p1 = s.ngdic['acqus']['P'][1]       # us

        self.dosy_par = {
                'gamma': sim.gamma[f'{s.acqus["nuc"]}'],    # MHz/T
                'D': d20,                                   # big delta /s
                'd': p30 * multi_p30 * 1e-6,                # little delta /s
                'tau': d16,                                 # tau /s
                'p90': p1 * 1e-6,                           # 90degree pulse /s
                }

    def select_model(self, dste, bp):
        """
        Select the model on the following scheme:

        -   ``dste = False`` and ``bp = False``: :func:`klassez.fit.model_ste`
        -   ``dste = False`` and ``bp = True``: :func:`klassez.fit.model_stebp`
        -   ``dste = True`` and ``bp = False``: :func:`klassez.fit.model_dste`
        -   ``dste = True`` and ``bp = True``: :func:`klassez.fit.model_dstebp`

        The selected function will be stored in ``self.model``.

        Parameters
        ----------
        dste : bool
            Single (``False``) or double (``True``) stimulated echo
        bp : bool
            Uses bipolar gradients (``True``) or not (``False``).
        """
        if dste and bp:
            self.model = fit.model_dstebp
        elif dste and not bp:
            self.model = fit.model_dste
        elif not dste and bp:
            self.model = fit.model_stebp
        elif not dste and not bp:
            self.model = fit.model_ste
        else:
            raise NameError('Unrecognized model')


@safe_kws
def model_ste(g, diff_c, gamma, D, d):
    r"""
    Model for Stimulated Echo.
    Basic Stejskal-Tanner equation.

    .. math::

        y(g) = \exp \{ - D (2 \pi \gamma g \delta)^2 \, (\Delta - \delta / 3) \}


    Parameters
    ----------
    g : 1darray
        Gradient strength T/m
    diff_c : float
        Diffusion coefficient m^2/s
    gamma : float
        Gyromagnetic ratio MHz/T
    D : float
        Big delta seconds
    d : float
        Little delta seconds

    Returns
    -------
    y : 1darray
        Computed model
    """
    # 1e6 is to convert MHz -> Hz for the gamma
    A = 2 * np.pi * gamma * d * 1e6
    B = (D - d/3)
    arg = - diff_c * g**2 * A**2 * B
    y = np.exp(arg)
    return y


@safe_kws
def model_stebp(g, diff_c, gamma, D, d, tau, p90):
    r"""
    Model for Stimulated Echo with Bipolar Gradients.
    Stejskal-Tanner equation modified by Jerschaw and Müller.

    .. math::

        y(g) = \exp \{ - D (2 \pi \gamma g \delta)^2 \, (\Delta - \delta / 3 - \tau - 4p_{90}) \}


    Parameters
    ----------
    g : 1darray
        Gradient strength T/m
    diff_c : float
        Diffusion coefficient m^2/s
    gamma : float
        Gyromagnetic ratio MHz/T
    D : float
        Big delta seconds
    d : float
        Little delta seconds
    tau : float
        Tau seconds
    p90 : float
        90 degree pulse seconds

    Returns
    -------
    y : 1darray
        Computed model
    """
    # 1e6 is to convert MHz -> Hz for the gamma
    A = 2 * np.pi * gamma * d * 1e6
    B = (D - d/3 - tau/2 - 4*p90)
    arg = - diff_c * g**2 * A**2 * B
    y = np.exp(arg)
    return y


@safe_kws
def model_dste(g, diff_c, gamma, D, d, tau, p90):
    r"""
    Model for Double Stimulated Echo.
    Stejskal-Tanner equation modified by Jerschaw and Müller.

    .. math::

        y(g) = \exp \{ - D (2 \pi \gamma g \delta)^2 \, (\Delta - 5 \delta / 3 - \tau - 4 p_{90}) \}


    Parameters
    ----------
    g : 1darray
        Gradient strength T/m
    diff_c : float
        Diffusion coefficient m^2/s
    gamma : float
        Gyromagnetic ratio MHz/T
    D : float
        Big delta seconds
    d : float
        Little delta seconds
    tau : float
        Tau seconds
    p90 : float
        90 degree pulse seconds

    Returns
    -------
    y : 1darray
        Computed model
    """
    # 1e6 is to convert MHz -> Hz for the gamma
    A = 2 * np.pi * gamma * d * 1e6
    B = (D - 5 * d/3 - tau - 4*p90)
    arg = - diff_c * g**2 * A**2 * B
    y = np.exp(arg)
    return y


@safe_kws
def model_dstebp(g, diff_c, gamma, D, d, tau, p90):
    r"""
    Model for Double Stimulated Echo with Bipolar Gradients.
    Stejskal-Tanner equation modified by Jerschaw and Müller.

    .. math::

        y(g) = \exp \{ - D (2 \pi \gamma g \delta)^2 \, (\Delta - 5 \delta / 3 - 3 \tau - 8 p_{90}) \}


    Parameters
    ----------
    g : 1darray
        Gradient strength T/m
    diff_c : float
        Diffusion coefficient m^2/s
    gamma : float
        Gyromagnetic ratio MHz/T
    D : float
        Big delta seconds
    d : float
        Little delta seconds
    tau : float
        Tau seconds
    p90 : float
        90 degree pulse seconds

    Returns
    -------
    y : 1darray
        Computed model
    """
    # 1e6 is to convert MHz -> Hz for the gamma
    A = 2 * np.pi * gamma * d * 1e6
    B = (D - 5 * d/3 - 3 * tau - 8*p90)
    arg = - diff_c * g**2 * A**2 * B
    y = np.exp(arg)
    return y


class DosyFit_pp3D(fit.DosyFit):
    """
    Interface for the fitting of a :class:`klassez.Spectra.DOSY_T1` object, i.e. a 3D spectrum where the DOSY is acquired along the `31` dimension and the big delta (``d20``) is increased along F2.

    Attributes
    ----------
    datadir : str
        Path where to save files and figures
    filename : str
        Name for files, figures, and so on
    planes : list of :class:`klassez.Spectra.pDOSY` object
        Projection of the original spectrum along the `31` direction
    g : 1darray
        Gradient strength in T/m
    dosy_par : dict
        Dictionary of dosy parameters.
        ::

            dosy_par = {
                    'gamma' : # gyromagnetic ratio in MHz/T,
                    'D'     : # list of big delta in seconds,
                    'd'     : # little delta in seconds,
                    'tau'   : # tau in seconds,
                    'p90'    : # 90° pulse duration in seconds,
                    }

    keys : list of str
        Identifiers for the profiles to fit
    data : dict of 2darray
        ``data = {key : profile for each plane (plane, integrals) for key in self.keys}``
    i_guess : list of dict
        Dictionary of the initial guess, generated by ``self.iguess``.
        ::

            i_guess = [{
                'I' : # intensity factor (1darray, long as planes),
                'q' : # offset (1darray, long as planes),
                'diff_c' : # diffusion coefficients (1darray, long as components),
                'diff_f' : # fractions (2darray, (planes, components))
                'diff_e' : # fit errors for the diffusion coefficients (1darray, long as components),
                } for _ in self.keys]

    result : list of dict
        Dictionary of fit results, generated by either ``self.dofit`` or ``self.read_fit``.
        Same structure and shape of ``self.i_guess``

    """
    def __init__(self, S, datadir=None, filename=None):
        """
        Initialize the class.
        This function will slice ``S`` along the DOSY dimension (`31`) and store the planes in the ``self.planes`` attribute.
        Then, it will try to load the integrals. If these are not found, it will ask you to integrate the first plane, and
        it will compute all the integrals by itself.

        Parameters
        ----------
        S : klassez.Spectra.DOSY_T1 object
            Input dataset, after processing
        datadir : str
            Custom path where to save files and figures. If ``None``, *./``filename``* is created
        filename : str
            Custom filename for files and figures. If ``None``, ``S.filename`` is used.
        """
        # Filename
        if filename is None:    # Same name of the input dataset
            self.filename = S.filename
        else:
            self.filename = filename

        # Datadir
        if datadir is None:     # ./<filename>
            self.datadir = Path.cwd() / self.filename
        else:
            self.datadir = Path(datadir)

        self.datadir.mkdir(exist_ok=True, parents=True)

        # Get the DOSY parameters
        self.fetch_dosy_par(S)

        # Set the model
        dste, bp = self.parse_pprog(S.ngdic['acqus']['PULPROG'])
        self.select_model(dste, bp)

        # Take all the planes in the DOSY dimension (31)
        self.planes = [S.getplane(k, dim='31') for k in range(len(S.x_f2))]

        # Compute the integrals for all the planes
        self.integrate_planes()

    def integrate_planes(self, keys=None, use_bas=False):
        """
        Get/compute the integrals for all the planes.
        Stores the integrals in ``self.data``

        Parameters
        ----------
        keys : list of str
            Keys that identify the regions to integrate. If None, the integration is performed interactively through GUI
        use_bas : bool
            Use the baseline or not.

        Returns
        -------
        None

        .. seealso::

            :func:`klassez.misc.key_to_limits`

            :func:`klassez.Spectra.pDOSY.integrate`

        """
        # Cycle through the planes
        for k, plane in enumerate(self.planes):
            # Do the following only if plane does not have the integrals already
            if hasattr(plane, 'integrals'):
                continue
            # Look if there is the integral filename
            integrals_filename = self.datadir / f'{plane.filename}'
            integrals_filename_x = integrals_filename.with_suffix('.igrl')
            if integrals_filename_x.exists():
                # There is --> Read it
                plane.read_integrals(integrals_filename_x)
                if k == 0:  # First plane: save the regions identifiers and the integration limits
                    keys = deepcopy(list(plane.integrals.keys()))
                    lims = misc.key_to_limits(keys)
            else:   # There is not ---> We have to integrate
                # First plane and the keys are not given from outside
                if k == 0 and keys is None:
                    lims = None     # --> GUI
                # First plane and the keys are given, OR, another plane => same keys of the first plane
                elif (k == 0 and keys is not None) or k:
                    lims = misc.key_to_limits(keys)     # transform to limits

            # Compute the integrals
            plane.integrate(filename=integrals_filename, lims=lims, use_bas=use_bas)
            # Overwrite the keys
            keys = deepcopy(list(plane.integrals.keys()))

        # Store the data
        big_dic = {  # Placeholder to save the data
                key: np.array([plane.integrals[key] for plane in self.planes])
                for key in self.planes[0].integrals.keys()
                }
        self.data = big_dic

    def fetch_dosy_par(self, s, bp=True):
        """
        Reads the acquisition parameters to get the DOSY parameters.

        Creates the ``self.dosy_par`` attribute.

        Parameters
        ----------
        s : DOSY object
            Input spectrum that contains the ``ngdic`` attribute
        bp : bool
            True if the sequence uses bipolar gradients, False if it does not.
            It is needed because in the former case the little delta is ``2 * p30``
        """
        # Gradient list
        #   difflist is in G/cm -> we need T/m
        self.g = s.x_f1 * 1e-2

        # Bipolar gradient sequences have littledelta = 2 * p30
        if bp:
            multi_p30 = 2
        else:
            multi_p30 = 1

        # Reading the parameter from acqus
        d16 = s.ngdic['acqus']['D'][16]     # s
        p30 = s.ngdic['acqus']['P'][30]     # us
        p1 = s.ngdic['acqus']['P'][1]       # us

        # Store the dosy parameters
        self.dosy_par = {
                'gamma': sim.gamma[f'{s.acqus["nuc"]}'],    # MHz/T
                'D': s.x_f2,                                # big delta /s
                'd': p30 * multi_p30 * 1e-6,                # little delta /s
                'tau': d16,                                 # tau /s
                'p90': p1 * 1e-6,                           # 90degree pulse /s
                }

    def iguess(self, filename=None, ext='idy', diff_c_0=1e-10, ref=0):
        """
        Either makes or reads an initial guess file for the fit. Operates only on _one_ plane, chosen as reference.
        The resulting dictionary will be saved in ``self.i_guess``

        Parameters
        ----------
        filename : str
            Will try to read ``<filename>.<ext>``. If it does not exist, will write in ``<filename>.idy``
        ext : str
            Extension for the file, either `idy` or `fdy`
        diff_c_0 : float
            Initial default value for the diffusion coefficient.
        ref : int
            Index of the reference plane (python numbering)

        Returns
        -------
        None

        .. seealso ::

            :func:`klassez.fit.make_iguess_dosy`

            :func:`klassez.fit.write_dy`

            :func:`klassez.fit.read_dy`
        """
        if filename is None:
            filename = self.datadir / f'{self.planes[ref].filename}'
        else:
            filename = Path(filename)
        # Check if the file exists
        filename_x = filename.with_suffix(f'.{ext}')

        if not filename_x.exists():       # Read everything you need from the file
            filename_x = filename.with_suffix('.idy')
            dosy_par_ref = deepcopy(self.dosy_par)
            dosy_par_ref['D'] = self.dosy_par['D'][ref]
            fit.make_iguess_dosy(self.g, labels=self.keys, data=np.asarray(self._data)[:, ref],
                                 model=self._model, model_args=dosy_par_ref,
                                 diff_c_0=diff_c_0, filename=filename)
        regions = fit.read_dy(filename_x)
        # Store it
        for plane in self.planes:
            plane.D.i_guess = regions
        print(f'{filename_x} loaded as input file.\n', c='tab:blue')

        self.merge_planes('iguess')

    def merge_planes(self, what='iguess'):
        """
        Rewrites either the ``self.i_guess`` or ``self.result`` attribute by merging
        the information relative to each plane.

        Parameters
        ----------
        what : str
            ``'iguess'`` or ``'result'``. Determines which attribute to read and write.
        """
        if what == 'iguess':
            tostore = [
                {
                    'label': label,
                    'I': [plane.D.i_guess[k]['I'] for plane in self.planes],
                    'q': [plane.D.i_guess[k]['q'] for plane in self.planes],
                    'diff_c': self.planes[0].D.i_guess[k]['diff_c'],
                    'diff_f': [plane.D.i_guess[k]['diff_f'] for plane in self.planes],
                    'diff_e': self.planes[0].D.i_guess[k]['diff_e'],
                    }
                for k, label in enumerate(self.keys)
                ]
            self.i_guess = tostore
        else:
            tostore = [
                {
                    'label': label,
                    'I': [plane.D.result[k]['I'] for plane in self.planes],
                    'q': [plane.D.result[k]['q'] for plane in self.planes],
                    'diff_c': self.planes[0].D.result[k]['diff_c'],
                    'diff_f': [plane.D.result[k]['diff_f'] for plane in self.planes],
                    'diff_e': self.planes[0].D.result[k]['diff_e'],
                    }
                for k, label in enumerate(self.keys)
                ]
            self.result = tostore

    def dofit(self, seq=False, filename=None, d_bds=3, f_bds=[0, 3], vary_q=False):
        """
        Performs the fit of the profiles. Saves a `.fdy` file and stores the results
        in the attribute ``self.result``.

        Parameters
        ----------
        seq : bool
            If ``False``, the fit is performed by forcing the diffusion coefficient to be the same across the same region in different planes.
            If ``True``, the diffusion coefficient can vary, and the fit is performed using the internal fit method of the planes objects.
        filename : str or None
            File where to save the results of the fit: ``<filename>.fdy``
        d_bds : float or list
            See :func:`klassez.fit.fit_dosy`
        f_bds : float or list
            See :func:`klassez.fit.fit_dosy`
        vary_q : bool
            Include the offset in the fit **strongly discouraged**

        Returns
        -------
        None

        .. seealso::

            :func:`klassez.fit.fit_dosy`

            :func:`klassez.fit.write_dy`
        """
        if filename is None:
            filenames = [self.datadir / f'{plane.filename}' for plane in self.planes]
        else:
            filenames = [Path(f'{filename}_{k}') for k in range(len(self.planes))]

        if seq:
            for fn, plane in zip(filenames, self.planes):
                plane.D.dofit(filename=filename)
        else:
            dic_results = []
            for k, label in enumerate(self.keys):
                print(f'Fitting region {label} [ # {k+1} of {len(self.keys)}]', c='tab:orange')
                dic_result = fit.fit_dosy_multi(self.g, self.data[label], self.i_guess[k],
                                                self.model, self.dosy_par,
                                                d_bds=d_bds, f_bds=f_bds, vary_q=vary_q)
                for p, _ in enumerate(self.planes):
                    dic_result[p]['label'] = label
                dic_results.append(dic_result)

            for p, plane in enumerate(self.planes):
                to_save = [dic_results[w][p] for w, _ in enumerate(self.keys)]
                plane.D.result = to_save

                f = filenames[p].with_suffix('.fdy').open('a', buffering=1)
                # Info on the region to be fitted
                now = datetime.now()
                date_and_time = now.strftime("%d/%m/%Y at %H:%M:%S")
                f.write('! DOSY fit performed by {} on {}\n\n'.format(getpass.getuser(), date_and_time))

                for k, result in enumerate(plane.D.result):
                    fit.write_dy(filenames[p].with_suffix('.fdy'), result['diff_c'], result['diff_f'],
                                 result['diff_e'], label, result['I'], result['q'])
                f.close()

        self.merge_planes('result')

    def load_fit(self, filename=None, ext='fdy'):
        """
        Reads the output of an already performed fit by reading the files
        `<plane.filename>.<ext>` for all the planes, reorganizes them, and stores
        the outcome in ``self.result``

        Parameters
        ----------
        ext : str
            ``'fdy'`` or ``'idy'``

        Returns
        -------
        None

        .. seealso ::

            :func:`klassez.fit.read_dy`
        """
        if filename is None:
            filenames = [self.datadir / f'{plane.filename}' for plane in self.planes]
        else:
            filenames = [Path(f'{filename}_{k}') for k in range(len(self.planes))]

        for plane, fn in zip(self.planes, filenames):
            regions = fit.read_dy(fn.with_suffix(f'.{ext}'))
            # Store it
            plane.D.result = regions

        self.merge_planes('result')

    def plot(self, what='result', only_all=False, show_res=False, res_offset=0, figdir=None, filename=None, ext='svg', dpi=100, dim=None):
        """
        Plots either the initial guess or the result of the fit, and saves all the figures. Calls :func:`fit.plot_fit_dosy`.
        The figures will be saved in the directory `<figdir>/<what>/<label>.<ext>`.

        Parameters
        ----------
        what : str
            'iguess' to plot the initial guess, 'result' to plot the fitted data
        only_all = bool
            Plot only the figure with all the planes together
        show_res : bool
            Show the plot of the residuals
        res_offset : float
            Displacement of the residuals plot from 0, to be given as a fraction of the height of the experimental data.
            ``res_offset`` > 0 will move the residuals BELOW the zero-line!
        figdir : str
            Base path for the figure. This function will create a directory named `<figdir>/<what>` and will save all the plots therein.
            The default is `Figures_<self.filename>`.
        filename : str
            Determines the name of the directory where the figures will be saved. If None, `<self.filename>` is used
        ext : str
            Format of the saved figures
        dpi : int
            Resolution of the figures, in dots per inches
        dim : tuple
            Dimension of the figure in inches

        Returns
        -------
        None

        .. seealso::

            :func:`klassez.fit.plot_fit_dosy`

            :func:`klassez.fit.plot_fit_dosy_multi`
        """
        # select the correct object to plot
        if what == 'iguess':
            regions = deepcopy(self.i_guess)
        elif what == 'result':
            regions = deepcopy(self.result)
        else:
            raise ValueError('Specify what you want to plot: "iguess" or "result"')

        # Use the default figdir
        if figdir is None:
            base = Path(f'Figures_{self.filename}') / what
        else:
            base = Path(figdir) / what

        # Create the directories, if they do not exist
        base.mkdir(exist_ok=True, parents=True)

        print(f'Saving figures in "{base}".', c='tab:cyan')

        # Get the fitted traces
        totals, components = self.get_fit_lines(what)

        for k, (total, comps, region) in enumerate(zip(totals, components, regions)):
            print(f'Saving figures of the {region["label"]} region ({k+1}/{len(regions)})...')
            # Get the experimental data from the one inside
            exp_data = self.data[region['label']]
            # Example: testspectrum-R_2.000:1.641_all
            if filename is None:
                fn = f'{self.filename}-R_{region["label"]}_all'
            else:
                fn = f'{filename}-R_{region["label"]}_all'
            # Make the super giga figure
            fit.plot_fit_dosy_multi(self.g, exp_data, total, comps, region,
                                    bigdeltas=self.dosy_par['D'],
                                    filename=base / fn,
                                    ext=ext, dpi=dpi)

            # Break here
            if only_all:
                continue

            # Make the minifigures
            for p, plane in enumerate(self.planes):
                # Adjust the filename
                if filename is None:
                    # Use filename of the plane itself, which already contains the p
                    fn = plane.filename
                else:
                    # Use the given filename, but add the p
                    fn = f'{filename}_p{p}'
                fn += '-R_' + region['label']
                # Regions contains the I, the q and the f of all planes!
                region_p = {
                        'I': region['I'][p],
                        'q': region['q'][p],
                        'diff_c': region['diff_c'],
                        'diff_e': region['diff_e'],
                        'diff_f': region['diff_f'][p],
                        }

                # Make the minifigure
                fit.plot_fit_dosy(self.g, region['label'], exp_data[p], total[p], comps[p], region_p,
                                  show_res=show_res, res_offset=res_offset,
                                  filename=base/fn,
                                  ext=ext, dpi=dpi, dim=dim)

        print('Done.\n', c='tab:cyan')

    def get_fit_lines(self, what='result'):
        """
        Computes the fit lines using the internal methods of :class:`klassez.fit.DosyFit`, and then collects it together.

        Parameters
        ----------
        what : str
            ``'iguess'`` or ``'result'``

        Returns
        -------
        totals : 3darray
            Collection of the total dosy profiles, per plane. Dimension: ``(number of planes, number of regions, pts)``
        components : 4darray
            Collection of the components that generate the correspondant total trace. Dimension: ``(number of planes, number of regions, number of components, pts)``

        .. seealso::

            :func:`klassez.fit.DosyFit.get_fit_lines`
        """
        # Unpack the Parameters object into a normal dictionary
        if what == 'iguess':
            dic = self.i_guess
        elif what == 'result':
            dic = self.result
        else:
            raise NameError('what must be either "iguess" or "result"')

        Np = self._data.shape[1]    # Number of planes
        # Nr = self._data.shape[0]    # Number of regions

        all_totals, all_components = [], []     # Placeholders
        for r, region in enumerate(dic):
            # Make diffusion coefficient and fractions as lists
            diff_c = np.asarray(region['diff_c'])       # shape: (Np, )
            diff_f = np.asarray(region['diff_f'])       # shape: (Np, Nr)
            Nc = diff_c.shape[-1]

            # Compute the models
            yyc = np.array([
                    [region['q'][p] + region['I'][p] * diff_f[p, k] *
                     self.model(self.g, diff_c[k], k=p, **self.dosy_par)
                     for k in range(Nc)]
                    for p in range(Np)
                    ])

            # Sum them to get the total trace
            totals = np.sum(yyc, axis=1)

            # Apply intensity and offset
            all_totals.append(totals)
            all_components.append(yyc)

        return np.asarray(all_totals), np.asarray(all_components)

    def model(self, g, diff_c, gamma, D, d, tau=0, p90=0, k=None):
        """
        Calls ``self._model``, instanced by :func:`klassez.DosyFit_pp3D.select_model`, with the passed arguments and, if the bigdelta ``D`` is an array, to plane ``k``.

        Parameters
        ----------
        g : 1darray
            Gradient strength T/m
        diff_c : float
            Diffusion coefficient m^2/s
        gamma : float
            Gyromagnetic ratio MHz/T
        D : list or float
            Big delta seconds
        d : float
            Little delta seconds
        tau : float
            Tau seconds
        k : int or None
            Index for the difflist that corresponds to the actual bigdelta.
            If ``None`` and ``D`` is a sequence, all the models will be computed.

        Returns
        -------
        y : 1darray
            Computed model
        """
        # We need a copy to exploit @safe_kws in _model. We pass D explicitly
        in_par = dict(gamma=gamma, d=d, tau=tau, p90=p90)

        if isinstance(D, float):        # D is a number
            y = self._model(g, diff_c, D=D, **in_par)
        elif isinstance(k, int):        # D is a list -> kth entry
            y = self._model(g, diff_c, D=D[k], **in_par)
        else:                           # D is a list -> all entries
            y = np.array([
                self._model(g, diff_c, D=in_D, **in_par)
                for in_D in D])
        return y

    def select_model(self, dste, bp):
        """
        Select the model on the following scheme:

        -   ``dste = False`` and ``bp = False``: :func:`klassez.fit.model_ste`
        -   ``dste = False`` and ``bp = True``: :func:`klassez.fit.model_stebp`
        -   ``dste = True`` and ``bp = False``: :func:`klassez.fit.model_dste`
        -   ``dste = True`` and ``bp = True``: :func:`klassez.fit.model_dstebp`

        The selected function will be stored in ``self.model``.

        Parameters
        ----------
        dste : bool
            Single (``False``) or double (``True``) stimulated echo
        bp : bool
            Uses bipolar gradients (``True``) or not (``False``).
        """
        if dste and bp:
            self._model = fit.model_dstebp
        elif dste and not bp:
            self._model = fit.model_dste
        elif not dste and bp:
            self._model = fit.model_stebp
        elif not dste and not bp:
            self._model = fit.model_ste
        else:
            raise NameError('Unrecognized model')
