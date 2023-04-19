#! /usr/bin/env python3

import os
import sys
import numpy as np
from scipy import linalg, stats
from scipy.spatial import ConvexHull
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

from . import fit, misc, sim, figures, processing
#from .__init__ import CM
from .config import CM, COLORS, cron


"""
Functions for performing fits.
"""

s_colors=[ 'tab:cyan', 'tab:red', 'tab:green', 'tab:purple', 'tab:pink', 'tab:gray', 'tab:brown', 'tab:olive', 'salmon', 'indigo' ]

def f_t1(t, A, B, T1):
    """
    Function that models the buildup of magnetization due to T1 relaxation:
        $ f(t) = A (1 - e^{-t/T_1}) + B $
    """
    f = A * (1 - np.exp(-t/T1) ) + B
    return f

def f_t2(t, A, B, T2):
    """
    Function that models the decay of magnetization due to T2 relaxation
        $ f(t) = A e^{-t/T_2} + B $
    """
    f = A * np.exp(-t/T2) + B
    return f

def histogram(data, nbins=100, density=True, f_lims= None, xlabel=None, x_symm=False, name=None):
    """
    Computes an histogram of 'data' and tries to fit it with a gaussian lineshape.
    The parameters of the gaussian function are calculated analytically directly from 'data'
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
    - name : str
        name for the figure to be saved
    -------
    Returns:
    - m : float
        Mean of data
    - s : float
        Standard deviation of data.
    """

    if len(data.shape) > 1:
        data = data.flatten()

    if x_symm:
        lims = (- max(np.abs(data)), max(np.abs(data)) )
    else:
        lims = (min(data), max(data))
    
    hist, bin_edges = np.histogram(data, bins=nbins, range=lims, density=density)   # Computes the bins for the histogram

    lnspc = np.linspace(lims[0], lims[1], len(data))        # Scale for a smooth gaussian
    m, s = stats.norm.fit(data)                                 # Get mean and standard deviation of 'data'
   
    if density:
        A = 1
    else:
        A = np.trapz(hist, dx=bin_edges[1]-bin_edges[0])    # Integral
    fit_g = A / (np.sqrt(2 * np.pi) * s) * np.exp(-0.5 * ((lnspc - m) / s)**2) # Gaussian lineshape

    fig = plt.figure()
    fig.set_size_inches(3.96, 2.78)
    ax = fig.add_subplot(1,1,1)
    ax.hist(data, color='tab:blue', density=density, bins=bin_edges) 
    ax.plot(lnspc, fit_g, c='r', lw=0.6, label = '$\mu = ${:.3g}'.format(m)+'\n$\sigma = ${:.3g}'.format(s))
    ax.tick_params(labelsize=7)
    ax.ticklabel_format(axis='both', style='scientific', scilimits=(-3,3), useMathText=True)
    ax.yaxis.get_offset_text().set_size(7)
    ax.xaxis.get_offset_text().set_size(7)
    if density:
        ax.set_ylabel('Normalized count', fontsize=8)
    else:
        ax.set_ylabel('Count', fontsize=8)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=8)
    if f_lims:
        ax.set_xlim(f_lims)
    ax.legend(loc='upper right', fontsize=6)
    fig.tight_layout()
    if name:
        plt.savefig(name+'.png', format='png', dpi=600)
    else:
        plt.show()
    plt.close()

    return m, s

def ax_histogram(ax, data0, nbins=100, density=True, f_lims= None, xlabel=None, x_symm=False, barcolor='tab:blue'):
    """
    Computes an histogram of 'data' and tries to fit it with a gaussian lineshape.
    The parameters of the gaussian function are calculated analytically directly from 'data'
    --------
    Parameters:
    - ax : matplotlib.subplot Object
        panel of the figure where to put the histogram
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
    - name : str
        name for the figure to be saved
    -------
    Returns:
    - m : float
        Mean of data
    - s : float
        Standard deviation of data.
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

    lnspc = np.linspace(lims[0], lims[1], len(data))        # Scale for a smooth gaussian
    m, s = stats.norm.fit(data)                                 # Get mean and standard deviation of 'data'
   
    if density:
        A = 1
    else:
        A = np.trapz(hist, dx=bin_edges[1]-bin_edges[0])    # Integral
    fit_g = A / (np.sqrt(2 * np.pi) * s) * np.exp(-0.5 * ((lnspc - m) / s)**2) # Gaussian lineshape

    ax.hist(data, color=barcolor, density=density, bins=bin_edges) 
    ax.plot(lnspc, fit_g, c='r', lw=0.6, label = 'Theoretical values:\n$\mu = ${:.3g}'.format(m)+'\n$\sigma = ${:.3g}'.format(s))


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

    misc.mathformat(ax, limits=(-3,3))

    ax.legend(loc='upper right')

    misc.set_fontsizes(ax, 10)

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

def fit_int(y, y_c):
    """
    Calculate the intensity according to the least square fit as:
        I = sum( obs * calc ) / sum( calc^2 )
    --------
    Parameters:
    - y: ndarray
        Observed data.
    - y_c: ndarray
        Calculated data
    --------
    Returns:
    - I: float
        Calculated intensity
    """
    I = np.sum(y * y_c, axis=-1) / np.sum(y_c**2, axis=-1)
    return I

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

        L.set_xdata(left)
        R.set_xdata(right)
        if rev:
            ax.set_xlim(left+25*res, right-25*res)
        else:
            ax.set_xlim(left-25*res, right+25*res)
        T = max(data_inside.real)
        B = min(data_inside.real)
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

    # Creation of interactive figure panel
    fig = plt.figure(1)
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
    ax.set_xlabel('$\delta\,$ /ppm')
    ax.set_ylabel('Intensity /a.u.')
    L = ax.axvline(x=left, lw=0.5, c='r')           # Left selector
    R = ax.axvline(x=right, lw=0.5, c='g')          # Right selector

    # Call the 'update' functions upon interaction with the widgets
    left_slider.on_changed(update_region)
    right_slider.on_changed(update_region)
    button.on_clicked(save)
    l_tbox.on_submit(on_submit_l) 
    r_tbox.on_submit(on_submit_r) 

    misc.set_fontsizes(ax, 14)

    plt.show()
    plt.close(1)

    return left, right



def make_signal(t, u, s, k, x_g, phi, A, SFO1=701.125, o1p=0, N=None):
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
    - x_g : float
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
    sgn = sim.t_voigt(t, U, S, A=A*k, phi=phi, x_g=x_g) # make the signal
    if isinstance(N, int):
        sgn = processing.zf(sgn, N)         # zero-fill it
    sgn = processing.ft(sgn)                # transform it
    return sgn


def voigt_fit(S, ppm_scale, V, C, t_AQ, limits=None, SFO1=701.125, o1p=0, utol=0.5, vary_phi=False, vary_xg=True, vary_basl=True, hist_name=None, write_out='fit.out', test_res=True):
    """
    Fits an NMR spectrum with a set of signals, whose parameters are specifed in the V matrix.
    There is the possibility to use a baseline through the parameter C.
    The signals are computed in the time domain and then Fourier transformed.
    --------
    Parameters:
    - S : 1darray
        Spectrum to be fitted
    - ppm_scale : 1darray
        Self-explanatory
    - V : 2darray
        matrix (# signals, parameters)
    - C : 1darray or False
        Coefficients of the polynomion to be used as baseline correction. If the 'baseline' checkbox in the interactive figure panel is not checked, C_f is False.
    - t_AQ : 1darray
        Acquisition timescale
    - limits : tuple or None
        Trim limits for the spectrum (left, right). If None, the whole spectrum is used.
    - SFO1 : float
        Larmor frequency /MHz
    - o1p : float
        pulse carrier frequency /ppm
    - utol : float
        tolerance for the chemical shift. The peak center can move in the range [u-utol, u+utol].
    - vary_xg: bool
        If it is False, the parameter x_g cannot be varyied during the fitting procedure. Useful when fitting with pure Gaussians or pure Lorentzians.
    - vary_basl: bool
        If it is False, the baseline is kept fixed at the initial parameters.
    -------
    Returns:
    - C_f : 1darray or False
        Coefficients of the polynomion to be used as baseline correction, or just False if not used.
    - V_f : 2darray
        matrix (# signals, parameters) after the fit
    - result : lmfit.fit_result
        container of all information on the fit
    """

    ns = V.shape[0]         # Number of signals
    # the baseline is used if C is an array
    if isinstance(C, list):
        C = np.array(C)
    use_basl = isinstance(C, np.ndarray)
    
    # Compute limits and get indexes on ppm_scale
    if limits is None:
        limits = [max(ppm_scale), min(ppm_scale)]
    lim1 = misc.ppmfind(ppm_scale, limits[0])[0]
    lim2 = misc.ppmfind(ppm_scale, limits[1])[0]
    lim1, lim2 = min(lim1, lim2), max(lim1, lim2)
    
    # Total integral and spectral width, to be used as limits for the fit
    I = np.trapz(S[lim1:lim2], dx=misc.calcres(ppm_scale))
    SW = misc.ppm2freq(np.abs(ppm_scale[0]-ppm_scale[-1]), B0=SFO1, o1p=o1p)

    # polynomion x-scale
    x = np.linspace(0, 1, ppm_scale[lim1:lim2].shape[-1])[::-1]

    # Filling up the Parameters dictionary
    param = l.Parameters()
    peak_names = ['u', 'fwhm', 'k', 'x_g', 'phi']   # signal entries
    poly_names = ['a', 'b', 'c', 'd', 'e']          # baseline polynomion coefficients

    minima = np.array([         # lower thresholds
            [u-utol for u in V[:,0]],      # chemical shift
            [0 for s in range(ns)],        # fwhm
            [0 for k in range(ns)],        # rel int
            [0-1e-5 for xg in range(ns)],       # xg
            [-180 for phi in range(ns)]    # phase angle
            ]).T
    maxima = np.array([         # upper thresholds
            [u+utol for u in V[:,0]],      # chemical shift
            [SW for s in range(ns)],       # fwhm
            [5 for k in range(ns)],        # rel int
            [1+1e-5 for xg in range(ns)],       # xg
            [180 for phi in range(ns)]     # phase angle
            ]).T

    for i in range(V.shape[0]): # put variables in the dictionary
        idx = str(i+1)
        for j in range(len(peak_names)):
            param.add(peak_names[j]+idx, value=V[i,j], min=minima[i,j], max=maxima[i,j])
        param['x_g'+idx].set(vary=vary_xg)
        param['phi'+idx].set(vary=vary_phi)
    param.add('A', value=V[0,-1], vary=False)   # Unique, got from first row

    if C is not False:  # Add polynomion
        lim_poly = np.array([1e1, 1e1, 1e1, 1e1, 1e1])
        for i in range(len(poly_names)):
            param.add(poly_names[i], value=C[i], min=C[i]-lim_poly[i], max=C[i]+lim_poly[i], vary=vary_basl)


    def f2min_real(param, S, use_basl=False):
        # cost function for the fit.
        N = S.shape[-1]

        # unpack V and C from dictionary
        param = param.valuesdict()
        V = fit.dic2mat(param, peak_names, ns, param['A'])
        if use_basl is not False:
            C_in = np.array([param[w] for w in poly_names])
            y = misc.polyn(x, C_in)
        else:
            y = np.zeros_like(x)

        # Compute only total signal
        sgn = np.zeros(len(x))
        for i in range(V.shape[0]):
            temp_sgn = make_signal(t_AQ, *V[i], SFO1, o1p, N).real
            sgn += temp_sgn[lim1:lim2]

        # Calculate residual
        R = y + sgn - S[lim1:lim2]
        return R

    def f2min_cplx(param, S, use_basl=False):
        # cost function for the fit.
        N = S.shape[-1]

        # unpack V and C from dictionary
        param = param.valuesdict()
        V = fit.dic2mat(param, peak_names, ns, param['A'])
        if use_basl is not False:
            C_in = np.array([param[w] for w in poly_names])
            y = misc.polyn(x, C_in) + 1j*misc.polyn(x, C_in)
        else:
            y = np.zeros_like(x)

        # Compute only total signal
        sgn = np.zeros(len(x)).astype(S.dtype)
        for i in range(V.shape[0]):
            temp_sgn = make_signal(t_AQ, *V[i], SFO1, o1p, N)
            sgn += temp_sgn[lim1:lim2]

        # Calculate residual
        R = y + sgn - S[lim1:lim2]
        R_tot = np.concatenate((R.real, R.imag), axis=-1)
        return R_tot
        
    # Fit 
    print('Starting fit...')
    start_time = datetime.now()
    if np.iscomplexobj(S):
        minner = l.Minimizer(f2min_cplx, param, fcn_args=(S, use_basl))
    else:
        minner = l.Minimizer(f2min_real, param, fcn_args=(S, use_basl))
    result = minner.minimize(method='leastsq', max_nfev=10000, xtol=1e-15, ftol=1e-15)
    end_time = datetime.now()
    runtime = end_time - start_time

    print('{} Total runtime: {}.\nNumber of function evaluations: {:5.0f}'.format(result.message, runtime, result.nfev))
    
    popt = result.params.valuesdict()       # final parameters

    # Put all the results in final variables
    V_f = dic2mat(popt, peak_names, ns, popt['A'])
    if use_basl is True:
        C_f = np.array([popt[w] for w in poly_names])
    else:
        C_f = False

    # Print the parameters
    print_par(V_f, C_f, limits=limits)
    if isinstance(write_out, str):
        print('These values are saved in: {}'.format(write_out))
        write_par(V_f, C_f, limits=limits, filename=write_out)
    
    # Check for the gaussianity of the residual
    if test_res is True:
        # Get the info 
        if np.iscomplexobj(S):
            Npt = len(result.residual)
            R = result.residual[:Npt//2] + 1j*result.residual[Npt//2:]
            SYSDEV, Q_G = fit.test_residuals(R.real)
        else:
            R = result.residual
            SYSDEV, Q_G = fit.test_residuals(R)


        # Make the figure of the residual
        if hist_name:
            fig = plt.figure()
            fig.set_size_inches(6.60, 2.56)
            plt.subplots_adjust(left=0.10, bottom=0.15, top=0.90, right=0.95, wspace=0.30)
            axr = fig.add_subplot(1,2,1) 
            axh = fig.add_subplot(1,2,2) 
            
            axr.set_title('Fit residual')
            axr.plot(ppm_scale[lim1:lim2], R.real, c='tab:blue', lw=0.5)
            axr.axhline(0, c='k', lw=0.25)
            axr.set_xlabel('$\delta\,$ /ppm')
            axr.set_ylabel('Intensity /a.u.')
            misc.mathformat(axr)
            misc.pretty_scale(axr, (ppm_scale[lim1], ppm_scale[lim2]))
            Rlims = np.ceil(max(np.abs(R)))
            misc.set_ylim(axr, [S, R, -S, -R])
            misc.pretty_scale(axr, axr.get_ylim(), axis='y', n_major_ticks=8) 

            axh.set_title('Histogram of residual')
            # Compute the number of bins
            if R.shape[-1] < 200:
                n_bins = 20
            else:
                n_bins = R.shape[-1] // 10
            if n_bins > 2500:
                n_bins = 2500
            m_R, s_R = fit.ax_histogram(axh, R.real, nbins=n_bins, density=False, x_symm=True, barcolor='tab:blue')
            axh.axvline(0, c='k', lw=0.25)
            axh.set_xlabel('Intensity /a.u.')
            misc.pretty_scale(axh, axh.get_xlim(), axis='x', n_major_ticks=8) 
            misc.pretty_scale(axh, axh.get_ylim(), axis='y')
            misc.mathformat(axh, 'both')

            misc.set_fontsizes(axr, 10)
            misc.set_fontsizes(axh, 10)
            plt.savefig(hist_name+'.png', dpi=300)
            plt.close()
        else:
            fig = plt.figure()
            ax = fig.add_subplot()
            m_R, s_R = fit.ax_histogram(ax, R.real, nbins=100, density=False, x_symm=True, barcolor='tab:blue')
            plt.close()

        # Plot the statistics
        print('-' * 60)
        print('{:^60}'.format('Statistics of the fit'))
        print('{:<30} = {:=9.3e} | Optimal : 0'.format('Mean of residuals', m_R)) 
        print('{:<30} = {:=9.3e} | Optimal : 0'.format('Mean/STD of residuals', m_R / s_R)) 
        print('{:<30} = {:+9.3e} | Optimal : 1'.format('Systematic deviation', SYSDEV)) 
        print('{:<30} = {:+9.3e} | Optimal : 1'.format('Gaussianity of residuals', Q_G)) 
        print('-' * 60)

    return V_f, C_f, result, runtime


@cron
def voigt_fit_2D(x_scale, y_scale, data, parameters, lim_f1, lim_f2, acqus, N=None, procs=None, utol=(1,1), s1tol=(0,500), s2tol=(0,500), vary_xg=False, logfile=None):
    """
    Function that performs the fit of a 2D peak using multiple components. 
    The program reads a parameter matrix, that contains:
        u1 /ppm, u2 /ppm, fwhm1 /Hz, fwhm2 /Hz, I /a.u., x_g
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
    - vary_xg: bool
        Choose if to fix the x_g value or not
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
                P[f'xg_{i+1}'],
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
        param.add(f'xg_{i+1}', value=parameters[i,5], min=0-1e-5, max=1+1e-5)   # Fraction of gaussianity
    
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
    if vary_xg is False:    # fix it
        [param[f'xg_{i+1}'].set(vary=False) for i in range(n_sgn)]
        
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
    keys_list = 'u1', 'u2', 's1', 's2', 'k', 'xg'
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
        popt[f'xg_{i+1}'],
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
    fig = plt.figure(1)
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

def integrate(ppm0, data0, X_label='$\delta\,$F1 /ppm'):
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
    fig = plt.figure()
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
            'ref_int': int_f[-1],            # Reference peak integral, absolute value
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
        int_val = int_fun[-1]                                               # Value of the integral
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


def make_iguess(S_in, ppm_scale, t_AQ, limits=None, SFO1=701.125, o1p=0, rev=True, name='i_guess.inp'):
    """
    Compute the initial guess for the quantitative fit of 1D NMR spectrum in an interactive manner. 
    When the panel is closed, the values are saved in a file.
    -------
    Parameters:
    - S : 1darray
        Spectrum to be fitted
    - ppm_scale : 1darray
        Self-explanatory
    - t_AQ : 1darray
        Acquisition timescale
    - limits : tuple or None
        Trim limits for the spectrum (left, right). If None, the whole spectrum is used.
    - SFO1 : float
        Larmor frequency /MHz
    - o1p : float
        pulse carrier frequency /ppm
    - rev : bool
        choose if you want to reverse the x-axis scale (True) or not (False).
    - name : str
        name of the file where to save the parameters
    -------
    Returns:
    - V_f : 2darray
        matrix (# signals, parameters)
    - C_f : 1darray or False
        Coefficients of the polynomion to be used as baseline correction. If the 'baseline' checkbox in the interactive figure panel is not checked, C_f is False.
    """
    if np.iscomplexobj(S_in):
        S = np.copy(S_in).real
    else:
        S = np.copy(S_in)

    N = S.shape[-1]

    # Set limits according to rev
    if limits is None:
        if rev is True:
            limits = [max(ppm_scale), min(ppm_scale)]
        else:
            limits = [min(ppm_scale), max(ppm_scale)]

    
    # Get index for the limits
    lim1 = misc.ppmfind(ppm_scale, limits[0])[0]
    lim2 = misc.ppmfind(ppm_scale, limits[1])[0]
    I = np.trapz(S[lim1:lim2], dx=misc.calcres(ppm_scale))/SFO1

    # make boxes for widgets
    poly_box = plt.axes([0.72, 0.10, 0.10, 0.3])
    slider_box = plt.axes([0.68, 0.10, 0.01, 0.65])
    peak_box = plt.axes([0.72, 0.45, 0.10, 0.3])
    su_box = plt.axes([0.815, 0.825, 0.08, 0.075])
    giu_box = plt.axes([0.894, 0.825, 0.08, 0.075])
    save_box = plt.axes([0.7, 0.825, 0.085, 0.04])
    reset_box = plt.axes([0.7, 0.865, 0.085, 0.04])
    p_or_s_box = plt.axes([0.73, 0.78, 0.04, 0.03])
    check_box = plt.axes([0.85, 0.1, 0.1, 0.7])
    
    # Make widgets
    #   Buttons
    up_button = Button(su_box, '$\\uparrow$', hovercolor = '0.975')    
    down_button = Button(giu_box, '$\\downarrow$', hovercolor = '0.975')
    save_button = Button(save_box, 'SAVE', hovercolor = '0.975')
    reset_button = Button(reset_box, 'RESET', hovercolor = '0.975')
    
    #   Radio
    poly_name = ['a', 'b', 'c', 'd', 'e']
    poly_radio = RadioButtons(poly_box, poly_name, activecolor='tab:orange')       # Polynomion
    
    peak_name = ['$\delta$ /ppm', '$\Gamma$ /Hz', '$k$', '$x_{g}$', '$\phi$', '$A$']
    peak_radio = RadioButtons(peak_box, peak_name, activecolor='tab:blue')      # Signal parameters
    
    #   Sliders
    slider = Slider(ax = slider_box, label = 'Active\nSignal', valmin = 1, valmax = 10, valinit = 1, valstep = 1, orientation='vertical', color='tab:blue')
    p_or_s = Slider(p_or_s_box, '', valmin=0, valmax=1, valinit=0, valstep=1, track_color='tab:blue', color='tab:orange')

    #   Checkbox
    check_name = [str(w+1) for w in range(10)]+['Basl']
    check_status = [True] + [False for w in range(10)]
    check = CheckButtons(check_box, check_name, check_status)
    
    # Create variable for the 'active' status
    stats = [np.zeros(len(peak_name)), np.zeros(len(poly_name))]
    #    u   s   k   xg  phi A
    #    a   b   c   d   e
    stats[0][0] = 1
    stats[1][0] = 1

    # Initial values
    #   Polynomion coefficients
    C = np.zeros(len(poly_name))  
    #   Parameters of the peaks
    V = np.array([[(limits[0]+limits[1])/2, SFO1*0.5, 1, 1, 0, I/10] for w in range(10)])
    V_init = np.copy(V)   # Save for reset
    
    #   Increase step for the polynomion (order of magnitude)
    om = np.zeros(len(poly_name))
    #   Increase step for the peak parameters
    sens = np.array([0.1, 20, 0.05, 0.1, 5, I/100])    
    sens_init = np.copy(sens)        # Save for reset
    
    # Functions connected to the widgets
    def statmod(label):
        # Sets 'label' as active modifying 'stats'
        nonlocal stats
        if label in peak_name:  # if signal
            stats[0] = np.zeros(len(peak_name))
            for k, L in enumerate(peak_name):
                if label == L:
                    stats[0][k] = 1
        elif label in poly_name:    # if baseline
            stats[1] = np.zeros(len(poly_name))
            for k, L in enumerate(poly_name):
                if label == L:
                    stats[1][k] = 1
        update(0)       # Call update to redraw the figure
                
    def roll_up_p(event):
        # Increase polynomion with mouse scroll
        nonlocal C
        for k in range(len(poly_name)):
            if stats[1][k]:
                C[k]+=10**om[k]
                
    def roll_down_p(event):
        # Decrease polynomion with mouse scroll
        nonlocal C
        for k in range(len(poly_name)):
            if stats[1][k]:
                C[k]-=10**om[k]
    
    def up_om(event):
        # Increase the om of the active coefficient by 1
        nonlocal om
        for k in range(len(poly_name)):
            if stats[1][k]:
                om[k] += 1
        
    def down_om(event):
        # Decrease the om of the active coefficient by 1
        nonlocal om
        for k in range(len(poly_name)):
            if stats[1][k]:
                om[k] -= 1
                
    def roll_up_s(event):
        # Increase signal with mouse scroll
        nonlocal V
        maxima = [np.inf, np.inf, np.inf, 1, 180, np.inf]
        for k in range(len(peak_name)):
            if stats[0][k]:
                V[slider.val-1,k]+=sens[k]
                if V[slider.val-1,k]>=maxima[k]:
                    V[slider.val-1,k]=maxima[k]
        V[:,-1] = V[slider.val-1,-1]
                
    def roll_down_s(event):
        # Decrease signal with mouse scroll
        nonlocal V
        minima = [-np.inf, 0, 0, 0, -180, 0]
        for k in range(len(peak_name)):
            if stats[0][k]:
                V[slider.val-1,k]-=sens[k]
                if V[slider.val-1,k]<=minima[k]:
                    V[slider.val-1,k]=minima[k]
        V[:,-1] = V[slider.val-1,-1]
    
    def up_sens(event):
        # Doubles increase step
        nonlocal sens
        for k in range(len(peak_name)):
            if stats[0][k]:
                sens[k] *= 2
        
    def down_sens(event):
        # Halves increase step
        nonlocal sens
        for k in range(len(peak_name)):
            if stats[0][k]:
                sens[k] /= 2
                
    def switch_up(event):
        # Fork function for mouse scroll up
        if p_or_s.val == 0:
            up_sens(event)
        elif p_or_s.val == 1:
            up_om(event)
            
    def switch_down(event):
        # Fork function for mouse scroll down
        if p_or_s.val == 0:
            down_sens(event)
        elif p_or_s.val == 1:
            down_om(event)
            
    def on_scroll(event):
        # Mouse scroll
        if event.button == 'up':
            if p_or_s.val==0:
                roll_up_s(event)
            elif p_or_s.val==1:
                roll_up_p(event)
        elif event.button == 'down':
            if p_or_s.val==0:
                roll_down_s(event)
            elif p_or_s.val==1:
                roll_down_p(event)
        update(0)
                
    def set_visible(label):
        # Set line visible or invisible according to the checkbox
        index = check_name.index(label)
        if index < 10:
            s_plot[index].set_visible(not s_plot[index].get_visible())
            if s_plot[index].get_visible():
                slider.set_val(index+1)
        elif index == 10:
            poly_plot.set_visible(not poly_plot.get_visible())
        update(0)

    def head_color(null):
        if p_or_s.val:
            head_print.set_color(s_colors[-1])
        else:
            w = slider.val - 1
            head_print.set_color(s_colors[w])
        plt.draw()

    
    # polynomion
    x = np.linspace(0, 1, ppm_scale[lim1:lim2].shape[-1])[::-1]
    y = np.zeros_like(x)

    # Signals
    sgn = np.zeros((V.shape[0], ppm_scale.shape[-1]))       # array for the single signals
    Total = np.zeros(len(ppm_scale[lim1:lim2]))                    # total function
    for i in range(V.shape[0]):
        sgn[i] = make_signal(t_AQ, V[i,0], V[i,1], V[i,2], V[i,3], V[i,4], V[i,5], SFO1, o1p, N).real
        if check_status[i]:
            Total += sgn[i][lim1:lim2].real
    

    # Initial figure
    fig = plt.figure(1)
    fig.set_size_inches(15,8)
    plt.subplots_adjust(bottom=0.10, top=0.90, left=0.05, right=0.65)
    ax = fig.add_subplot(1,1,1)

    ax.plot(ppm_scale[lim1:lim2], S[lim1:lim2], label='Experimental', lw=1.0, c='k')  # experimental

    # signals, total, polynomion
    s_plot=[]
    for i in range(V.shape[0]):
        temp, = ax.plot(ppm_scale[lim1:lim2], sgn[i][lim1:lim2].real, c=s_colors[i], lw=1.0, ls='--')
        s_plot.append(temp)
    total_plot, = ax.plot(ppm_scale[lim1:lim2], y+Total, label = 'Fit', c='tab:blue', lw=1.0)
    poly_plot, = ax.plot(ppm_scale[lim1:lim2], y, label = 'Baseline', lw=0.8, c='tab:orange')
    s_colors.append(poly_plot.get_color())

    # make pretty scale
    ax.set_xlim(max(limits[0],limits[1]),min(limits[0],limits[1]))
    misc.pretty_scale(ax, ax.get_xlim(), axis='x', n_major_ticks=10)

    # Header for current values print
    head_print = ax.text(0.1, 0.04, 
            '{:_>7s} ,{:_>5} ,{:_>5} ,{:_>5} ,{:_>7} ,{:_>11} | {:_^11}, {:_^11}, {:_^11}, {:_^11}, {:_^11}'.format(
                'u', 's', 'k', 'xg', 'phi', 'A', 'a', 'b', 'c', 'd', 'e'),
            ha='left', va='bottom', transform=fig.transFigure, fontsize=10, color=s_colors[0])
    values_print = ax.text(0.1, 0.01,
            '{:+7.2f}, {:5.0f}, {:5.3f}, {:5.2f}, {:+07.2f}, {:5.2e} | {:+5.2e}, {:+5.2e}, {:+5.2e}, {:+5.2e}, {:+5.2e}'.format(
                V[0,0], V[0,1],V[0,2],V[0,3],V[0,4],V[0,5], C[0], C[1], C[2], C[3], C[4]),
            ha='left', va='bottom', transform=fig.transFigure, fontsize=10)

    # Customize checkbox appearance
    #   make boxes more squared
    HBOX = check_box.dataLim.bounds[-1]
    misc.edit_checkboxes(check, xadj=0, yadj=0.001, length=0.1, height=(HBOX-0.5*HBOX)/len(check_name), color=s_colors)

    def update(val):
        # Calculates and draws all the figure elements
        Total_inside = np.zeros_like(Total)
        check_status = check.get_status()
        sgn = []
        if check_status[-1]:    # baseline check
            y = misc.polyn(x, C)
        else:
            y = np.zeros_like(x)

        # Make the signals
        for i in range(V.shape[0]):
            if check_status[i]:
                sgn.append(make_signal(t_AQ, V[i,0], V[i,1], V[i,2], V[i,3], V[i,4], V[i,5], SFO1, o1p, N))
                Total_inside += sgn[i][lim1:lim2].real
            else:
                sgn.append(np.zeros_like(ppm_scale))

        # Update the plot
        for i in range(V.shape[0]):
            if check_status[i]:
                s_plot[i].set_ydata(sgn[i][lim1:lim2].real)
        poly_plot.set_ydata(y)
        total_plot.set_ydata(y+Total_inside)
        w = slider.val - 1 
        # print the current values
        values_print.set_text(
                '{:+7.2f}, {:5.0f}, {:5.3f}, {:5.2f}, {:+07.2f}, {:5.2e} | {:+5.2e}, {:+5.2e}, {:+5.2e}, {:+5.2e}, {:+5.2e}'.format(
                    V[w,0], V[w,1],V[w,2],V[w,3],V[w,4],V[w,5], C[0], C[1], C[2], C[3], C[4]))
        plt.draw()
    
    def reset(event):
        # Sets all the widgets to their starting values
        nonlocal C, V, om, sens 
        C = np.zeros(len(poly_name))
        V = np.copy(V_init)
        om = np.zeros_like(C)
        sens = np.copy(sens_init)
        update(0)       # to update the figure
    
    # Declare variables to store the final values
    V_f = []
    C_f = np.zeros_like(C)
    def save(event):
        # Put current values in the final variables that are returned
        nonlocal V_f, C_f
        V_f = []
        check_status=check.get_status()
        for i in range(len(check_status)-1):    # last one is baseline
            if check_status[i]:
                V_f.append(V[i])    
        V_f = np.array(V_f)
        if check_status[i]:
            C_f = np.copy(C)


    # Connect widgets to functions
    poly_radio.on_clicked(statmod)
    peak_radio.on_clicked(statmod)
    up_button.on_clicked(switch_up)
    down_button.on_clicked(switch_down)
    scroll = fig.canvas.mpl_connect('scroll_event', on_scroll)
    check.on_clicked(set_visible)
    save_button.on_clicked(save)
    reset_button.on_clicked(reset)
    p_or_s.on_changed(head_color)
    slider.on_changed(head_color)

    # Ruler for slider
    for i, H in enumerate(np.linspace(0.10, 0.75, 10)):
        plt.text(0.685, H, '$-$', ha='center', va='center', fontsize=20, color=s_colors[i], transform=fig.transFigure)
    
    # Set visibility 
    for i in range(V.shape[0]):
        s_plot[i].set_visible(check_status[i])
    poly_plot.set_visible(False)

    ax.legend()
    plt.show()

    # correct the intensities
    V_f[:,2], Acorr = misc.molfrac(V_f[:,2])
    V_f[:,-1] *= Acorr


    # Write the input file and return the values
    check_status=check.get_status()
    if check_status[-1]:
        write_par(V_f, C_f, limits, filename=name)
        return V_f, C_f
    else:
        write_par(V_f, False, limits, filename=name)
        return V_f, False 



# --------------------------------------------------------------------

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
    f.write('{:<4}\t{:>7}\t{:>5}\t{:>5}\t{:>5}\t{:>5}\t{:<9}\n'.format('#', 'u', 's', 'k', 'x_g', 'phi', 'A'))
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
    print('{:<4}\t{:>7}\t{:>5}\t{:>5}\t{:>5}\t{:>5}\t{:<9}'.format('#', 'u', 's', 'k', 'x_g', 'phi', 'A'))
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
    #   u   s   k   xg  phi A   
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


def test_residuals(R, nbins=100, density=False):
    """
    Test the residuals of a fit to see if it was reliable.
    Returns two values, SYSDEV and Q_G.
    SYSDEV is inspired by Svergun's Gnom, and it tells if there are systematic deviations basing on the number of sign changes in the residual. Optimal value must be 1.
    Q_G is to see the discrepancy between a gaussian function built with the mean and standard deviation of the residuals and the gaussian built fitting the histogram of the residuals. Values go from 0 (worst case) to 1 (best case).
    -------
    Parameters:
    - R : 1darray
        Array of the residuals
    - nbins : int
        number of bins of the histogram, i.e. the number of points that will be used to fit the histogram.
    - density : bool
        True to normalize the histogram, False otherwise.
    -------
    Returns:
    - SYSDEV : float
        Read full caption
    - Q_G : float
        Read full caption
    """
    # Get theoretical mean and std of the residue
    m_t, s_t = np.mean(R), np.std(R)
    
    # Calculate SYSDEV
    N_s = np.sum((np.diff(np.sign(R)) != 0)*1)
    SYSDEV = N_s / (len(R)/2)
    
    # Make histogram 
    hist, bin_edges = np.histogram(R, bins=nbins, density=density)   # Computes the bins for the histogram

    # Set A according to density
    if density:
        A_t = 1
    else:
        A_t = np.trapz(hist, dx=bin_edges[1]-bin_edges[0])    # Integral

    # center point of the bin bar
    x = np.array( [(bin_edges[i]+bin_edges[i+1])/2 for i in range(len(bin_edges)-1)])

    # Theoretical gaussian function and its integral
    G_t = sim.f_gaussian(x, m_t, s_t, A_t)
    I_t = np.trapz(G_t, dx=misc.calcres(x))

    # Fitted gaussian and its integral
    m_f, s_f, A_f = gaussian_fit(x, hist)
    G_f = sim.f_gaussian(x, m_f, s_f, A_f)
    I_f = np.trapz(G_f, dx=misc.calcres(x))
    
    # Calculate Q_G
    Q_G = np.trapz(np.abs(G_t - G_f), dx=misc.calcres(x))
    # Normalize it. 1- is to make it similar to SYSDEV
    Q_G = 1 - (Q_G / (I_t + I_f))

    return SYSDEV, Q_G

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
    - SFO1: float
        Larmor frequency of the nucleus
    - o1p : float
        Pulse carrier frequency
    - nuc : str or None
        Nucleus. Used to write the X_scale of the plot.
    - input_file : str
        filename of the input file
    - output_file : str
        filename of the output file
    - log_file : str
        filename of the log file
    - limits : tuple
        borders of the fitting window
    - Vi : 2darray
        array with the values of the signals used as initial guess
    - Ci : 1darray
        coefficients of the baseline polynomion as initial guess
    - Vf : 2darray
        array with the values of the signals after the fit
    - Cf : 1darray
        coefficients of the baseline polynomion after the fit
    - s_labels : list
        legend entries for the single signals.
    """

    def __init__(self, ppm_scale, S, t_AQ, SFO1, o1p, nuc=None):
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
        """
        self.ppm_scale = ppm_scale
        self.S = S
        self.t_AQ = t_AQ
        self.SFO1 = SFO1
        self.o1p = o1p
        if nuc is None:
            self.X_label = '$\delta\,$ /ppm'
        elif isinstance(nuc, str):
            fnuc = misc.nuc_format(nuc)
            self.X_label = '$\delta$ ' + fnuc +' /ppm'

    def iguess(self, input_file=None, limits=None):
        """
        Reads, or computes, the initial guess for the fit.
        The window limits can be set interactively or specified through the parameter 'limits'.
        --------
        Parameters:
        - input_file: str or None
            Path to the input file. If None, the default name 'ppm1:ppm2.inp' is set.
        - limits: tuple or None
            (lim_sx, lim_dx) in ppm. If None and input_file does not exist yet, they are set interactively.
        """
        # If input_file is not given, set in_file_exist to False.
        # If input_file is passed as argument, check if there is already, and set in_file_exist accordingly.
        if input_file is None:
            in_file_exist = False
        else:
            in_file_exist = os.path.exists(input_file)


        if in_file_exist is True:       # Read everything you need from the file
            self.Vi, self.Ci, self.limits = fit.read_par(input_file)
        else:                           # Make the initial guess interactively and save the file.
            # Get the limits interactively if they are not given, and the input file is to be created
            if limits is None:
                self.limits = fit.get_region(self.ppm_scale, self.S)
            else:
                self.limits = limits
            if input_file is None:      # If the input file name was not passed, set a default name
                input_file = f'{self.limits[0]:.2f}:{self.limits[1]:.2f}.inp'
            self.Vi, self.Ci = fit.make_iguess(self.S, self.ppm_scale, self.t_AQ, self.limits, self.SFO1, self.o1p, name=input_file)

        self.input_file = input_file
        print(f'{input_file} loaded as input file.')

    def load_fit(self, output_file=None):
        """
        Reads the output file of a fit and stores the values in self.Vf, self.Cf and self.limits.
        self.output_file is also replaced with output_file.
        ---------
        Parameters:
        - output_file: str
            Path to the output file to be read
        """
        out_file_exist = os.path.exists(output_file)
        if out_file_exist is True:       # Read everything you need from the file
            self.Vf, self.Cf, self.limits = fit.read_par(output_file)
            self.output_file = output_file
            print(f'{output_file} loaded as output file.')
        else:
            raise NameError(f'{output_file} does not exist.')

    def dofit(self, input_file=None, output_file=None, log_file='fit.log', on_cplex=True, **kwargs):
        """
        Performs the fit using fit.voigt_fit with self.Vi and self.Ci as initial parameters.
        --------
        Parameters:
        - input_file: str 
            Path to the input file. If None, self.input_file is used.
        - output_file: str 
            Path to the output file. If None, self.output_file is used.
        - log_file: str
            Path to the log file.
        - on_cplex: bool
            Choose if to fit on the complex data (True) or on the real data (False).
        - kwargs: keyworded parameters
            See fit.voigt_fit. 
        """

        if input_file is None:
            input_file = self.input_file
        if output_file is None:
            output_file = f'out_{self.limits[0]:.2f}:{self.limits[1]:.2f}'

        self.output_file = output_file
        if np.iscomplexobj(self.S) and on_cplex:
            S = np.copy(self.S)
        else:
            S = np.copy(self.S.real)

        self.Vf, self.Cf, self._result, self._runtime = fit.voigt_fit(
                S, self.ppm_scale, self.Vi, self.Ci, self.t_AQ, self.limits, self.SFO1, self.o1p,
                write_out=self.output_file, **kwargs)
     
        fit.write_log(input_file=self.input_file, output_file=self.output_file,
                limits=self.limits, V_i=self.Vi, C_i=self.Ci, V_f=self.Vf, C_f=self.Cf,
                result=self._result, runtime=self._runtime, test_res=True, log_file=log_file)

    def plot(self, what='fit', s_labels=None, **kwargs):
        """
        Makes a figure of either the initial guess or the fit.
        Calls figures.fitfigure.
        --------
        Parameters:
        - what: str
            'iguess' or 'fit'. self.Vi and self.Ci or self.Vf and self.Cf are read, respectively.
        - s_labels: list
            Labels for the components that will appear in the legend of the figure.
        - kwargs: keyworded parameters
            See figures.fitfigure for the additional parameters.
        """
        if what == 'iguess':
            V = self.Vi
            C = self.Ci
        elif what == 'fit':
            V = self.Vf
            C = self.Cf
        limits = self.limits
               
        if s_labels is not None:
            self.s_labels = s_labels

        if np.iscomplexobj(self.S):
            S = np.copy(self.S.real)
        else:
            S = np.copy(self.S)


        figures.fitfigure(S, self.ppm_scale, self.t_AQ,
                V, C, SFO1=self.SFO1, o1p=self.o1p, limits=limits, 
                s_labels=s_labels, X_label=self.X_label, **kwargs)
        
    def _join(self, files, joined_name=None, flag=0):
        joined = fit.join_par(files, self.ppm_scale, joined_name)
        self.limits = joined[2]
        if flag == 0:   #input
            self.Vi = joined[0]
            self.Ci = joined[1]
            self.ibasl = joined[3]
        elif flag == 1:   #output
            self.Vf = joined[0]
            self.Cf = joined[1]
            self.fbasl = joined[3]

    def get_fit_lines(self, what='fit'):
        """
        Calculates the components, total fit curve and baseline used for iguess or fit.
        Calls fit.calc_fit_lines to do it.
        --------
        Parameters:
        - what: str
            'iguess' or 'fit'. self.Vi and self.Ci or self.Vf and self.Cf are read, respectively.
        --------
        Returns:
        - signals: list of 1darray
            Components used for the fit
        - Total: 1darray
            Sum of all the signals
        - baseline: 1darray
            Baseline.
        """
        if what == 'iguess':
            V = self.Vi
            C = self.Ci
        elif what == 'fit':
            V = self.Vf
            C = self.Cf

        N = self.S.shape[-1]

        signals, Total, baseline = fit.calc_fit_lines(self.ppm_scale, self.limits, self.t_AQ, self.SFO1, self.o1p, N=N, V=V, C=C)

        return signals, Total, baseline


def gen_iguess(x, experimental, param, model, model_args=[]):
    """
    GUI for the interactive setup of a Parameters object to be used in a fitting procedure. 
    Once you initialized the Parameters object with the name of the parameters and a dummy value, you are allowed to set the value, minimum, maximum and vary status through the textboxes given in the right column, and see their effects in real time.
    Upon closure of the figure, the Parameters object with the updated entries is returned.
    A maximum of 18 parameters will fit the figure.
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
    ---------
    Returns:
    - param: lmfit.Parameters Object
        Updated Parameters Object
    """

    # Declare some stuff to be used multiple times
    L_box = 0.08        # Length of the textboxes
    H_box = 0.04        # Height of the textboxes

    #   Y position of the rows
    y0box = 0.85
    list_Y_box = []
    for k in range(len(param)):
        space = H_box + 0.01 
        list_Y_box.append(y0box - k*space)


    # ---------------------------------------------------------------------------------------
    # Functions connected to the widgets
    def update(text):
        """ Called upon writing something in the textboxes """
        def get_val(tb):
            """ Overwrite inf with np.inf otherwise raises error """
            if 'inf' in tb.text:
                return eval(tb.text.replace('inf', 'np.inf'))
            else:
                return eval(tb.text)

        nonlocal param
        # Read all textboxes at once and set Parameters accordingly
        for p, tb_val, tb_min, tb_max in zip(labels, val_tb, min_tb, max_tb):
            param[p].set(value=get_val(tb_val), min=get_val(tb_min), max=get_val(tb_max))

        # Compute and redraw the model function
        newmodel = model(param, *model_args)
        model_plot.set_ydata(newmodel)
        plt.draw()

    def set_vary(null):
        """ Called by the checkboxes """
        nonlocal param
        # Read all textboxes at once, set Parameters.vary accordingly
        for cb, p in zip(var_cb, labels):
            param[f'{p}'].set(vary=cb.get_status()[0])

    # ---------------------------------------------------------------------------------------


    # Make the figure
    fig = plt.figure()
    fig.set_size_inches(figures.figsize_large)
    plt.subplots_adjust(left=0.1, right=0.6, top=0.9, bottom=0.1)
    ax = fig.add_subplot(1,1,1)
    
    # Draw the widgets
    #   Header row
    [plt.text(X, 0.925, f'{head}', ha='center', transform=fig.transFigure) 
            for X, head in zip((0.635, 0.72, 0.81, 0.90, 0.96), ('Parameter', 'Value', 'Min', 'Max', 'Vary'))]
    #   First column
    labels = [f'{p}' for p in param]    # Name of the parameters
    #       Write them in the first column, right-aligned
    [plt.text(0.675, Y_box+H_box/2, f'{label}', ha='right', va='center', transform=fig.transFigure) for Y_box, label in zip(list_Y_box, labels)]
    #   Textboxes for 'value'
    val_boxes = [plt.axes([0.68, Y_box, L_box, H_box]) for Y_box in list_Y_box]
    val_tb = [TextBox(box, '', textalignment='center', initial=f'{param[p].value}') for box, p in zip(val_boxes, labels)]
    #   Textboxes for 'min'
    min_boxes = [plt.axes([0.77, Y_box, L_box, H_box]) for Y_box in list_Y_box]
    min_tb = [TextBox(box, '', textalignment='center', initial=f'{param[p].min}') for box, p in zip(min_boxes, labels)]
    #   Textboxes for 'max'
    max_boxes = [plt.axes([0.86, Y_box, L_box, H_box]) for Y_box in list_Y_box]
    max_tb = [TextBox(box, '', textalignment='center', initial=f'{param[p].max}') for box, p in zip(max_boxes, labels)]
    #   Checkboxes for 'vary'
    var_boxes = [plt.axes([0.95, Y_box, 0.025, H_box]) for Y_box in list_Y_box]
    var_cb = [CheckButtons(box, labels=[''], actives=[f'{param[p].vary}']) for box, p in zip(var_boxes, labels)]
    [misc.edit_checkboxes(cb, 0.2, 0.2, 0.6, 0.6, color='tab:blue') for cb in var_cb] # make bigger squares

    # Plot the data and the model
    ax.plot(x, experimental, '.', markersize=2, c='tab:red', label='Observed data')
    model_plot, = ax.plot(x, model(param, *model_args), c='tab:blue', label='Model')

    # Fancy shit
    misc.pretty_scale(ax, ax.get_xlim(), 'x')
    misc.pretty_scale(ax, ax.get_ylim(), 'y')
    misc.mathformat(ax)
    ax.legend()
    misc.set_fontsizes(ax, 15)

    # Connect the widgets to their functions
    for column in zip(val_tb, min_tb, max_tb):
        for box in column:
            box.on_submit(update)
    for cb in var_cb:
        cb.on_clicked(set_vary)

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
        Matrix of dimension (# signals, 6) that contains, for each row: v1(Hz), v2(Hz), fwhm1(Hz), fwhm2(Hz), A, x_g
    - fit_interval: tuple of tuple
        Fitting window. ( (left_f1, right_f1), (left_f2, right_f2) )
    """

    ### FIRST OF ALL, THE FIGURE!
    fig = plt.figure()
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
        #'k': 0.1,   # relative intensity
        'k': 0.1,   # relative intensity
        'x_g': 0.5, # fraction of gaussianity
        } for w in range(10)]
    I1 = processing.integral(tr1, x=ppm_f1, lims=lim_f1)[-1]
    I2 = processing.integral(tr2, x=ppm_f2, lims=lim_f2)[-1]
    A = (I1 + I2) / (2*np.pi*fwhm0)


    # Sensitivity for mouse
    sens = {
        'u1': 0.25,   # ppm
        'u2': 0.25,   # ppm
        'fwhm1': 10, # Hz
        'fwhm2': 10, # Hz
        'k': 0.01,  # 1%
        'x_g': 0.1, # 10%
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
            r'$x_{g}$': 'x_g',
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
    up_button = Button(su_box, '$\\uparrow$', hovercolor = '0.975')         # increase sensitivity button
    down_button = Button(giu_box, '$\\downarrow$', hovercolor = '0.975')    # decrease sensitivity button
    save_button = Button(save_box, 'SAVE', hovercolor = '0.975')            # SAVE button
    reset_button = Button(reset_box, 'RESET', hovercolor = '0.975')         # RESET button
    
    #   textboxes
    TB1 = [TextBox(box, '', initial=f'{value:.2f}', textalignment='center') for box, value in zip(tb1_boxes, lim_f1)]   # set limits for F1
    TB2 = [TextBox(box, '', initial=f'{value:.2f}', textalignment='center') for box, value in zip(tb2_boxes, lim_f2)]   # set limits for F2

    #   Radiobuttons for parameter selection
    peak_name = [r'$\delta$ /ppm', r'$\Gamma$ /Hz', r'$k$', r'$x_{g}$', r'$A$']         # Labels for the parameters
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
    misc.edit_checkboxes(check, xadj=0, yadj=0.001, length=0.1, height=(HBOX-0.5*HBOX)/len(check_name), color=COLORS)

    # Text that shows the current values of the parameters
    head_print = plt.text(0.725, 0.4,
            '{:9s}:'.format(r'$\delta$ F2') + f'{V[0]["u2"]:-9.2f}\n'+
            '{:9s}:'.format(r'$\delta$ F1') + f'{V[0]["u1"]:-9.2f}\n'+
            '{:9s}:'.format(r'$\Gamma$ F2') + f'{V[0]["fwhm2"]:-9.2f}\n'+
            '{:9s}:'.format(r'$\Gamma$ F1') + f'{V[0]["fwhm1"]:-9.2f}\n'+
            '{:9s}:'.format(r'$k$') + f'{V[0]["k"]:-9.2f}\n'+
            '{:9s}:'.format(r'$x_g$') + f'{V[0]["x_g"]:-9.2f}\n'+
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
            sequence of the parameters: u1, u2, fwhm1, fwhm2, I, x_g
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
                values[5],   # x_g
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
            if 'k' not in key2edit and 'x_g' not in key2edit:   # i.e. it is u or fwhm
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
        values = [V[s_idx][f'{key}'] for key in ('u1', 'u2', 'fwhm1', 'fwhm2', 'k', 'x_g')]
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
        # Safety check for x_g
        if V[s_idx]['x_g'] <= 0:
            V[s_idx]['x_g'] = 0
        elif V[s_idx]['x_g'] >= 1:
            V[s_idx]['x_g'] = 1
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
        value_string = '{:9s}:'.format(r'$\delta$ F2') + f'{V[s_idx]["u2"]:-9.2f}\n'+ '{:9s}:'.format(r'$\delta$ F1') + f'{V[s_idx]["u1"]:-9.2f}\n'+ '{:9s}:'.format(r'$\Gamma$ F2') + f'{V[s_idx]["fwhm2"]:-9.2f}\n'+ '{:9s}:'.format(r'$\Gamma$ F1') + f'{V[s_idx]["fwhm1"]:-9.2f}\n'+ '{:9s}:'.format(r'$k$') + f'{V[s_idx]["k"]:-9.2f}\n'+ '{:9s}:'.format(r'$x_g$') + f'{V[s_idx]["x_g"]:-9.2f}\n'+ '{:9s}:'.format(r'$A$') + f'{A:-9.2e}\n'
        head_print.set_text(value_string)
        head_print.set_color(COLORS[s_idx]) # color of the active signal
        fig.canvas.draw()

    def save(event):
        """ Slot for the save button: store the current values into the final variables """
        nonlocal final_parameters, fit_interval
        final_parameters = [[   # u1, u2, fwhm1, fwhm2, k*A, x_g
                    misc.ppm2freq(V[x]['u1'], acqus['SFO1'], acqus['o1p']),
                    misc.ppm2freq(V[x]['u2'], acqus['SFO2'], acqus['o2p']),
                    V[x]['fwhm1'],
                    V[x]['fwhm2'],
                    V[x]['k'] * A,
                    V[x]['x_g'],
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
        values = [Vline[f'{key}'] for key in ('u1', 'u2', 'fwhm1', 'fwhm2', 'k', 'x_g')]
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
        sequence of the parameters: u1, u2, fwhm1, fwhm2, I, x_g. Multiple components are allowed
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
        fig = plt.figure()
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
            plt.savefig(f'{name}.png', dpi=dpi)
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

    def draw_coord(self, filename=None, labelsize=8, dpi=600, **kwargs):
        """
        Makes a figure with the experimental dataset and the peak-picked signals as crosshairs.
        --------
        Parameters:
        - filename: str or None
            Filename for the figure to be saved. If None, it is shown instead.
        - labelsize: float
            Font size for the peak index
        - dpi: int
            Resolution of the saved image in dots per inches
        - kwargs: keyworded arguments
            Additional options for figures.ax2D
        """

        fig = plt.figure()
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
            plt.savefig(f'{filename}.png', dpi=dpi)
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
            > Vi will contain [u1, u2, fwhm1, fwhm2, Im, x_g] for each peak,
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
            > Vf will contain [u1, u2, fwhm1, fwhm2, Im, x_g] for each peak,
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
            parameters = [[   # u1, u2, fwhm1, fwhm2, k*A, x_g
                misc.ppm2freq(u1, acqus['SFO1'], acqus['o1p']), # v1 /Hz
                misc.ppm2freq(u2, acqus['SFO2'], acqus['o2p']), # v2 /Hz
                fwhm0,  # fwhm1 /Hz
                fwhm0,  # fwhm2 /Hz
                1,     # I
                0.5,    # x_g
                ]] 
            # Integral
            """
            _, _, data_exp = misc.trim_data_2D(ppm_f2, ppm_f1, self.data, lim_f2, lim_f1)
            data_calc = fit.build_2D_sgn(parameters, acqus, N=(len(tr1), len(tr2)), procs=self.procs)
            _, _, data_calc = misc.trim_data_2D(ppm_f2, ppm_f1, data_calc, lim_f2, lim_f1)
            #A0 = fit.fit_int(data_exp.flatten(), data_calc.flatten())
            """
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
        f.write(f'{"#":<4s}\t{"clu":>4s}\t{"u1":>8s}\t{"u2":>8s}\t{"fwhm1":>8s}\t{"fwhm2":>8s}\t{"I":>8s}\t{"x_g":>8s}\t{"Fit. interv.":>20s}\n')

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
            u1, u2, fwhm1, fwhm2, I, xg
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

        v1, v2, fwhm1, fwhm2, I, x_g = values   # Unpack
        if conv_u:
            # Convert u1, u2 from Hz to ppm
            u1, u2 = [misc.freq2ppm(x, acqus[f'SFO{y}'], acqus[f'o{y}p']) for x, y in zip([v1, v2], [1, 2])]
        else:
            u1, u2 = v1, v2

        # Write the line
        f.write(f'{index:4.0f}\t{clu:4.0f}\t{u1:8.2f}\t{u2:8.2f}\t{fwhm1:8.1f}\t{fwhm2:8.1f}\t{I:8.3e}\t{x_g:8.4f}\t{interval}\n')

    def _read_par_line(self, line):
        """ Splits the line into the three attributes """
        split = line.split('\t')
        index = eval(split[0])
        clu = eval(split[1])
        values = [eval(w) for w in split[2:-1]]
        fit_interval = eval(split[-1])
        return index, clu, values, fit_interval



