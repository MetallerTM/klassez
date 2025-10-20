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

from . import fit, misc, sim, figures, processing, anal
from .config import CM, COLORS, cron

def select_traces(ppm_f1, ppm_f2, data, Neg=True, grid=False):
    """
    Select traces from a 2D spectrum, save the coordinates in a list. 
    Left click to select a point, right click to remove it.


    Parameters:
    -----------
    ppm_f1 : 1darray
        ppm scale of the indirect dimension
    ppm_f2 : 1darray
        ppm scale of the direct dimension
    data : 2darray
        Spectrum
    Neg : bool
        Choose if to show the negative contours ( True) or not ( False )
    grid : bool
        Choose if to display the grid ( True) or not ( False )

    Returns:
    --------
    coord : list
        List containing the ``[x,y]`` coordinates of the selected points.
    """
    cmaps = 'Blues_r', 'Reds_r'
    # Select traces from a 2D spectrum, save the coordinates in a list
    lvlstep = 1.4                  # for mouse scroll

    # Make the figure
    fig = plt.figure('Traces Selector')
    fig.set_size_inches(figures.figsize_large)
    ax = fig.add_subplot(1,1,1)
    ax.set_title('Left double click (or middle click) to add point, right click to remove point')
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.90)

    # Set figure borders
    xsx = max(ppm_f2)
    xdx = min(ppm_f2)
    ysx = max(ppm_f1)
    ydx = min(ppm_f1)

    # set level for contour
    livello = 0.2
    cnt = figures.ax2D(ax, ppm_f2, ppm_f1, data, xlims=(xsx, xdx), ylims=(ysx, ydx), cmap=cmaps[0], c_fac=1.4, lvl=livello, lw=0.5, X_label='', Y_label='')
    if Neg:
        Ncnt = figures.ax2D(ax, ppm_f2, ppm_f1, -data, xlims=(xsx, xdx), ylims=(ysx, ydx), cmap=cmaps[1], c_fac=1.4, lvl=livello, lw=0.5)
    else: 
        Ncnt = None

    # Make pretty scales
    misc.pretty_scale(ax, (xsx, xdx), 'x')
    misc.pretty_scale(ax, (ysx, ydx), 'y')

    xgrid = ppm_f2
    ygrid = ppm_f1
    if grid:        # Set grid to visible
        for i in xgrid:
            ax.axvline(i, color='grey', lw=0.1)
        for j in ygrid:
            ax.axhline(j, color='grey', lw=0.1)

    # Parameters: to save coordinates
    coord = []          # Final list of coordinates
    dot = []            # Bullets in figure
    dothline = []       # Horizontal lines
    dotvline = []       # Vertical lines

    def on_click(event):
        # What happens if you click?
        x, y = event.xdata, event.ydata     # x,y position of cursor
        if event.inaxes == ax:     # You are inside the figure
            ix, iy = misc.find_nearest(xgrid, x), misc.find_nearest(ygrid, y)       # Handle to the grid
            if (event.button == 1 and event.dblclick) or event.button == 2:     # Left click: add point
                if [ix, iy] not in coord:       # Avoid superimposed peaks
                    coord.append([ix,iy])       # Update list
                    # Update figure:
                    #   add bullet
                    line, = ax.plot(ix, iy, 'ro', markersize=2)
                    dot.append(line)
                    #   add horizontal line
                    dothline.append(ax.axhline(iy, c='r', lw=0.4))
                    #   add vertical line
                    dotvline.append(ax.axvline(ix, c='r', lw=0.4))
            if event.button == 3:     # Right click: remove point
                if [ix, iy] in coord:       # only if the point is already selected
                    # Remove coordinates and all figure elements
                    i = coord.index([ix, iy])
                    coord.remove([ix, iy])
                    killd = dot.pop(i)
                    killd.remove()
                    killh = dothline.pop(i)
                    killh.remove()
                    killv = dotvline.pop(i)
                    killv.remove()

        fig.canvas.draw()
    
    def on_scroll(event):
        # Zoom
        nonlocal livello, cnt
        if Neg:
            nonlocal Ncnt

        xsx, xdx = ax.get_xlim()
        ysx, ydx = ax.get_ylim()

        if event.button == 'up':
            livello *= lvlstep 
        if event.button == 'down':
            livello /= lvlstep
        if livello > 1:
            livello = 1
        cnt, Ncnt = figures.redraw_contours(ax, ppm_f2, ppm_f1, data, lvl=livello, cnt=cnt, Neg=Neg, Ncnt=Ncnt, lw=0.5, cmap=cmaps)
        misc.pretty_scale(ax, (xsx, xdx), 'x')
        misc.pretty_scale(ax, (ysx, ydx), 'y')
        fig.canvas.draw()

    # Widgets
    cursor = Cursor(ax, useblit=True, color='red', linewidth=0.4)
    mouse = fig.canvas.mpl_connect('button_press_event', on_click)
    scroll = fig.canvas.mpl_connect('scroll_event', on_scroll)

    plt.show()
    plt.close()

    return coord

def select_for_integration(ppm_f1, ppm_f2, data, Neg=True):
    """
    Select the peaks of a 2D spectrum to integrate.
    First, select the area where your peak is located by dragging the red square.
    Then, select the center of the peak by right_clicking. 
    Finally, click 'ADD' to store the peak. Repeat the procedure for as many peaks as you want.

    Parameters:
    -----------
    ppm_f1 : 1darray
        ppm scale of the indirect dimension
    ppm_f2 : 1darray
        ppm scale of the direct dimension
    data : 2darray
        Spectrum
    Neg : bool
        Choose if to show the negative contours ( True) or not ( False )

    Returns:
    -----------
    peaks : list of dict
        For each peak there are two keys, 'f1' and 'f2', whose meaning is obvious. 
        For each of these keys, you have 'u': center of the peak /ppm, and 'lim': the limits of the square you drew before.
    """

    cmaps = CM['Blues_r'], CM['Reds_r']
    lvlstep = 1.4                  # Increase step for contours when scroll the mouse1

    # Make an underlying grid to snap the pointer
    xgrid = np.copy(ppm_f2)
    ygrid = np.copy(ppm_f1)
    # Parameters: to save coordinates
    coord = []          # Final list of coordinates
    rekt = []           # Rectangles
    # Set figure borders
    xsx, xdx = max(ppm_f2), min(ppm_f2)
    ysx, ydx = max(ppm_f1), min(ppm_f1)
    # set base level for contour
    lvl0 = 0.2

    # -----------------------------------------------------------------------------------------------------------------
    # Functions connected to the widgets
    def add_crosshair(coord, ix, iy):
        """ Add blue crosshair in (ix, iy) """
        if [ix, iy] not in coord:       # Avoid superimposed peaks
            coord.append([ix,iy])       # Update list
            ax.plot(ix, iy, 'bo', markersize=2) # add dot
            ax.axhline(iy, c='b', lw=0.4)   # add horizontal line
            ax.axvline(ix, c='b', lw=0.4)   # add vertical line
            for obj in (tmp_dot, tmp_hline, tmp_vline):
                obj.set_visible(False)      # Set the red crosshair invisible
        return coord

    def on_click(event):
        """ Right click moves the red crosshair """
        x, y = event.xdata, event.ydata     # x,y position of cursor
        if event.inaxes == ax: # You are inside the figure
            ix, iy = misc.find_nearest(xgrid, x), misc.find_nearest(ygrid, y)       # Snap to the grid
            if event.button == 3:    
                # Update figure:
                tmp_dot.set_data((ix,), (iy,))
                tmp_hline.set_ydata((iy,))
                tmp_vline.set_xdata((ix,))
                # Make visible the red crosshair
                for obj in (tmp_dot, tmp_hline, tmp_vline):
                    obj.set_visible(True)
        else:
            pass
        fig.canvas.draw()
    
    def on_scroll(event):
        """ Redraw contours with more/less levels """
        nonlocal lvl0, cnt
        if Neg:
            nonlocal Ncnt

        # Read the input
        if event.button == 'up':
            lvl0 *= lvlstep 
        if event.button == 'down':
            lvl0 /= lvlstep
        if lvl0 > 1:
            lvl0 = 1

        # Redraw contours
        if Neg:
            cnt, Ncnt = figures.redraw_contours(ax, ppm_f2, ppm_f1, data, lvl=lvl0, cnt=cnt, Neg=Neg, Ncnt=Ncnt, lw=0.5, cmap=[cmaps[0], cmaps[1]])
        else:
            cnt, _ = figures.redraw_contours(ax, ppm_f2, ppm_f1, data, lvl=lvl0, cnt=cnt, Neg=Neg, Ncnt=None, lw=0.5, cmap=[cmaps[0], cmaps[1]])
        # Draw the pretty things again
        misc.pretty_scale(ax, (xsx, xdx), 'x')
        misc.pretty_scale(ax, (ysx, ydx), 'y')
        misc.set_fontsizes(ax, 14)
        fig.canvas.draw()

    def onselect(epress, erelease):
        """ Drag rectangle """
        if epress.button == 1: # left click
            # Vertices of the rectangle, counterclockwise
            X = np.array(span.extents[0:2])
            Y = np.array(span.extents[2:4])
            vertX = X[0], X[1], X[1], X[0]
            vertY = Y[0], Y[0], Y[1], Y[1]

            # Make visible the red rectangle
            if not tmp_rekt.get_visible():
                tmp_rekt.set_visible(True)
            tmp_rekt.set_xy(np.array((vertX, vertY)).T) # .T because (vertX, vertY).shape = (2, 4)
        else:
            pass
        fig.canvas.draw()

    def add_func(event):
        """ ADD button """
        nonlocal tmp_rekt, coord
        # Draw blue crosshair reading data from the red dot
        ix, iy = tmp_dot.get_data()
        coord = add_crosshair(coord, ix, iy)    # Update coord with the new peak

        # Draw blue rectangle reading data from the red rectangle
        verts = np.array(tmp_rekt.get_xy())[:-1]    # Skip the latter because it knows it has to close the perimeter
        dummy_rekt, = ax.fill(verts[:,0], verts[:,1], 'tab:blue', alpha=0.25)
        rekt.append(dummy_rekt)
        # Set red rectangle to invisible
        tmp_rekt.set_visible(False)
        fig.canvas.draw()

    # -----------------------------------------------------------------------------------------------------------------

    # Make the figure
    fig = plt.figure('Manual Peak Picking')
    fig.set_size_inches(figures.figsize_large)
    ax = fig.add_subplot(1,1,1)
    ax.set_title('Drag with left peak for region; select peak with right click')
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.875, top=0.90)
    
    # ADD button
    add_box = plt.axes([0.925, 0.70, 0.05, 0.05])
    add_button = Button(add_box, 'ADD', hovercolor='0.975')

    # Draw contour
    cnt = figures.ax2D(ax, ppm_f2, ppm_f1, data, cmap=cmaps[0], c_fac=1.4, lvl=lvl0, lw=0.5)
    if Neg:   
        Ncnt = figures.ax2D(ax, ppm_f2, ppm_f1, -data, cmap=cmaps[1], c_fac=1.4, lvl=lvl0, lw=0.5)

    # Initialize the red curves
    tmp_rekt, = ax.fill(np.array([0.1,0.2,0.3]), np.array([0.1,0.2,0.3]), 'tab:red', alpha=0.25, visible=False) # Rectangle
    tmp_dot, = ax.plot(0, 0, 'ro', markersize=2, visible=False) # Dot
    tmp_hline = ax.axhline(0, 0, c='r', lw=0.4, visible=False)  # Horizontal line
    tmp_vline = ax.axvline(0, 0, c='r', lw=0.4, visible=False)  # Vertical line

    # Pretty things
    misc.pretty_scale(ax, (xsx, xdx), 'x')
    misc.pretty_scale(ax, (ysx, ydx), 'y')
    misc.set_fontsizes(ax, 14)

    # Widgets
    cursor = Cursor(ax, useblit=True, color='red', linewidth=0.4)       # Moving crosshair
    mouse = fig.canvas.mpl_connect('button_press_event', on_click)      # Right click
    scroll = fig.canvas.mpl_connect('scroll_event', on_scroll)          # Mouse scroll
    span = RectangleSelector(ax, onselect, useblit=False, props=dict(facecolor='tab:red', alpha=0.5)) # Draggable rectangle
    add_button.on_clicked(add_func) # Button

    plt.show()
    plt.close()

    # -----------------------------------------------------------------------------------------------------------------

    # collect results
    peaks = [] 

    def calc_borders(rect):
        """ Calculate the limits of the rectangle """
        vert = rect.get_xy()
        vertX, vertY = vert[:,0], vert[:,1]
        x_lims = min(vertX), max(vertX)
        y_lims = min(vertY), max(vertY)
        return x_lims, y_lims

    for dot, rec in zip(coord, rekt):
        x_lims, y_lims = calc_borders(rec)
        # Create an entry for each peak as stated in the description
        peaks.append({
            'f1' : {
                'u' : dot[1],
                'lim' : y_lims,
                },
            'f2' : { 
                'u' : dot[0], 
                'lim' : x_lims,
                },
            })
    return peaks


def noise_std(y):
    r"""
    Calculates the standard deviation of the noise using the Bruker formula: 

    Taken :math:`y` as an array of :math:`N` points, and :math:`y[i]` its i-th entry:

    .. math::
       \sigma_N = \frac{1}{\sqrt{r-1}} \sqrt{ \sum_{k=0}^{r-1} (y[k]^2) - \frac{1}{r} [ ( \sum_{k=0}^{r-1} y[k] )^2 + \frac{3}{r^2 -1}( \sum_{k=0}^{r / 2 - 1} (k+1) (y[r/ 2 + k] - y[r/ 2 - k -1 ] ) )^2 ] }


    Parameters:
    -----------
    y : 1darray
        The spectral region you would like to use to calculate the standard deviation of the noise.

    Returns:
    ---------
    noisestd : float
        The standard deviation of the noise.
    """
    N = len(y)
    n = N//2
    # W
    W = np.sum(y)**2
    # Y
    Y = np.sum(y**2)
    # X
    if N%2 == 0:
        X = np.sum([(k+1) * (y[n+k] - y[n-k-1]) for k in range(n)])
    else:
        X = np.sum([(k+1) * (y[n+k] - y[n-k-1]) for k in range(n)])
    noisestd = (N-1)**(-0.5) * np.sqrt( Y - 1/N * (W + 3 * X**2 / (N**2 -1) ))
    return noisestd


def snr(data, x=None, signal=None, n_reg=None):
    """
    Computes the signal to noise ratio of a 1D spectrum as height of the signal over twice the noise standard deviation.

    Parameters:
    -----------
    data : 1darray
        The spectrum of which you want to compute the SNR
    x : 1darray
        Scale of the spectrum to use. If given, the values in ``n_reg`` are searched according to this scale
    signal : float, optional
        If provided, uses this value as maximum signal. Otherwise, it is selected as the maximum value in ``data``
    n_reg : list or tuple, optional
        If provided, contains the points that delimit the noise region. Otherwise, the whole spectrum is used.
    
    Returns:
    --------
    snr : float
        The SNR of the spectrum
    """
    # Computes the SNR of a 1D spectrum (or 2D projection).
    # n_reg is a list/tuple of 2 values that delimitates the noise region
    if signal is None:
        signal = np.max(data)

    if x is None:
        x = np.arange(data.shape[-1])

    if n_reg is None:
        y = data
    else:
        A = misc.ppmfind(x, n_reg[0])[0]
        B = misc.ppmfind(x, n_reg[1])[0]
        w = slice(min(A,B), max(A,B))
        y = data[w]
    snr = signal / (2 * anal.noise_std(y))
    return snr

def snr_2D(data, n_reg=None):
    """
    Computes the signal to noise ratio of a 2D spectrum.
   
    Parameters:
    -----------
    data : 1darray
        The spectrum of which you want to compute the SNR
    n_reg : list or tuple
        If provided, the points of F1 scale and F2 scale, respectively, of which to extract the projections.
        Otherwise, opens the tool for interactive selection.
  
    Returns:
    --------
    snr_f1 : float
        The SNR of the indirect dimension
    snr_f2 : float
        The SNR of the direct dimension
    """
    # Computes the SNR of a 2D spectrum.
    # n_reg is: (ppmf1 for f2 trace, ppmf2 for f1 trace)
    if n_reg is None:
        x_scale = np.arange(data.shape[-1])
        y_scale = np.arange(data.shape[0])
        coord = anal.select_traces(y_scale, x_scale, data)
        n_reg = (coord[0][0], coord[0][1])
        print('index for SNR (F1 | F2): ',n_reg)

    f1_trace = data[:,n_reg[0]]
    f2_trace = data[n_reg[1],:]

    snr_f1 = anal.snr(f1_trace, signal=np.max(data))
    snr_f2 = anal.snr(f2_trace, signal=np.max(data))

    return snr_f1, snr_f2


def get_trace(data, ppm_f2, ppm_f1, a, b=None, column=True):
    """

    Takes as input a 2D dataset and the ppm scales of direct and indirect dimensions respectively.
    Calculates the projection on the given axis summing from ``a`` (ppm) to ``b`` (ppm). 
    Default: indirect dimension projection (i.e. ``column=True``), change it to False for the direct dimension projection.


    Parameters:
    -----------
    data : 2darray
        Spectrum of which to extract the projections
    ppm_f2 : 1darray
        ppm scale of the direct dimension
    ppm_f1 : 1darray
        ppm scale of the indirect dimension
    a : float
        The ppm value from which to start extracting the projection.
    b : float, optional
        If provided, the ppm value at which to stop extracting the projection. Otherwise, returns only the ``a`` trace.
    column : bool
        If True, extracts the F1 projection. If False, extracts the F2 projection.

    Returns:
    --------
    y : 1darray
        Computed projection
    """
    if b is None:
        b = a

    if column:
        A = misc.ppmfind(ppm_f2, a)[0]
        B = misc.ppmfind(ppm_f2, b)[0]+1
        w = slice(min(A,B), max(A,B))
        y = np.sum(data[...,w],axis=1)
    else:
        A = misc.ppmfind(ppm_f1, a)[0]
        B = misc.ppmfind(ppm_f1, b)[0]+1
        w = slice(min(A,B), max(A,B))
        y = np.sum(data[w,...],axis=0)
    return y



def integral_2D(ppm_f1, t_f1, SFO1, ppm_f2, t_f2, SFO2, u_1=None, fwhm_1=200, utol_1=0.5, u_2=None, fwhm_2=200, utol_2=0.5, plot_result=False):
    """
    Calculate the integral of a 2D peak. The idea is to extract the traces correspondent to the peak center and fit them with a gaussian function in each dimension. Then, once got the intensity of each of the two gaussians, multiply them together in order to obtain the 2D integral. 
    This procedure should be equivalent to what CARA does.

    .. note :: 

        In development!!!

    
    Parameters:
    -----------
    ppm_f1 : 1darray
        PPM scale of the indirect dimension
    t_f1 : 1darray 
        Trace of the indirect dimension, real part
    SFO1 : float
        Larmor frequency of the nucleus in the indirect dimension
    ppm_f2 : 1darray 
        PPM scale of the direct dimension
    t_f2 : 1darray 
        Trace of the direct dimension, real part
    SFO2 : float
        Larmor frequency of the nucleus in the direct dimension
    u_1 : float
        Chemical shift in F1 /ppm. Defaults to the center of the scale
    fwhm_1 : float
        Starting FWHM /Hz in the indirect dimension
    utol_1 : float
        Allowed tolerance for u_1 during the fit. (u_1-utol_1, u_1+utol_1)
    u_2 : float
        Chemical shift in F2 /ppm. Defaults to the center of the scale
    fwhm_2 : float
        Starting FWHM /Hz in the direct dimension
    utol_2 : float
        Allowed tolerance for u_2 during the fit. (u_2-utol_2, u_2+utol_2)
    plot_result : bool
        True to show how the program fitted the traces.

    Returns:
    -----------
    I_tot : float
        Computed integral.
    """

    def f2min(param, T, x, SFO1):
        """ Cost function """
        par = param.valuesdict()
        sigma = misc.freq2ppm(par['fwhm'], np.abs(SFO1)) / (2 * (2 * np.log(2))**0.5)     # Convert FWHM to ppm and then to std
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
        sigma = misc.freq2ppm(popt['fwhm'], np.abs(SFO1)) / (2 * (2 * np.log(2))**0.5) 
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


def integrate(ppm0, data0, X_label=r'$\delta\,$F1 /ppm'):
    r"""
    Allows interactive integration of a NMR spectrum through a dedicated GUI. Returns the values as a dictionary, where the keys are the selected regions truncated to the 2nd decimal figure.
    The returned dictionary contains pre-defined keys, as follows:

    * total:    total integrated area
    * ref_pos:  location of the reference peak /ppm1:ppm2
    * ref_int:  absolute integral of the reference peak
    * ref_val:  for how many nuclei the reference peak integrates

    The absolute integral of the x-th peak, :math:`I_x`, must be calculated according to the formula:

    .. math::

        I_x = I_x^{\text(relative)} \frac{\text{ ref_int }}{ \text{ ref_val }}


    Parameters:
    -----------
    ppm : 1darray
        PPM scale of the spectrum
    data : 1darray
        Spectrum to be integrated.
    X_label : str
        Label of the x-axis

    Returns:
    -----------
    f_vals : dict
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
    
    .. error::

        Old function!! Legacy


    Parameters:
    -----------
    ppm_f1 : 1darray
        PPM scale of the indirect dimension
    ppm_f2 : 1darray 
        PPM scale of the direct dimension
    data : 2darray 
        real part of the spectrum
    SFO1 : float
        Larmor frequency of the nucleus in the indirect dimension
    SFO2 : float
        Larmor frequency of the nucleus in the direct dimension
    fwhm_1 : float
        Starting FWHM /Hz in the indirect dimension
    fwhm_2 : float
        Starting FWHM /Hz in the direct dimension
    utol_1 : float
        Allowed tolerance for u_1 during the fit. (u_1-utol_1, u_1+utol_1)
    utol_2 : float
        Allowed tolerance for u_2 during the fit. (u_2-utol_2, u_2+utol_2)
    plot_result : bool
        True to show how the program fitted the traces.

    Returns:
    -----------
    I : dict
        Computed integrals. The keys are ``'<ppm f1>:<ppm f2>'`` with 2 decimal figures.
    """

    # Get all the information that integral_2D needs
    peaks = anal.select_for_integration(ppm_f1, ppm_f2, data, Neg=True)

    I = {}      # Declare empty dictionary
    for P in peaks:
        # Extract trace F1
        T1 = anal.get_trace(data, ppm_f2, ppm_f1, P['f2']['u'], column=True)
        x_T1, y_T1 = misc.trim_data(ppm_f1, T1, *P['f1']['lim'])    # Trim according to the rectangle
        # Extract trace F2
        T2 = anal.get_trace(data, ppm_f2, ppm_f1, P['f1']['u'], column=False)
        x_T2, y_T2 = misc.trim_data(ppm_f2, T2, *P['f2']['lim'])    # Trim according to the rectangle

        # Compute the integrals
        I_p = processing.integral_2D(x_T1, y_T1, SFO1, x_T2, y_T2, SFO2,
                u_1=P['f1']['u'], fwhm_1=fwhm_1, utol_1=utol_1, 
                u_2=P['f2']['u'], fwhm_2=fwhm_2, utol_2=utol_2,
                plot_result=plot_result)

        # Store the integral in the dictionary
        I[f'{P["f2"]["u"]:.2f}:{P["f1"]["u"]:.2f}'] = I_p
    return I


