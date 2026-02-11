#! /usr/bin/env python3

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Cursor, SpanSelector, RectangleSelector, CheckButtons
import lmfit
from datetime import datetime
import getpass

from . import fit, misc, sim, figures, processing, anal
from .config import CM, COLORS


def select_traces(ppm_f1, ppm_f2, data, Neg=True, grid=False):
    """
    Select traces from a 2D spectrum, save the coordinates in a list.
    Left click to select a point, right click to remove it.


    Parameters
    ----------
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

    Returns
    -------
    coord : list
        List containing the ``[x,y]`` coordinates of the selected points.
    """
    cmaps = 'Blues_r', 'Reds_r'
    # Select traces from a 2D spectrum, save the coordinates in a list
    lvlstep = 1.4                  # for mouse scroll

    # Make the figure
    fig = plt.figure('Traces Selector')
    fig.set_size_inches(figures.figsize_large)
    ax = fig.add_subplot()
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

    # Parameters to save coordinates
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
                    coord.append([ix, iy])       # Update list
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
    Cursor(ax, useblit=True, color='red', linewidth=0.4)
    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('scroll_event', on_scroll)

    plt.show()
    plt.close()

    return coord


def select_for_integration(ppm_f1, ppm_f2, data, Neg=True):
    """
    Select the peaks of a 2D spectrum to integrate.
    First, select the area where your peak is located by dragging the red square.
    Then, select the center of the peak by right_clicking.
    Finally, click 'ADD' to store the peak. Repeat the procedure for as many peaks as you want.

    Parameters
    ----------
    ppm_f1 : 1darray
        ppm scale of the indirect dimension
    ppm_f2 : 1darray
        ppm scale of the direct dimension
    data : 2darray
        Spectrum
    Neg : bool
        Choose if to show the negative contours ( True) or not ( False )

    Returns
    ----------
    peaks : list of dict
        For each peak there are two keys, 'f1' and 'f2', whose meaning is obvious.
        For each of these keys, you have 'u': center of the peak /ppm, and 'lim': the limits of the square you drew before.
    """

    cmaps = CM['Blues_r'], CM['Reds_r']
    lvlstep = 1.4                  # Increase step for contours when scroll the mouse1

    # Make an underlying grid to snap the pointer
    xgrid = np.copy(ppm_f2)
    ygrid = np.copy(ppm_f1)
    # Parameters to save coordinates
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
            coord.append([ix, iy])       # Update list
            ax.plot(ix, iy, 'bo', markersize=2)  # add dot
            ax.axhline(iy, c='b', lw=0.4)   # add horizontal line
            ax.axvline(ix, c='b', lw=0.4)   # add vertical line
            for obj in (tmp_dot, tmp_hline, tmp_vline):
                obj.set_visible(False)      # Set the red crosshair invisible
        return coord

    def on_click(event):
        """ Right click moves the red crosshair """
        x, y = event.xdata, event.ydata     # x,y position of cursor
        if event.inaxes == ax:   # You are inside the figure
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
        if epress.button == 1:   # left click
            # Vertices of the rectangle, counterclockwise
            X = np.array(span.extents[0:2])
            Y = np.array(span.extents[2:4])
            vertX = X[0], X[1], X[1], X[0]
            vertY = Y[0], Y[0], Y[1], Y[1]

            # Make visible the red rectangle
            if not tmp_rekt.get_visible():
                tmp_rekt.set_visible(True)
            tmp_rekt.set_xy(np.array((vertX, vertY)).T)  # .T because (vertX, vertY).shape = (2, 4)
        else:
            pass
        fig.canvas.draw()

    def add_func(event):
        """ ADD button """
        nonlocal coord
        # Draw blue crosshair reading data from the red dot
        ix, iy = tmp_dot.get_data()
        coord = add_crosshair(coord, ix, iy)    # Update coord with the new peak

        # Draw blue rectangle reading data from the red rectangle
        verts = np.array(tmp_rekt.get_xy())[:-1]    # Skip the latter because it knows it has to close the perimeter
        dummy_rekt, = ax.fill(verts[:, 0], verts[:, 1], 'tab:blue', alpha=0.25)
        rekt.append(dummy_rekt)
        # Set red rectangle to invisible
        tmp_rekt.set_visible(False)
        fig.canvas.draw()

    # -----------------------------------------------------------------------------------------------------------------

    # Make the figure
    fig = plt.figure('Manual Peak Picking')
    fig.set_size_inches(figures.figsize_large)
    ax = fig.add_subplot()
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
    tmp_rekt, = ax.fill(np.array([0.1, 0.2, 0.3]), np.array([0.1, 0.2, 0.3]), 'tab:red', alpha=0.25, visible=False)  # Rectangle
    tmp_dot, = ax.plot(0, 0, 'ro', markersize=2, visible=False)  # Dot
    tmp_hline = ax.axhline(0, 0, c='r', lw=0.4, visible=False)  # Horizontal line
    tmp_vline = ax.axvline(0, 0, c='r', lw=0.4, visible=False)  # Vertical line

    # Pretty things
    misc.pretty_scale(ax, (xsx, xdx), 'x')
    misc.pretty_scale(ax, (ysx, ydx), 'y')
    misc.set_fontsizes(ax, 14)

    # Widgets
    Cursor(ax, useblit=True, color='red', linewidth=0.4)       # Moving crosshair
    fig.canvas.mpl_connect('button_press_event', on_click)      # Right click
    fig.canvas.mpl_connect('scroll_event', on_scroll)          # Mouse scroll
    span = RectangleSelector(ax, onselect, useblit=False, props=dict(facecolor='tab:red', alpha=0.5))    # Draggable rectangle
    add_button.on_clicked(add_func)  # Button

    plt.show()
    plt.close()

    # -----------------------------------------------------------------------------------------------------------------

    # collect results
    peaks = []

    def calc_borders(rect):
        """ Calculate the limits of the rectangle """
        vert = rect.get_xy()
        vertX, vertY = vert[:, 0], vert[:, 1]
        x_lims = min(vertX), max(vertX)
        y_lims = min(vertY), max(vertY)
        return x_lims, y_lims

    for dot, rec in zip(coord, rekt):
        x_lims, y_lims = calc_borders(rec)
        # Create an entry for each peak as stated in the description
        peaks.append({
            'f1': {
                'u': dot[1],
                'lim': y_lims,
                },
            'f2': {
                'u': dot[0],
                'lim': x_lims,
                },
            })
    return peaks


def noise_std(y):
    r"""
    Calculates the standard deviation of the noise using the Bruker formula:

    Taken :math:`y` as an array of :math:`N` points, and :math:`y[i]` its i-th entry:

    .. math::

       \sigma_N = \frac{1}{\sqrt{r-1}} \sqrt{ \sum_{k=0}^{r-1} (y[k]^2) - \frac{1}{r}
       [ ( \sum_{k=0}^{r-1} y[k] )^2 + \frac{3}{r^2 -1}( \sum_{k=0}^{r / 2 - 1} (k+1) (y[r/ 2 + k] - y[r/ 2 - k -1 ] ) )^2 ] }


    Parameters
    ----------
    y : 1darray
        The spectral region you would like to use to calculate the standard deviation of the noise.

    Returns
    --------
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
    if N % 2 == 0:
        X = np.sum([(k+1) * (y[n+k] - y[n-k-1]) for k in range(n)])
    else:
        X = np.sum([(k+1) * (y[n+k] - y[n-k-1]) for k in range(n)])
    noisestd = (N-1)**(-0.5) * np.sqrt(Y - 1/N * (W + 3 * X**2 / (N**2 - 1)))
    return noisestd


def snr(data, x=None, signal=None, n_reg=None):
    """
    Computes the signal to noise ratio of a 1D spectrum as height of the signal over twice the noise standard deviation.

    Parameters
    ----------
    data : 1darray
        The spectrum of which you want to compute the SNR
    x : 1darray
        Scale of the spectrum to use. If given, the values in ``n_reg`` are searched according to this scale
    signal : float, optional
        If provided, uses this value as maximum signal. Otherwise, it is selected as the maximum value in ``data``
    n_reg : list or tuple, optional
        If provided, contains the points that delimit the noise region. Otherwise, the whole spectrum is used.

    Returns
    -------
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
        w = slice(min(A, B), max(A, B))
        y = data[w]
    snr = signal / (2 * anal.noise_std(y))
    return snr


def snr_2D(data, n_reg=None):
    """
    Computes the signal to noise ratio of a 2D spectrum.

    Parameters
    ----------
    data : 1darray
        The spectrum of which you want to compute the SNR
    n_reg : list or tuple
        If provided, the points of F1 scale and F2 scale, respectively, of which to extract the projections.
        Otherwise, opens the tool for interactive selection.

    Returns
    -------
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
        print('index for SNR (F1 | F2): ', n_reg)

    f1_trace = data[:, n_reg[0]]
    f2_trace = data[n_reg[1], :]

    snr_f1 = anal.snr(f1_trace, signal=np.max(data))
    snr_f2 = anal.snr(f2_trace, signal=np.max(data))

    return snr_f1, snr_f2


def get_trace(data, ppm_f2, ppm_f1, a, b=None, column=True):
    """

    Takes as input a 2D dataset and the ppm scales of direct and indirect dimensions respectively.
    Calculates the projection on the given axis summing from ``a`` (ppm) to ``b`` (ppm).
    Default: indirect dimension projection (i.e. ``column=True``), change it to False for the direct dimension projection.


    Parameters
    ----------
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

    Returns
    -------
    y : 1darray
        Computed projection
    """
    if b is None:
        b = a

    if column:
        A = misc.ppmfind(ppm_f2, a)[0]
        B = misc.ppmfind(ppm_f2, b)[0]+1
        w = slice(min(A, B), max(A, B))
        y = np.sum(data[..., w], axis=1)
    else:
        A = misc.ppmfind(ppm_f1, a)[0]
        B = misc.ppmfind(ppm_f1, b)[0]+1
        w = slice(min(A, B), max(A, B))
        y = np.sum(data[w, ...], axis=0)
    return y


def integral_2D(ppm_f1, t_f1, SFO1, ppm_f2, t_f2, SFO2, u_1=None, fwhm_1=200, utol_1=0.5, u_2=None, fwhm_2=200, utol_2=0.5, plot_result=False):
    """
    Calculate the integral of a 2D peak. The idea is to extract the traces correspondent to the peak center and fit them with a gaussian function in each dimension.
    Then, once got the intensity of each of the two gaussians, multiply them together in order to obtain the 2D integral.
    This procedure should be equivalent to what CARA does.

    .. note ::

        In development!!!


    Parameters
    ----------
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

    Returns
    ----------
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
        param = lmfit.Parameters()
        param.add('u', value=u_0, min=u_0-utol, max=u_0+utol)
        param.add('fwhm', value=fwhm_0, min=0)
        param.add('I', value=1, vary=False)         # Do not vary as it is adjusted during the fit

        minner = lmfit.Minimizer(f2min, param, fcn_args=(T, ppm, SFO1))
        result = minner.minimize(method='leastsq', max_nfev=10000, xtol=1e-10, ftol=1e-10)
        popt = result.params.valuesdict()

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

    if plot_result:  # Do the plot
        xlim = [(max(ppm_f2), min(ppm_f2)),
                (max(ppm_f1), min(ppm_f1))]

        # Make the figure
        fig = plt.figure('Computed Integrals')
        fig.set_size_inches(figures.figsize_large)
        plt.subplots_adjust(left=0.05, right=0.95, bottom=0.10, top=0.90, wspace=0.20)

        axes = [fig.add_subplot(1, 2, w+1) for w in range(2)]
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


def integrate(ppm0, data0, SFO1, filename='integrals', X_label=r'$\delta\,$F1 /ppm', dx=1):
    r"""
    Allows interactive integration of a NMR spectrum through a dedicated GUI. Returns the values as a dictionary,
    where the keys are the selected regions truncated to the 3nd decimal figure.
    The values are saved in the ``<filename>.igrl`` file.

    In the GUI, draw the integration region around the peak.
    The shape of the integral function appears in red in that region, and the value of the integral
    is reported under the "current integral" label on the right side.
    The integral can be corrected with a baseline, that is the straight line that connects the border of
    the integration window. It can be activated and deactivated using the checkbox on the right side.
    Press "ADD" to save the integral value. Upon pressing, the integral function becomes green and the
    value appears on top of the figure.
    To remove an integral from the list, first click on the corresponding value on the top to select it: the number
    and the integral function should become purple. Then, click on the "REMOVE" button. Use the right click to reset
    the selection.

    At the end of the procedure, click on the "SAVE" button to write the `.igrl` file.

    Parameters
    ----------
    ppm : 1darray
        PPM scale of the spectrum
    data : 1darray
        Spectrum to be integrated.
    SFO1 : float
        Larmor frequency of the observed nucleus.
        For conversion to the correct integration scale.
    filename : str
        Name for the `.igrl` file to be written. Without extension!
    X_label : str
        Label of the x-axis
    dx : float
        Correction for the integral values. It should be ``dx = 2 * dw``

    Returns
    ----------
    abs_vals : dict
        Dictionary containing the values of the integrated peaks.


    .. seealso ::

        :func:`klassez.anal.write_igrl`

    """

    # Copy to prevent overwriting
    ppm = np.copy(ppm0)
    data = np.copy(data0)

    # Create the frequency scale to use for integration. The offset is not important
    x = misc.ppm2freq(ppm, SFO1)

    # Calculate the total integral function
    int_f = processing.integral(data, x) * dx

    # Make the figure
    fig = plt.figure('Spectrum Integration')
    fig.set_size_inches(figures.figsize_large)
    plt.subplots_adjust(left=0.05, bottom=0.10, top=0.90, right=0.88)
    ax = fig.add_subplot()

    # Transform for text coordinates for integral: x is in ax reference frame, y is in fig reference frame
    mixed_transform = mpl.transforms.blended_transform_factory(ax.transData, fig.transFigure)

    # Declare variables
    abs_vals = {}               # dictionary: integrals of the peaks, absolute values
    text_integrals = {}         # dictionary: text that appears at the top
    perm_integrals = {}         # dictionary: green functions
    sel_key = None              # selected integral to be removed with the REMOVE button

    # Make boxes for widgets
    add_box = plt.axes([0.915, 0.80, 0.05, 0.06])
    remove_box = plt.axes([0.915, 0.72, 0.05, 0.06])
    save_box = plt.axes([0.915, 0.20, 0.05, 0.06])
    check_box = plt.axes([0.89, 0.35, 0.10, 0.06])
    # Make buttons
    add_button = Button(add_box, 'ADD', hovercolor='0.875')
    save_button = Button(save_box, 'SAVE', hovercolor='0.875')
    remove_button = Button(remove_box, 'REMOVE', hovercolor='0.875')

    # Make the checkbox
    checkbox = CheckButtons(check_box, ['Use baseline'], actives=[False])
    #   Adjust the position and the dimension of the tick box
    misc.edit_checkboxes(checkbox, xadj=0, yadj=0.05, dim=100, color='violet')

    # ---------------------------------------------------------------------------------------
    # Functions connected to the widgets

    def check_clicked(label):
        """ Slot for the checkbox """
        # Get the region of the red curve
        x_values = tmp_plot.get_xdata()
        # Draw it again with/without baseline and correct the value
        onselect(x_values[0], x_values[-1])

    def calc_bas(isx, idx):
        """ Compute straight line baseline """
        # Two points
        bas_points = np.array([data[isx], data[idx]])
        # Use the linear regression to connect them, employ the point scale
        _, (bas_m, bas_q) = fit.lr(bas_points, x=np.asarray([isx, idx]))
        # full-length x-scale for baseline
        xb = np.arange(isx, idx+1, 1)
        # baseline = mx + q
        bas = xb * bas_m + bas_q
        return bas

    def onselect(vsx, vdx):
        """ When you drag and release """
        # Take indexes of the borders of the selected window and sort them
        isx = misc.ppmfind(ppm, vsx)[0]
        idx = misc.ppmfind(ppm, vdx)[0]
        isx, idx = min(isx, idx), max(isx, idx)
        # Make a slice
        sl = slice(isx, idx+1)
        # If you select less than three points you are not integrating anything
        if idx - isx < 3:
            return

        # Compute the baseline according to the checkbox status
        if checkbox.get_status()[0]:        # do
            bas = calc_bas(isx, idx)
            # and make it visible
            bas_plot.set_visible(True)
        else:                               # don't
            bas = np.zeros_like(ppm[sl])
            # and make it disappear
            bas_plot.set_visible(False)
        # Update baseline plot
        bas_plot.set_data(ppm[sl], bas)

        # Compute the integral
        int_fun = processing.integral(data[sl] - bas, x[sl])    # Integral function ( primitive )
        int_val = (int_fun[-1] - int_fun[0]) * dx               # Value of the integral

        # Update the plot
        #   compute correction factor to make the curve actually visible
        yscale = (max(data[sl]) - min(data[sl])) / (max(int_fun) - min(int_fun))
        #   Plot the red curve
        tmp_plot.set_data(ppm[sl], yscale * int_fun)
        #   and set it visible
        tmp_plot.set_visible(True)
        #   update text under the red label
        tmp_text.set_text(f'{int_val:12.5g}')
        fig.canvas.draw()

    def f_add(event):
        """ When you click 'ADD' """
        # Set the integral function as invisible so that it does not overlay with the permanent one
        tmp_plot.set_visible(False)
        # Get the data from the red curve
        xdata, ydata = tmp_plot.get_data()
        # Get the number from under the red label
        int_val = eval(tmp_text.get_text())
        # Key comes from xdata, value comes from the red label
        abs_vals['{:.3f}:{:.3f}'.format(xdata[0], xdata[-1])] = int_val

        # Update the plot
        #   draw the green curve in place of the red one and store the artist in a dictionary
        perm_integrals['{:.3f}:{:.3f}'.format(xdata[0], xdata[-1])] = ax.plot(xdata, ydata, c='tab:green', lw=2.2)[-1]
        #   draw the text at the top
        text_integrals['{:.3f}:{:.3f}'.format(xdata[0], xdata[-1])] = ax.text(
                np.mean(xdata), 0.90, '{:12.5g}'.format(int_val),       # in the center of the window, at the top panel
                ha='center', va='bottom', transform=mixed_transform, fontsize=10, rotation=60)
        fig.canvas.draw()

    def update_selkey(event):
        """ For the selection. There are a ton of early returns! """
        nonlocal sel_key
        # Right click resets the selection
        if event.button == 3:
            sel_key = None
            return

        # MouseButtonEvent gives the position in pixels -> conversion needed
        figw_px, figh_px = fig.get_figwidth() * fig.dpi, fig.get_figheight() * fig.dpi
        if event.y / figh_px < 0.9:     # y coordinate below the top border of the subplot
            return

        # You made until here. Nice!

        # Placeholders for the selection
        keys, locs = [], []
        # Loop over all the texts
        for key, text_obj in text_integrals.items():
            # find the center position of the text in fig frame
            bbox_display = text_obj.get_window_extent(fig.canvas.get_renderer())
            bbox_fig = bbox_display.transformed(fig.transFigure.inverted())
            # Save the keys of the dictionary and the positions in the lists
            keys.append(key)
            locs.append((bbox_fig.x0 + bbox_fig.x1) / 2)

        # Mouse click position in fig frame
        relp = event.x / figw_px
        # Find the closest text from where you clicked
        nearest = misc.find_nearest(locs, relp)
        # You need to click sufficiently close!
        if abs(relp - nearest) > 0.005:
            return
        # Selection worked! Find the index and put the selection
        i_nearest = locs.index(nearest)
        sel_key = keys[i_nearest]

    def on_mouse_click(event):
        """ Click to select a certain integral """
        # DO NOT GO FURTHER if there are not integrals
        if len(text_integrals.keys()) == 0:
            return
        # Select the integral
        update_selkey(event)

        if sel_key is not None:
            # Highlight in blue
            text_integrals[sel_key].set_color('indigo')
            perm_integrals[sel_key].set_color('indigo')
        else:
            # Reset: text in black, function in green
            for key in text_integrals.keys():
                text_integrals[key].set_color('k')
                perm_integrals[key].set_color('tab:green')
        fig.canvas.draw()

    def f_remove(event):
        """ Remove the selected integral """
        nonlocal sel_key
        # DO NOT GO FURTHER if an integral is not selected
        if sel_key is None:
            return
        # Kill the artists
        text_integrals[sel_key].remove()
        perm_integrals[sel_key].remove()
        # Remove the things from the dictionaries
        abs_vals.pop(sel_key)
        text_integrals.pop(sel_key)
        perm_integrals.pop(sel_key)
        # Update the plot
        fig.canvas.draw()
        # Reset the selection
        sel_key = None

    def f_save(event):
        """ When you click 'SAVE' """
        # Do NOT close the panel if there were not integrals computed
        if len(abs_vals.keys()) == 0:
            print('No integrals taken yet!')
            return
        # Write the .igrl file and close
        anal.write_igrl(filename, abs_vals, indirect_scale=None, header=True)
        plt.close()
        print(f'Integrals saved in {filename}.igrl.')

    # ---------------------------------------------------------------------------------------

    # Add things to the figure panel

    ax.plot(ppm, data, c='tab:blue', lw=0.8)        # Spectrum
    # Draw the total integral function but set to invisible because it is useless, needed as placeholder for the red curve
    tmp_plot, = ax.plot(ppm, int_f/max(int_f)*max(data), c='tab:red', lw=2.2, visible=False)
    bas_plot, = ax.plot(ppm, np.zeros_like(data), c='violet', lw=1.8, visible=False)

    # Draw text labels in the figure, on the right
    ax.text(0.94, 0.68, 'Current integral', ha='center', va='center', transform=fig.transFigure, fontsize=14, color='tab:red')
    tmp_text = ax.text(0.94, 0.65, '0', horizontalalignment='center', verticalalignment='center', transform=fig.transFigure, fontsize=14)

    # Fancy shit
    ax.set_xlim(max(ppm), min(ppm))
    ax.set_xlabel(X_label)
    ax.set_ylabel('Intensity /a.u.')
    misc.pretty_scale(ax, ax.get_xlim(), 'x')
    misc.pretty_scale(ax, ax.get_ylim(), 'y')
    misc.mathformat(ax, 'y')
    misc.set_fontsizes(ax, 14)

    # Add more widgets and connect the buttons to their functions
    add_button.on_clicked(f_add)
    remove_button.on_clicked(f_remove)
    save_button.on_clicked(f_save)
    checkbox.on_clicked(check_clicked)
    fig.canvas.mpl_connect('button_press_event', on_mouse_click)
    # Here I need to keep a reference to it otherwise the span selector and the cursor do not show
    cursor = Cursor(ax, c='tab:red', lw=0.8, horizOn=False)
    span = SpanSelector(ax, onselect, 'horizontal', props=dict(facecolor='tab:red', alpha=0.5))
    # Useless but used to avoid flake to become angry
    span.set_visible(True)
    cursor.vertOn = True

    # Show the figure
    plt.show()

    return abs_vals


def integrate_p2D(ppm0, data0, SFO1, ref=0, indirect_scale=None, filename='integrals', X_label=r'$\delta\,$F1 /ppm', dx=1):
    r"""
    Allows interactive integration of a (series of) NMR spectrum through a dedicated GUI. Returns the values as a dictionary,
    where the keys are the selected regions truncated to the 3nd decimal figure.
    The values are saved in the ``<filename>.igrl`` file.

    In the GUI, draw the integration region around the peak.
    The shape of the integral function appears in red in that region, and the value of the integral
    is reported under the "current integral" label on the right side.
    Both are referred to the ``ref``-th row of ``data``.
    The trend of the integrals appear in the inset plot at the bottom left corner.
    The integral can be corrected with a baseline, that is the straight line that connects the border of
    the integration window. It can be activated and deactivated using the checkbox on the right side.
    Press "ADD" to save the integral values. Upon pressing, the integral function becomes green and the
    value appears on top of the figure.
    To remove an integral from the list, first click on the corresponding value on the top to select it: the number
    and the integral function should become purple. Then, click on the "REMOVE" button. Use the right click to reset
    the selection.

    At the end of the procedure, click on the "SAVE" button to write the `.igrl` file.

    Parameters
    ----------
    ppm : 1darray
        PPM scale of the spectrum
    data : 1darray
        Spectrum to be integrated.
    SFO1 : float
        Larmor frequency of the observed nucleus.
        For conversion to the correct integration scale.
    ref : int
        Index of the transient to be used as reference when drawing the functions
    indirect_scale : 1darray
        Scale of the indirect dimension, if any. If ``None``, 1, 2, 3 ... are used.
    filename : str
        Name for the `.igrl` file to be written. Without extension!
    X_label : str
        Label of the x-axis
    dx : float
        Correction for the integral values. It should be ``dx = 2 * dw``

    Returns
    ----------
    abs_vals : dict
        Dictionary containing the values of the integrated peaks.


    .. seealso ::

        :func:`klassez.anal.write_igrl`

    """

    # Copy to prevent overwriting
    ppm = np.copy(ppm0)
    all_data = np.copy(data0)
    data = np.copy(all_data[ref])

    # Create the frequency scale to use for integration. The offset is not important
    x = misc.ppm2freq(ppm, SFO1)
    if indirect_scale is None:
        indirect_scale = np.arange(all_data.shape[0]) + 1
    labels = [f'{label:>12.5g}' for label in indirect_scale]

    # Calculate the total integral function
    int_f = processing.integral(data, x)

    # Make the figure
    fig = plt.figure('Spectrum Integration')
    fig.set_size_inches(figures.figsize_large)
    plt.subplots_adjust(left=0.205, bottom=0.10, top=0.90, right=0.88)
    ax = fig.add_subplot()
    axy = fig.add_axes([0.020, 0.10, 0.15, 0.25])

    # Transform for text coordinates for integral: x is in ax reference frame, y is in fig reference frame
    mixed_transform = mpl.transforms.blended_transform_factory(ax.transData, fig.transFigure)

    # Declare variables
    abs_vals = {}               # dictionary: integrals of the peaks, absolute values
    text_integrals = {}         # dictionary: text that appears at the top
    perm_integrals = {}         # dictionary: green functions
    sel_key = None              # selected integral to be removed with the REMOVE button

    # Make boxes for widgets
    add_box = plt.axes([0.915, 0.80, 0.05, 0.06])
    remove_box = plt.axes([0.915, 0.72, 0.05, 0.06])
    save_box = plt.axes([0.915, 0.20, 0.05, 0.06])
    check_box = plt.axes([0.89, 0.35, 0.10, 0.06])
    # Make buttons
    add_button = Button(add_box, 'ADD', hovercolor='0.875')
    save_button = Button(save_box, 'SAVE', hovercolor='0.875')
    remove_button = Button(remove_box, 'REMOVE', hovercolor='0.875')

    # Make the checkbox
    checkbox = CheckButtons(check_box, ['Use baseline'], actives=[False])
    #   Adjust the position and the dimension of the tick box
    misc.edit_checkboxes(checkbox, xadj=0, yadj=0.05, dim=100, color='violet')

    # ---------------------------------------------------------------------------------------
    # Functions connected to the widgets

    def check_clicked(label):
        """ Slot for the checkbox """
        # Get the region of the red curve
        x_values = tmp_plot.get_xdata()
        # Draw it again with/without baseline and correct the value
        onselect(x_values[0], x_values[-1])

    def calc_bas(y, isx, idx):
        """ Compute straight line baseline """
        # Two points
        bas_points = np.array([y[isx], y[idx]])
        # Use the linear regression to connect them, employ the point scale
        _, (bas_m, bas_q) = fit.lr(bas_points, x=np.asarray([isx, idx]))
        # full-length x-scale for baseline
        xb = np.arange(isx, idx+1, 1)
        # baseline = mx + q
        bas = xb * bas_m + bas_q
        return bas

    def calc_all_bas(isx, idx):
        """ Compute the straight line baseline for all the experiments """
        all_bas = np.array([
            calc_bas(y, isx, idx)
            for y in all_data])
        return all_bas

    def axy_plot(y):
        """ Draw a series in the miniplot """
        # Set the data and make it visible
        int_plot.set_ydata(y)
        int_plot.set_visible(True)
        # Update the limits
        misc.set_ylim(axy, y)
        fig.canvas.draw()

    def axy_reset():
        """ Turn off the miniplot """
        # Draw zeros and make it invisible
        int_plot.set_ydata(np.zeros_like(indirect_scale))
        int_plot.set_visible(False)
        # Update the limits
        misc.set_ylim(axy, np.zeros_like(indirect_scale))
        fig.canvas.draw()

    def onselect(vsx, vdx):
        """ When you drag and release """
        # Take indexes of the borders of the selected window and sort them
        isx = misc.ppmfind(ppm, vsx)[0]
        idx = misc.ppmfind(ppm, vdx)[0]
        isx, idx = min(isx, idx), max(isx, idx)
        # Make a slice
        sl = slice(isx, idx+1)
        # If you select less than three points you are not integrating anything
        if idx - isx < 3:
            return

        # Compute the baseline according to the checkbox status
        if checkbox.get_status()[0]:        # do
            bas = calc_bas(data, isx, idx)
            # and make it visible
            bas_plot.set_visible(True)
        else:                               # don't
            bas = np.zeros_like(ppm[sl])
            # and make it disappear
            bas_plot.set_visible(False)
        # Update baseline plot
        bas_plot.set_data(ppm[sl], bas)

        # Compute the integral
        int_fun = processing.integral(data[sl] - bas, x[sl])    # Integral function ( primitive )
        int_val = (int_fun[-1] - int_fun[0]) * dx               # Value of the integral

        # Compute all integrals
        if checkbox.get_status()[0]:        # do
            all_bas = calc_all_bas(isx, idx)
        else:                               # don't
            all_bas = np.zeros_like(all_data[..., sl])
        all_int = processing.integrate(all_data[..., sl] - all_bas, x[sl], dx=dx)

        # Update the plot
        #   compute correction factor to make the curve actually visible
        yscale = (max(data[sl]) - min(data[sl])) / (max(int_fun) - min(int_fun))
        #   Plot the red curve
        tmp_plot.set_data(ppm[sl], yscale * int_fun)
        #   and set it visible
        tmp_plot.set_visible(True)
        #   update text under the red label
        tmp_text.set_text(f'{int_val:12.5g}')

        # Update the minipanel
        axy_plot(all_int)
        fig.canvas.draw()

    def f_add(event):
        """ When you click 'ADD' """
        # Set the integral function as invisible so that it does not overlay with the permanent one
        tmp_plot.set_visible(False)
        # Get the data from the red curve
        xdata, ydata = tmp_plot.get_data()
        int_val = eval(tmp_text.get_text())

        # Compute all the integrals of the series
        #   Get the window limits from the red curve
        isx = misc.ppmfind(ppm, min(xdata))[0]
        idx = misc.ppmfind(ppm, max(xdata))[0]
        isx, idx = min(isx, idx), max(isx, idx)
        #   Make a slice
        sl = slice(isx, idx+1)

        #   Baseline
        if checkbox.get_status()[0]:        # do
            all_bas = calc_all_bas(isx, idx)
        else:                               # don't
            all_bas = np.zeros_like(all_data[..., sl])
        # Compute the integrals
        all_int = dx * processing.integrate(all_data[..., sl] - all_bas, x[sl])

        # Key comes from xdata, value comes from the red label
        abs_vals['{:.3f}:{:.3f}'.format(xdata[0], xdata[-1])] = all_int

        # Update the plot
        #   draw the green curve in place of the red one and store the artist in a dictionary
        perm_integrals['{:.3f}:{:.3f}'.format(xdata[0], xdata[-1])] = ax.plot(xdata, ydata, c='tab:green', lw=2.2)[-1]
        #   draw the text at the top
        text_integrals['{:.3f}:{:.3f}'.format(xdata[0], xdata[-1])] = ax.text(
                np.mean(xdata), 0.90, '{:12.5g}'.format(int_val),       # in the center of the window, at the top panel
                ha='center', va='bottom', transform=mixed_transform, fontsize=10, rotation=60)
        #   make the miniplot invisible
        axy_reset()
        fig.canvas.draw()

    def update_selkey(event):
        """ For the selection. There are a ton of early returns! """
        nonlocal sel_key
        # Right click resets the selection
        if event.button == 3:
            sel_key = None
            return

        # MouseButtonEvent gives the position in pixels -> conversion needed
        figw_px, figh_px = fig.get_figwidth() * fig.dpi, fig.get_figheight() * fig.dpi
        if event.y / figh_px < 0.9:     # y coordinate below the top border of the subplot
            return

        # You made until here. Nice!

        # Placeholders for the selection
        keys, locs = [], []
        # Loop over all the texts
        for key, text_obj in text_integrals.items():
            # find the center position of the text in fig frame
            bbox_display = text_obj.get_window_extent(fig.canvas.get_renderer())
            bbox_fig = bbox_display.transformed(fig.transFigure.inverted())
            # Save the keys of the dictionary and the positions in the lists
            keys.append(key)
            locs.append((bbox_fig.x0 + bbox_fig.x1) / 2)

        # Mouse click position in fig frame
        relp = event.x / figw_px
        # Find the closest text from where you clicked
        nearest = misc.find_nearest(locs, relp)
        # You need to click sufficiently close!
        if abs(relp - nearest) > 0.005:
            return
        # Selection worked! Find the index and put the selection
        i_nearest = locs.index(nearest)
        sel_key = keys[i_nearest]

    def on_mouse_click(event):
        """ Click to select a certain integral """
        # DO NOT GO FURTHER if there are not integrals
        if len(text_integrals.keys()) == 0:
            return
        # Select the integral
        update_selkey(event)

        if sel_key is not None:
            # Highlight in blue
            text_integrals[sel_key].set_color('indigo')
            perm_integrals[sel_key].set_color('indigo')
            axy_plot(abs_vals[sel_key])
        else:
            # Reset: text in black, function in green
            for key in text_integrals.keys():
                text_integrals[key].set_color('k')
                perm_integrals[key].set_color('tab:green')
        fig.canvas.draw()

    def f_remove(event):
        """ Remove the selected integral """
        nonlocal sel_key
        # DO NOT GO FURTHER if an integral is not selected
        if sel_key is None:
            return
        # Kill the artists
        text_integrals[sel_key].remove()
        perm_integrals[sel_key].remove()
        axy_reset()
        # Remove the things from the dictionaries
        abs_vals.pop(sel_key)
        text_integrals.pop(sel_key)
        perm_integrals.pop(sel_key)
        # Update the plot
        fig.canvas.draw()
        # Reset the selection
        sel_key = None

    def f_save(event):
        """ When you click 'SAVE' """
        # Do NOT close the panel if there were not integrals computed
        if len(abs_vals.keys()) == 0:
            print('No integrals taken yet!')
            return
        # Write the .igrl file and close
        anal.write_igrl(filename, abs_vals, indirect_scale=indirect_scale, header=True)
        plt.close()
        print(f'Integrals saved in {filename}.igrl.')

    # ---------------------------------------------------------------------------------------

    # Add things to the figure panel

    for k, (y, c, label) in enumerate(zip(all_data, COLORS, labels)):
        ax.plot(ppm, y, c=c, lw=0.8, label=label)        # Spectrum
    # Draw the total integral function but set to invisible because it is useless, needed as placeholder for the red curve
    tmp_plot, = ax.plot(ppm, int_f/max(int_f)*max(data), c='tab:red', lw=2.2, visible=False)
    bas_plot, = ax.plot(ppm, np.zeros_like(data), c='violet', lw=1.8, visible=False)
    int_plot, = axy.plot(indirect_scale, np.zeros_like(indirect_scale), '.-', ms=5, c='tab:red', visible=False)

    # Draw text labels in the figure, on the right
    ax.text(0.94, 0.68, 'Current integral', ha='center', va='center', transform=fig.transFigure, fontsize=14, color='tab:red')
    tmp_text = ax.text(0.94, 0.65, '0', horizontalalignment='center', verticalalignment='center', transform=fig.transFigure, fontsize=14)

    # Fancy shit
    ax.set_xlim(max(ppm), min(ppm))
    ax.set_xlabel(X_label)
    ax.set_ylabel('Intensity /a.u.')
    misc.pretty_scale(ax, ax.get_xlim(), 'x')
    misc.pretty_scale(ax, ax.get_ylim(), 'y')
    misc.mathformat(ax, 'y')
    misc.set_fontsizes(ax, 14)
    ax.legend(loc='upper left', bbox_to_anchor=(0.0125, 0.975), bbox_transform=fig.transFigure, fontsize=9, ncols=2)
    axy.set_xlabel('Indirect dimension')
    misc.pretty_scale(axy, axy.get_xlim(), 'x', 4)
    misc.pretty_scale(axy, axy.get_ylim(), 'y', 4)
    misc.mathformat(axy, 'y')
    misc.set_fontsizes(axy, 14)

    # Add more widgets and connect the buttons to their functions
    add_button.on_clicked(f_add)
    remove_button.on_clicked(f_remove)
    save_button.on_clicked(f_save)
    checkbox.on_clicked(check_clicked)
    fig.canvas.mpl_connect('button_press_event', on_mouse_click)
    # Here I need to keep a reference to it otherwise the span selector and the cursor do not show
    cursor = Cursor(ax, c='tab:red', lw=0.8, horizOn=False)
    span = SpanSelector(ax, onselect, 'horizontal', props=dict(facecolor='tab:red', alpha=0.5))
    # Useless but used to avoid flake to become angry
    span.set_visible(True)
    cursor.vertOn = True

    # Show the figure
    plt.show()

    return abs_vals


def integrate_2D(ppm_f1, ppm_f2, data, SFO1, SFO2, fwhm_1=200, fwhm_2=200, utol_1=0.5, utol_2=0.5, plot_result=False):
    """
    Function to select and integrate 2D peaks of a spectrum, using dedicated GUIs.
    Calls integral_2D to do the dirty job.

    .. error::

        Old function!! Legacy


    Parameters
    ----------
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

    Returns
    ----------
    I : dict
        Computed integrals. The keys are ``'<ppm f1>:<ppm f2>'`` with 2 decimal figures.
    """

    # Get all the information that integral_2D needs
    peaks = anal.select_for_integration(ppm_f1, ppm_f2, data, Neg=True)

    Int = {}      # Declare empty dictionary
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
        Int[f'{P["f2"]["u"]:.2f}:{P["f1"]["u"]:.2f}'] = I_p
    return Int


def write_igrl(filename, dic, indirect_scale=None, header=False):
    """
    Write a section in a integral report file, which shows the integrated regions and the values of the peaks. It allows

    Parameters
    ----------
    filename : str
        Path to the file to be written
    dic : dict
        Dictionary of integral values. The keys are 'ppm1:ppm2' and the
        associated values are the integrals, as floats or as sequences.
    indirect_scale : 1darray
        Scale of the indirect dimension. To be used in the future for fitting
    header : bool
        If True, adds a "!" starting line to separate fit trials

    Returns
    -------
    None

    .. seealso::

        :func:`klassez.anal.read_igrl`

        :func:`klassez.anal.integrate`

    """
    # Understand how many integrals per window we have to write
    for key, value in dic.items():
        if isinstance(value, (int, float)):
            # Integrals of only one spectrum
            n_spectra = 1
        else:
            # Pseudo 2D
            n_spectra = len(value)
        # Information achieved, exit from this useless loop
        break

    # Header line of the table
    spectra_idx = [f'I {w+1}' for w in range(n_spectra)]
    firstline = '; '.join([
        f'{"#":4s}', f'{"Window /ppm":>18s}', *[f'{w:>12s}' for w in spectra_idx]
        ])

    # 4 [#] + 18 [Window] + 12*n_spectra [I] + 2*n_spectra [; ] + 2 [first ;] + 1 [extra]
    n_dashes = 25 + 14 * n_spectra

    # Open the file
    f = open(f'{filename}.igrl', 'a', buffering=1)

    # Info on the region to be fitted
    if header:
        now = datetime.now()
        date_and_time = now.strftime("%d/%m/%Y at %H:%M:%S")
        f.write('! Integrals computed by {} on {}\n\n'.format(getpass.getuser(), date_and_time))

    # Write the indirect scale
    if indirect_scale is not None and n_spectra > 1:
        f.write('x = [')
        for x in indirect_scale:
            f.write(f'{x:.5g}, ')
        f.write(']\n\n')

    # Start writing the table
    f.write(firstline+'\n')
    f.write('-' * n_dashes + '\n')

    # Write the lines of the table
    for k, (key, value) in enumerate(dic.items()):
        # First two entries: number of the window and ppm extremes
        things_to_write = [f'{k+1:4.0f}', f'{key:>18s}']
        # Fork if value is a number or a sequence
        if isinstance(value, (int, float)):     # number
            things_to_write.append(f'{value:12.5e}')
        else:                                   # sequence
            things_to_write.extend([f'{x:12.5e}' for x in value])
        # Make a single line to write
        line = '; '.join(things_to_write)
        # Write it
        f.write(line + '\n')
    # \bottomrule
    f.write('-' * n_dashes + '\n\n')

    # End the section
    f.write('=' * 96 + '\n\n')
    # Close the file
    f.close()


def read_igrl(filename, n=-1):
    """
    Reads a `.igrl` file, containing the integrals of either one or a series of spectra.
    The file is separated and unpacked into a dictionary, each of which contains the integration windows with three decimal figures as keys
    and the integrals as their associated value.
    If present, the indirect timescale is also read and returned.

    Parameters
    ----------
    filename : str
        Path to the filename to be read
    n : int
        Number of performed integrating procedure to be read. Default: last one. The breakpoints are lines that start with "!".
        For this reason, ``n=0`` returns an empty dictionary, hence the first attempt is ``n=1``.

    Returns
    ----------
    dic : dict
        Integrals
    indirect_scale : 1darray or None
        If there is a line that starts with ``'x = '``, it is interpreted as the scale for the indirect dimension.
        If there is not, ``None`` is returned.


    .. seealso::

        :func:`klassez.fit.write_igrl`

    """
    def read_region(R):
        """ Creates a dictionary of parameters from a section of the input file.  """
        # Placeholders
        indirect_scale = None
        dic_r = {}
        # Separate the lines and remove the empty ones
        R = R.split('\n')
        for k, r in enumerate(R):
            if len(r) == 0 or r.isspace():
                _ = R.pop(k)

        n_bp = 0        # Number of breaking points (----)
        for k, r in enumerate(R):
            if '------' in r:   # Increase breakpoint and store the line number
                n_bp += 1
                continue

            if n_bp == 0:
                # Before the table, seek for the indirect timescale
                if 'x = ' in r:
                    indirect_scale = np.array(eval(r.split('=')[-1]))

            if n_bp == 1:       # Second section: integrals
                line = r.split(';')  # Separate the values
                # Get the key from the second column
                key = line[1].replace(' ', '')
                # Unpack the rest of the line
                values = [eval(w) for w in line[2:]]
                # Substitute a float if there is only one value, otherwise convert to array
                if len(values) > 1:
                    values = np.asarray(values)
                else:
                    values = values[-1]

                # Put the values in the dictionary
                dic_r[key] = values

            if n_bp == 2:   # End of file: stop reading
                break

        return dic_r, indirect_scale

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

    # Get the values. R has only one entry!
    dic, indirect_scale = read_region(R[0])
    return dic, indirect_scale
