#! /usr/bin/env python3

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Cursor, RadioButtons, TextBox, Slider, CheckButtons, SpanSelector
from scipy.signal import find_peaks, peak_widths
from datetime import datetime
from pathlib import Path
import getpass
import warnings
from copy import deepcopy

from . import misc, processing, figures, anal, fit, sim
from .config import CM, COLORS
from .Spectra import Spectrum_1D

# =================================================================================
# PROCESSING GUI
# =================================================================================


def interactive_basl_windows(ppm, data):
    """
    Allows for interactive partitioning of a spectrum in windows.
    Double left click to add a bar, double right click to remove it.
    Returns the location of the red bars as a list.

    Parameters
    ----------
    ppm : 1darray
        PPM scale of the spectrum
    data : 1darray
        Spectrum to be partitioned

    Returns
    -------
    coord : list
        List containing the coordinates of the windows, plus ``ppm[0]`` and ``ppm[-1]``
    """

    # Make the figure
    fig = plt.figure('Manual Computation of Polynomial Baseline')
    fig.set_size_inches(15, 8)
    ax = fig.add_subplot()
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.95)

    ax.set_title('Divide the spectrum into windows. Double click to set a wall, right click to remove it')

    # Set figure borders

    figures.ax1D(ax, ppm, data)

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
            if (event.button == 1 and event.dblclick) or event.button == 2:     # Left click: add point
                if ix not in coord:       # Avoid superimposed peaks
                    coord.append(ix)       # Update list
                    # Update figure:
                    #   add bullet
                    dotvline.append(ax.axvline(ix, c='r', lw=0.4))
            if event.button == 3:    # Right click: remove point
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
    cursor.vertOn = True
    fig.canvas.mpl_connect('button_press_event', on_click)

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

    Parameters
    ----------
    ppm : 1darray
        PPM scale of the spectrum
    data : 1darray
        spectrum
    limits : tuple
        Window limits (left, right).

    Returns
    -------
    mode : str
        Baseline correction mode: ``'polynomion'`` as default, ``'spline'`` if you press the button
    C_f : 1darray or str
        Baseline polynomion coefficients, or ``'callintsmooth'`` if you press the spline button
    """

    # Initialize mode
    mode = 'polynomion'

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
    up_button = Button(su_box, r'$\uparrow$', hovercolor='0.975')
    down_button = Button(giu_box, r'$\downarrow$', hovercolor='0.975')
    save_button = Button(save_box, 'SAVE', hovercolor='0.975')
    reset_button = Button(reset_box, 'RESET', hovercolor='0.975')
    callspline_button = Button(callspline_box, 'SPLINE BASELINE\nCORRECTION', hovercolor='0.975')

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
        for k in range(len(poly_name)):
            if stats[k]:
                C[k] += 10**om[k]

    def roll_down_p(event):
        # Decrease polynomion with mouse scroll
        for k in range(len(poly_name)):
            if stats[k]:
                C[k] -= 10**om[k]

    def up_om(event):
        # Increase the om of the active coefficient by 1
        for k in range(len(poly_name)):
            if stats[k]:
                om[k] += 1

    def down_om(event):
        # Decrease the om of the active coefficient by 1
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
    fig.set_size_inches(15, 8)
    plt.subplots_adjust(bottom=0.10, top=0.90, left=0.05, right=0.80)
    ax = fig.add_subplot()

    ax.plot(ppm[lim1:lim2], data[lim1:lim2], label='Spectrum', lw=1.0, c='tab:blue')     # experimental

    poly_plot, = ax.plot(ppm[lim1:lim2], y, label='Baseline', lw=0.8, c='tab:orange')    # Polynomion

    # make pretty scale
    ax.set_xlim(max(limits), min(limits))
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
    fig.canvas.mpl_connect('scroll_event', on_scroll)
    save_button.on_clicked(save)
    reset_button.on_clicked(reset)
    callspline_button.on_clicked(use_spline_instead)

    ax.legend()
    plt.show()
    plt.close()

    return mode, C_f


def interactive_echo_param(data0):
    """
    Interactive plot that allows to select the parameters needed to process a CPMG-like FID.
    Use the TextBox or the arrow keys to adjust the values.
    You can call :func:`klassez.processing.sum_echo_train` or :func:`klassez.processing.split_echo_train` by starring the return statement of this function, i.e.:

    .. code-block:: python

        processing.sum_echo_train(data0, *interactive_echo_train(data0))

    as they are in the correct order to be used in this way.

    Parameters
    ----------
    data0 : ndarray
        CPMG FID

    Returns
    -------
    n : int
        Distance between one echo and the next one
    n_echoes : int
        Number of echoes to sum/split
    i_p : int
        Offset points from the start of the FID

    .. seealso::

        :func:`klassez.processing.sum_echo_train`

        :func:`klassez.processing.split_echo_train`
    """

    # Check for data dimension and safety copy
    if len(data0.shape) == 1:
        data = np.copy(data0)
    elif len(data0.shape) == 2:
        data = np.copy(data0[0, :])
    else:
        raise ValueError('Data shape not supported')

    # Make the figure
    fig = plt.figure('Echo Splitter')
    fig.set_size_inches(figures.figsize_large)
    plt.subplots_adjust(left=0.25, right=0.95, top=0.90, bottom=0.15)
    ax = fig.add_subplot(2, 3, (1, 5))    # Square plot
    axs = fig.add_subplot(2, 3, 3)        # Right top
    axt = fig.add_subplot(2, 3, 6)        # Right bottom

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

    # Initialize the three values in a dictionary
    param = {
            'n': 20,
            'n_echoes': 2,
            'i_p': 0,
            }

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


def interactive_fp(fid0, acqus, procs):
    """
    Perform the processing of a 1D NMR spectrum interactively. The GUI offers the opportunity to test different window functions, as well as different ``tdeff`` values and final sizes.
    The active parameters appear as blue text.

    .. warning::

        The rendering can be *very slow*. Use with great care.

    .. error::

        To be checked, might not work as expected

    Parameters
    ----------
    fid0 : 1darray
        FID to process
    acqus : dict
        Dictionary of acquisition parameters
    procs : dict
        Dictionary of processing parameters

    Returns
    -------
    pdata : 1darray
        Processed spectrum
    procs : dict
        Updated dictionary of processing parameters.
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

    # Calculate starting values
    data = processing.fp(fid, wf=procs['wf'], zf=procs['zf'], tdeff=procs['tdeff'])

    # Get WF
    apodf = get_apod(fid0.shape[-1], procs)

    # Calculate the ppm scale
    fq_scale = processing.make_scale(data.shape[-1], acqus['dw'], rev=True)
    ppm_scale = misc.freq2ppm(fq_scale, acqus['SFO1'], acqus['o1p'])

    # Make the figure panel
    fig = plt.figure('Interactive Processing')
    fig.set_size_inches(15, 9)
    plt.subplots_adjust(left=0.1, bottom=0.05, right=0.8, top=0.95, hspace=0.4)
    ax = fig.add_subplot(4, 1, (1, 3))     # spectrum
    axf = fig.add_subplot(4, 1, 4)        # fid
    # Define useful things
    modes = ['No', 'em', 'sin', 'qsin', 'gm', 'gmb']   # entries for the radiobuttons
    act_keys = {    # Active Parameters
            'No': [],
            'em': ['lb'],
            'sin': ['ssb'],
            'qsin': ['ssb'],
            'gm': ['lb_gm', 'gb_gm', 'gc'],
            'gmb': ['lb', 'gb'],
            }
    tx = {}  # Dictionary of the texts

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
        try:
            SI = eval(v)
            procs['zf'] = SI
        except Exception:
            pass
        update()

    def update_tdeff(v):
        try:
            val = eval(v)
            procs['tdeff'] = int(val)
        except Exception:
            pass
        tx['tdeff'].set_text('{:.0f}'.format(procs['tdeff']))
        update()

    def update_mode(label):
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
        try:
            lb = eval(v)
            procs['wf']['lb'] = lb
        except Exception:
            pass
        tx['lb'].set_text('{:.0f}'.format(procs['wf']['lb']))
        update()

    def update_gb(v):
        try:
            gb = eval(v)
            procs['wf']['gb'] = gb
        except Exception:
            pass
        tx['gb'].set_text('{:.2f}'.format(procs['wf']['gb']))
        update()

    def update_lb_gm(v):
        try:
            lb = eval(v)
            procs['wf']['lb_gm'] = lb
        except Exception:
            pass
        tx['lb_gm'].set_text('{:.0f}'.format(procs['wf']['lb_gm']))
        update()

    def update_gb_gm(v):
        try:
            gb = eval(v)
            procs['wf']['gb_gm'] = gb
        except Exception:
            pass
        tx['gb_gm'].set_text('{:.2f}'.format(procs['wf']['gb_gm']))
        update()

    def update_gc(v):
        try:
            gc = eval(v)
            procs['wf']['gc'] = gc
        except Exception:
            pass
        tx['gc'].set_text('{:.2f}'.format(procs['wf']['gc']))
        update()

    def update_ssb(v):
        try:
            ssb = eval(v)
            procs['wf']['ssb'] = ssb
        except Exception:
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
    misc.set_ylim(axf, (-1, 1))
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


def interactive_xfb(fid0, acqus, procs, lvl0=0.1, show_cnt=True):
    """
    Perform the processing of a 2D NMR spectrum interactively. The GUI offers the opportunity to test different window functions, as well as different tdeff values and final sizes.
    The active parameters appear as blue text.
    When changing the parameters, give it some time to compute.

    .. warning::

        *Extremely slow rendering!* Use with care

    Parameters
    ----------
    fid0 : 2darray
        FID to process
    acqus : dict
        Dictionary of acquisition parameters
    procs : dict
        Dictionary of processing parameters
    lvl0 : float
        Starting level of the contours
    show_cnt : bool
        Choose if to display data using contours (True) or heatmap (False)

    Returns
    -------
    pdata : 2darray
        Processed spectrum
    procs : dict
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
    act_keys = {    # Active Parameters
            'No': [],
            'em': ['lb'],
            'sin': ['ssb'],
            'qsin': ['ssb'],
            'gm': ['lb', 'gb', 'gc'],
            'gmb': ['lb', 'gb'],
            }
    tx = [{}, {}]  # Dictionary of the texts. [Left column i.e. F2, Right column i.e. F1]

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
        fid02 = np.copy(fid0[0, :])      # F2 FID
        fid01 = np.copy(fid0[:, 0])      # F1 FID
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
        ax.set_xlabel(r'$\delta\,$' + misc.nuc_format(acqu2s['nuc'])+' /ppm')
        ax.set_ylabel(r'$\delta\,$' + misc.nuc_format(acqu1s['nuc'])+' /ppm')
        misc.set_ylim(axf2, (apodf2, -apodf2))
        misc.set_ylim(axf1, (apodf1, -apodf1))
        misc.set_fontsizes(ax, 14)

        # Redraw
        fig.canvas.draw()

    # --------------------------------------------------
    # update_SI = [update_SI_f2, update_SI_f1]
    def update_SI_f2(v):
        try:
            SI = eval(v)
            procs['zf'][1] = SI
        except Exception:
            pass
        update()

    def update_SI_f1(v):
        try:
            SI = eval(v)
            procs['zf'][0] = SI
        except Exception:
            pass
        update()
    update_SI = [update_SI_f2, update_SI_f1]

    # --------------------------------------------------
    # update_tdeff = [update_tdeff_f2, update_tdeff_f1]
    def update_tdeff_f2(v):
        try:
            val = eval(v)
            procs['tdeff'][1] = int(val)
        except Exception:
            pass
        tx[0]['tdeff'].set_text('{:.0f}'.format(procs['tdeff'][1]))
        update()

    def update_tdeff_f1(v):
        try:
            val = eval(v)
            procs['tdeff'][0] = int(val)
        except Exception:
            pass
        tx[1]['tdeff'].set_text('{:.0f}'.format(procs['tdeff'][0]))
        update()
    update_tdeff = [update_tdeff_f2, update_tdeff_f1]

    # --------------------------------------------------
    # update_mode = [update_mode_f2, update_mode_f1]
    def update_mode_f2(label):
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
        try:
            ssb = eval(v)
            procs['wf'][1]['ssb'] = ssb
        except Exception:
            pass
        tx[0]['ssb'].set_text('{:.0f}'.format(procs['wf'][1]['ssb']))
        update()

    def update_ssb_f1(v):
        try:
            ssb = eval(v)
            procs['wf'][0]['ssb'] = ssb
        except Exception:
            pass
        tx[1]['ssb'].set_text('{:.0f}'.format(procs['wf'][0]['ssb']))
        update()
    update_ssb = [update_ssb_f2, update_ssb_f1]

    # --------------------------------------------------
    # update_lb = [update_lb_f2, update_lb_f1]
    def update_lb_f2(v):
        try:
            lb = eval(v)
            procs['wf'][1]['lb'] = lb
        except Exception:
            pass
        tx[0]['lb'].set_text('{:.0f}'.format(procs['wf'][1]['lb']))
        update()

    def update_lb_f1(v):
        try:
            lb = eval(v)
            procs['wf'][0]['lb'] = lb
        except Exception:
            pass
        tx[1]['lb'].set_text('{:.0f}'.format(procs['wf'][0]['lb']))
        update()
    update_lb = [update_lb_f2, update_lb_f1]

    # --------------------------------------------------
    # update_gb = [update_gb_f2, update_gb_f1]
    def update_gb_f2(v):
        try:
            gb = eval(v)
            procs['wf'][1]['gb'] = gb
        except Exception:
            pass
        tx[0]['gb'].set_text('{:.2f}'.format(procs['wf'][1]['gb']))
        update()

    def update_gb_f1(v):
        try:
            gb = eval(v)
            procs['wf'][0]['gb'] = gb
        except Exception:
            pass
        tx[1]['gb'].set_text('{:.2f}'.format(procs['wf'][0]['gb']))
        update()
    update_gb = [update_gb_f2, update_gb_f1]

    # --------------------------------------------------
    # update_gc = [update_gc_f2, update_gc_f1]
    def update_gc_f2(v):
        try:
            gc = eval(v)
            procs['wf'][1]['gc'] = gc
        except Exception:
            pass
        tx[0]['gc'].set_text('{:.2f}'.format(procs['wf'][1]['gc']))
        update()

    def update_gc_f1(v):
        try:
            gc = eval(v)
            procs['wf'][0]['gc'] = gc
        except Exception:
            pass
        tx[1]['gc'].set_text('{:.2f}'.format(procs['wf'][0]['gc']))
        update()
    update_gc = [update_gc_f2, update_gc_f1]

    # ------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------

    # Draw the figure panel
    fig = plt.figure('Interactive Processing')
    fig.set_size_inches(15, 9)
    plt.subplots_adjust(left=0.1, bottom=0.05, right=0.725, top=0.95, hspace=0.75, wspace=0.25)
    ax = fig.add_subplot(4, 3, (1, 9))     # spectrum
    axf2 = fig.add_subplot(4, 3, 10)        # fid F2
    axf1 = fig.add_subplot(4, 3, 11)        # fid F1
    axhm = fig.add_subplot(4, 3, 12)        # fid total

    # Spectrum plot
    ax.set_title('Spectrum')
    if CNT:
        cnt = figures.ax2D(ax, ppm_f2, ppm_f1, data, lvl=lvl0, X_label='', Y_label='', fontsize=14)
    else:
        cnt, axcbar = figures.ax_heatmap(ax, data, zlim='auto', z_sym=True, cmap=None,
                                         xscale=ppm_f2, yscale=ppm_f1, rev=(True, True),
                                         n_xticks=10, n_yticks=10, n_zticks=10, fontsize=14)
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
    fidp1, = axf1.plot(np.arange(fid.shape[0]), fid0[:, 0].real/max(fid0[:, 0].real), c='tab:blue', lw=0.6)  # FID
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
    misc.set_ylim(axf2, (-1, 1))
    misc.mathformat(axf2)
    #       x
    misc.pretty_scale(axf2, (0, fid.shape[1]), n_major_ticks=4)
    #   FID F1 y-axis
    #       y
    misc.set_ylim(axf1, (-1, 1))
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


def interactive_phase_1D(ppmscale, S, reference=None):
    """
    This function allow to adjust the phase of 1D spectra interactively. Use the mouse scroll to regulate the values.
    Press the "Z" key to toggle between automatic and manual adjustment of the window.

    Parameters
    ----------
    ppmscale : 1darray
        ppm scale of the spectrum. Used to regulate the pivot position
    S :  1darray
        Spectrum to be phased. Must be complex!
    reference : list of 1darray or Spectrum_1D object
        Reference spectrum to be used for phasing. Can be also given as ``[ppm, spectrum]``

    Returns
    -------
    phased_data : 1darray
        Phased spectrum
    final_values: tuple
        ``(p0, p1, pivot)``
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
    fig.set_size_inches(15, 8)
    plt.subplots_adjust(left=0.075, bottom=0.10, right=0.8, top=0.9)    # Make room for the sliders
    ax = fig.add_subplot()

    if reference is not None:
        if isinstance(reference, Spectrum_1D):
            ax.plot(reference.ppm, reference.S.real, c='k', lw=1, label='Reference')
        else:
            ax.plot(reference[0], reference[1].real, c='k', lw=1, label='Reference')

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

    box_p90 = plt.axes([0.81, 0.225, 0.085, 0.05])      # +90
    box_m90 = plt.axes([1-0.095, 0.225, 0.085, 0.05])  # -90

    box_z = plt.axes([0.05, 0.02, 0.02, 0.06])          # Zoom state button
    # Remove ticks
    box_z.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
    # Make it red
    box_z.set_alpha(0.4)
    box_z.set_facecolor('tab:red')
    box_z.text(0.5, 0.5, 'Z', ha='center', va='center', fontsize=15, transform=box_z.transAxes)

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
    p90_button = Button(box_p90, '+90°', hovercolor='0.975')
    m90_button = Button(box_m90, '-90°', hovercolor='0.975')
    saveandexit = Button(box_sande, 'SAVE AND EXIT', hovercolor='0.975')
    #   Radiobuttons
    radio = RadioButtons(box_radio, radiolabels)

    # Array 'status': 1 means active, 0 means inactive.
    stat = np.array([1, 0, 0])
    # values:     p0 p1 pivot
    P = np.array([0, 0, float(round(np.mean(ppmscale), 2))])

    zoom_adj = False

    def statmod(label):
        # changes the 'stat' array according to the radiobutton
        nonlocal stat
        stat = np.zeros(3, dtype=int)
        for k, L in enumerate(radiolabels):
            if label == L:
                stat[k] = 1

    def p90(event):
        """ add 90 degrees to phase 0 or phase 1 """
        if bool(int(stat[-1])):
            return
        for j in range(len(stat)):
            if stat[j]:
                P[j] += 90
        on_scroll(None)

    def m90(event):
        """ removes 90 degrees to phase 0 or phase 1 """
        if bool(int(stat[-1])):
            return
        for j in range(len(stat)):
            if stat[j]:
                P[j] -= 90
        on_scroll(None)

    def roll_up(event):
        # Increase the active value of its 'sens'
        for k in range(3):
            if stat[k]:
                P[k] += sens[k]

    def roll_down(event):
        # Decrease the active value of its 'sens'
        for k in range(3):
            if stat[k]:
                P[k] -= sens[k]

    def sens_up(event):
        # Doubles the active 'sens'
        for k in range(3):
            if stat[k]:
                sens[k] = sens[k]*2

    def sens_down(event):
        # Halves the active 'sens'
        for k in range(3):
            if stat[k]:
                sens[k] = sens[k]/2

    def on_scroll(event):
        # When you move the mouse scroll
        if event is not None:
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
            idxs = []
            for x in ax.get_xlim():
                if x < min(ppmscale):
                    xval = min(ppmscale)
                elif x > max(ppmscale):
                    xval = max(ppmscale)
                else:
                    xval = x
                idxs.append(misc.ppmfind(ppmscale, xval)[0])
            sl = slice(*sorted(idxs))

            T = max(data_inside[sl].real)
            B = min(data_inside[sl].real)
            ax.set_ylim(B - 0.05*T, T + 0.05*T)
        # Update
        fig.canvas.draw()

    def reset(event):
        # Reset the phase and pivot values to their starting point
        nonlocal P, sens
        P = np.array([0, 0, round(np.mean(ppmscale), 2)])
        sens = [5, 5, 0.1]
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
            zoom_adj = not zoom_adj
            if zoom_adj:
                box_z.set_facecolor('tab:green')
            else:
                box_z.set_facecolor('tab:red')
            plt.draw()

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
    ax.set_title('Use mouse scroll to adjust the phases. Press Z to toggle zoom adjustment')

    # Link widgets to functions
    up_button.on_clicked(sens_up)
    down_button.on_clicked(sens_down)
    radio.on_clicked(statmod)
    reset_button.on_clicked(reset)
    save_button.on_clicked(save)
    p90_button.on_clicked(p90)
    m90_button.on_clicked(m90)
    saveandexit.on_clicked(save_and_exit)
    fig.canvas.mpl_connect('scroll_event', on_scroll)
    fig.canvas.mpl_connect('key_press_event', zoom_onoff)

    plt.show()

    phased_data = phase(S, p0=p0_f, p1=p1_f, pivot=pivot_f)
    final_values = p0_f, p1_f, pivot_f
    print('p0: {:.3f}, p1: {:.3f}, pv: {:.3f}\n'.format(*final_values), c='violet')
    return phased_data, final_values


def interactive_phase_2D(ppm_f1, ppm_f2, S, hyper=True):
    """
    Interactively adjust the phases of a 2D spectrum.
    First select the traces you want to use as reference, then use the mouse scroll to adjust the phase angles.
    ``S`` must be complex or hypercomplex, so BEFORE TO UNPACK with :func:`klassez.processing.unpack_2D`

    Parameters
    ----------
    ppm_f1 : 1darray
        ppm scale of the indirect dimension
    ppm_f2 : 1darray
        ppm scale of the direct dimension
    S : 2darray
        Data to be phase-adjusted
    hyper : bool
        True if ``S`` is hypercomplex, False if ``S`` is just complex

    Returns
    -------
    S : 2darray
        Phased data
    final_values_f1 : tuple
        ``(p0_f1, p1_f1, pivot_f1)``
    final_values_f2 : tuple
        ``(p0_f2, p1_f2, pivot_f2)``

    .. seealso::

        :func:`klassez.anal.select_traces`

        :func:`klassez.processing.ps`

        :func:`klassez.gui.interactive_phase_1D`
    """

    # Unpack the hyperser
    if hyper:
        S_rr, S_ri, S_ir, S_ii = processing.unpack_2D(S)
    else:
        S_rr, _ = S.real, S.imag

    zoom_adj = False

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
            S_rr, _ = S.real, S.imag
        # Create empty lists for the traces
        f1, f2 = [], []
        npk = len(coord)
        for i in range(npk):
            y = anal.get_trace(S_rr, ppm_f2, ppm_f1, coord[i][0], column=True)
            f1.append(y)
            x = anal.get_trace(S_rr, ppm_f2, ppm_f1, coord[i][1], column=False)
            f2.append(x)
        return f1, f2

    # Get the traces on which to see the effects of phase adjustment
    coord = anal.select_traces(ppm_f1, ppm_f2, S_rr)
    npk = len(coord)

    # Make the figure
    fig = plt.figure('Phase Correction')
    fig.set_size_inches(15, 8)
    plt.subplots_adjust(left=0.125, bottom=0.125, right=0.8, top=0.9, wspace=0.10, hspace=0.20)    # Make room for the sliders

    # Get the traces
    f1, f2 = maketraces(coord, S, ppm_f2, ppm_f1, hyper)

    # Set initial values

    # Create boxes
    #   for sentitivity sliders
    box_us = plt.axes([0.815, 0.825, 0.08, 0.075])      # increase sensitivity
    box_ds = plt.axes([0.905, 0.825, 0.08, 0.075])      # decrease sensitivity
    #   for buttons
    box_save = plt.axes([0.81, 0.15, 0.085, 0.04])      # save button
    box_reset = plt.axes([1-0.095, 0.15, 0.085, 0.04])  # reset button
    box_sande = plt.axes([0.81, 0.10, 0.18, 0.04])      # save and exit button
    box_radio = plt.axes([0.81, 0.55, 0.18, 0.25])      # radio buttons
    box_dimen = plt.axes([0.81, 0.35, 0.18, 0.18])      # radio buttons
    box_z = plt.axes([0.05, 0.02, 0.02, 0.06])          # Zoom state button
    # Remove ticks
    box_z.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
    # Make it red
    box_z.set_alpha(0.4)
    box_z.set_facecolor('tab:red')
    box_z.text(0.5, 0.5, 'Z', ha='center', va='center', fontsize=15, transform=box_z.transAxes)

    box_p90 = plt.axes([0.81, 0.225, 0.085, 0.05])      # +90
    box_m90 = plt.axes([1-0.095, 0.225, 0.085, 0.05])  # -90

    radiolabels = [     # labels for the radio buttons
            '0$^{th}$-order\nphase correction',
            '1$^{st}$-order\nphase correction',
            '1$^{st}$-order\npivot'
            ]

    # Make the sliders
    #   for sensitivity
    up_button = Button(box_us, r'$\uparrow$', hovercolor='0.975')
    down_button = Button(box_ds, r'$\downarrow$', hovercolor='0.975')
    # Make the buttons
    save_button = Button(box_save, 'SAVE', hovercolor='0.975')
    reset_button = Button(box_reset, 'RESET', hovercolor='0.975')
    saveandexit = Button(box_sande, 'SAVE AND EXIT', hovercolor='0.975')

    p90_button = Button(box_p90, '+90°', hovercolor='0.975')
    m90_button = Button(box_m90, '-90°', hovercolor='0.975')
    #   Radiobuttons
    radio = RadioButtons(box_radio, radiolabels)
    seldim = RadioButtons(box_dimen, ['F2', 'F1'])

    # Array "sensitivity":
    sens = [    # p0 p1 pivot
            [5, 5, 0.1],    # F2
            [5, 5, 0.1]     # F1
            ]

    # "status" arrays:
    stat = np.array([1, 0, 0])  # p0, p1, pivot
    statf = np.array([1, 0])    # f2, f1

    P = np.array([  # Values
        [0, 0, round(np.mean([min(ppm_f2), max(ppm_f2)]), 2)],  # F2
        [0, 0, round(np.mean([min(ppm_f1), max(ppm_f1)]), 2)]   # F1
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
            statf = np.array([1, 0])
        if label == 'F1':
            statf = np.array([0, 1])

    def roll_up(event):
        # Increase the active value of its 'sens'
        for i in range(2):
            for k in range(3):
                if statf[i] and stat[k]:
                    P[i, k] += sens[i][k]
                    # Manage out-of-border
                    if P[0][2] > max(ppm_f2):
                        P[0][2] = round(np.floor(max(ppm_f2)), 2)
                    if P[1][2] > max(ppm_f1):
                        P[1][2] = round(np.floor(max(ppm_f1)), 2)

    def roll_down(event):
        # Decrease the active value of its 'sens'
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
        for i in range(2):
            for k in range(3):
                if statf[i] and stat[k]:
                    sens[i][k] = sens[i][k]*2

    def sens_down(event):
        # Halves the active 'sens'
        for i in range(2):
            for k in range(3):
                if statf[i] and stat[k]:
                    sens[i][k] = sens[i][k]/2

    def p90(event):
        """ add 90 degrees to phase 0 or phase 1 """
        if bool(int(stat[-1])):
            return
        for j in range(len(stat)):
            if stat[j]:
                if statf[0]:
                    P[0][j] += 90
                else:
                    P[1][j] += 90
                break
        on_scroll(None)

    def m90(event):
        """ removes 90 degrees to phase 0 or phase 1 """
        if bool(int(stat[-1])):
            return
        for j in range(len(stat)):
            if stat[j]:
                if statf[0]:
                    P[0][j] -= 90
                else:
                    P[1][j] -= 90
                break
        on_scroll(None)

    def on_scroll(event):
        # When you move the mouse scroll
        if event is not None:
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
            zoom_adj = not zoom_adj
            if zoom_adj:
                box_z.set_facecolor('tab:green')
            else:
                box_z.set_facecolor('tab:red')
            fig.canvas.draw()

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
        if np.mod(i+1, 2) != 0:
            ax[i].set_xlim(max(ppm_f2), min(ppm_f2))
        else:
            ax[i].set_xlim(max(ppm_f1), min(ppm_f1))
    # Set vertical limits
    for i in range(npk):
        for j in range(2):
            if j == 0:    # left
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
    plt.text(0.5, 0.98, 'Use mouse scroll to adjust the phases. Press Z to toggle zoom adjustment',
             ha='center', va='top', transform=fig.transFigure, fontsize=16)

    phases_text = plt.text(0.975, 0.015,
                           'p02={:7.2f} | p12={:7.2f} | pv2={:7.2f} || p01={:7.2f} | p11={:7.2f} | pv1={:7.2f}'.format(*P[0], *P[1]),
                           ha='right', va='bottom', transform=fig.transFigure, fontsize=10)

    # Connect the widgets to the functions
    reset_button.on_clicked(reset)
    save_button.on_clicked(save)
    saveandexit.on_clicked(save_and_exit)

    p90_button.on_clicked(p90)
    m90_button.on_clicked(m90)

    up_button.on_clicked(sens_up)
    down_button.on_clicked(sens_down)
    radio.on_clicked(statmod)
    seldim.on_clicked(change_dim)
    fig.canvas.mpl_connect('scroll_event', on_scroll)
    fig.canvas.mpl_connect('key_press_event', zoom_onoff)

    plt.show()

    # Phase the spectrum with the final Parameters
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
    print('F2 - p0: {:.3f}, p1: {:.3f}, pv: {:.3f}'.format(*final_values_f2), c='violet')
    print('F1 - p0: {:.3f}, p1: {:.3f}, pv: {:.3f}\n'.format(*final_values_f1), c='violet')

    return S, final_values_f1, final_values_f2


def interactive_qfil(ppm, data_in, SFO1):
    """
    Interactive function to design a gaussian filter with the aim of suppressing signals in the spectrum.
    You can adjust position and width of the filter scrolling with the mouse.

    If you want to use the frequency scale instead of the ppm one, pass the scale as ``ppm`` and set ``SFO1=1``.
    When using these values for :func:`klassez.processing.qfil`, remember to pass the same scale and ``SFO1``!

    Parameters
    ----------
    ppm : 1darray
        ppm scale of the spectrum
    data_in : 1darray
        Spectrum on which to apply the filter.
    SFO1 : float
        Spectrometer Larmor frequency

    Returns
    -------
    u : float
        Position of the gaussian filter /ppm
    s : float
        Width of the gaussian filter (FWHM) /Hz
    """

    # Safe copy
    data = np.copy(data_in.real)

    # Initialize the values: u at the center of the spectrum, s as 100 points
    cnv = 2 * (2 * np.log(2))**0.5      # conversion factor sigma -> FWHM
    u = np.mean(ppm)
    fwhm = misc.freq2ppm(50, SFO1)     # in ppm
    s = fwhm / cnv

    sens = misc.freq2ppm(10, SFO1)  # one mouse 'tick'
    stat = 1    # move s

    # Make the filter with start values
    G = sim.f_gaussian(ppm, u, s)
    G /= max(G)     # Normalize it to preserve intensities

    # Make the figure
    fig = plt.figure('Adjust Position and Width for QFIL')
    fig.set_size_inches(figures.figsize_large)
    plt.subplots_adjust(left=0.10, bottom=0.15, right=0.85, top=0.90)
    ax = fig.add_subplot()

    # Plot
    #   Original spectrum
    figures.ax1D(ax, ppm, data, c='k', lw=0.8, X_label=r'$\delta\, $/ppm', Y_label='Intensity /a.u.', label='Original')
    #   Filter
    G_plot, = ax.plot(ppm, G*np.max(data), c='tab:red', lw=0.6, ls='--', label='Filter')
    #   Processed data
    pdata = data * (1 - G)      # Compute it
    p_spect, = ax.plot(ppm, pdata, c='tab:blue', lw=0.7, label='Processed')

    # --------------------------------------------------

    # WIDGETS
    #   Radio-buttons to select which value to modify
    radio_box = plt.axes([0.875, 0.40, 0.10, 0.20])
    radio_labels = ['$u$', r'$\Gamma$']
    radio = RadioButtons(radio_box, radio_labels, active=1)

    # Modify sensitivity buttons
    up_box = plt.axes([0.875, 0.70, 0.05, 0.05])
    up_button = Button(up_box, r'$\uparrow$')
    dn_box = plt.axes([0.925, 0.70, 0.05, 0.05])
    dn_button = Button(dn_box, r'$\downarrow$')

    # Text
    values_text = '\n'.join([
        r'$u = $' + f'{u:12.5f} ppm',
        r'$\Gamma = $' + f'{misc.ppm2freq(s*cnv, SFO1):12.5f}  Hz',
        ])
    v_text = ax.text(0.855, 0.35, values_text, ha='left', va='top', transform=fig.transFigure, fontsize=14)

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
        values_text = '\n'.join([
            f'u: {u:12.5f} ppm',
            f's: {misc.ppm2freq(s, SFO1)*cnv:12.5f}  Hz',
            ])
        v_text.set_text(values_text)
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

    shz = misc.ppm2freq(s, SFO1)
    fwhmhz = shz * cnv

    return u, fwhmhz


# =================================================================================
# FIT GUI
# =================================================================================

def gen_iguess_2D(ppm_f1, ppm_f2, tr1, tr2, u1, u2, acqus, fwhm0=100, procs=None):
    """
    Generate the initial guess for the fit of a 2D signal.
    The employes model is the one of a 2D Voigt signal, acquired with the States-TPPI scheme in the indirect dimension (i.e. :func:`klassez.sim.t_2DVoigt`).
    The program allows for the inclusion of up to 10 components for the signal, in order to improve the fit.
    The acqus dictionary must contain the following keys:

        * t1: acquisition timescale in the indirect dimension (States)
        * t2: acquisition timescale in the direct dimension
        * SFO1: Larmor frequency of the nucleus in the indirect dimension
        * SFO2: Larmor frequency of the nucleus in the direct dimension
        * o1p: carrier position in the indirect dimension /ppm
        * o2p: carrier position in the direct dimension /ppm

    The signals will be processed according to the values in the ``procs`` dictionary, if given; otherwise, they will be just zero-filled up to the data size (i.e. ``(len(ppm_f1), len(ppm_f2))`` ).

    Parameters
    ----------
    ppm_f1 : 1darray
        ppm scale for the indirect dimension
    ppm_f2 : 1darray
        ppm scale for the direct dimension
    tr1 : 1darray
        Trace of the original 2D peak in the indirect dimension
    tr2 : 1darray
        Trace of the original 2D peak in the direct dimension
    u1 : float
        Chemical shift of the original 2D peak in the indirect dimension /ppm
    u2 : float
        Chemical shift of the original 2D peak in the direct dimension /ppm
    acqus : dict
        Dictionary of acquisition parameters
    fwhm0 : float
        Initial value for FWHM in both dimensions
    procs : dict
        Dictionary of processing parameters

    Returns
    -------
    final_parameters : 2darray
        Matrix of dimension (# signals, 6) that contains, for each row: v1(Hz), v2(Hz), fwhm1(Hz), fwhm2(Hz), A, b
    fit_interval : tuple of tuple
        Fitting window. ( (left_f1, right_f1), (left_f2, right_f2) )
    """

    # FIRST OF ALL, THE FIGURE!
    fig = plt.figure('Manual Computation of Initial Guess - 2D')
    fig.set_size_inches(figures.figsize_large)
    plt.subplots_adjust(left=0.05, right=0.65, top=0.90, bottom=0.10, wspace=0.2)
    ax2, ax1 = [fig.add_subplot(1, 2, w+1) for w in range(2)]

    # INITIALIZE THE VALUES

    # Values to be returned
    final_parameters = []
    fit_interval = None, None

    # limits of the window
    lim_f1 = u1 + 100/np.abs(acqus['SFO1']), u1 - 100/np.abs(acqus['SFO1'])
    lim_f2 = u2 + 100/np.abs(acqus['SFO2']), u2 - 100/np.abs(acqus['SFO2'])

    V = [{
        'u1': u1,   # ppm
        'u2': u2,   # ppm
        'fwhm1': fwhm0,  # Hz
        'fwhm2': fwhm0,  # Hz
        'k': 0.1,   # relative intensity
        'b': 0.5,   # fraction of gaussianity
        } for w in range(10)]
    I1 = processing.integrate(tr1, x=ppm_f1, dx=(2 * acqus['dw']), lims=lim_f1)
    I2 = processing.integrate(tr2, x=ppm_f2, dx=(2 * acqus['dw']), lims=lim_f2)
    A = (I1 + I2) / (2*np.pi*fwhm0)

    # Sensitivity for mouse
    sens = {
        'u1': 0.25,   # ppm
        'u2': 0.25,   # ppm
        'fwhm1': 10,     # Hz
        'fwhm2': 10,    # Hz
        'k': 0.01,      # 1%
        'b': 0.1,   # 10%
        'A': np.floor(np.log10(A)) - 1      # This goes according to order of magnitudes
        }

    # Copy initial values for reset
    V_in = [dict(q) for q in V]
    A_in = np.copy(A)
    sens_in = dict(sens)

    # conversion of the names from radio labels to dict keys
    conv_r2d = {
            r'$\delta$ /ppm': 'u',
            r'$\Gamma$ /Hz': 'fwhm',
            r'$k$': 'k',
            r'$\beta$': 'b',
            }

    # --------------------------------------------------------------------------
    # SETUP OF THE INTERACTIVE FIGURE PANEL

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
    up_button = Button(su_box, r'$\uparrow$', hovercolor='0.975')         # increase sensitivity button
    down_button = Button(giu_box, r'\downarrow$', hovercolor='0.975')    # decrease sensitivity button
    save_button = Button(save_box, 'SAVE', hovercolor='0.975')            # SAVE button
    reset_button = Button(reset_box, 'RESET', hovercolor='0.975')         # RESET button

    #   textboxes
    TB1 = [TextBox(box, '', initial=f'{value:.2f}', textalignment='center') for box, value in zip(tb1_boxes, lim_f1)]   # set limits for F1
    TB2 = [TextBox(box, '', initial=f'{value:.2f}', textalignment='center') for box, value in zip(tb2_boxes, lim_f2)]   # set limits for F2

    #   Radiobuttons for parameter selection
    peak_name = [r'$\delta$ /ppm', r'$\Gamma$ /Hz', r'$k$', r'$\beta$', r'$A$']         # Labels for the parameters
    peak_radio = RadioButtons(peak_box, peak_name, active=2, activecolor='tab:blue')      # Actual radiobuttons

    #   Sliders
    #       Peak selector
    slider = Slider(ax=slider_box, label='Active\nSignal', valmin=1, valmax=10, valinit=1, valstep=1, orientation='vertical', color='tab:blue')
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
    misc.edit_checkboxes(check, xadj=0, yadj=0.001, dim=100, color=COLORS)

    # Text that shows the current values of the parameters
    head_print = plt.text(0.725, 0.4,
                          '{:9s}:'.format(r'$\delta$ F2') + f'{V[0]["u2"]:-9.2f}\n' +
                          '{:9s}:'.format(r'$\delta$ F1') + f'{V[0]["u1"]:-9.2f}\n' +
                          '{:9s}:'.format(r'$\Gamma$ F2') + f'{V[0]["fwhm2"]:-9.2f}\n' +
                          '{:9s}:'.format(r'$\Gamma$ F1') + f'{V[0]["fwhm1"]:-9.2f}\n' +
                          '{:9s}:'.format(r'$k$') + f'{V[0]["k"]:-9.2f}\n' +
                          '{:9s}:'.format(r'$\beta$') + f'{V[0]["b"]:-9.2f}\n' +
                          '{:9s}:'.format(r'$A$') + f'{A:-9.2e}\n',
                          ha='left', va='top', transform=fig.transFigure, fontsize=12, color=COLORS[0])

    # --------------------------------------------------------------------------
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

        Parameters
        ----------
        final_parameters: list or 2darray
            sequence of the parameters: u1, u2, fwhm1, fwhm2, I, b
        acqus: dict
            2D-like acqus dictionary containing the acquisition timescales (keys t1 and t2)
        N: tuple of int
            Zero-filling values (F1, F2). Not read if procs is not None
        procs: dict
            2D-like procs dictionary.

        Returns
        -------
        peaks: list of 2darray
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
        tr_f1 = anal.get_trace(peak, ppm_f2, ppm_f1, a=values[1], column=True)   # F2 @ u1 ppm
        tr_f2 = anal.get_trace(peak, ppm_f2, ppm_f1, a=values[0], column=False)  # F1 @ u2 ppm

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
        # Organize the parameters
        values = [V[s_idx][f'{key}'] for key in ('u1', 'u2', 'fwhm1', 'fwhm2', 'k', 'b')]
        values[-2] *= A
        # Compute the 2D signal and extract the traces
        sgn_1[s_idx], sgn_2[s_idx] = make_sgn_2D(values, acqus, N=(N1, N2), procs=procs)

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
        nonlocal A
        s_idx = slider.val - 1  # active signal
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
        key2edit = get_key2edit()
        if key2edit == 'A':  # increase it by one order of magnitude
            sens['A'] += 1
        else:    # double
            sens[key2edit] *= 2

    def down_sens(event):
        """ Slot for the down-arrow button"""
        key2edit = get_key2edit()
        if key2edit == 'A':
            sens['A'] -= 1  # decrease it by one order of magnitude
        else:   # halve
            sens[key2edit] /= 2

    def redraw_text(event):
        """ Updates the text according to the current values. Also changes its color. """
        s_idx = slider.val - 1  # python numbering
        value_string = '\n'.join([
            '{:9s}:'.format(r'$\delta$ F2') + f'{V[s_idx]["u2"]:-9.2f}',
            '{:9s}:'.format(r'$\delta$ F1') + f'{V[s_idx]["u1"]:-9.2f}',
            '{:9s}:'.format(r'$\Gamma$ F2') + f'{V[s_idx]["fwhm2"]:-9.2f}',
            '{:9s}:'.format(r'$\Gamma$ F1') + f'{V[s_idx]["fwhm1"]:-9.2f}',
            '{:9s}:'.format(r'$k$') + f'{V[s_idx]["k"]:-9.2f}',
            '{:9s}:'.format(r'$\beta$') + f'{V[s_idx]["b"]:-9.2f}',
            '{:9s}:'.format(r'$A$') + f'{A:-9.2e}',
            '',
            ])
        head_print.set_text(value_string)
        head_print.set_color(COLORS[s_idx])  # color of the active signal
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

    # --------------------------------------------------------------------------

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


def get_region(ppmscale, S, fig_title='Region Selector'):
    """
    Interactively select the spectral region to be fitted.
    Returns the border ppm values.

    Parameters
    ----------
    ppmscale : 1darray
        The ppm scale of the spectrum
    S : 1darray
         The spectrum to be trimmed
    fig_title : str
        Title for the interactive figure panel

    Returns
    -------
    reg_lims : list of tuple
        Limits on ``ppmscale`` selected by the GUI.
    """

    # Creation of interactive figure panel
    fig = plt.figure(fig_title)
    fig.set_size_inches(15, 8)
    plt.subplots_adjust(left=0.065, bottom=0.115, right=0.84, top=0.90)    # Make room for the sliders
    ax = fig.add_subplot()

    # Limits of the x axis
    xlims = [max(ppmscale), min(ppmscale)]

    # Placeholders
    reg_lims = []       # container for the regions
    reg_spans = []      # container for the green spans
    sel_idx = None      # Selected region

    # Placeholder for the red text
    sel_text = fig.text(0.92, 0.80, '', ha='center', va='center', transform=fig.transFigure, fontsize=14, color='tab:red')

    # Make boxes for widgets
    add_box = plt.axes([0.855, 0.84, 0.06, 0.06])
    remove_box = plt.axes([0.925, 0.84, 0.06, 0.06])
    save_box = plt.axes([0.860, 0.05, 0.12, 0.06])

    # Box to list the selected regions
    list_box = plt.axes([0.855, 0.15, 0.13, 0.55])
    list_box.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    list_box.set_title('Selected regions')

    all_text = list_box.text(0.5, 0.95, '', ha='center', va='top', transform=list_box.transAxes, fontsize=14, color='tab:green')

    # Make buttons
    add_button = Button(add_box, 'ADD', hovercolor='0.875')
    save_button = Button(save_box, 'SAVE and EXIT', hovercolor='0.875')
    remove_button = Button(remove_box, 'REMOVE', hovercolor='0.875')

    #   SLOTS
    def onselect(xmin, xmax):
        """ Connected to the slider, changes the red text """
        sel_text.set_text(f'{xmax:8.3f}:{xmin:8.3f}')
        fig.canvas.draw()

    def write_rlims():
        """ Writes the limits in the box """
        w_str = [f'{max(w):8.3f}:{min(w):8.3f}' for w in reg_lims]
        all_text.set_text('\n'.join(w_str))
        fig.canvas.draw()

    def add(event):
        """ ADD button """
        # Get limits from the span selector
        xmin, xmax = span.extents
        # Do not allow less than 5 points selection
        if xmax - xmin < 5 * misc.calcres(ppmscale):
            return
        # Update the limits
        reg_lims.append((xmax, xmin))
        # Draw the green span and add it to the list
        tmp_span = ax.axvspan(xmin, xmax, color='tab:green', alpha=0.2, zorder=10)
        reg_spans.append(tmp_span)
        # Set the spanselector invisible otherwise it looks ugly as shit
        span.set_visible(False)
        # Write the green text
        write_rlims()

    def remove(event):
        """ REMOVE button """
        # You have selected a peak
        if sel_text.get_color() == 'violet' and sel_idx is not None:
            #  Remove the selected entry from the list of limits
            _ = reg_lims.pop(sel_idx)
            # Extract the green span and do not draw it anymore
            kill_span = reg_spans.pop(sel_idx)
            kill_span.remove()
            # Update the green text
            write_rlims()
            fig.canvas.draw()

    def get_index_selected(x):
        """ Looks in the limits list if ``x`` is inside one region """
        for idx, lims in enumerate(reg_lims):
            if min(lims) < x and x < max(lims):
                return idx

    def on_click(event):
        """ Mouse click """
        # Do stuff only if you click inside the main panel
        if event.inaxes == ax:
            nonlocal sel_idx
            if event.dblclick:  # only if you double click
                # idx might be either a number or None
                idx = get_index_selected(event.xdata)
                sel_idx = idx
                if idx is not None:
                    # Set the selected span and the selection text to violet
                    reg_spans[idx].set_color('violet')
                    onselect(min(reg_lims[idx]), max(reg_lims[idx]))
                    sel_text.set_color('violet')
            else:
                # single click: reset everything to normal
                sel_idx = None
                for reg_span in reg_spans:
                    reg_span.set_color('tab:green')
                    sel_text.set_color('tab:red')
            fig.canvas.draw()

    def save(event):
        """ SAVE button """
        nonlocal reg_lims
        # Sort the values to be returned from left to right in the ppmscale
        reg_lims = sorted(reg_lims, key=lambda w: w[0])[::-1]
        plt.close()

    # Draw the spectrum
    ax.plot(ppmscale, S.real, c='tab:blue', lw=0.8)        # Plot the data

    # Cosmetic stuff
    ax.set_xlabel(r'$\delta\,$ /ppm')
    ax.set_ylabel('Intensity /a.u.')
    misc.pretty_scale(ax, xlims, 'x')
    misc.pretty_scale(ax, ax.get_ylim(), 'y')
    misc.mathformat(ax, 'y')
    misc.set_fontsizes(ax, 20)

    # Connect the widgets to the slots
    span = SpanSelector(ax, onselect, onmove_callback=onselect, minspan=5*misc.calcres(ppmscale),
                        direction='horizontal', interactive=True, drag_from_anywhere=True,
                        props={'facecolor': 'tab:red', 'alpha': 0.3})
    add_button.on_clicked(add)
    remove_button.on_clicked(remove)
    save_button.on_clicked(save)
    fig.canvas.mpl_connect('button_press_event', on_click)

    # Start event loop
    plt.show()
    plt.close()

    return reg_lims


def interactive_smoothing(x, y, cmap='RdBu'):
    """
    Interpolate the given data with a 3rd-degree spline. Type the desired smoothing factor in the box and see the outcome directly on the figure.
    When the panel is closed, the smoothed function is returned.

    .. warning::

        Extremely slow rendering!!! There is a problem that must be fixed somewhen.


    Parameters
    ----------
    x : 1darray
        Scale of the data
    y : 1darray
        Data to be smoothed
    cmap : str
        Name of the colormap to be used to represent the weights. Must be present in ``CM``

    Returns
    -------
    sx : 1darray
        Location of the spline points
    sy : 1darray
        Smoothed ``y``
    s_f : float
        Employed smoothing factor for the spline
    weights : 1darray
        Weights vector
    """
    # Make the figure
    fig = plt.figure('Interactive Smoothing with spline')
    fig.set_size_inches(figures.figsize_large)
    plt.subplots_adjust(left=0.1, right=0.85, top=0.95, bottom=0.15)
    ax = fig.add_subplot()

    cmap = CM[f'{cmap}']        # Read the colormap

    # Get the limits for the figure
    lims = x[0], x[-1]

    # Initialize data
    s_f = 0.95                       # Smoothing factor
    size = x.shape[-1]               # Spline size
    weights = np.ones_like(x) * 0.5  # Weights vector
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
        label='Weight',
        valmin=1e-5,
        valmax=1,
        valinit=0.5,
        valstep=0.05,
        orientation='vertical'
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
        except Exception:
            pass
        size_text.set_text('Size:\n{:.0f}'.format(size))
        update_plot()

    def update_sf(text):
        """ Update s_f, write it, call update_plot"""
        nonlocal s_f
        try:
            s_f = eval(text)
        except Exception:
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
    ax.legend(fontsize=12, loc='upper right')

    # Connect widget to function
    sf_tb.on_submit(update_sf)
    size_tb.on_submit(update_size)

    # Declare span selector
    span = SpanSelector(ax, onselect, "horizontal", useblit=True,
                        props=dict(alpha=0.25, facecolor="tab:blue"),
                        interactive=True, drag_from_anywhere=True)

    # Press space and mouse left button
    fig.canvas.mpl_connect('key_press_event', press_space)
    fig.canvas.mpl_connect('button_press_event', onselect)
    fig.canvas.mpl_connect('scroll_event', mouse_scroll)

    plt.show()

    # Compute final output
    sx, sy = fit.smooth_spl(x, y, size=size, s_f=s_f, weights=weights)

    return sx, sy, s_f, weights


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

    The baseline can be computed first by initializing the x-scale on the selected window through the "SET BASL" button. The informer light next to the button becomes green if it is properly set.
    The baseline coefficients can be set with the mouse scroll analogously to any other parameter.

    When you are satisfied with your fit, press "SAVE" to write the information in the output file.
    Then, the GUI is brought back to the initial situation, and the region you were working on will be marked with a green rectangle.
    You can repeat the procedure as many times as you wish, to prepare the guess on multiple spectral windows.

    Keyboard shortcuts:

    * "increase sensitivity" : '>'
    * "decrease sensitivity" : '<'
    * mouse scroll up: 'up arrow key'
    * mouse scroll down: 'down arrow key'
    * "add a component": '+'
    * "remove the active component": '-'
    * "change component, forward": 'page up'
    * "change component, backward": 'page down'


    Parameters
    ----------
    S_in : 1darray
        Experimental spectrum
    ppm_scale : 1darray
        PPM scale of the spectrum
    t_AQ : 1darray
        Acquisition timescale
    SFO1 : float
        Nucleus Larmor frequency /MHz
    o1p : float
        Carrier frequency /ppm
    filename : str or Path
        Path to the filename where to save the information. The '.ivf' extension is added automatically.


    Returns
    -------
    None

    .. seealso::

        :func:`klassez.gui.make_iguess_auto`
    """

    # -----------------------------------------------------------------------
    # USEFUL STRUCTURES
    def rename_dic(dic, Np):
        """
        Change the keys of a dictionary with a sequence of increasing numbers, starting from 1.

        Parameters
        ----------
        dic : dict
            Dictionary to edit
        Np : int
            Number of peaks, i.e. the sequence goes from 1 to Np

        Returns
        -------
        new_dic : dict
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

        Parameters
        ----------
        peaks : dict
            Components

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

    # -------------------------------------------------------------------------------
    # Write the info on the file
    filename = Path(filename)
    filename_x = filename.with_suffix('.ivf')
    with filename_x.open('a', buffering=1) as f:
        now = datetime.now()
        date_and_time = now.strftime("%d/%m/%Y at %H:%M:%S")
        f.write('! Initial guess computed by {} on {}\n\n'.format(getpass.getuser(), date_and_time))

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

    # Baseline scale
    x_bsl = np.linspace(0, 1, N)
    # Baseline - from where to where?
    x_bsl_lims = [max(ppm_scale), min(ppm_scale)]
    x_bsl_lims_pts = [misc.ppmfind(ppm_scale, w)[0] for w in x_bsl_lims]
    if x_bsl_lims_pts[0] > x_bsl_lims_pts[1]:
        x_bsl_lims = x_bsl_lims[::-1]
    # for the plot
    x_bsl_2plot = np.linspace(*sorted(x_bsl_lims), len(x_bsl))
    whole_basl = np.zeros_like(ppm_scale)

    # Set limits
    limits = [max(ppm_scale), min(ppm_scale)]

    # Get point indices for the limits
    lim1, lim2 = sorted([misc.ppmfind(ppm_scale, limit)[0] for limit in limits])
    # Calculate the absolute intensity (or something that resembles it)
    A = np.trapezoid(np.abs(S)[lim1:lim2], dx=misc.calcres(ppm_scale*SFO1))*2*misc.calcres(acqus['t1'])
    _A = 1 * A
    # Baseline constant
    B = 10**np.floor(np.log10(A))
    _B = 1 * B
    # Make a sensitivity dictionary
    sens = {
            'u': np.abs(limits[0] - limits[1]) / 50,    # 1/50 of the SW
            'fwhm': 2.5,
            'k': 0.05,
            'b': 0.1,
            'phi': 10,
            'A': 10**(np.floor(np.log10(A)-1)),    # approximately
            'c0': 0.1,
            'c1': 0.1,
            'c2': 0.1,
            'c3': 0.1,
            'c4': 0.1,
            'B': 1,
            }
    _sens = dict(sens)                          # RESET value
    # baseline coefficients
    bas_c = np.zeros(5)
    basl = np.zeros_like(x_bsl)

    # Peaks dictionary
    peaks = {}

    # Initial figure
    fig = plt.figure('Manual Computation of Inital Guess')
    fig.set_size_inches(15, 8)
    plt.subplots_adjust(bottom=0.10, top=0.90, left=0.05, right=0.65)
    ax = fig.add_subplot()

    # make boxes for widgets
    slider_box = plt.axes([0.68, 0.10, 0.01, 0.65])     # Peak selector slider
    peak_box = plt.axes([0.875, 0.275, 0.10, 0.425])       # Radiobuttons
    up_box = plt.axes([0.815, 0.825, 0.08, 0.075])      # Increase sensitivity button
    down_box = plt.axes([0.894, 0.825, 0.08, 0.075])    # Decrease sensitivity button
    save_box = plt.axes([0.7, 0.825, 0.085, 0.04])      # Save button
    reset_box = plt.axes([0.7, 0.865, 0.085, 0.04])     # Reset button
    group_box = plt.axes([0.72, 0.50, 0.06, 0.04])      # Textbox for the group selection
    plus_box = plt.axes([0.72, 0.65, 0.08, 0.075])     # Add button
    minus_box = plt.axes([0.72, 0.55, 0.08, 0.075])    # Minus button
    basset_box = plt.axes([0.875, 0.71, 0.08, 0.04])    # baseline box
    basset_flagbox = plt.axes([0.955, 0.71, 0.02, 0.04])
    basset_flagbox.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    basset_flagbox.set_fc('tab:red')

    # Make widgets
    #   Buttons
    up_button = Button(up_box, r'$\uparrow$', hovercolor='0.975')
    down_button = Button(down_box, r'$\downarrow$', hovercolor='0.975')
    save_button = Button(save_box, 'SAVE', hovercolor='0.975')
    reset_button = Button(reset_box, 'RESET', hovercolor='0.975')
    plus_button = Button(plus_box, '$+$', hovercolor='0.975')
    minus_button = Button(minus_box, '$-$', hovercolor='0.975')
    basset_button = Button(basset_box, 'Set BASL', hovercolor='0.975')

    #   Textbox
    group_tb = TextBox(group_box, 'Group', textalignment='center')

    #   Radiobuttons
    peak_name = [r'$\delta$ /ppm', r'$\Gamma$ /Hz', '$k$', r'$\beta$', r'$\phi$', '$A$',
                 r'$c_0$', r'$c_1$', r'$c_2$', r'$c_3$', r'$c_4$', r'$B$']
    peak_radio = RadioButtons(peak_box, peak_name, activecolor='tab:blue')      # Signal parameters
    peak_box.text(0, 0.5, '-'*30, ha='left', va='center', transform=peak_box.transAxes)

    #   Slider
    slider = Slider(ax=slider_box, label='Active\nSignal', valmin=0, valmax=1-1e-3, valinit=0, valstep=1e-10, orientation='vertical', color='tab:blue')

    # -------------------------------------------------------------------------------
    # SLOTS
    def make_x_basl(event):
        nonlocal x_bsl_lims, x_bsl, basl

        x_bsl_lims = sorted(ax.get_xlim())
        n_pts_basl = int(np.abs(x_bsl_lims[1] - x_bsl_lims[0]) / misc.calcres(ppm_scale))
        x_bsl = np.linspace(0, 1, n_pts_basl)
        x_bsl_2plot = np.linspace(*x_bsl_lims, n_pts_basl)[::-1]
        basl = B * misc.polyn(x_bsl, bas_c)
        basl_plot.set_data(x_bsl_2plot, basl)
        basset_flagbox.set_fc('tab:green')
        plt.draw()

    def redraw():
        for v1, v2 in zip(sorted(ax.get_xlim()), sorted(x_bsl_lims)):
            if np.abs(v1 - v2) > 1e-4:
                basset_flagbox.set_fc('tab:red')
                break
        plt.draw()

    def radio_changed(event):
        """ Change the printed value of sens when the radio changes """
        active = peak_name.index(peak_radio.value_selected)
        param = list(sens.keys())[active]
        write_sens(param)

    def up_sens(event):
        """ Doubles sensitivity of active parameter """
        active = peak_name.index(peak_radio.value_selected)
        param = list(sens.keys())[active]
        sens[param] *= 2
        write_sens(param)

    def down_sens(event):
        """ Halves sensitivity of active parameter """
        active = peak_name.index(peak_radio.value_selected)
        param = list(sens.keys())[active]
        sens[param] /= 2
        write_sens(param)

    def up_value(param, idx):
        """ Increase the value of param of idx-th peak """
        if param == 'A':        # It is outside the peaks dictionary!
            nonlocal A
            A += sens['A']
        elif param == 'B':
            nonlocal B
            B += sens['B']
        elif 'c' in param:
            i_c = int(eval(param.replace('c', '')))
            bas_c[i_c] += sens[param]
        else:
            peaks[idx].__dict__[param] += sens[param]
            # Make safety check for b
            if peaks[idx].b > 1:
                peaks[idx].b = 1

    def down_value(param, idx):
        """ Decrease the value of param of idx-th peak """
        if param == 'A':    # It is outside the peaks dictionary!
            nonlocal A
            A -= sens['A']
        elif param == 'B':
            nonlocal B
            B -= sens['B']
        elif 'c' in param:
            i_c = int(eval(param.replace('c', '')))
            bas_c[i_c] -= sens[param]
        else:
            peaks[idx].__dict__[param] -= sens[param]
            # Safety check for fwhm
            if peaks[idx].fwhm < 0:
                peaks[idx].fwhm = 0
            # Safety check for b
            if peaks[idx].b < 0:
                peaks[idx].b = 0

    def scroll(event):
        """ Connection to mouse scroll """
        # Get the active parameter and convert it into Peak's attribute
        active = peak_name.index(peak_radio.value_selected)
        param = list(sens.keys())[active]
        # Get the active peak
        if Np == 0:  # No peaks!
            idx = 0
        else:
            idx = int(np.floor(slider.val * Np) + 1)
        if Np != 0 or 'c' in param or 'B' in param:
            nonlocal whole_basl
            # Fork for up/down
            if event.button == 'up':
                up_value(param, idx)
            if event.button == 'down':
                down_value(param, idx)

            # Recompute the baseline
            basl = B * misc.polyn(x_bsl, bas_c)
            whole_basl = misc.sum_overlay(np.zeros_like(ppm_scale), basl, x_bsl_lims[1], ppm_scale)
            basl_plot.set_ydata(basl)

        if Np != 0:
            # Recompute the components
            for k, _ in enumerate(peaks):
                p_sgn[k+1].set_ydata(peaks[k+1](A)[lim1:lim2])

            # Recompute the total trace
            p_fit.set_ydata(whole_basl[lim1:lim2]+calc_total(peaks)[lim1:lim2])
        # Update the text
        write_par(idx)
        write_bpar()
        redraw()

    def write_bpar():
        valuesb_print.set_text('{:+7.3f}\n{:+7.3f}\n{:+7.3f}\n{:+7.3f}\n{:+7.3f}\n{:5.2e}'.format(*bas_c, B))

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
        if not Np:  # Clear the textbox and do nothing more
            group_tb.text_disp.set_text('')
            plt.draw()
            return
        # Get active peak
        idx = int(np.floor(slider.val * Np) + 1)
        try:
            group = int(eval(text))
        except Exception:
            group = peaks[idx].group
        group_tb.text_disp.set_text('')
        peaks[idx].group = group
        write_par(idx)
        write_bpar()
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
            write_bpar()
        redraw()

    def key_binding(event):
        """ Keyboard """
        key = event.key
        if key == 'w':
            make_x_basl(0)
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
        nonlocal A, sens, bas_c, basl, whole_basl, B
        bas_c = np.zeros(5)
        basl = np.zeros_like(x_bsl)
        basl_plot.set_ydata(basl)
        whole_basl = np.zeros_like(ppm_scale)
        Q = Np
        for k in range(Q):
            remove_peak(event)
        A = _A
        B = _B

        sens = dict(_sens)
        ax.set_xlim(*_xlim)
        ax.set_ylim(*_ylim)
        active = peak_name.index(peak_radio.value_selected)
        param = list(sens.keys())[active]
        write_sens(param)
        redraw()

    def add_peak(event):
        """ Add a component """
        nonlocal Np
        # Increase the number of peaks
        Np += 1
        # Add an entry to the dictionary labelled as last
        peaks[Np] = fit.Peak(acqus, u=np.mean(ax.get_xlim()), N=N, group=lastgroup)
        # Plot it and add the trace to the plot dictionary
        p_sgn[Np] = ax.plot(ppm_scale[lim1:lim2], peaks[Np](A)[lim1:lim2], lw=0.8)[-1]
        # Move the slider to the position of the new peak
        slider.set_val((Np - 1) / Np)
        # Recompute the step of the slider
        slider.valstep = 1 / Np
        # Calculate the total trace with the new peak
        total = calc_total(peaks)
        # Update the total trace plot
        p_fit.set_ydata(whole_basl[lim1:lim2] + total[lim1:lim2])
        # Update the text
        write_par(Np)
        write_bpar()
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
        p_fit.set_ydata(whole_basl[lim1:lim2] + total[lim1:lim2])
        # Change the slider position
        if Np == 0:  # to zero and do not make it move
            slider.set_val(0)
            slider.valstep = 1e-10
            write_par(0)
            write_bpar()
        elif Np == 1:   # To zero and that's it
            slider.set_val(0)
            slider.valstep = 1 / Np
            write_par(1)
            write_bpar()
        else:   # To the previous point
            if idx == 1:
                slider.set_val(0)
            else:
                slider.set_val((idx - 2) / Np)     # (idx - 1) -1
            slider.valstep = 1 / Np
            write_par(int(np.floor(slider.val * Np) + 1))
            write_bpar()
        redraw()

    def save(event):
        """ Write a section in the output file """
        nonlocal prev
        # Adjust the intensities
        # Convert the baseline coefficients
        bas_c_norm = B / A * bas_c
        fit.write_vf(filename_x, peaks, ax.get_xlim(), A, prev, bas_c=bas_c_norm)
        prev += len(peaks)

        # Mark a region as "fitted" with a green box
        ax.axvspan(*ax.get_xlim(), color='tab:green', alpha=0.1)
        # Call reset to return at the initial situation
        reset(event)

    # -------------------------------------------------------------------------------

    ax.plot(ppm_scale[lim1:lim2], S[lim1:lim2], label='Experimental', lw=1.0, c='k')  # experimental
    p_fit = ax.plot(ppm_scale[lim1:lim2], np.zeros_like(S)[lim1:lim2], label='Fit', lw=0.9, c='b', zorder=10)[-1]  # Total trace
    basl_plot = ax.plot(x_bsl_2plot, basl, label='Baseline', lw=1.0, c='mediumorchid')[-1]  # Baseline
    p_sgn = {}  # Components

    # Header for current values print
    head_print = ax.text(0.75, 0.4750,
                         '{:>7s}\n{:>5}\n{:>5}\n{:>5}\n{:>7}\n{:>7}\n{:>7}'.format(
                             r'$\delta$', r'$\Gamma$', '$k$', r'$\beta$', r'$\phi$', '$A$', 'Group'),
                         ha='right', va='top', transform=fig.transFigure, fontsize=14, linespacing=1.5)
    # Text placeholder for the values - linspacing is different to align with the header
    values_print = ax.text(0.85, 0.4750, '',
                           ha='right', va='top', transform=fig.transFigure,
                           fontsize=14, linespacing=1.55)
    ax.text(0.75, 0.2250,
            '{:>7s}\n{:>5}\n{:>5}\n{:>5}\n{:>7}\n{:>7}'.format(
                r'$c_0$', r'$c_1$', '$c_2$', '$c_3$', '$c_4$', '$B$'),
            ha='right', va='top', transform=fig.transFigure, fontsize=14, linespacing=1.5, color='mediumorchid')
    # Text placeholder for the values - linspacing is different to align with the header
    valuesb_print = ax.text(0.85, 0.2250, '',
                            ha='right', va='top', transform=fig.transFigure,
                            fontsize=14, linespacing=1.55)
    write_bpar()
    # Text to display the active sensitivity values
    sens_print = ax.text(0.875, 0.775, f'Sensitivity: $\\pm${sens["u"]:10.4g}',
                         ha='center', va='bottom', transform=fig.transFigure, fontsize=14)
    # Text to remind keyboard shortcuts
    t_uparrow = r'$\uparrow$'
    t_downarrow = r'$\downarrow$'
    keyboard_text = '\n'.join([
        f'{"KEYBOARD SHORTCUTS":^50s}',
        f'{"Key":>5s}: Action',
        '-'*50,
        f'{"<":>5s}: Decrease sens.',
        f'{">":>5s}: Increase sens.',
        f'{"+":>5s}: Add component',
        f'{"-":>5s}: Remove component',
        f'{"Pg"+t_uparrow:>5s}: Change component, up',
        f'{"Pg"+t_downarrow:>5s}: Change component, down',
        f'{t_uparrow:>5s}: Increase value',
        f'{t_downarrow:>5s}: Decrease value',
        '-'*50,
        ])
    ax.text(0.86, 0.025, keyboard_text,
            ha='left', va='bottom', transform=fig.transFigure, fontsize=8, linespacing=1.55)

    # make pretty scales
    ax.set_xlim(max(limits[0], limits[1]), min(limits[0], limits[1]))
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
    basset_button.on_clicked(make_x_basl)
    fig.canvas.mpl_connect('scroll_event', scroll)
    fig.canvas.mpl_connect('key_press_event', key_binding)

    plt.show()  # Start event loop
    plt.close()


def make_iguess_auto(ppm, data, SW, SFO1, o1p, filename='iguess'):
    """
    GUI to create a `.ivf` file, used as initial guess for Voigt_Fit.
    The computation of the peak positions and linewidths employs ``scipy.signal.find_peaks`` and ``scipy.signal.peak_widths``, respectively.
    In addition, peak features may be added manually by clicking with the left button twice. Unwanted features can be removed with right clicks.
    If the FWHM of a peak cannot be computed automatically, a dummy FWHM of 1 Hz is assigned automatically.
    The file `<filename>.ivf` is written upon pressing the SAVE button.
    Press Z to activate/deactivate the cursor snap.

    Parameters
    ----------
    ppm : 1darray
        PPM scale of the spectrum
    data : 1darray
        real part of the spectrum to fit
    SW : float
        Spectral width /Hz
    SFO1 : float
        Nucleus Larmor Frequency /MHz
    o1p : float
        Carrier position /ppm
    filename : str or Path
        Path to the file where to save the initial guess. The .ivf extension is added automatically.

    Returns
    -------
    None

    .. seealso::

        :func:`klassez.gui.make_iguess`

        :func:`scipy.signal.find_peaks`

        :func:`scipy.signal.peak_widths`
    """

    # MISCELLANEOUS FUNCTIONS
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
        Find the position of the peaks given height and prominence with ``scipy.signal.find_peaks``

        Parameters
        ----------
        x : 1darray
            array of x values
        y : 1darray
            array of y values
        H : float
            Threshold values (height)
        P : float
            Threshold values (prominence)

        Returns
        -------
        ks : list
            List of indices where the program found peaks
        """
        ks, *_ = find_peaks(y, height=H, prominence=P)
        return ks

    def maketotal(xj):
        """
        Compute the model trace, given the peak positions

        Parameters
        ----------
        xj: list
            Indices of peak positions

        Returns
        -------
        peak_in: list of fit.Peak objects
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
            lims = (freq[u] - IW * fwhms[k]/2, freq[u] + IW * fwhms[k]/2)
            try:
                As.append(processing.integrate(s, freq, dx=(0.5 * SW), lims=lims))
            except Exception:
                As.append(1)
                print(lims)

        # Make the fit.Peak objects with the estimated parameters
        q = [(fit.Peak(acqus,
                       ppm[x],
                       fwhms[j],
                       k=As[j],
                       ))
             for j, x in enumerate(xj)]
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

    #  Initialize variables
    # Write the info on the file
    filename = Path(filename)
    filename_x = filename.with_suffix('.ivf')
    with filename_x.open('a', buffering=1) as f:
        now = datetime.now()
        date_and_time = now.strftime("%d/%m/%Y at %H:%M:%S")
        f.write('! Initial guess computed by {} on {}\n\n'.format(getpass.getuser(), date_and_time))

    prev = 0
    # Dwell time
    dw = 1/SW
    # Acquisition points
    TD = data.shape[-1]
    # Acquisition timescale
    t_aq = np.linspace(0, TD*dw, TD)
    # acquisition parameters for the fit.Peak class
    acqus = {'t1': t_aq, 'SFO1': SFO1, 'o1p': o1p, 'N': TD, }

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

    # SLOTS

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
            snap = not snap
        if key == '<':      # halves sensitivity
            sens_fork(0.5)
        if key == '>':      # doubles sensitivity
            sens_fork(2)
        if key == 'pageup':  # cycle radiobuttons down
            i = radio_labels.index(radio.value_selected)
            j = i - 1
            if j < 0:   # e.g. j=-1
                j = len(radio.labels) + j
            radio.set_active(j)
        if key == 'pagedown':   # cycle radiobuttons up
            i = radio_labels.index(radio.value_selected)
            j = i + 1
            if j > len(radio.labels) - 1:    # e.g. j = 3, len(radio.labels) = 3
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
        if (event.button == 1 and event.dblclick) or event.button == 2:    # Left double click or middle click
            xi.append(x)    # Append the value to the manual peak list
            if x in x_blacklist:    # Remove it
                i = x_blacklist.index(x)    # First find the index of x in the blacklist
                x_blacklist.pop(i)          # Then, remove it
            # Redraw everything
            update(0)
        if event.button == 3:   # Right single click
            # Find the closest peak to the point you clicked
            if len(xi):  # first in the manual list
                closest_i = misc.find_nearest(xi, x)
            else:   # if it is empty, set None
                closest_i = None
            if len(xj):  # then in the automatic list
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
        fit.write_vf(filename_x, peaks, ax.get_xlim(), 1, prev)
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

    # -------------------------------------------------------------------------------------------------------------

    # Make the figure panel
    fig = plt.figure('Automatic Computation of Initial Guess')
    fig.set_size_inches(figures.figsize_large)
    ax = fig.add_subplot()
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
    cursor.horizOn = False

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
        '-'*50,
        f'{"<":>5s}: Decrease sens.',
        f'{">":>5s}: Increase sens.',
        f'{"Pg"+uparrow_text:>5s}: Change parameter, up',
        f'{"Pg"+downarrow_text:>5s}: Change parameter, down',
        f'{uparrow_text:>5s}: Increase value',
        f'{downarrow_text:>5s}: Decrease value',
        f'{"z":>5s}: Toggle snap on cursor',
        '-'*50,
        ])
    ax.text(0.86, 0.2, keyboard_text,
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


def edit_vf(S_in, ppm_scale, regions, t_AQ, SFO1=701.125, o1p=0, filename='edited', ext='ivf'):
    """
    Allows for interactive modification of either initial guesses or fit results, read by `.vf` files.

    The GUI displays the experimental spectrum in black and the total model function in blue.
    The already-modelled regions are highlighted by a green vertical span.

    Double-click with the left button of the mouse on a region to select it: from green it will become violet.
    To change the limits of the region, press the "Change Window Limits" button. The highlighted region will become red, and you can drag it to adjust the position of the window limits.
    Press the button again to store the new values.

    To edit the model of a region, after you selected it, press the "UNLOCK" button.
    The peaks that contribute to that region gets unpacked, and become editable.
    Use the slider to move between the various components, and the mouse scroll to change the model parameters.
    Use the "+" and "-" buttons to add or remove components.
    Press "LOCK" to save your modification. Press "RESET" to restore the region model as it was before pressing "UNLOCK".

    To add a new region, select the region of the spectrum you want to fit by focusing the zoom on it using the lens button.
    Then, press the "New Region" button, that will become yellow.
    Add and move the components as you would do to edit a pre-existing region.
    Press "LOCK" to save the new region.

    To remove a region, UNLOCK it and remove all the components, then press "LOCK". The associated green span will disappear.

    At the end, press "SAVE and EXIT" to write the edited `.vf` file and close the GUI.


    Keyboard shortcuts:

    * "increase sensitivity" : '>'
    * "decrease sensitivity" : '<'
    * mouse scroll up: 'up arrow key'
    * mouse scroll down: 'down arrow key'
    * "add a component": '+'
    * "remove the active component": '-'
    * "change component, forward": 'page up'
    * "change component, backward": 'page down'


    Parameters
    ----------
    S_in : 1darray
        Experimental spectrum
    ppm_scale : 1darray
        PPM scale of the spectrum
    region : list of dict
        Regions containing the starting point, as read from :func:`klassez.fit.read_vf`
    t_AQ : 1darray
        Acquisition timescale
    SFO1 : float
        Nucleus Larmor frequency /MHz
    o1p : float
        Carrier frequency /ppm
    filename : str or Path
        Path to the filename where to save the information.
        The extension (for convention, ``'ivf'`` or ``.fvf``) can be specified through ``ext``
    ext : str
        Extension for the file, so that the final filename will be ``<filename>.<ext>``

    Returns
    -------
    None

    .. seealso::

        :func:`klassez.gui.make_iguess`
        :func:`klassez.gui.make_iguess_auto`
    """

    # -----------------------------------------------------------------------
    # USEFUL STRUCTURES
    def rename_dic(dic, Np):
        """
        Change the keys of a dictionary with a sequence of increasing numbers, starting from 1.

        Parameters
        ----------
        dic : dict
            Dictionary to edit
        Np : int
            Number of peaks, i.e. the sequence goes from 1 to Np

        Returns
        -------
        new_dic : dict
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
        This is to be used when computing the trace of the UNLOCKED zone.

        Parameters
        ----------
        peaks : dict
            Components

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

    def make_total(regions, exclude=None):
        """
        Calculate the sum trace from the ``regions`` list of dictionaries.
        This is to be used when computing the trace of the LOCKED zone.
        You can ``exclude`` a certain region of ``regions`` to prevent the rendering

        Parameters
        ----------
        region : list of dict
            Components
        exclude : int
            Index of the region to be excluded from the computation.
            If ``None``, no region is ignored.

        Returns
        -------
        total : 1darray
            Sum spectrum
        whole_basl : 1darray
            Baseline
        """
        # Placeholders
        signals = []
        whole_basl = np.zeros_like(ppm_scale)
        total = np.zeros_like(ppm_scale)
        # Loop on the regions
        for k, region in enumerate(regions):
            if k == exclude:
                continue
            # Remove the limits and the intensity from the region dictionary
            param = deepcopy(region)
            limits = param.pop('limits')
            Int = param.pop('I')
            # Take out the baseline coefficients, if there are any
            if 'bas_c' in region.keys():
                bas_c = Int * region['bas_c']
                param.pop('bas_c')
            else:
                bas_c = np.zeros(5)
            # Convert the limits from ppm to points and make the slice
            limits_pt = misc.ppmfind(ppm_scale, limits[0])[0], misc.ppmfind(ppm_scale, limits[1])[0]
            # Baseline x-scale
            x = np.linspace(0, 1, int(np.abs(limits_pt[1]-limits_pt[0])))
            # Compute baseline
            basl = misc.polyn(x, bas_c)
            whole_basl = misc.sum_overlay(whole_basl, basl, max(limits), ppm_scale)
            # Make the fit.Peak objects
            peaks = {key: fit.Peak(acqus, N=N, **value) for key, value in param.items()}
            # Get the arrays from the dictionary and put them in the list
            signals.extend([p(Int, cplx=False, get_fid=False) for _, p in peaks.items()])
        # Compute the total trace and the baseline
        total += np.sum(signals, axis=0)
        total += whole_basl
        return total, whole_basl

    def extract_unlocked(regions):
        """
        When the highlighted regions becomes UNLOCKED, this function takes the peak parameters of that region
        and computes things that can be edited through the GUI.

        Parameters
        ----------
        regions : list of dict
            Regions

        Returns
        -------
        peaks : dict of fit.Peak objects
            Editable peaks
        bas_c : 1darray
            Baseline coefficients
        total : 1darray
            Total trace of the locked region only
        whole_basl : 1darray
            Baseline of the locked region only
        """
        # Do nothing if I selected nothing
        if r_selected is None:
            return

        # Take the unlocked region out
        u_region = regions[r_selected]
        # Make a shallow copy to prevent damages
        param = deepcopy(u_region)
        # Intensity of the region -> for the baseline coefficients
        Int = param.pop('I')
        # Take the limits out as well
        _ = param.pop('limits')
        # Correct the baseline coefficients and take them out
        if 'bas_c' in u_region.keys():
            bas_c = Int * u_region['bas_c']
            param.pop('bas_c')
        else:   # instantiate them at 0
            bas_c = np.zeros(5)

        # Make the fit.Peak objects
        peaks = {key: fit.Peak(acqus, N=N, **value) for key, value in param.items()}
        # Compute the total trace EXCLUDING the selected region
        total, whole_basl = make_total(regions, r_selected)
        return peaks, bas_c, total, whole_basl

    def edit_btn_props(button, color='0.85', fw='normal', *, active=666):
        """
        Change appearance and active status of a button.

        Parameters
        ----------
        button : mpl.widget.Button
            Button to edit
        color : str
            Facecolor of the button
        fw : str
            Fontweight -> 'bold' or 'normal'
        active : bool
            Interactable or not. ``666`` leaves it as it was before.

        """
        # Store active status
        active_before = button.get_active()
        # Activate in order to be able to change the stuff without running in a block
        button.set_active(True)
        # Change the color both in the ax and as the attribute otherwise it does not render
        button.ax.set_facecolor(color)
        button.color = color
        # Set fontweight to the text
        button.label.set_fontweight(fw)

        if active != 666:   # apply active status
            button.set_active(bool(active))
        else:   # set as it was before
            button.set_active(active_before)
        # Apply changes
        button.ax.figure.canvas.draw_idle()

    # ===============================================================================
    # INITIALIZATION OF THE PROCESS

    # Get the filename as path object
    filename = Path(filename)
    filename_x = filename.with_suffix(f'.{ext}')

    # Set limits for the plot
    limits = [max(ppm_scale), min(ppm_scale)]
    # Get point indices for the limits
    lim1, lim2 = sorted([misc.ppmfind(ppm_scale, limit)[0] for limit in limits])

    # Remove the imaginary part from the experimental data and make a shallow copy
    if np.iscomplexobj(S_in):
        S = np.copy(S_in).real
    else:
        S = np.copy(S_in)
    # Make an acqus dictionary based on the input parameters.
    acqus = {'t1': t_AQ, 'SFO1': SFO1, 'o1p': o1p}

    # Variables to be used and modified during the whole GUI execution
    N = S.shape[-1]     # Number of points
    Np = 0              # Number of peaks
    lastgroup = 0       # Placeholder for last group added
    prev = 0            # Number of previous peaks
    r_selected = None   # Region selected (index)
    exit_status = 1     # Becomes "0" if you press "SAVE"

    # Baseline scale
    x_bsl = np.linspace(0, 1, N)
    # Baseline - from where to where?
    x_bsl_lims = [max(ppm_scale), min(ppm_scale)]
    x_bsl_lims_pts = [misc.ppmfind(ppm_scale, w)[0] for w in x_bsl_lims]
    if x_bsl_lims_pts[0] > x_bsl_lims_pts[1]:
        x_bsl_lims = x_bsl_lims[::-1]
    # for the plot
    x_bsl_2plot = np.linspace(*sorted(x_bsl_lims), len(x_bsl))
    whole_basl = np.zeros_like(ppm_scale)

    # PEAKS EDITING DEFAULTS
    # Peaks dictionary -> to be populated when unlocked
    peaks = {}
    # Calculate the absolute intensity (or something that resembles it)
    A = np.trapezoid(np.abs(S)[lim1:lim2],
                     dx=misc.calcres(ppm_scale*SFO1)) * 2 * misc.calcres(acqus['t1'])
    _A = 1 * A      # for reset
    # Baseline constant
    B = 10**np.floor(np.log10(A))
    # Baseline coefficients
    bas_c = np.zeros(5)
    basl = np.zeros_like(x_bsl)
    # Make a sensitivity dictionary
    sens = {
            'u': np.abs(limits[0] - limits[1]) / 50,    # 1/50 of the SW
            'fwhm': 2.5,
            'k': 0.05,
            'b': 0.1,
            'phi': 10,
            'A': 10**(np.floor(np.log10(A)-1)),    # approximately
            'c0': 0.1,
            'c1': 0.1,
            'c2': 0.1,
            'c3': 0.1,
            'c4': 0.1,
            'B': 1,
            }
    _sens = dict(sens)                          # RESET value

    # Compute the total trace and the baseline as everything is locked
    # equivalent to "loading the starting point"
    fit_total, whole_basl = make_total(regions)

    # Initial figure
    fig = plt.figure('Editing of a Voigt Fit')
    fig.set_size_inches(15, 8)
    plt.subplots_adjust(bottom=0.10, top=0.90, left=0.05, right=0.65)
    ax = fig.add_subplot()

    # WIDGETS
    # make boxes for widgets
    newwin_box = plt.axes([0.875, 0.185, 0.10, 0.06])   # New region button
    lock_box = plt.axes([0.875, 0.115, 0.10, 0.06])     # Lock/Unlock button
    editwin_box = plt.axes([0.875, 0.045, 0.10, 0.06])  # Change window limits button
    up_box = plt.axes([0.815, 0.825, 0.08, 0.075])      # Increase sensitivity button
    down_box = plt.axes([0.894, 0.825, 0.08, 0.075])    # Decrease sensitivity button
    save_box = plt.axes([0.7, 0.825, 0.085, 0.04])      # Save button
    reset_box = plt.axes([0.7, 0.865, 0.085, 0.04])     # Reset button
    plus_box = plt.axes([0.72, 0.65, 0.08, 0.075])     # Add button
    minus_box = plt.axes([0.72, 0.55, 0.08, 0.075])    # Minus button
    basset_box = plt.axes([0.875, 0.71, 0.08, 0.04])    # baseline box
    group_box = plt.axes([0.72, 0.50, 0.06, 0.04])      # Textbox for the group selection

    basset_flagbox = plt.axes([0.955, 0.71, 0.02, 0.04])    # Box that becomes red/green
    # Clean decorator to make it look like a square, red
    basset_flagbox.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    basset_flagbox.set_fc('tab:red')

    peak_box = plt.axes([0.875, 0.275, 0.10, 0.425])    # Radiobuttons
    slider_box = plt.axes([0.68, 0.10, 0.01, 0.65])     # Peak selector slider

    # Make widgets
    #   Buttons -- deactivate some by default, activated by slots
    newwin_button = Button(newwin_box, 'New Region', hovercolor='0.975')
    newwin_button.label.set_fontweight('bold')      # Make label bold
    lock_button = Button(lock_box, 'UNLOCK', hovercolor='0.975')
    lock_button.set_active(False)
    editwin_button = Button(editwin_box, 'Change window\nlimits', hovercolor='0.975')
    editwin_button.set_active(False)
    up_button = Button(up_box, r'$\uparrow$', hovercolor='0.975')
    down_button = Button(down_box, r'$\downarrow$', hovercolor='0.975')
    save_button = Button(save_box, 'SAVE and EXIT', hovercolor='0.975')
    reset_button = Button(reset_box, 'RESET', hovercolor='0.975')
    reset_button.set_active(False)
    plus_button = Button(plus_box, '$+$', hovercolor='0.975')
    plus_button.set_active(False)
    minus_button = Button(minus_box, '$-$', hovercolor='0.975')
    minus_button.set_active(False)
    basset_button = Button(basset_box, 'Set BASL', hovercolor='0.975')
    #   Textbox
    group_tb = TextBox(group_box, 'Group', textalignment='center')
    #   Radiobuttons
    peak_name = [r'$\delta$ /ppm', r'$\Gamma$ /Hz', '$k$', r'$\beta$', r'$\phi$', '$A$',
                 r'$c_0$', r'$c_1$', r'$c_2$', r'$c_3$', r'$c_4$', r'$B$']
    peak_radio = RadioButtons(peak_box, peak_name, activecolor='tab:blue')      # Signal parameters
    peak_box.text(0, 0.5, '-'*30, ha='left', va='center', transform=peak_box.transAxes)     # Horizontal line
    #   Slider
    slider = Slider(ax=slider_box, label='Active\nSignal', valmin=0, valmax=1-1e-3,
                    valinit=0, valstep=1e-10, orientation='vertical', color='tab:blue')

    # ===============================================================================
    # SLOTS

    # For figure rendering

    def redraw():
        """ Check if the limits of the baseline correspond to the figure zoom """
        for v1, v2 in zip(sorted(ax.get_xlim()), sorted(x_bsl_lims)):
            if np.abs(v1 - v2) > 1e-4:
                basset_flagbox.set_fc('tab:red')
                break
        fig.canvas.draw()

    def rspansel_onselect(vsx, vdx):
        """ Slot of the spanselector, it does NOTHING """
        pass

    # --------------------------------------------------------------------------------
    # For text updates

    def radio_changed(event):
        """ Change the printed value of sens when the radio changes """
        active = peak_name.index(peak_radio.value_selected)
        param = list(sens.keys())[active]
        write_sens(param)

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

    def write_bpar():
        """ Write the baseline coefficients values """
        valuesb_print.set_text('{:+7.3f}\n{:+7.3f}\n{:+7.3f}\n{:+7.3f}\n{:+7.3f}\n{:5.2e}'.format(*bas_c, B))

    def write_sens(param):
        """ Updates the current sensitivity value in the text """
        text = f'Sensitivity: $\\pm${sens[param]:10.4g}'
        # Update the text
        sens_print.set_text(text)
        # Redraw the figure
        plt.draw()

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
            write_bpar()
        redraw()

    # --------------------------------------------------------------------------------
    # Calculations and plotting in real time

    def make_x_basl(event):
        """ Compute the baseline x scale """
        nonlocal x_bsl_lims, x_bsl, basl
        # Get the limits from the figure zoom
        x_bsl_lims = sorted(ax.get_xlim())
        # Determine how many points do I need
        n_pts_basl = int(np.abs(x_bsl_lims[1] - x_bsl_lims[0]) / misc.calcres(ppm_scale))
        # Scale for the computation
        x_bsl = np.linspace(0, 1, n_pts_basl)
        # Scale for the plot
        x_bsl_2plot = np.linspace(*x_bsl_lims, n_pts_basl)[::-1]

        # Compute the baseline according to bas_c
        basl = B * misc.polyn(x_bsl, bas_c)
        # Update the plot
        basl_plot.set_data(x_bsl_2plot, basl)
        basset_flagbox.set_fc('tab:green')
        fig.canvas.draw()

    # Peak movements
    def up_sens(event):
        """ Doubles sensitivity of active parameter """
        active = peak_name.index(peak_radio.value_selected)
        param = list(sens.keys())[active]
        sens[param] *= 2
        write_sens(param)

    def down_sens(event):
        """ Halves sensitivity of active parameter """
        active = peak_name.index(peak_radio.value_selected)
        param = list(sens.keys())[active]
        sens[param] /= 2
        write_sens(param)

    def up_value(param, idx):
        """ Increase the value of param of idx-th peak """
        if param == 'A':        # It is outside the peaks dictionary!
            nonlocal A
            A += sens['A']
        elif param == 'B':      # Also B like A
            nonlocal B
            B += sens['B']
        elif 'c' in param:      # Baseline coefficients
            i_c = int(eval(param.replace('c', '')))
            bas_c[i_c] += sens[param]
        else:   # all the rest
            peaks[idx].__dict__[param] += sens[param]
            # Make safety check for b
            if peaks[idx].b > 1:
                peaks[idx].b = 1

    def down_value(param, idx):
        """ Decrease the value of param of idx-th peak """
        if param == 'A':    # It is outside the peaks dictionary!
            nonlocal A
            A -= sens['A']
        elif param == 'B':  # Also B like A
            nonlocal B
            B -= sens['B']
        elif 'c' in param:  # Baseline coefficients
            i_c = int(eval(param.replace('c', '')))
            bas_c[i_c] -= sens[param]
        else:   # all the rest
            peaks[idx].__dict__[param] -= sens[param]
            # Safety check for fwhm
            if peaks[idx].fwhm < 0:
                peaks[idx].fwhm = 0
            # Safety check for b
            if peaks[idx].b < 0:
                peaks[idx].b = 0

    def scroll(event):
        """ Connection to mouse scroll """
        # Get the active parameter and convert it into Peak's attribute
        active = peak_name.index(peak_radio.value_selected)
        param = list(sens.keys())[active]
        # Get the active peak
        if Np == 0:  # No peaks!
            idx = 0
        else:
            idx = int(np.floor(slider.val * Np) + 1)

        # Do things only if there are peaks or I am trying to move the baseline
        if Np != 0 or 'c' in param or 'B' in param:
            nonlocal whole_basl
            # Fork for up/down
            if event.button == 'up':
                up_value(param, idx)
            if event.button == 'down':
                down_value(param, idx)

            # Recompute the baseline
            basl = B * misc.polyn(x_bsl, bas_c)
            whole_basl = misc.sum_overlay(np.zeros_like(ppm_scale), basl, x_bsl_lims[1], ppm_scale)
            basl_plot.set_ydata(basl)

        if Np != 0:     # Only if there are peaks
            # Update the plot of the components
            for k, _ in enumerate(peaks):
                p_sgn[k+1].set_ydata(peaks[k+1](A)[lim1:lim2])

            # Recompute the total trace
            p_fit.set_ydata(whole_basl[lim1:lim2]+calc_total(peaks)[lim1:lim2])
        # Update the text
        write_par(idx)
        write_bpar()
        redraw()

    def set_group(text):
        """ Set the attribute 'group' of the active signal according to the textbox """
        if not Np:  # Clear the textbox and do nothing more
            group_tb.text_disp.set_text('')
            plt.draw()
            return
        # Get active peak
        idx = int(np.floor(slider.val * Np) + 1)
        try:
            group = int(eval(text))
        except Exception:
            group = peaks[idx].group
        group_tb.text_disp.set_text('')
        peaks[idx].group = group
        write_par(idx)
        write_bpar()
        redraw()

    def add_peak(event):
        """ Add a component """
        nonlocal Np
        # Increase the number of peaks
        Np += 1
        # Add an entry to the dictionary labelled as last
        peaks[Np] = fit.Peak(acqus, u=np.mean(ax.get_xlim()), N=N, group=lastgroup)
        # Plot it and add the trace to the plot dictionary
        p_sgn[Np] = ax.plot(ppm_scale[lim1:lim2], peaks[Np](A)[lim1:lim2], lw=0.8)[-1]
        # Move the slider to the position of the new peak
        slider.set_val((Np - 1) / Np)
        # Recompute the step of the slider
        slider.valstep = 1 / Np
        # Calculate the total trace with the new peak
        total = calc_total(peaks)
        # Update the total trace plot
        p_fit.set_ydata(whole_basl[lim1:lim2] + total[lim1:lim2])
        # Update the text
        write_par(Np)
        write_bpar()
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
        p_fit.set_ydata(whole_basl[lim1:lim2] + total[lim1:lim2])
        # Change the slider position
        if Np == 0:  # to zero and do not make it move
            slider.set_val(0)
            slider.valstep = 1e-10
            write_par(0)
            write_bpar()
        elif Np == 1:   # To zero and that's it
            slider.set_val(0)
            slider.valstep = 1 / Np
            write_par(1)
            write_bpar()
        else:   # To the previous point
            if idx == 1:
                slider.set_val(0)
            else:
                slider.set_val((idx - 2) / Np)     # (idx - 1) -1
            slider.valstep = 1 / Np
            write_par(int(np.floor(slider.val * Np) + 1))
            write_bpar()
        redraw()

    # -------------------------------------------------------------------------------
    # Behavior of the EDIT buttons

    def click(event):
        """ Mouse click behavior -> region selector """
        # Double left click in the figure panel
        if event.button == 1 and event.inaxes == ax and event.dblclick:
            # Highlight a region as selected or reset
            select_region(event.xdata)

    def select_region(x):
        """ Search if ``x`` is inside a region, if yes, marks it as selected """
        nonlocal r_selected
        # Set as "nothing is selected", useful for early return
        r_selected = None
        # Set all the regions to green -> non selected
        for vspan in reg_span:
            vspan.set_color('tab:green')

        # Get the limits of all the regions
        region_limits = [w['limits'] for w in regions]
        r_idx = None    # Nothing is selected
        # Check if the x coordinate where I clicked are within a region
        for k, limits in enumerate(region_limits):
            if min(limits) < x and x < max(limits):
                # I am inside! That's the region, store the index and break the loop
                r_idx = k
                break

        if r_idx is None:   # => The for never checked the if statement
            # Reset the appearance and behavior of all the editing buttons to default
            edit_btn_props(lock_button, '0.85', 'normal', active=False)
            edit_btn_props(editwin_button, '0.85', 'normal', active=False)
            edit_btn_props(newwin_button, '0.85', 'bold', active=True)
            fig.canvas.draw()
            # Then do nothing else
            return
        else:   # The selected region is the r_idx-th
            r_selected = r_idx

        # Highlighted region becomes purple
        reg_span[r_idx].set_color('purple')
        # With a selected region I can unlock and change limits,
        edit_btn_props(lock_button, 'lightgreen', 'bold', active=True)
        edit_btn_props(editwin_button, 'lightgreen', 'bold', active=True)
        # I cannot create a new region
        edit_btn_props(newwin_button, '0.85', 'normal', active=False)
        fig.canvas.draw()

    def lockunlock(event):
        """ Fork function for LOCKING/UNLOCKING """
        if lock_button.label.get_text() == 'LOCK':  # I am in UNLOCKED state
            lock(event)
        elif lock_button.label.get_text() == 'UNLOCK':  # I am in LOCKED state
            unlock(event)

    def unlock(event):
        """ Pass from LOCKED to UNLOCKED state """
        nonlocal peaks, Np, bas_c, A, B, fit_total, whole_basl
        # Do absolutely NOTHING if there is not a region selected
        if r_selected is None:
            return
        # Update appearance of the buttons
        #   "unlock" becomes "lock", and red
        lock_button.label.set_text('LOCK')
        edit_btn_props(lock_button, 'lightcoral', 'bold', active=True)
        #   Activate the reset
        edit_btn_props(reset_button, 'lightgreen', 'normal', active=True)
        #   Turn off new window and edit window
        edit_btn_props(editwin_button, '0.85', 'normal', active=False)
        edit_btn_props(newwin_button, newwin_button.ax.get_facecolor(), 'normal', active=False)
        # Turn off the purple vertical span
        reg_span[r_selected].set_visible(False)

        # Unpack the highlighted region and update the editable variables
        peaks, bas_c, total, whole_basl = extract_unlocked(regions)
        Np = len(peaks)
        limits = regions[r_selected]['limits']
        A = regions[r_selected]['I']
        B = regions[r_selected]['I']

        # Zoom into the region
        ax.set_xlim(sorted(limits, reverse=True))
        misc.set_ylim(ax, misc.trim_data(ppm_scale, S, limits)[1])

        # Make the baseline
        make_x_basl(event)
        # Compute the model and the baseline of whatever is not unlocked
        fit_total, whole_basl = make_total(regions, r_selected)

        # Update the plot
        #   Peak traces
        for key, _ in peaks.items():
            p_sgn[key] = ax.plot(ppm_scale[lim1:lim2], np.zeros_like(ppm_scale)[lim1:lim2])[-1]
        #   Total model
        whole_fit.set_ydata(fit_total[lim1:lim2])
        #   Populate the values and prepare for the editing
        scroll(event)
        #   Set the total model of the unlocked region visible
        p_fit.set_visible(True)
        #   Set the slider to the first component of the region
        slider.set_val(0)
        # Update the valstep of the slider according to how many peaks I have in the region
        if Np == 0:  # to zero and do not make it move
            slider.valstep = 1e-10
        else:
            slider.valstep = 1 / Np
        # Write the values next to the editable parameters
        selector(None)
        # Deactivate the dangerous buttons
        plus_button.set_active(True)
        minus_button.set_active(True)
        fig.canvas.draw()

    def lock(event):
        """ Pass from UNLOCKED to LOCKED state """
        nonlocal fit_total, whole_basl, sens
        # If the unlocked region does not have peaks, remove the whole region
        if len(peaks) == 0:
            _ = regions.pop(r_selected)
            # And the span as well
            active_span = reg_span.pop(r_selected)
            active_span.remove()
            del active_span
        else:
            # Correct the baseline coefficients
            bas_c_norm = B / A * bas_c
            # Instanciate a new dictionary for the edited values
            new_region = {
                    # Limits stay the same
                    'limits': regions[r_selected]['limits'],
                    'I': A,     # Intensity is the one of the edited region
                    'bas_c': bas_c_norm,    # Baseline from the GUI
                    **peaks                 # These are Peak objects
                    }
            # Replace the Peak object with their dictionary of parameters
            for key, item in new_region.items():
                # Avoid 'limits', 'I' and 'bas_c'
                if isinstance(item, fit.Peak):
                    new_region[key] = item.par()
            # Replace the edited region with the new one
            regions[r_selected] = new_region
            # Make the region span visible again
            reg_span[r_selected].set_visible(True)

        # Empty the peaks dictionary in a "safe" way
        for k in range(Np):
            remove_peak(event)

        # Reset sensitivity to defaults
        sens = deepcopy(_sens)

        # Change appearance and behavior of the buttons
        #   UNLOCK -> LOCK, green
        lock_button.label.set_text('UNLOCK')
        edit_btn_props(lock_button, 'lightgreen', 'bold', active=True)
        #   As in the initial state
        edit_btn_props(newwin_button, '0.85', 'bold', active=True)
        edit_btn_props(editwin_button, 'lightgreen', 'bold', active=True)
        edit_btn_props(reset_button, '0.85', 'normal', active=False)
        #   Clear selection
        select_region(np.inf)

        # Reset the zoom of the figure
        ax.set_xlim(_xlim)
        ax.set_ylim(_ylim)

        # Make the model trace of the unlocked region invisible
        p_fit.set_visible(False)

        # Compute total trace including the edited region
        fit_total, whole_basl = make_total(regions)
        # Update the plots
        whole_fit.set_ydata(fit_total[lim1:lim2])
        basl_plot.set_data(ppm_scale[lim1:lim2], whole_basl[lim1:lim2])
        # Deactivate the dangerous buttons
        plus_button.set_active(False)
        minus_button.set_active(False)
        fig.canvas.draw()

    def activate_span_region(event):
        """ Use a spanselector to change the limits of an existing window """
        # Do NOTHING if a region is not selected
        if r_selected is None:
            return

        # Activation
        if editwin_button.label.get_text() == 'Change window\nlimits':
            # Change button label and appearance
            editwin_button.label.set_text('CLICK TO\nCONFIRM')
            edit_btn_props(editwin_button, 'lightcoral', 'normal')

            # Deactivate the UNLOCK and the NEW REGION buttons
            edit_btn_props(lock_button, '0.85', 'normal', active=False)
            edit_btn_props(newwin_button, '0.85', 'normal', active=False)

            # Set the corresponding axvspan invisible
            reg_span[r_selected].set_visible(False)
            # Transform the original span into a red spanselector
            vspan_x, vspan_w = reg_span[r_selected].get_x(), reg_span[r_selected].get_width()
            original_extents = vspan_x, vspan_x + vspan_w
            rspansel.extents = original_extents
            # Make it visible and active
            rspansel.set_visible(True)
            rspansel.set_active(True)

        # Deactivation
        elif editwin_button.label.get_text() == 'CLICK TO\nCONFIRM':
            # Compute the new limits from the coordinates of the spanselector
            new_x = min(rspansel.extents)
            new_width = max(rspansel.extents)-min(rspansel.extents)
            # Do not update if region is less than 10 Hz
            if new_width * SFO1 < 10:
                return

            # Change button label and appearance
            editwin_button.label.set_text('Change window\nlimits')
            edit_btn_props(editwin_button, '0.85', 'normal')
            edit_btn_props(newwin_button, '0.85', 'normal', active=True)
            edit_btn_props(newwin_button, '0.85', 'normal', active=True)

            # Update the axvspan that was invisible
            reg_span[r_selected].set_x(new_x)
            reg_span[r_selected].set_width(new_width)
            # Set it visible
            reg_span[r_selected].set_visible(True)

            # Update the regions dict with the new limits
            regions[r_selected]['limits'] = sorted(rspansel.extents, reverse=True)

            # Make the spanselector invisible and inactive
            rspansel.set_visible(False)
            rspansel.set_active(False)

            # Reset the selection by passing an impossible value
            select_region(np.inf)

        # Update the plot
        fig.canvas.draw()

    def create_new_region(event):
        """ Create the new region as you would do in the normal GUI for creating the initial guess """
        nonlocal r_selected
        # Get the limits from the figure zoom
        limits = ax.get_xlim()
        # Set the placeholders by adding a default entry to the regions list
        regions.append({'limits': limits, 'I': _A})
        # Put the selected region as the last one
        r_selected = len(regions) - 1
        # Append it to region_limits to keep everything consistent
        region_limits.append(limits)
        # Add a vspan, green, to the list
        reg_span.append(ax.axvspan(*limits, color='tab:green', alpha=0.1))
        # Make the "New Region" button yellow as informative marker
        edit_btn_props(newwin_button, 'wheat', 'bold', active=False)
        # Open the new region as it was pre-existing using unlock
        unlock(event)

    # ------------------------------------------------------------------------
    # SAVE AND RESET
    def save(event):
        """ Write a section in the output file """
        # Write a whole section of the vf file in here, so start with prev=0
        nonlocal prev, exit_status
        prev = 0
        # Write the header of the file
        with filename_x.open('a', buffering=1) as f:
            now = datetime.now()
            date_and_time = now.strftime("%d/%m/%Y at %H:%M:%S")
            f.write('! Edited by {} on {}\n\n'.format(getpass.getuser(), date_and_time))

        for k, region in enumerate(regions):
            # get limits, intensity and baseline coefficients from "regions"
            limits = region.pop('limits')
            A = region.pop('I')
            bas_c = region.pop('bas_c')
            # Create the Peaks dictionary
            peaks = {key: fit.Peak(acqus, **dic_p) for key, dic_p in region.items()}
            # Write the vf file, the keys of the peaks are updated automatically
            fit.write_vf(filename_x, peaks, limits, A, prev, bas_c=bas_c)
            # Update the number of peaks
            prev += len(peaks)

        # Close the GUI
        plt.close()
        exit_status = 0
        print(f'Edited parameters written in {filename_x}.\n', c='tab:blue')

    def reset(event):
        """ Return the active region as it was before unlocking """
        nonlocal fit_total, whole_basl
        # Continue only if I am in the UNLOCKED situation
        if r_selected is None or lock_button.label.get_text() == 'UNLOCK':
            return
        # Update the GUI:
        #   Make the region highlight visible again
        reg_span[r_selected].set_visible(True)
        #   Change UNLOCKED -> LOCKED
        lock_button.label.set_text('UNLOCK')
        #   Correct appearance and active status of the buttons
        edit_btn_props(lock_button, 'lightgreen', 'bold', active=True)
        edit_btn_props(newwin_button, '0.85', 'bold', active=True)
        edit_btn_props(editwin_button, 'lightgreen', 'bold', active=True)
        edit_btn_props(reset_button, '0.85', 'normal', active=False)
        #   Reset the selection to nothing
        select_region(np.inf)

        # Remove all the peaks from the active region -> empties peaks dict
        for k in range(Np):
            remove_peak(event)

        # Reset the zoom of the figure
        ax.set_xlim(_xlim)
        ax.set_ylim(_ylim)

        # Compute the model and the baseline as they were before unlocking
        fit_total, whole_basl = make_total(regions)
        # Update the plot with them
        #   Make the plot of the model of the unlocked region invisible
        p_fit.set_visible(False)
        #   Plot the whole model
        whole_fit.set_ydata(fit_total[lim1:lim2])
        #   Plot the whole baseline
        basl_plot.set_data(ppm_scale[lim1:lim2], whole_basl[lim1:lim2])

        # Deactivate the dangerous buttons
        plus_button.set_active(False)
        minus_button.set_active(False)
        fig.canvas.draw()

    def key_binding(event):
        """ Keyboard """
        key = event.key
        if key == 'w':
            make_x_basl(0)
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

    # -------------------------------------------------------------------------------

    # Make the plots
    #   Experimental trace
    ax.plot(ppm_scale[lim1:lim2], S[lim1:lim2], label='Experimental', lw=1.0, c='k')
    #   Total trace of active region
    p_fit = ax.plot(ppm_scale[lim1:lim2], fit_total[lim1:lim2], lw=0.9, c='b', zorder=10)[-1]
    #   Total trace without active region
    whole_fit = ax.plot(ppm_scale[lim1:lim2], fit_total[lim1:lim2], label='Fit', lw=0.9, c='b', zorder=10)[-1]
    #   Baseline
    basl_plot = ax.plot(x_bsl_2plot, whole_basl, label='Baseline', lw=1.0, c='mediumorchid')[-1]  # Baseline
    #   Components (empty because it is populated only when you UNLOCK)
    p_sgn = {}
    #   Span that highlight the regions
    region_limits = [w['limits'] for w in regions]
    reg_span = [ax.axvspan(*lims, color='tab:green', alpha=0.1) for lims in region_limits]

    # Instantiate the text
    #   Header for current values print
    head_print = ax.text(0.75, 0.4750,
                         '{:>7s}\n{:>5}\n{:>5}\n{:>5}\n{:>7}\n{:>7}\n{:>7}'.format(
                             r'$\delta$', r'$\Gamma$', '$k$', r'$\beta$', r'$\phi$', '$A$', 'Group'),
                         ha='right', va='top', transform=fig.transFigure, fontsize=14, linespacing=1.5)
    #   Text placeholder for the values - linspacing is different to align with the header
    values_print = ax.text(0.85, 0.4750, '',
                           ha='right', va='top', transform=fig.transFigure,
                           fontsize=14, linespacing=1.55)
    ax.text(0.75, 0.2250,
            '{:>7s}\n{:>5}\n{:>5}\n{:>5}\n{:>7}\n{:>7}'.format(
                r'$c_0$', r'$c_1$', '$c_2$', '$c_3$', '$c_4$', '$B$'),
            ha='right', va='top', transform=fig.transFigure, fontsize=14, linespacing=1.5, color='mediumorchid')
    #    Text placeholder for the values - linspacing is different to align with the header
    valuesb_print = ax.text(0.85, 0.2250, '',
                            ha='right', va='top', transform=fig.transFigure,
                            fontsize=14, linespacing=1.55)
    #   Populate
    write_bpar()

    #   Text to display the active sensitivity values
    sens_print = ax.text(0.875, 0.775, f'Sensitivity: $\\pm${sens["u"]:10.4g}',
                         ha='center', va='bottom', transform=fig.transFigure, fontsize=14)

    # Make pretty scales and bigger fontsizes
    ax.legend()
    ax.set_xlim(max(limits[0], limits[1]), min(limits[0], limits[1]))
    misc.pretty_scale(ax, ax.get_xlim(), axis='x', n_major_ticks=10)
    misc.pretty_scale(ax, ax.get_ylim(), axis='y', n_major_ticks=10)
    misc.mathformat(ax)
    misc.set_fontsizes(ax, 14)

    # RESET values for xlim and ylim
    _xlim = ax.get_xlim()
    _ylim = ax.get_ylim()

    # Connect the widgets to their slots
    #   Buttons
    plus_button.on_clicked(add_peak)
    minus_button.on_clicked(remove_peak)
    up_button.on_clicked(up_sens)
    down_button.on_clicked(down_sens)
    reset_button.on_clicked(reset)
    save_button.on_clicked(save)
    basset_button.on_clicked(make_x_basl)
    editwin_button.on_clicked(activate_span_region)
    lock_button.on_clicked(lockunlock)
    newwin_button.on_clicked(create_new_region)
    #   Slider and textbox for the group
    slider.on_changed(selector)
    group_tb.on_submit(set_group)
    # Radiobuttons
    peak_radio.on_clicked(radio_changed)
    # Interactions
    fig.canvas.mpl_connect('button_press_event', click)         # Mouse clicks
    fig.canvas.mpl_connect('scroll_event', scroll)              # Mouse scroll
    fig.canvas.mpl_connect('key_press_event', key_binding)      # Keyboard
    # Span selector for the limits of the region
    rspansel = SpanSelector(ax, rspansel_onselect, direction='horizontal',
                            interactive=True, snap_values=ppm_scale,
                            props={'color': 'red', 'alpha': 0.1})
    rspansel.set_active(False)      # ...that starts unactive

    # Start event loop
    plt.show()

    # Clear memory on closing
    plt.close()

    # If exit_status is not 0 you did not SAVE!
    if exit_status:
        print('WARNING: changes were not saved!', c='yellow')


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

    When you are satisfied with your fit, press "SAVE" to write the information in the output file.
    Then, the GUI is brought back to the initial situation, and the region you were working on will be marked with a green rectangle.
    You can repeat the procedure as many times as you wish, to prepare the guess on multiple spectral windows.

    Keyboard shortcuts:

    * "increase sensitivity" : '>'
    * "decrease sensitivity" : '<'
    * mouse scroll up: 'up arrow key'
    * mouse scroll down: 'down arrow key'
    * "add a component": '+'
    * "remove the active component": '-'
    * "change component, forward": 'page up'
    * "change component, backward": 'page down'


    Parameters
    ----------
    S_in : 1darray
        Experimental spectrum
    ppm_scale : 1darray
        PPM scale of the spectrum
    expno : int
        Index of experiment of the pseudo 2D on which to compute the initial guess, in python numbering
    t_AQ : 1darray
        Acquisition timescale
    SFO1 : float
        Nucleus Larmor frequency /MHz
    o1p : float
        Carrier frequency /ppm
    filename : str
        Path to the filename where to save the information. The '.ivf' extension is added automatically.
    """

    # -----------------------------------------------------------------------
    # USEFUL STRUCTURES
    def rename_dic(dic, Np):
        """
        Change the keys of a dictionary with a sequence of increasing numbers, starting from 1.

        Parameters
        ----------
        dic : dict
            Dictionary to edit
        Np : int
            Number of peaks, i.e. the sequence goes from 1 to Np

        Returns
        -------
        new_dic : dict
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

        Parameters
        ----------
        peaks : dict
            Components

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

    # -------------------------------------------------------------------------------

    # Initial figure
    fig = plt.figure('Manual Computation of Initial Guess - Pseudo2D')
    fig.set_size_inches(15, 8)
    plt.subplots_adjust(bottom=0.10, top=0.90, left=0.05, right=0.65)
    ax = fig.add_subplot()

    # Write the info on the file
    with open(f'{filename}.ivf', 'a', buffering=1) as f:
        now = datetime.now()
        date_and_time = now.strftime("%d/%m/%Y at %H:%M:%S")
        f.write('! Initial guess computed by {} on {}\n\n'.format(getpass.getuser(), date_and_time))

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
    A = processing.integrate(S, x=ppm_scale*SFO1, dx=(2 * misc.calcres(acqus['t1'])), lims=[w*SFO1 for w in limits])
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
    up_button = Button(up_box, r'$\uparrow$', hovercolor='0.975')
    down_button = Button(down_box, r'$\downarrow$', hovercolor='0.975')
    save_button = Button(save_box, 'SAVE', hovercolor='0.975')
    reset_button = Button(reset_box, 'RESET', hovercolor='0.975')
    plus_button = Button(plus_box, '$+$', hovercolor='0.975')
    minus_button = Button(minus_box, '$-$', hovercolor='0.975')

    #   Textbox
    group_tb = TextBox(group_box, 'Group', textalignment='center')

    #   Radiobuttons
    peak_name = [r'$\delta$ /ppm', r'$\Gamma$ /Hz', '$k$', '$x_{g}$', r'$\phi$', '$A$']
    peak_radio = RadioButtons(peak_box, peak_name, activecolor='tab:blue')      # Signal parameters

    #   Slider
    slider = Slider(ax=slider_box, label='Active\nSignal', valmin=0, valmax=1-1e-3, valinit=0, valstep=1e-10, orientation='vertical', color='tab:blue')

    # -------------------------------------------------------------------------------
    # SLOTS

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
        active = peak_name.index(peak_radio.value_selected)
        param = list(sens.keys())[active]
        sens[param] *= 2
        write_sens(param)

    def down_sens(event):
        """ Halves sensitivity of active parameter """
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
            peaks[idx].__dict__[param] -= sens[param]
            # Safety check for fwhm
            if peaks[idx].fwhm < 0:
                peaks[idx].fwhm = 0
            # Safety check for b
            if peaks[idx].b < 0:
                peaks[idx].b = 0

    def scroll(event):
        """ Connection to mouse scroll """
        if Np == 0:  # No peaks!
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
        if not Np:  # Clear the textbox and do nothing more
            group_tb.text_disp.set_text('')
            plt.draw()
            return
        # Get active peak
        idx = int(np.floor(slider.val * Np) + 1)
        try:
            group = int(eval(text))
        except Exception:
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
        nonlocal A, sens
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
        nonlocal Np
        # Increase the number of peaks
        Np += 1
        # Add an entry to the dictionary labelled as last
        peaks[Np] = fit.Peak(acqus, u=np.mean(ax.get_xlim()), N=N, group=lastgroup)
        # Plot it and add the trace to the plot dictionary
        p_sgn[Np] = ax.plot(ppm_scale[lim1:lim2], peaks[Np](A)[lim1:lim2], lw=0.8)[-1]
        # Move the slider to the position of the new peak
        slider.set_val((Np - 1) / Np)
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
        if Np == 0:  # to zero and do not make it move
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
                slider.set_val((idx - 2) / Np)     # (idx - 1) -1
            slider.valstep = 1 / Np
            write_par(int(np.floor(slider.val * Np) + 1))
        redraw()

    def save(event):
        """ Write a section in the output file """
        nonlocal prev
        # Adjust the intensities
        for _, peak in peaks.items():
            peak.k *= A
        fit.write_vf_P2D(f'{filename}.ivf', [peaks for w in range(n_exp)], ax.get_xlim(), prev)
        prev += len(peaks)

        # Mark a region as "fitted" with a green box
        ax.axvspan(*ax.get_xlim(), color='tab:green', alpha=0.1)
        # Call reset to return at the initial situation
        reset(event)

    # -------------------------------------------------------------------------------

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
        '-'*50,
        f'{"<":>5s}: Decrease sens.',
        f'{">":>5s}: Increase sens.',
        f'{"+":>5s}: Add component',
        f'{"-":>5s}: Remove component',
        f'{"Pg"+t_uparrow:>5s}: Change component, up',
        f'{"Pg"+t_downarrow:>5s}: Change component, down',
        f'{t_uparrow:>5s}: Increase value',
        f'{t_downarrow:>5s}: Decrease value',
        '-'*50,
        ])
    ax.text(0.86, 0.025, keyboard_text,
            ha='left', va='bottom', transform=fig.transFigure, fontsize=8, linespacing=1.55)

    # make pretty scales
    ax.set_xlim(max(limits[0], limits[1]), min(limits[0], limits[1]))
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


def make_iguess_dosy_panel(x, label, y, model, model_args, diff_c_0=1e-10, filename='dosy_fit'):
    """
    Make the initial guess for the fit of a DOSY profile by using a GUI to visually adjust the value of
    the diffusion coefficient and the number of components to use.

    The goal is to try to match the thin solid blue line to the trend of the black dots.

    Use the mouse scroll to modify the values and redraw the figure.
    The radio buttons at the bottom will make you choose if to edit the diffusion coefficient or the fraction
    of the given component.
    The sensitivity can be modified by using the radiobutton as well (coarse/fine) or using the up/down buttons.

    Use the + button to add a component. Use the - button to remove a the currently selected component.
    You can use the slider to change the active component. The headers above the current values of diffusion
    coefficient and fraction will appear of the same color as the active component.

    The intensity to match the model with the experimental data is computed and applied automatically.
    Disabling this option will allow you to play with the values more freely.
    There is also the option to include an offset, however **this should be used only in case of severe systematic errors**
    during the integration procedure.

    Upon pressing the "SAVE" button, a section of the ``<filename>.idy`` file will be written and the GUI will close.


    Parameters
    ----------
    x : 1darray
        Independent variable for the model (usually the gradient list)
    label : str
        Identifier for the region, typically the integration window or peak number
    y : 1darray
        Integrated dosy profile
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

        :func:`klassez.fit.make_iguess_dosy`

        :func:`klassez.fit.write_dy`
    """
    # Initialize variables
    diff_c = [diff_c_0]         # Initial diffusion coefficient
    diff_f = [1]                # Fraction is obviously 1

    A, q = 0, 0                 # Intensity and offset
    Np = 1                      # Number of components

    # "_" version are the originals for the "reset" function
    #   Multiplier for the diffc [coarse, fine]
    lvl_step = [1.6, 1.05]
    _lvl_step = deepcopy(lvl_step)
    #   Added factor for the sensitivity
    lvl_step_incr = [0.05, 0.005]
    _lvl_step_incr = deepcopy(lvl_step_incr)
    #   Adding factor for the difff [coarse, fine]
    frc_step = [0.1, 0.01]
    _frc_step = deepcopy(frc_step)

    # ---------------------------------------------------------------
    # Make the figure panel
    fig = plt.figure('Initialization of diffusion coefficient')
    fig.set_size_inches(figures.figsize_large)
    plt.subplots_adjust(left=0.10, bottom=0.10, right=0.80, top=0.90)
    ax = fig.add_subplot()

    # Make boxes for widgets
    check_box = plt.axes([0.825, 0.80, 0.15, 0.10])         # Toggle intensity and offset
    up_box = plt.axes([0.825, 0.15, 0.04, 0.05])            # Increase sensitivity
    down_box = plt.axes([0.825, 0.10, 0.04, 0.05])          # Decrease sensitivity
    plus_box = plt.axes([0.88, 0.675, 0.045, 0.075])        # Add component
    minus_box = plt.axes([0.93, 0.675, 0.045, 0.075])       # Remove component
    save_box = plt.axes([0.905, 0.015, 0.07, 0.06])         # Save button
    reset_box = plt.axes([0.825, 0.015, 0.07, 0.06])        # Reset button
    radio_box = plt.axes([0.88, 0.10, 0.095, 0.10])         # Coarse/fine
    dorf_box = plt.axes([0.88, 0.225, 0.095, 0.10])         # Move diffc or difff
    slider_box = plt.axes([0.84, 0.25, 0.01, 0.50])         # Selector slider

    # Widgets
    #   Only factor is toggled at the beginning
    checkbox = CheckButtons(check_box, labels=['Calc. factor', 'Use offset'], actives=[True, False])
    misc.edit_checkboxes(checkbox, xadj=0, yadj=0.05, dim=100)
    #   Buttons to increase and decrease sensitivity
    up_button = Button(up_box, r'$\uparrow$', hovercolor='0.895')
    down_button = Button(down_box, r'$\downarrow$', hovercolor='0.895')
    #   Add/remove component
    plus_button = Button(plus_box, '$+$', hovercolor='0.975')
    minus_button = Button(minus_box, '$-$', hovercolor='0.975')
    #   Save and reset
    save_button = Button(save_box, 'SAVE', hovercolor='0.975')
    reset_button = Button(reset_box, 'RESET', hovercolor='0.975')
    #   Radiobuttons
    radio = RadioButtons(radio_box, ['Coarse', 'Fine'], active=0)
    dorf = RadioButtons(dorf_box, ['Diff. Coeff.', 'Fraction'], active=0)
    #   Selector slider, these are just random number inside.
    #   The actual slider will go from 0 to slightly less than 1 in Np step
    slider = Slider(ax=slider_box, label='# Component',
                    valmin=0, valmax=1-1e-3, valinit=0, valstep=1e-10,
                    orientation='vertical', color='tab:blue')

    # -----------------------------------------------------------------------------------------
    # USEFUL FUNCTIONS
    def calc_f(x, y, diffc, difff=1):
        """ Model for a single component """
        # Compute the model and multiply it by its fraction
        f = difff * model(x, diffc, **model_args)
        return f

    def calc_t(x, y, diff_c, diff_f):
        """ Compute all the components """
        # Loop over calc_f using all the values
        yc = [calc_f(x, y, diffc, difff) for diffc, difff in zip(diff_c, diff_f)]
        # The total trace is the sum of all the components
        t = np.sum(yc, axis=0)
        # Update I and q only if the option is toggled in the checkbox
        if checkbox.get_status()[0]:
            nonlocal A, q
            A, q = fit.fit_int(y, t, q=checkbox.get_status()[1])
        # Apply A and q to the total. If the option is unactive, it uses the previous one
        t = A * t + q
        # Update all the components as well
        for k, y_c in enumerate(yc):
            yc[k] = A * y_c + q

        return t, yc

    # SLOTS
    def selector(event):
        """ When you move the slider """
        # Get the index of the selected component
        idx = int(np.floor(slider.val * Np))

        for k, line in enumerate(fit_plot):
            # Set the linewidth of the active plot to 3 and the rest to 1
            if k == idx:
                line.set_lw(3)
            else:
                line.set_lw(1.5)
        # Update the values
        update_text(idx)

    def update_frac_increment(w):
        """ Update the sensitivity value for the fraction. w = +/-1 """
        # radio -> coarse / fine
        # increase -> w = +1 -> *2
        # decrease -> w = -1 -> /2
        frc_step[radio.index_selected] *= 2**w

    def update_diff_increment(w):
        """ Update the sensitivity value for the diffusion coefficient. w = +/-1 """
        # radio -> coarse / fine
        # increase -> w = +1 -> + lvl_step_incr[radio]
        # decrease -> w = -1 -> - lvl_step_incr[radio]
        lvl_step[radio.index_selected] += w * lvl_step_incr[radio.index_selected]

        # make sure these do not reach 1 otherwise the plot will not update anymore
        for k in range(2):
            if lvl_step[k] <= 1:
                lvl_step[k] = 1.01

    def up_sens(event):
        """ Fork for the UP button """
        if dorf.index_selected == 0:    # diff_c
            update_diff_increment(+1)
        else:       # diff_f
            update_frac_increment(+1)

    def down_sens(event):
        """ Fork for the DOWN button """
        if dorf.index_selected == 0:    # diff_c
            update_diff_increment(-1)
        else:       # diff_f
            update_frac_increment(-1)

    def update_diff(event):
        """ Main function to redraw the plot and update the values """
        # event can be None to not update the values, or ScrollEvent
        # Get active component from the slider
        idx = int(np.floor(slider.val * Np))

        if event is not None:       # Fork -> direction of the scroll
            if event.button == 'up':
                w = +1
            elif event.button == 'down':
                w = -1

            if dorf.index_selected == 0:    # Update diffc
                # w = +1 => * // w = -1 => /
                diff_c[idx] *= lvl_step[radio.index_selected]**w
            else:                           # Update fraction
                # w = +1 => + // w = -1 => -
                diff_f[idx] += frc_step[radio.index_selected] * w

        # Recompute the total trace and the components using the current values
        t, yc = calc_t(x, y, diff_c, diff_f)

        # Update the plots
        tot_plot.set_ydata(t)
        for k, y_c in enumerate(yc):
            fit_plot[k].set_ydata(y_c)
        # Update the texts and redraw the artists
        update_text(idx)
        fig.canvas.draw()

    def toggle_check(label):
        """ Redraw everything knowing that either I or q behavior changed """
        update_diff(None)

    def add_comp(event):
        """ Add a component """
        nonlocal Np
        # Increase the number of components
        Np += 1
        # add a default entry to the values lists
        diff_c.append(diff_c_0)
        diff_f.append(1)

        # Add a placeholder in the component plot lists
        fit_plot.append(ax.plot(x, np.zeros_like(x), '--', lw=1)[-1])
        # Redraw everything taking also the new component into account
        update_diff(None)

        # Move the slider to the position of the new component
        slider.set_val((Np - 1) / Np)
        # Recompute the step of the slider
        slider.valstep = 1 / Np
        # Update linewidths
        selector(None)

    def remove_comp(event):
        """ Remove the active component """
        nonlocal Np
        if Np == 1:     # At least one must remain!
            return

        # Get the active peak
        idx = int(np.floor(slider.val * Np))
        # Decrease Np by 1
        Np -= 1
        # Remove the current values from the values lists
        _ = diff_c.pop(idx)
        _ = diff_f.pop(idx)
        # Remove the correspondant line from the plot list
        del_p = fit_plot.pop(idx)
        del_p.remove()      # Erase the artist

        if Np == 1:   # To zero and that's it
            slider.set_val(0)
            slider.valstep = 1 / Np
        else:   # To the previous point
            if idx == 1:
                slider.set_val(0)
            else:
                slider.set_val((idx - 1) / Np)
            slider.valstep = 1 / Np

        # Redraw everything and adjust linewidths
        update_diff(None)
        selector(None)

    def update_text(idx):
        """ Change colors to the headers and update the values """
        legend_text.set_color(fit_plot[idx].get_color())
        value_text.set_text(f'\n{diff_c[idx]:.5g}\n\n\n{diff_f[idx]:.5g}\n\n')
        fig.canvas.draw()

    def reset(event):
        """ Restore all values to the default """
        nonlocal diff_c, diff_f, lvl_step, lvl_step_incr, frc_step

        # Remove the curves one by one
        for _ in range(Np):
            remove_comp(None)

        # Reset the values
        diff_c = [diff_c_0]
        diff_f = [1]

        # Reset the increments
        lvl_step = deepcopy(_lvl_step)
        lvl_step_incr = deepcopy(_lvl_step_incr)
        frc_step = deepcopy(_frc_step)

        # Redraw everything
        update_diff(None)
        selector(None)
        fig.canvas.draw()

    def save(event):
        """ Write a section in the output file """
        # Placeholder for the errors: initial guess is errorless by definition :)
        diff_e = [None for w in range(Np)]
        fit.write_dy(f'{filename}.idy', diff_c, diff_f, diff_e, label, A, q)
        # Close everything
        plt.close()

    # -------------------------------------------------------------------------------
    # Get the initial total trace and the components (it is only one but I like consistency)
    t, yc = calc_t(x, y, diff_c, diff_f)

    # Plots
    ax.plot(x, y, '.', ms=10, c='k', label='Experimental')
    # The total is thinner hence it can appear on top
    tot_plot, = ax.plot(x, t, '-', lw=1, label='Total Fit', zorder=1000)
    # Components as a list
    fit_plot = [ax.plot(x, y_c, '--', lw=3)[-1] for y_c in yc]

    # Text on the right
    # Same position for header and value, the text itself is interleaved
    legend_text = fig.text(0.925, 0.60,     # here 1st and 4th line
                           'Diff. C.\n\n\nFraction\n\n\n',
                           ha='center', va='top', transform=fig.transFigure,
                           fontsize=16)
    # Here just a placeholder because...
    value_text = fig.text(0.925, 0.60, '',  # here 2nd and 5th line
                          ha='center', va='top', transform=fig.transFigure,
                          fontsize=16)
    # ... this is the function that applies color to header and correct text in value
    update_text(0)

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
    misc.set_fontsizes(ax, 14)

    # Connect widgets to slots
    up_button.on_clicked(up_sens)
    down_button.on_clicked(down_sens)
    plus_button.on_clicked(add_comp)
    minus_button.on_clicked(remove_comp)
    slider.on_changed(selector)
    reset_button.on_clicked(reset)
    save_button.on_clicked(save)
    checkbox.on_clicked(toggle_check)
    fig.canvas.mpl_connect('scroll_event', update_diff)

    # Start event loop
    plt.show()
