#! /usr/bin/env python3

import os
import sys
import numpy as np
from scipy import linalg, stats
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
s_colors=[ 'tab:cyan', 'tab:red', 'tab:green', 'tab:purple', 'tab:pink', 'tab:gray', 'tab:brown', 'tab:olive', 'salmon', 'indigo' ]

from .config import CM, COLORS, cron

figsize_small = (3.59, 2.56)
figsize_large = (15, 8)

warnings.filterwarnings("ignore", message="No contour levels were found within the data range.")


def heatmap(data, zlim='auto', z_sym=True, cmap=None, xscale=None, yscale=None, rev=(False, False), n_xticks=10, n_yticks=10, n_zticks=10, fontsize=10, name=None):
    """
    Computes a heatmap of data.
    --------
    Parameters:
    - data: 2darray
        Input data
    - zlim: tuple or 'auto' or 'abs'
        Vertical limits of the heatmap, that determines the extent of the colorbar. 'auto' means (min(data), max(data)), 'abs' means(min(|data|), max(|data|)). 
    - z_sym: bool
        True to symmetrize the vertical scale around 0. 
    - cmap: matplotlib.cm object
        Colormap of the heatmap. 
    - xscale: 1darray or None
        x-scale. None means np.arange(data.shape[1])
    - yscale: 1darray or None
        y-scale. None means np.arange(data.shape[0])
    - rev: tuple of bool
        Reverse scale (x, y).
    - n_xticks: int
        Number of ticks of the x axis
    - n_yticks: int
        Number of ticks of the y axis
    - n_zticks: int
        Number of ticks of the color bar 
    - fontsize: float
        Biggest font size to apply to the figure.
    - name: str or None
        Filename for the figure. Set to None to show the figure.
    """
    print('Computing heatmap...', end='\r')

    # Check if data is real
    if np.iscomplexobj(data):
        data = data.real

    # Set zlim
    if zlim == 'auto':
        zlim = np.min(data), np.max(data)
    elif zlim == 'abs':
        zlim = np.min(np.abs(data)), np.max(np.abs(data))

    # Symmetrize z
    if z_sym is True:
        zlim = -max(zlim), max(zlim)

    # Set default cmap
    if cmap is None:
        cmap = CM['icefire_r']

    # Set default scales
    if xscale is None:
        xscale = np.arange(data.shape[-1])
    if yscale is None:
        yscale = np.arange(data.shape[0])

    # Set extent according to rev
    if rev == (False, False):   # do not reverse
        extent = min(xscale), max(xscale), min(yscale), max(yscale)
    elif rev == (True, False):  # reverse only x
        extent = max(xscale), min(xscale), min(yscale), max(yscale)
    elif rev == (False, True):  # reverse only y
        extent = min(xscale), max(xscale), max(yscale), min(yscale)
    elif rev == (True, True):   # reverse both
        extent = max(xscale), min(xscale), max(yscale), min(yscale)

    # Create figure panel
    fig = plt.figure()
    fig.set_size_inches(figsize_small)
    plt.subplots_adjust(left=0.15, bottom=0.15, top=0.90, right=0.85)
    ax = fig.add_subplot()

    # Divide the ax subplot to make space for the colorbar
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='2.5%', pad=0.10)

    # Plot data
    im = ax.imshow(data, aspect='auto', cmap=cmap, vmin=zlim[0], vmax=zlim[1], extent=extent)

    # Make colorbar
    plt.colorbar(im, cax=cax, orientation='vertical')

    # Customize appearance
    #   x-axis 
    misc.pretty_scale(ax, (extent[0], extent[1]), axis='x', n_major_ticks=n_xticks)
    #   y-axis 
    misc.pretty_scale(ax, (extent[2], extent[3]), axis='y', n_major_ticks=n_yticks)
    #   colorbar y-axis 
    misc.pretty_scale(cax, zlim, axis='y', n_major_ticks=n_zticks)
    misc.mathformat(cax)
    #   fontsizes
    misc.set_fontsizes(ax, fontsize)
    misc.set_fontsizes(cax, fontsize)

    if name:
        # Save the figure
        print('Saving {}.png...'.format(name), end='\r')
        plt.savefig(name+'.png', dpi=600)
        print('{}.png saved.\n'.format(name))
    else:
        # Make figure larger
        fig.set_size_inches(figsize_large)
        # Increase fontsize
        misc.set_fontsizes(ax, 14)
        misc.set_fontsizes(cax, 14)
        # Show
        plt.show()
        print('\n')
    plt.close()


def ax_heatmap(ax, data, zlim='auto', z_sym=True, cmap=None, xscale=None, yscale=None, rev=(False, False), n_xticks=10, n_yticks=10, n_zticks=10, fontsize=10):
    """
    Computes a heatmap of data on the given 'ax'
    --------
    Parameters:
    - ax: matplotlib.Subplot object
        Panel where to draw the heatmap
    - data: 2darray
        Input data
    - zlim: tuple or 'auto' or 'abs'
        Vertical limits of the heatmap, that determines the extent of the colorbar. 'auto' means (min(data), max(data)), 'abs' means(min(|data|), max(|data|)). 
    - z_sym: bool
        True to symmetrize the vertical scale around 0. 
    - cmap: matplotlib.cm object
        Colormap of the heatmap. 
    - xscale: 1darray or None
        x-scale. None means np.arange(data.shape[1])
    - yscale: 1darray or None
        y-scale. None means np.arange(data.shape[0])
    - rev: tuple of bool
        Reverse scale (x, y).
    - n_xticks: int
        Number of ticks of the x axis
    - n_yticks: int
        Number of ticks of the y axis
    - n_zticks: int
        Number of ticks of the color bar 
    - fontsize: float
        Biggest font size to apply to the figure.
    -------
    Returns:
    - im: matplotlib.AxesImage
        The heatmap
    - cax: figure panel where the colorbar is drawn
    """

    # Check if data is real
    if np.iscomplexobj(data):
        data = data.real

    # Set zlim
    if zlim == 'auto':
        zlim = np.min(data), np.max(data)
    elif zlim == 'abs':
        zlim = np.min(np.abs(data)), np.max(np.abs(data))

    # Symmetrize z
    if z_sym is True:
        zlim = -max(zlim), max(zlim)

    # Set default cmap
    if cmap is None:
        cmap = CM['icefire_r']

    # Set default scales
    if xscale is None:
        xscale = np.arange(data.shape[-1])
    if yscale is None:
        yscale = np.arange(data.shape[0])

    # Set extent according to rev
    if rev == (False, False):   # do not reverse
        extent = min(xscale), max(xscale), min(yscale), max(yscale)
    elif rev == (True, False):  # reverse only x
        extent = max(xscale), min(xscale), min(yscale), max(yscale)
    elif rev == (False, True):  # reverse only y
        extent = min(xscale), max(xscale), max(yscale), min(yscale)
    elif rev == (True, True):   # reverse both
        extent = max(xscale), min(xscale), max(yscale), min(yscale)

    # Divide the ax subplot to make space for the colorbar
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from matplotlib.ticker import StrMethodFormatter
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='2.5%', pad=0.10)

    # Plot data
    im = ax.imshow(data, aspect='auto', cmap=cmap, vmin=zlim[0], vmax=zlim[1], extent=extent)

    # Make colorbar
    plt.colorbar(im, cax=cax, orientation='vertical')

    # Customize appearance
    #   x-axis 
    misc.pretty_scale(ax, (extent[0], extent[1]), axis='x', n_major_ticks=n_xticks)
    #   y-axis 
    misc.pretty_scale(ax, (extent[2], extent[3]), axis='y', n_major_ticks=n_yticks)
    #   colorbar y-axis 
    misc.pretty_scale(cax, zlim, axis='y', n_major_ticks=n_zticks)
    misc.mathformat(cax)
    #   fontsizes
    misc.set_fontsizes(ax, fontsize)
    misc.set_fontsizes(cax, fontsize)

    # Return the heatmap and the colorbar axis
    return im, cax



def sns_heatmap(data, name=None):
    """
    Computes a heatmap of data, which is a matrix. 
    This function employs the seaborn package.
    Specify name if you want to save the figure.
    ---------
    Parameters:
    - data: 2darray
        Data of which to compute the heatmap. Make sure the entries are real numbers.
    - name: str or None
        Filename of the figure to be saved. If None, the figure is shown instead.
    """
    data = data.real

    fig = plt.figure()
    fig.set_size_inches(figsize_small)
    ax = fig.add_subplot(1,1,1)
    formatter = matplotlib.ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-2, 2))
    
    ax = sns.heatmap(data, center=0, linewidth=0, cbar_kws={'format': formatter})
    ax.tick_params(labelsize=7)
    ax.set_xlabel('F2', fontsize=8)
    ax.set_ylabel('F1', fontsize=8)
    ax.figure.axes[-1].yaxis.get_offset_text().set_size(7)
    ax.figure.axes[-1].tick_params(labelsize=7)
    fig.tight_layout()
    if name:
        plt.savefig(name+'.png', format='png', dpi=600)
    else:
        plt.show()
    plt.close()


def plot_fid_re(fid, scale=None, c='b', lims=None, name=None):
    """
    Makes a single-panel figure that shows either the real or the imaginary part of the FID.
    The x-scale and y-scale are automatically adjusted.
    """


    size = fid.shape[-1]
    fid = fid.flatten()
    n_trans = fid.shape[-1]//size
    if lims is None:
        if n_trans >1:
            lims = (0,n_trans)
        else:
            lims = (0,size)

    if scale is None:
        if n_trans > 1:
            scale = np.empty(1)
            for i in range(n_trans):
                temp_scale = i + np.linspace(0, 1, size)
                scale = np.concatenate((scale, temp_scale), axis=-1)
            scale = np.delete(scale, 0)
        else:
            scale = np.arange(size)


    fig = plt.figure()
    fig.set_size_inches(figsize_small)
    plt.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.90)
    ax1 = fig.add_subplot(1,1,1)
    ax1.axhline(0, ls='-', c='k', lw=0.2)
    ax1.plot(scale, fid.real, c=c, lw=0.5)
    ax1.set_xlim(lims)
    ax1.set_xlabel('# points', fontsize=8)
    ax1.set_ylabel('Intensity /a.u.', fontsize=8)

    misc.set_ylim(ax1, [np.abs(fid.real), -np.abs(fid.real)])
    misc.mathformat(ax1, axis='y')


    if name:
        misc.set_fontsizes(ax1, 10)
        plt.savefig(name+'.png', format='png', dpi=600)
    else:
        fig.set_size_inches(figsize_large)
        misc.set_fontsizes(ax1, 14)
        plt.show()
    plt.close()

def plot_fid(fid, name=None):
    """
    Makes a two-panel figure that shows on the left the real part of the FID, on the right the imaginary part.
    The x-scale and y-scale are automatically adjusted.
    """

    size = fid.shape[-1]
    fid = fid.flatten()
    n_trans = fid.shape[-1]//size
    scale = np.empty(1)
    for i in range(n_trans):
        temp_scale = i + np.linspace(0, 1, size)
        scale = np.concatenate((scale, temp_scale), axis=-1)
    scale = np.delete(scale, 0)

    fig = plt.figure()
    fig.set_size_inches(5.50, 2.56)
    plt.subplots_adjust(left=0.1, bottom=0.1, top=0.95, right=0.95, wspace=0.20)
    ax1 = fig.add_subplot(1,2,1)
    ax2 = fig.add_subplot(1,2,2)

    ax1.set_title('Real channel', fontsize=8)
    ax2.set_title('Imaginary channel', fontsize=8)

    ax1.axhline(0, ls='-', c='k', lw=0.2)
    ax1.plot(scale, fid.real, c='b', lw=0.1)
    ax2.axhline(0, ls='-', c='k', lw=0.2)
    ax2.plot(scale, fid.imag, c='r', lw=0.1)

    ax1.set_xticks(np.linspace(0, n_trans, n_trans+1))
    ax1.set_xticks(np.arange(0, n_trans, 0.2), minor=True)
    ax1.set_xlim(0, n_trans)
    ax2.set_xticks(np.linspace(0, n_trans, n_trans+1))
    ax2.set_xticks(np.arange(0, n_trans, 0.2), minor=True)
    ax2.set_xlim(0, n_trans)
    ax2.tick_params(axis='y', labelleft=False)

    misc.set_ylim(ax1, [np.abs(fid.real), -np.abs(fid.real)])
    misc.set_ylim(ax2, [np.abs(fid.real), -np.abs(fid.real)])
    misc.mathformat(ax1, axis='y')
    misc.mathformat(ax2, axis='y')
    
    if name:
        misc.set_fontsizes(10)
        plt.savefig(name+'.png', format='png', dpi=600)
    else:
        fig.set_size_inches(figsize_large)
        misc.set_fontsizes(14)
        plt.show()
    plt.close()

def figure2D(ppm_f2, ppm_f1, datax, xlims=None, ylims=None, cmap=None, c_fac=1.4, lvl=0.09, name=None, X_label='$\delta\ $ F2 /ppm', Y_label='$\delta\ $ F1 /ppm', lw=0.5, Negatives=False, cmapneg=None, n_xticks=10, n_yticks=10, fontsize=10):
    """
    Makes a 2D contour plot. 
    Allows for the buildup of modular figures. 
    The contours are drawn according to the formula:
        cl = contour_start * contour_factor ** np.arange(contour_num)
    where contour_start = np.max(data) * lvl, contour_num = 16 and contour_factor = c_fac.
    Increasing the value of c_fac will decrease the number of contour lines, whereas decreasing the value of c_fac will increase the number of contour lines.
    -----------
    Parameters:
    - ppm_f2: 1darray
        ppm scale of the direct dimension
    - ppm_f1: 1darray
        ppm scale of the indirect dimension
    - datax: 2darray
        the 2D NMR spectrum to be plotted
    - xlims: tuple
        limits for the x-axis (left, right). If None, the whole scale is used.
    - ylims: tuple
        limits for the y-axis (left, right). If None, the whole scale is used.
    - cmap: matplotlib.cm Object
        Colour for the contour
    - c_fac: float
        Contour factor parameter
    - lvl: float
        height with respect to maximum at which the contour are computed
    - name: str
        Filename for the figure.
    - X_label: str
        text of the x-axis label;
    - Y_label: str
        text of the y-axis label;
    - lw: float
        linewidth of the contours
    - Negatives: bool
        Choose if to plot the negative contours or not
    - cmapneg: matplotlib.cm Object
        Colour for the negative contours
    - n_xticks: int
        Number of numbered ticks on the x-axis of the figure
    - n_yticks: int
        Number of numbered ticks on the x-axis of the figure
    - fontsize: float
        Biggest font size in the figure.
    """

    swapped_scales = len(ppm_f2) == datax.shape[0] and len(ppm_f1) == datax.shape[1]
    if swapped_scales:
        raise AssertionError('Swapped scales!')

    if cmap is None:
        cmap = cm.Greys_r

    if xlims is None:
        xsx, xdx = max(ppm_f2), min(ppm_f2)
    else:
        xsx, xdx = max(xlims), min(xlims)
    if ylims is None:
        ysx, ydx = max(ppm_f1), min(ppm_f1)
    else:
        ysx, ydx = max(ylims), min(ylims)

    norm = np.max(datax)
    contour_start = norm*lvl
    contour_num = 16 
    contour_factor = c_fac 
    # calculate contour levels
    cl = contour_start * contour_factor ** np.arange(contour_num)


    fig = plt.figure()
    fig.set_size_inches(figsize_small)
    plt.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.95)
    ax = fig.add_subplot(1,1,1)
    ax.contour(ppm_f2, ppm_f1, datax, cl, cmap=cmap, extent=(min(ppm_f2), max(ppm_f2), max(ppm_f1), min(ppm_f1)), linewidths=lw)
    
    if Negatives:       # Plot the negative part of the spectrum
        if cmapneg is None:
            cmapneg = cm.Reds_r
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="No contour levels were found within the data range.")
            ax.contour(ppm_f2, ppm_f1, -datax, cl, cmap=cmapneg, extent=(min(ppm_f2), max(ppm_f2), max(ppm_f1), min(ppm_f1)), linewidths=lw)

    ax.set_xlabel(X_label)
    ax.set_ylabel(Y_label)

    misc.pretty_scale(ax, (xsx, xdx), axis='x', n_major_ticks=n_xticks)
    misc.pretty_scale(ax, (ysx, ydx), axis='y', n_major_ticks=n_yticks)

    misc.set_fontsizes(ax, fontsize)

    if name:
        print( 'Saving '+name+'.png...')
        plt.savefig(name+'.png', format='png', dpi=600)
    else:
        fig.set_size_inches(figsize_large)
        misc.set_fontsizes(ax, 14)
        plt.show()
    plt.close()
    print( 'Done.')

def ax2D(ax, ppm_f2, ppm_f1, datax, xlims=None, ylims=None, cmap=None, c_fac=1.4, lvl=0.1, lw=0.5, X_label='$\delta\,$F2 /ppm', Y_label='$\delta\,$F1 /ppm', title=None, n_xticks=10, n_yticks=10, fontsize=10):
    """
    Makes a 2D contour plot like the one in figures.figure2D, but in a specified panel. 
    Allows for the buildup of modular figures. 
    The contours are drawn according to the formula:
        cl = contour_start * contour_factor ** np.arange(contour_num)
    where contour_start = np.max(data) * lvl, contour_num = 16 and contour_factor = c_fac.
    Increasing the value of c_fac will decrease the number of contour lines, whereas decreasing the value of c_fac will increase the number of contour lines.
    -----------
    Parameters:
    - ax: matplotlib.subplot Object
        panel where to put the figure
    - ppm_f2: 1darray
        ppm scale of the direct dimension
    - ppm_f1: 1darray
        ppm scale of the indirect dimension
    - datax: 2darray
        the 2D NMR spectrum to be plotted
    - xlims: tuple
        limits for the x-axis (left, right). If None, the whole scale is used.
    - ylims: tuple
        limits for the y-axis (left, right). If None, the whole scale is used.
    - cmap: matplotlib.cm Object
        Colour for the contour
    - c_fac: float
        Contour factor parameter
    - lvl: float
        height with respect to maximum at which the contour are computed
    - X_label: str
        text of the x-axis label;
    - Y_label: str
        text of the y-axis label;
    - lw: float
        linewidth of the contours
    - title: str
        Figure title.
    - n_xticks: int
        Number of numbered ticks on the x-axis of the figure
    - n_yticks: int
        Number of numbered ticks on the x-axis of the figure
    - fontsize: float
        Biggest font size in the figure.
    ---------
    Returns:
    - cnt: matplotlib.QuadContour object
        Drawn contour lines
    """

    swapped_scales = len(ppm_f2) == datax.shape[0] and len(ppm_f1) == datax.shape[1]
    if swapped_scales:
        raise AssertionError('Swapped scales!')

    if cmap is None:
        cmap = cm.Greys_r

    if xlims is None:
        xsx, xdx = max(ppm_f2), min(ppm_f2)
    else:
        xsx, xdx = max(xlims), min(xlims)
    if ylims is None:
        ysx, ydx = max(ppm_f1), min(ppm_f1)
    else:
        ysx, ydx = max(ylims), min(ylims)

    norm = np.max(np.abs(datax))
    contour_start = norm*lvl
    contour_num = 16 
    contour_factor = c_fac
    # calculate contour levels
    cl = contour_start * contour_factor ** np.arange(contour_num)

    cnt = ax.contour(ppm_f2, ppm_f1, datax, cl, cmap=cmap, extent=(min(ppm_f2), max(ppm_f2), max(ppm_f1), min(ppm_f1)), linewidths=lw)

    if X_label is not None:
        ax.set_xlabel(X_label)
    if Y_label is not None:
        ax.set_ylabel(Y_label)

    misc.pretty_scale(ax, (xsx, xdx), axis='x', n_major_ticks=n_xticks)
    misc.pretty_scale(ax, (ysx, ydx), axis='y', n_major_ticks=n_yticks)

    if title:
        ax.set_title(title)
    misc.set_fontsizes(ax, fontsize)

    return cnt




def figure2D_multi(ppm_f2, ppm_f1, datax, xlims=None, ylims=None, lvl='default', name=None, X_label='$\delta\ $ F2 /ppm', Y_label='$\delta\ $ F1 /ppm', lw=0.5, Negatives=False, n_xticks=10, n_yticks=10, labels=None):
    """
    Generates the figure of multiple, superimposed spectra.
    --------
    Parameters:
    - ppm_f2: 1darray
        ppm scale of the direct dimension
    - ppm_f1: 1darray
        ppm scale of the indirect dimension
    - datax: list
        the 2D NMR spectra to be plotted
    - xlims: tuple
        limits for the x-axis (left, right). If None, the whole scale is used.
    - ylims: tuple
        limits for the y-axis (left, right). If None, the whole scale is used.
    - lvl: "default" or list
        height with respect to maximum at which the contour are computed. If "default", each spectrum is at 10% of maximum height. Otherwise, each entry of the list corresponds to the contour height of the respective spectrum.
    - name: str
        filename of the figure, if it has to be saved;
    - X_label: str
        text of the x-axis label;
    - Y_label: str
        text of the y-axis label;
    - lw: float
        linewidth of the contours
    - Negatives: bool
        set it to True if you want to see the negative part of the spectrum
    - n_xticks: int
        Number of numbered ticks on the x-axis of the figure
    - n_yticks: int
        Number of numbered ticks on the x-axis of the figure
    - labels: list
        entries of the legend. If None, the legend is not drawn.
    """
    nsp = len(datax)
    cmaps = [cm.Blues_r, cm.Reds_r, cm.Greens_r, cm.Greys_r, cm.Purples_r, cm.Oranges_r, cm.YlOrBr_r, cm.YlOrRd_r, cm.OrRd_r, cm.PuRd_r, cm.RdPu_r, cm.BuPu_r, cm.GnBu_r, cm.PuBu_r, cm.YlGnBu_r, cm.PuBuGn_r, cm.BuGn_r, cm.YlGn]

    # Labels of the spectra that appear in the legend
    if not labels:
        labels = []
        for k in range(nsp):
            labels.append(str(k+1))

    if xlims is None:
        xsx, xdx = max(ppm_f2), min(ppm_f2)
    else:
        xsx, xdx = max(xlims), min(xlims)
    if ylims is None:
        ysx, ydx = max(ppm_f1), min(ppm_f1)
    else:
        ysx, ydx = max(ylims), min(ylims)


    fig = plt.figure()
    fig.set_size_inches(figsize_small)
    plt.subplots_adjust(left=0.15, bottom=0.2)
    ax = fig.add_subplot(1,1,1)

    contour_num = 16 
    contour_factor = 1.40
    if lvl == 'default':
        lvl = np.ones(nsp) * 0.1 
    norm = []
    contour_start = []
    cl = []
    cnt = []
    for k in range(nsp):
        norm.append(np.max(np.abs(datax[k])))
        contour_start.append(norm[k] * lvl[k])
        # calculate contour levels
        cl.append(norm[k] * lvl[k] * contour_factor ** np.arange(contour_num))
        cntt = ax.contour(ppm_f2, ppm_f1, datax[k], cl[k], cmap=cmaps[k], extent=(min(ppm_f2), max(ppm_f2), max(ppm_f1), min(ppm_f1)), linewidths=0.5)
        cnt.append(cntt)
    for i in range(len(labels)):
        cnt[i].collections[i].set_label(labels[i])

    ax.set_xlabel(X_label)
    ax.set_ylabel(Y_label)

    misc.pretty_scale(ax, (xsx, xdx), axis='x', n_major_ticks=n_xticks)
    misc.pretty_scale(ax, (ysx, ydx), axis='y', n_major_ticks=n_yticks)
    ax.legend()

    if name:
        misc.set_fontsizes(10)
        print( 'Saving '+name+'.png...')
        plt.savefig(name+'.png', format='png', dpi=600)
    else:
        fig.set_size_inches(figsize_large)
        misc.set_fontsizes(14)
        plt.show()
    plt.close()
    print( 'Done.')






def figure1D(ppm, data, norm=False, xlims=None, ylims=None, c='b', lw=0.5, name=None, X_label='$\delta\ $ F1 /ppm', Y_label='Intensity /a.u.', n_xticks=10, n_yticks=10, hideylabels=False):
    """
    Makes the figure of a 1D NMR spectrum.

    The plot can be customized in a very flexible manner by setting the function keywords properly.
    --------
    Parameters:
	- ppm: 1darray
		ppm scale of the spectrum
	- data: 1darray
		spectrum to be plotted
	- norm: bool
		if True, normalizes the intensity to 1.
	- xlims: list or tuple
		Limits for the x-axis. If None, the whole scale is used.
	- ylims: list or tuple
		Limits for the y-axis. If None, the whole scale is used.
	- c: str
		Colour of the line.
	- lw: float
		 linewidth
    - name: str or None
        Filename for the figure to be saved. If None, the figure is shown instead.
	- X_label: str
		 text of the x-axis label;
	- Y_label: str
		 text of the y-axis label;
	- n_xticks: int
		 Number of numbered ticks on the x-axis of the figure
	- n_yticks: int
		 Number of numbered ticks on the x-axis of the figure
	- fontsize: float
		 Biggest font size in the figure.
    --------
    Returns:
	- line: Line2D Object
		Line object returned by plt.plot.
    """
    if np.iscomplexobj(data):
        data = np.copy(data.real)

    if xlims is None:
        xsx, xdx = max(ppm), min(ppm)
    else:
        xsx, xdx = max(xlims[0], xlims[1]), min(xlims[0], xlims[1])

    if norm:
        data = data/np.max(np.abs(data))
        if Y_label=='Intensity /a.u.':
            Y_label='Normalized Intensity /a.u.'

    fig = plt.figure()
    fig.set_size_inches(figsize_small)
    plt.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.90)
    ax = fig.add_subplot(1,1,1)
    ax.plot(ppm, data, lw=lw, c=c)

    misc.set_ylim(ax, data)

    if ylims is None:
        ysx, ydx = ax.get_ylim()
    else:
        ysx, ydx = min(ylims), max(ylims)

    misc.pretty_scale(ax, (xsx, xdx), axis='x', n_major_ticks=n_xticks)
    misc.pretty_scale(ax, (ysx, ydx), axis='y', n_major_ticks=n_yticks)
    
    if hideylabels:
        ax.tick_params(axis='y', which='both', left=False, labelleft=False)

    ax.set_xlabel(X_label)
    if not hideylabels:
        ax.set_ylabel(Y_label)
        misc.mathformat(ax, axis='y')

    if name:
        misc.set_fontsizes(ax, 10)
        print( 'Saving '+name+'.png...')
        plt.savefig(name+'.png', format='png', dpi=600)
    else:
        fig.set_size_inches(figsize_large)
        misc.set_fontsizes(ax, 14)
        plt.show()
    plt.close()
    print( 'Done.')


def ax1D(ax, ppm, data, norm=False, xlims=None, ylims=None, c='b', lw=0.5, X_label='$\delta\ $ F1 /ppm', Y_label='Intensity /a.u.', n_xticks=10, n_yticks=10, label=None, fontsize=10):
    """
    Makes the figure of a 1D NMR spectrum, placing it in a given figure panel.
    This allows the making of modular figures.

    The plot can be customized in a very flexible manner by setting the function keywords properly.
    --------
    Parameters:
    - ax: matplotlib.subplot Object
        panel where to put the figure
	- ppm: 1darray
		ppm scale of the spectrum
	- data: 1darray
		spectrum to be plotted
	- norm: bool
		if True, normalizes the intensity to 1.
	- xlims: list or tuple
		Limits for the x-axis. If None, the whole scale is used.
	- ylims: list or tuple
		Limits for the y-axis. If None, the whole scale is used.
	- c: str
		Colour of the line.
	- lw: float
		 linewidth
	- X_label: str
		 text of the x-axis label;
	- Y_label: str
		 text of the y-axis label;
	- n_xticks: int
		 Number of numbered ticks on the x-axis of the figure
	- n_yticks: int
		 Number of numbered ticks on the x-axis of the figure
	- label: str
		 label to be put in the legend.
	- fontsize: float
		 Biggest font size in the figure.
    --------
    Returns:
	- line: Line2D Object
		Line object returned by plt.plot.
    """
    if np.iscomplexobj(data):
        data = np.copy(data.real)

    if xlims is None:
        xsx, xdx = max(ppm), min(ppm)
    else:
        xsx, xdx = max(xlims), min(xlims)

    if norm:
        data = data/np.max(data)
        if Y_label=='Intensity /a.u.':
            Y_label='Normalized Intensity /a.u.'

    line, = ax.plot(ppm, data, lw=lw, c=c)
    if isinstance(label, str):
        line.set_label(label) 

    misc.set_ylim(ax, data)

    if ylims is None:
        ysx, ydx = ax.get_ylim()
    else:
        ysx, ydx = min(ylims), max(ylims)

    misc.pretty_scale(ax, (xsx, xdx), axis='x', n_major_ticks=n_xticks)
    misc.pretty_scale(ax, (ysx, ydx), axis='y', n_major_ticks=n_yticks)
    misc.mathformat(ax, axis='y')

    ax.set_xlabel(X_label)
    misc.set_fontsizes(ax, fontsize)
    return line


def figure1D_multi(ppm0, data0, xlims=None, ylims=None, norm=False, c=None, name=None, X_label='$\delta\ $ F1 /ppm', Y_label='Intensity /a.u.', n_xticks=10, n_yticks=10, hideylabels=False, labels=None):
    """
    Creates the superimposed plot of a series of 1D NMR spectra.
    -------
    Parameters:
    - ppm0: list or 1darray
        ppm scale of the spectra
    - data0: list
        List containing the spectra to be plotted
    - xlims: tuple or None
        Limits for the x-axis. If None, the whole scale is used.
    - ylims: tuple or None
        Limits for the y-axis. If None, they are automatically set.
    - norm: False or float or str
        If it is False, it does nothing. If it is float, divides all spectra for that number. If it is str('#'), normalizes all the spectra to the '#' spectrum. If it is whatever else string, normalizes all spectra to themselves.
    - c: tuple or None
        List of the colors to use for the traces. None uses the default ones.
    - name: str or None
        Filename of the figure, if it has to be saved. If it is None, the figure is shown instead.
    - X_label: str
        text of the x-axis label
    - Y_label: str
        text of the y-axis label
    - n_xticks: int
        Number of numbered ticks on the x-axis of the figure
    - n_yticks: int
        Number of numbered ticks on the x-axis of the figure
    - hideylabels: bool
        if True, does not show label and tick labels of the y axis.
    - labels: list or None or False
        List of the labels to be shown in the legend. If it is None, the default entries are used (i.e., '1, 2, 3,...'). If it is False, the legend is not shown.
    """

    # Check input data format and transform into a list if it is not already
    if isinstance(data0, list):
        nsp = len(data0)
    elif isinstance(data0, np.ndarray):
        if len(data0.shape) == 1:
            nsp = 1
        elif len(data0.shape) == 2:
            nsp = data0.shape[0]
        else:
            raise ValueError('Unknown input data. Aborting...')

    # Check ppm scale format and transform into a list if it is not already
    if isinstance(ppm0, np.ndarray):
        if len(ppm0.shape) == 1:
            ppm = [ppm0 for k in range(nsp)]
        elif len(ppm0.shape) ==2:
            ppm = [ppm0[k] for k in ppm0.shape[0]]
        else:
            raise ValueError('Unknown input scale. Aborting...')
    elif isinstance(ppm0, list):
        if len(ppm0) == nsp:
            ppm = [ppm0[k] for k in range(nsp)]
        else:
            raise ValueError('The provided ppm scales do not match the number of spectra')
    else:
        raise ValueError('Unknown input scale. Aborting...')

    # Build the labels if not given
    if labels is None:
        labels = ['{}'.format(w+1) for w in range(nsp)]
    elif labels is False:
        pass
    elif len(labels) == nsp:
        pass
    else:
        raise ValueError('The number of provided labels do not match the number of spectra')

    # Build the list of spectra
    if nsp == 1:
        print('You provided only one spectrum. You should call figure1D instead.')
        return 0
    else:
        data = [data0[k] for k in range(nsp)]     # copy to prevent overwriting

    # Delete the imaginary part of the spectra, if there is
    for k, spectrum in enumerate(data):
        if np.iscomplexobj(spectrum):
            data[k] = np.copy(spectrum.real)

    # Handle the 'norm' flag
    if norm is not False:
        if isinstance(norm, float) or isinstance(norm, int): # norm is a number
            normval = [norm for k in range(nsp)]    # normalize for that number
            print('Spectra were normalized to {}.'.format(normval[0]))
        elif isinstance(norm, str): # norm is a string
            if xlims is None:           
                idx1 = [0 for k in range(nsp)]
                idx2 = [len(ppm[k]) for k in range(nsp)]
            else:
                idx1 = [misc.ppmfind(ppm[k], max(xlims))[0] for k in range(nsp)]
                idx2 = [misc.ppmfind(ppm[k], min(xlims))[0] for k in range(nsp)]
            try:    # Check if norm can be interpreted as a list index
                idx = int(eval(norm)) - 1
                # If so, normalize all the spectra to the maximum of the norm-th spectrum (ordinary numbering)
                normval = [np.max(data[idx][min(idx1[k], idx2[k]):max(idx1[k],idx2[k])]) for k in range(nsp)]
                print('Spectra were normalized to the {}° spectrum'.format(norm))
            except: # If you write anything else
                # normalize all spectra to themselves
                normval = [np.max(data[k][min(idx1[k], idx2[k]):max(idx1[k],idx2[k])]) for k in range(nsp)]
                print('Spectra were normalized to themselves.')
        data = [data[k]/normval[k] for k in range(nsp)]
        # Correct the Y-label if left to the default one
        if Y_label == 'Intensity /a.u.':
            Y_label = 'Normalized intensity /a.u.'

    # Set the colors
    if isinstance(c, tuple) or isinstance(c, list):
        if len(c) < nsp:
            raise ValueError('The provided colors are not enough for the spectra.')
    else:
        c = COLORS
        # If the default colors are not enough, cycle between them
        while len(c) < nsp:
            c = list(c)
            c += list(COLORS)
            c = tuple(c)

    # Make the figure
    fig = plt.figure()
    fig.set_size_inches(figsize_small)
    ax = fig.add_subplot(1,1,1)
    plt.subplots_adjust(left=0.20, bottom=0.15, right=0.95, top=0.90)
    # Add the traces
    for k, s in enumerate(data):
        line = figures.ax1D(ax, ppm[k], data[k], norm=False, xlims=None, ylims=None, c=c[k], lw=0.5, X_label='', Y_label='', n_xticks=10, n_yticks=10)
        if labels is not False:
            line.set_label(labels[k])

    # Adjust the limits
    misc.set_ylim(ax, data)

    if xlims is None:
        xsx, xdx = ax.get_xlim()
    else:
        xsx, xdx = max(xlims), min(xlims)

    if ylims is None:
        ysx, ydx = ax.get_ylim()
    else:
        ysx, ydx = min(ylims), max(ylims)

    # Make pretty scales
    misc.pretty_scale(ax, (xsx, xdx), axis='x', n_major_ticks=n_xticks)
    misc.pretty_scale(ax, (ysx, ydx), axis='y', n_major_ticks=n_yticks)
    
    # Set the labels for the axes
    if hideylabels:
        ax.tick_params(axis='y', which='both', left=False, labelleft=False)

    ax.set_xlabel(X_label)
    if not hideylabels:
        ax.set_ylabel(Y_label)
        misc.mathformat(ax, axis='y')

    # Legend
    if labels is not False:
        ax.legend()

    # Save / Show the figure
    if name:
        misc.set_fontsizes(ax, 10)
        print( 'Saving '+name+'.png...')
        plt.savefig(name+'.png', format='png', dpi=600)
    else:
        fig.set_size_inches(figsize_large)
        plt.subplots_adjust(left=0.10)
        misc.set_fontsizes(ax, 14)
        plt.show()
    plt.close()
    print( 'Done.')


def fitfigure(S, ppm_scale, t_AQ, V, C=False, SFO1=701.125, o1p=0, limits=None, s_labels=None, X_label='$\delta\,$ F1 /ppm', n_xticks=10, name=None):
    """
    Makes the figure to show the result of a quantitative fit.
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
    - limits : tuple or None
        Trim limits for the spectrum (left, right). If None, the whole spectrum is used.
    - s_labels : list or None or False
        Legend entries for the single components. If None, they are computed automatically as 1, 2, 3, etc. If False, they are not shown in the legend.
    - X_label : str
        label for the x-axis.
    - n_xticks : int
        number of numbered ticks that will appear in the ppm scale. An oculated choice can be very satisfying.
    - name : str or None
        Name with which to save the figure. If None, the picture is shown instead of being saved.
    """
    N = S.shape[-1]

    # Set the limits
    if limits is None:
        limits = (max(ppm_scale), min(ppm_scale))

    # Get limit indexes
    lim1 = misc.ppmfind(ppm_scale, limits[0])[0]
    lim2 = misc.ppmfind(ppm_scale, limits[1])[0]
    lim1, lim2 = min(lim1, lim2), max(lim1, lim2)

    # Compute legend labels, if not already present
    if s_labels is None:
        s_labels = [str(w+1) for w in np.arange(V.shape[0])]

    x = np.linspace(0, 1, ppm_scale[lim1:lim2].shape[-1])[::-1]
    # Make the polynomion only if C contains its coefficients
    if C is False:
        y = np.zeros_like(x)
    else:
        y = misc.polyn(x, C)

    # Make the signals
    sgn = []
    Total = np.zeros_like(x)
    for i in range(V.shape[0]):
        sgn.append(fit.make_signal(t_AQ, V[i,0], V[i,1], V[i,2], V[i,3], V[i,4], V[i,5], SFO1=SFO1, o1p=o1p, N=N))
        Total += sgn[i][lim1:lim2].real

    # Initial figure
    fig = plt.figure(1)
    fig.set_size_inches(figsize_small)
    plt.subplots_adjust(bottom=0.15, top=0.90, left=0.15, right=0.95)
    ax = fig.add_subplot(1,1,1)

    # Experimental and total
    ax.plot(ppm_scale[lim1:lim2], S[lim1:lim2], label='Experimental', lw=0.8, c='k')
    ax.plot(ppm_scale[lim1:lim2], y+Total, label = 'Fit', c='tab:blue', lw=0.7)

    # Single components
    for i in range(V.shape[0]):
        s_plot, = ax.plot(ppm_scale[lim1:lim2], sgn[i][lim1:lim2].real, c=COLORS[i], lw=0.4, ls='--')
        if bool(s_labels[i]):
            s_plot.set_label(s_labels[i])

    # Baseline
    if C is not False:
        ax.plot(ppm_scale[lim1:lim2], y, label = 'Baseline', lw=0.4, c='tab:orange', ls='-.')

    # Customize picture appearance
    misc.pretty_scale(ax, limits, axis='x', n_major_ticks=n_xticks)

    ax.set_xlabel(X_label)
    ax.set_ylabel('Intensity /a.u.')

    misc.mathformat(ax, axis='y')

    ax.legend(framealpha=0.2)
    # Save/show the figure
    if name:
        misc.set_fontsizes(ax, 8)
        plt.savefig(name+'.png', dpi=600)
    else:
        fig.set_size_inches(figsize_large)
        misc.set_fontsizes(ax, 14)
        plt.show()
    plt.close()



def stacked_plot(ppmscale, S, xlims=None, lw=0.5, name=None, X_label='$\delta\ $ F1 /ppm', Y_label='Normalized intensity /a.u.', n_xticks=10, labels=None):
    """
    Creates a stacked plot of all the spectra contained in the list S. Note that S MUST BE a list. All the spectra must share the same scale.
    --------
    Parameters:
    - ppmscale: 1darray
        ppm scale of the spectrum
    - S: list
        spectra to be plotted
    - xlims: list or tuple
        Limits for the x-axis. If None, the whole scale is used.
    - lw: float
        linewidth
    - name: str
        filename of the figure, if it has to be saved;
    - X_label: str
        text of the x-axis label;
    - Y_label: str
        text of the y-axis label;
    - n_xticks: int
        Number of numbered ticks on the x-axis of the figure
    - labels: list
        labels to be put in the legend.
    """
    nsp = len(S)                                # number of spectra in the lsit
    if not labels:                              # auto-builds the labels for the spectra if not specified
        labels=[]
        for k in range(nsp):
            labels.append(str(k+1))
    
    # Normalizes all the spectra to the biggest value of the series
    norm_factor = np.max(np.abs(np.array(S)))
    for k in range(nsp):
        S[k] /= norm_factor

    # Define limits for the x-scale
    if xlims is None:
        xsx=max(ppmscale)
        xdx=min(ppmscale)
    else:
        xsx, xdx = max(xlims), min(xlims)

    # Define the figure
    fig = plt.figure()
    fig.set_size_inches(figsize_small)
    plt.subplots_adjust(left=0.15, bottom=0.2, right=0.95, top=0.95)
    ax = fig.add_subplot(1,1,1)
    # plot the spectra
    for k in range(nsp):
        ax.plot(ppmscale, S[k]+k, lw=lw, label=labels[k])

    misc.pretty_scale(ax, (xsx, xdx), axis='x', n_major_ticks=n_xticks)
    misc.mathformat(ax)

    # Decorate the axes
    ax.set_xlabel(X_label)
    ax.set_ylabel(Y_label)

    ax.set_ylim(-0.25, nsp+0.25)
    ax.legend()

    # Shows or saves the figure
    if name:
        misc.set_fontsizes(ax, 10)
        print( 'Saving '+name+'.png...')
        plt.savefig(name+'.png', format='png', dpi=600)
    else:
        fig.set_size_inches(figsize_large)
        plt.subplots_adjust(left=0.10, bottom=0.1, right=0.95, top=0.95)
        misc.set_fontsizes(ax, 14)
        cursor = Cursor(ax, useblit=True, horizOn=False, c='tab:red', lw=0.8)
        plt.show()
    plt.close()
    print( 'Done.')


def dotmd(ppmscale, S, labels=None, lw=0.8, n_xticks=10):
    """
    Interactive display of multiple 1D spectra.
    --------
    Parameters:
    - ppmscale: 1darray or list
        ppm scale of the spectra. If only one scale is supplied, all the spectra are plotted using the same scale. Otherwise, each spectrum is plotted using its scale. 
    - S: list
        spectra to be plotted
    - labels: list
        labels to be put in the legend.
    - n_xticks: int
        Number of numbered ticks on the x-axis of the figure
    """
    if isinstance(S, list):
        S = [S[w].real for w in range(len(S))]
    elif isinstance(S, np.ndarray):
        if len(S.shape) == 1:
            S = [S.real]
        elif len(S.shape) == 2:
            S = [S[w].real for w in range(S.shape[0])]
        if len(S.shape) == 3:
            raise ValueError('Maybe you should use figures.dotmd_2D')
        else:
            raise ValueError('{}D arrays are not allowed.'.format(len(S.shape)))

    nsp = len(S)

    if isinstance(ppmscale, np.ndarray):
        if len(ppmscale.shape) == 2:
            if ppmscale.shape[0] != nsp:
                raise ValueError('Number of scales do not match the number of spectra')
        elif len(ppmscale.shape) == 1:
            ppmscale = [ppmscale for w in range(nsp)]
        else:
            raise ValueError('There is a problem in the shape of the scale.')

    # flags for the activation of scroll zoom
    flags = np.ones(nsp)
    lvlstep = 0.1

    # Labels of the spectra that appear in the legend
    if not labels:
        labels = []
        for k in range(nsp):
            labels.append(str(k+1))
    elif len(labels) != nsp:
        raise ValueError('Shape mismatch: you provided {} labels for {} spectra.'.format(len(labels), nsp)) 

    # define boxes for sliders
    u_box = plt.axes([0.025, 0.85, 0.080, 0.05])
    d_box = plt.axes([0.025, 0.25, 0.080, 0.05])
    adj_box = plt.axes([0.025, 0.55, 0.080, 0.05])
    iz_box = plt.axes([0.025, 0.10, 0.05, 0.05])
    dz_box = plt.axes([0.025, 0.05, 0.05, 0.05]) 
    check_box = plt.axes([0.87, 0.20, 0.12, 0.04*nsp])

    # Functions connected to the sliders

    def increase_zoom(event): 
        nonlocal lvlstep
        lvlstep *= 2

    def decrease_zoom(event):
        nonlocal lvlstep
        lvlstep /= 2

    def y_autoscale(val):
        misc.set_ylim(ax, np.concatenate([s * scale_factor[k] for k, s in enumerate(S)]))
        D, U = ax.get_ylim()
        u_tb.set_val('{:.3e}'.format(U))
        d_tb.set_val('{:.3e}'.format(D))
        misc.pretty_scale(ax, ax.get_xlim(), axis='x', n_major_ticks=10)
        fig.canvas.draw()

    def update_ylim(val):
        U = eval(u_tb.text)
        D = eval(d_tb.text)
        misc.pretty_scale(ax, (D,U), axis='y', n_major_ticks=n_xticks)
        misc.pretty_scale(ax, ax.get_xlim(), axis='x', n_major_ticks=10)
        fig.canvas.draw()

    def on_scroll(event):
        nonlocal scale_factor
        for k in range(nsp):
            if flags[k]:
                if event.button == 'up':
                    scale_factor[k] += lvlstep 
                if event.button == 'down':
                    scale_factor[k] += -lvlstep
                if scale_factor[k] < 0:
                    scale_factor[k] = 0
        for k in range(nsp):
            spectrum[k].set_ydata(S[k].real * scale_factor[k])
            scale_text[k].set_text(f'{scale_factor[k]:.2f}')
        misc.pretty_scale(ax, ax.get_xlim(), axis='x', n_major_ticks=10)
        fig.canvas.draw()

    def radioflag(label):
        nonlocal flags
        status = radio.get_status()
        for k, stat in enumerate(status):
            flags[k] = stat

    # Make the figure
    fig = plt.figure(1)
    fig.set_size_inches(figsize_large)
    plt.subplots_adjust(left = 0.15, bottom=0.10, right=0.85, top=0.95)    # Make room for the sliders
    ax = fig.add_subplot(1,1,1)

    # Auto-adjusts the limits for the y-axis
    misc.set_ylim(ax, np.concatenate(S))
    # Make pretty scales
    misc.pretty_scale(ax, (np.max(np.concatenate(ppmscale)), np.min(np.concatenate(ppmscale))), axis='x', n_major_ticks=n_xticks)
    misc.pretty_scale(ax, ax.get_ylim(), axis='y', n_major_ticks=10)

    # Pretty y-axis numbers
    misc.mathformat(ax)
    # Adjust fonts
    misc.set_fontsizes(ax, 14)

    scale_factor = np.ones(nsp)
    spectrum = []
    # Plot the data
    for k in range(nsp):
        spect, = ax.plot(ppmscale[k], S[k].real*scale_factor[k], c=COLORS[k], lw=lw)      
        spectrum.append(spect)
    for k, spectr in enumerate(spectrum):
        spectr.set_label(labels[k])
    ax.legend(loc='upper right')

    # TextBoxes to set the ylims
    y_l = ax.get_ylim()
    u_tb = TextBox(ax=u_box, label='', initial='{:.3e}'.format(y_l[1]), textalignment='center')
    d_tb = TextBox(ax=d_box, label='', initial='{:.3e}'.format(y_l[0]), textalignment='center')

    # Create labels for the checkbox
    checklabels = []
    for k in range(nsp):
        checklabels.append(spectrum[k].get_label()[:12])
    radio = CheckButtons(check_box, checklabels, list(np.ones(nsp)))
    HBOX = check_box.dataLim.bounds[-1]
    misc.edit_checkboxes(radio, xadj=0, yadj=0.005, length=0.1, height=(HBOX-0.2*HBOX)/nsp, 
            color=[spec.get_color() for spec in spectrum])

    lbl_y = [ Q.get_position()[1] for Q in radio.labels]
    scale_text = []
    for Y, value in zip(lbl_y, scale_factor):
        scale_text.append(ax.text(0.995, Y, f'{value:.3f}',
            ha='right', va='center', transform=check_box.transAxes, fontsize=10))

    # Create buttons
    iz_button = Button(iz_box, label='$\\uparrow$') #!!!
    dz_button = Button(dz_box, label='$\downarrow$') #!!!
    adj_button = Button(adj_box, label='Adjust') #!!!

    # Connect the widgets to functions
    radio.on_clicked(radioflag)
    scroll = fig.canvas.mpl_connect('scroll_event', on_scroll)
        
    u_tb.on_submit(update_ylim)
    d_tb.on_submit(update_ylim)
    adj_button.on_clicked(y_autoscale)
    iz_button.on_clicked(increase_zoom)
    dz_button.on_clicked(decrease_zoom)
    cursor = Cursor(ax, useblit=True, color='red', horizOn=False, linewidth=0.4)

    plt.show()
    plt.close()


 
def dotmd_2D(ppm_f1, ppm_f2, S0, labels=None, name='dotmd_2D', X_label='$\delta\ $ F2 /ppm', Y_label='$\delta\ $ F1 /ppm', n_xticks=10, n_yticks=10, Neg=False):
    """
    Interactive display of multiple 2D spectra. They have to share the same scales.
    -------
    Parameters:
    - ppm_f1: 1darray
        ppm scale of the indirect dimension. If only one scale is supplied, all the spectra are plotted using the same scale. Otherwise, each spectrum is plotted using its scale. There is a 1:1 correspondance between ppm_f1 and S.
    - ppm_f2: 1darray
        ppm scale of the direct dimension. If only one scale is supplied, all the spectra are plotted using the same scale. Otherwise, each spectrum is plotted using its scale. There is a 1:1 correspondance between ppm_f2 and S.
    - S: list
        spectra to be plotted
    - labels: list
        labels to be put in the legend.
    - name: str
        If you choose to save the figure, this is its filename.
    - X_label: str
        text of the x-axis label;
    - Y_label: str
        text of the y-axis label;
    - n_xticks: int
        Number of numbered ticks on the x-axis of the figure
    - n_yticks: int
        Number of numbered ticks on the x-axis of the figure
    - Neg: bool
        If True, show the negative contours.
    """

    cmaps = [CM[key] for key in CM.keys() if '_r' in key]   # Use only _r cmaps otherwise you don't see a thing

    # Checks on dimensions of S0
    if isinstance(S0, list):
        S = [S0[w].real for w in range(len(S0))]
    elif isinstance(S0, np.ndarray):
        if len(S0.shape) == 1:
            raise ValueError('Maybe you should use figures.dotmd')
        elif len(S0.shape) == 2:
            S = [S0.real]
        elif len(S.shape) == 3:
            S = [S0[w].real for w in range(S0.shape[0])]
        else:
            raise ValueError('{}D arrays are not allowed.'.format(len(S0.shape)))

    nsp = len(S)        # Number of SPectra

    # Checks on scales dimensions
    if isinstance(ppm_f1, np.ndarray):
        if len(ppm_f1.shape) == 2:
            if ppm_f1.shape[0] != nsp:
                raise ValueError('Number of scales do not match the number of spectra')
        elif len(ppm_f1.shape) == 1:
            ppm_f1 = [ppm_f1 for w in range(nsp)]
        else:
            raise ValueError('There is a problem in the shape of the scale.')
    if isinstance(ppm_f2, np.ndarray):
        if len(ppm_f2.shape) == 2:
            if ppm_f2.shape[0] != nsp:
                raise ValueError('Number of scales do not match the number of spectra')
        elif len(ppm_f2.shape) == 1:
            ppm_f2 = [ppm_f2 for w in range(nsp)]
        else:
            raise ValueError('There is a problem in the shape of the scale.')
    # ----------------------------------------------------------------------------------

    # flags for the activation of scroll zoom
    flags = np.ones(nsp)
    scale_factor = np.ones(nsp)
    # Start level contour 
    lvl = [0.1 for k in range(nsp)]
    # Initialize lvlstep
    lvlstep = 0.02

    # Labels of the spectra that appear in the legend
    if not labels:
        labels = []
        for k in range(nsp):
            labels.append(str(k+1))
    elif len(labels) != nsp:
        raise ValueError('Shape mismatch: you provided {} labels for {} spectra.'.format(len(labels), nsp)) 

    # define boxes for sliders
    iz_box = plt.axes([0.025, 0.10, 0.05, 0.05])
    dz_box = plt.axes([0.025, 0.05, 0.05, 0.05])
    check_box = plt.axes([0.87, 0.20, 0.12, 0.04*nsp])
    save_box = plt.axes([0.15, 0.90, 0.10, 0.05])

    # ----------------------------------------------------------------------------------
    # Functions connected to the sliders
    def increase_zoom(event):
        """ double it """
        nonlocal lvlstep
        lvlstep *= 2 

    def decrease_zoom(event):
        """ halve it """
        nonlocal lvlstep
        lvlstep /= 2 

    def on_scroll(event):
        """ What happens when you scroll """
        nonlocal lvl, cnt
        if Neg:
            nonlocal Ncnt
        # Get limits of the figure, to reset them later
        xsx, xdx = ax.get_xlim()
        ysx, ydx = ax.get_ylim()
        # Move only the active spectra
        for k in range(nsp):
            if flags[k]:
                if event.button == 'up':
                    lvl[k] += lvlstep 
                if event.button == 'down':
                    lvl[k] += -lvlstep
                if lvl[k] < 1e-5:
                    lvl[k] = 1e-5
                if lvl[k] > 1:
                    lvl[k] = 1
        # Clear ax because cnt cannot be overwritten as list
        ax.cla()
        # Redraw the contours
        cnt = [figures.ax2D(ax, ppm_f2[k], ppm_f1[k], S[k], 
            xlims=(max(ppm_f2[k]), min(ppm_f2[k])), ylims=(max(ppm_f1[k]), min(ppm_f1[k])), 
            cmap=cmaps[k], c_fac=1.4, lvl=lvl[k], lw=0.5, X_label=X_label, Y_label=Y_label)
                for k in range(nsp)]
        if Neg:
            Ncnt = [figures.ax2D(ax, ppm_f2[k], ppm_f1[k], -S[k], 
                xlims=(max(ppm_f2[k]), min(ppm_f2[k])), ylims=(max(ppm_f1[k]), min(ppm_f1[k])), 
                cmap=cmaps[k], c_fac=1.4, lvl=lvl[k], lw=0.5, X_label=X_label, Y_label=Y_label)
                for k in range(nsp)]
        else: 
            Ncnt = None
        # Redraw the legend because of ax.cla()
        ax.legend(legend, Labels, loc='upper right', fontsize=14)

        # Set the limits as they were before
        misc.pretty_scale(ax, (xsx, xdx), 'x')
        misc.pretty_scale(ax, (ysx, ydx), 'y')

        # Update the zoom values in the legend
        [scale_text[k].set_text(f'{value:.3f}') for k, value in enumerate(lvl)]
        # Bigger fonts
        misc.set_fontsizes(ax, 14)
        fig.canvas.draw()

    def radioflag(label):
        """ Change the flags array according to the checkbox """
        nonlocal flags
        status = radio.get_status()
        for k, stat in enumerate(status):
            flags[k] = stat

    def makefigure(event):
        """ Make a figure """
        if nsp == 1:
            figures.figure2D(ppm_f2[0], ppm_f1[0], S[0], xlims=(l_slider.val, r_slider.val), ylims=(u_slider.val, d_slider.val), lvl=lvl[0], name=name, X_label=X_label, Y_label=Y_label, n_xticks=10, n_yticks=10)
        else:
            figures.figure2D_multi(ppm_f2, ppm_f1, S, xlims=(l_slider.val, r_slider.val), ylims=(u_slider.val, d_slider.val), lvl=lvl, name=name, X_label=X_label, Y_label=Y_label, n_xticks=10, n_yticks=10, labels=labels)

    # ----------------------------------------------------------------------------------

    # Make the figure
    fig = plt.figure(1)
    fig.set_size_inches(figsize_large)
    plt.subplots_adjust(left = 0.15, bottom=0.10, right=0.85, top=0.95)    # Make room for the sliders
    ax = fig.add_subplot(1,1,1)

    # Draw the contours
    cnt = [figures.ax2D(ax, ppm_f2[k], ppm_f1[k], S[k], xlims=(max(ppm_f2[k]), min(ppm_f2[k])), ylims=(max(ppm_f1[k]), min(ppm_f1[k])), cmap=cmaps[k], c_fac=1.4, lvl=lvl[k], lw=0.5, X_label=X_label, Y_label=Y_label)
            for k in range(nsp)]
    if Neg:
        Ncnt = [figures.ax2D(ax, ppm_f2[k], ppm_f1[k], -S[k], xlims=(max(ppm_f2[k]), min(ppm_f2[k])), ylims=(max(ppm_f1[k]), min(ppm_f1[k])), cmap=cmaps[k], c_fac=1.4, lvl=lvl[k], lw=0.5, X_label=X_label, Y_label=Y_label)
            for k in range(nsp)]
    else: 
        Ncnt = None

    # Set the legend
    legend = []
    for i in range(len(labels)):
        h,_ = cnt[i].legend_elements()
        legend.append(h[0])
    if Neg:
        for i in range(len(labels)):
            Nh,_ = Ncnt[i].legend_elements()
            legend.append(Nh[0])
    # Draw the legend
    if Neg:
        Nlabels = ['$-$ '+labels[w] for w in range(len(labels))]
        Labels = [*labels, *Nlabels]
    else:
        Labels = labels
    ax.legend(legend, Labels, loc='upper right', fontsize=14)

    # Make pretty x-scale
    xsx, xdx = max(np.concatenate(ppm_f2)), min(np.concatenate(ppm_f2))
    ysx, ydx = max(np.concatenate(ppm_f1)), min(np.concatenate(ppm_f1))
    misc.pretty_scale(ax, (xsx, xdx), axis='x')
    misc.pretty_scale(ax, (ysx, ydx), axis='y')


    # Create labels for the checkbox
    checklabels = []
    for k in range(nsp):
        checklabels.append(labels[k][:12])
    radio = CheckButtons(check_box, checklabels, list(np.ones(nsp)))
    HBOX = check_box.dataLim.bounds[-1]
    misc.edit_checkboxes(radio, xadj=0, yadj=0.005, length=0.1, height=(HBOX-0.2*HBOX)/nsp)

    lbl_y = [ Q.get_position()[1] for Q in radio.labels]
    scale_text = []
    for Y, value in zip(lbl_y, scale_factor):
        scale_text.append(check_box.text(0.995, Y, f'{value:.3f}',
            ha='right', va='center', transform=check_box.transAxes, fontsize=10))

    # Create buttons
    iz_button = Button(iz_box, label='$\\uparrow$')
    dz_button = Button(dz_box, label='$\downarrow$')
    save_button = Button(ax=save_box, label='Make\nfigure')

    # Connect the widgets to functions
    radio.on_clicked(radioflag)
    scroll = fig.canvas.mpl_connect('scroll_event', on_scroll)
        
    iz_button.on_clicked(increase_zoom)
    dz_button.on_clicked(decrease_zoom)
    save_button.on_clicked(makefigure)

    cursor = Cursor(ax, useblit=True, color='red', linewidth=0.4)

    misc.set_fontsizes(ax, 14)

    plt.show()
    plt.close()


def redraw_contours(ax, ppm_f2, ppm_f1, S, lvl, cnt, Neg=False, Ncnt=None, lw=0.5, cmap=[None, None], verb=False):
    """
    Redraws the contours in interactive 2D visualizations.
    --------
    Parameters:
    - ax: matplotlib.Subplot Object
        Panel of the figure where to draw the contours
    - ppm_f2: 1darray
        ppm scale of the direct dimension
    - ppm_f1: 1darray
        ppm scale of the indirect dimension
    - S: 2darray
        Spectrum
    - lvl: float
        Level at which to draw the contours
    - cnt: matplotlib.contour.QuadContourSet object
        Pre-existing contours
    - Neg: bool
        Choose if to draw the negative contours (True) or not (False)
    - Ncnt: matplotlib.contour.QuadContourSet object
        Pre-existing negative contours
    - lw: float
        Linewidth
    - cmap: list
        Colour of the contours. [cmap +, cmap -]
    -------
    Returns:
    - cnt: matplotlib.contour.QuadContourSet object
        Updated contours
    - Ncnt: matplotlib.contour.QuadContourSet object or None
        Updated negative contours if Neg is True, None otherwise
    """

    # Suppress the 'I cannot find the contours' warning
    warnings.filterwarnings("ignore", message="No contour levels were found within the data range.")

    for c in cnt.collections:
        # try to remove the positive contours
        try:
            c.remove()
        except Exception as e:
            if verb:
                print(e)
            pass
    if Neg:
        # try to remove the negative contours
        for Nc in Ncnt.collections:
            try:
                Nc.remove()
            except Exception as e:
                if verb:
                    print(e)
                pass
    # Draw new positive contours
    cnt = figures.ax2D(ax, ppm_f2, ppm_f1, S, lvl=lvl, cmap=cmap[0])
    if Neg:
        # Draw new negative contours
        Ncnt = figures.ax2D(ax, ppm_f2, ppm_f1, -S, lvl=lvl, cmap=cmap[1])
    else:
        Ncnt = None

    # Return things
    return cnt, Ncnt

