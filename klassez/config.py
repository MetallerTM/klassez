#! /usr/bin/env python3

import sys
import os
from pathlib import Path
from numpy import loadtxt
import seaborn as sns
from datetime import datetime
from functools import wraps
import inspect
from importlib.resources import open_text, files

_print = print


class XtermColors:
    """
    Container for the pretty customization of printed text, in terms of colors and style.
    Uses the `xterm` version of the standard `matplotlib` colors.

    Usage:

    .. code-block::

        from klassez import textcolor

        text = 'my text'
        print(textcolor(text, 'red')


    Use

    .. code-block::

        print(klassez.textcolor)

    to see the supported colors and styles.

    Attributes
    ----------
    colors : dict
        Name of the supported colors and correspondant formatting string.
    styles : dict
        Name of the supported font styles
    reset : str
        String to reset the output to normal

    """
    styles = {
            'bold': '\033[1m',
            'italic': '\033[3m',
            'bold_italic': '\033[1;3m',
            'underline': '\033[4m',
            }
    reset = '\033[0m'

    def __str__(self):
        """ Prints the name of the supported colors and styles. """
        N = len(list(self.colors.keys()))
        n_percol = 6
        color_keys = [list(self.colors.keys())[n_percol*j: n_percol*j+n_percol]
                      for j in range(N // n_percol)]
        color_keys += [list(self.colors.keys())[-1 * (N % n_percol):]]

        color_doc_string = '\n'.join([
            ' '.join([self(f'{w:>15s}', color=w) for w in splitted_colors])
            for splitted_colors in color_keys
            ])

        style_doc_string = ' '.join([self(f'{w:>15s}', style=w) for w in list(self.styles.keys())])

        dash_line = '-' * 96

        doc_string = '\n'.join([
            'List of supported colors',
            dash_line,
            color_doc_string,
            '\n',
            'List of supported styles',
            dash_line,
            style_doc_string,
            ])
        return doc_string

    def __init__(self, dic_xcolor):
        """
        Store ``dic_xcolor`` in ``self.colors``.

        Parameters
        ----------
        dic_xcolor : dict
            Name of the colors and rendering strings

        """
        self.colors = dic_xcolor

    def __call__(self, text: str, color=None, style=None):
        """
        Formats the text.

        Parameters
        ----------
        text : str
            Text to format
        color, c : str or None
            Color to format the text. If ``None``, the default color is kept
        style, s : str or None
            Style to format the text. If ``None``, the default style is kept

        Returns
        -------
        fmt_text : str
            Formatted text with chosen style and color
        """
        color_str = self.colors[color] if color else ''
        style_str = self.styles[style] if style else ''
        fmt_text = color_str + style_str + text + self.reset
        return fmt_text


def cron(f):
    r""" Decorator: use it to monitor the runtime of a function.  """
    @wraps(f)
    def wrapper(*args, **kws):
        start_time = datetime.now()

        return_values = f(*args, **kws)

        end_time = datetime.now()
        run_time = end_time - start_time
        print(f'Runtime: {run_time}\n')
        return return_values
    return wrapper


def safe_kws(f):
    """
    Decorator.

    Let us assume we want to run the following code:

    .. code-block::

        def f(a, b=1):
            print(a, b)

        kws = {'a': 1, 'b': 2, 'c': 3}
        f(**kws)

    This will raise an error, because there is not a parameter ``c`` in the
    signature of ``f``. Decorating ``f`` with ``@safe_kws`` will filter the passed
    ``kws`` dictionary to include only the parameters that are also present in ``f``,
    in order to avoid this error to appear.
    """

    # Take the signature of the function as OrderedDict {arg_name: default_value}
    sig = inspect.signature(f)
    # Take only the names of the arguments
    all_args_names = set(sig.parameters.keys())

    @wraps(f)
    def wrapper(*args, **kws):
        # From kws, keep only the arguments that also appear in the signature
        new_kws = {arg: value for arg, value in kws.items()
                   if arg in all_args_names}
        return f(*args, **new_kws)
    return wrapper


def cprint(*args, c=None, s=None, sep=' ', end='\n', file=None, flush=False):
    """
    This function can override the default :func:`print` function, allowing to format the text easily.
    Calls :func:`klassez.textcolor` with ``color=c`` and ``style=s``.

    Parameters
    ----------
    args : sequence
        Things to print, will be converted in text.
    c : str or None
        Color of the text
    s : str or None
        Style of the text
    All the rest are the parameters of the print function
    """
    # Format the text
    str_args = [textcolor(str(arg), color=c, style=s) for arg in args]

    # Join using the correct separator
    text = sep.join(str_args)

    # Print with the original print
    _print(text, end=end, file=file, flush=flush)


# Use seaborn's colormaps and save it to a dictionary
CMapsNames = [
    'Accent', r'Accent_r', r'Blues', r'Blues_r', r'BrBG', r'BrBG_r', r'BuGn', r'BuGn_r', r'BuPu', r'BuPu_r',
    r'CMRmap', r'CMRmap_r', r'Dark2', r'Dark2_r', r'GnBu', r'GnBu_r', r'Greens', r'Greens_r', r'Greys', r'Greys_r',
    r'OrRd', r'OrRd_r', r'Oranges', r'Oranges_r', r'PRGn', r'PRGn_r', r'Paired', r'Paired_r', r'Pastel1', r'Pastel1_r',
    r'Pastel2', r'Pastel2_r', r'PiYG', r'PiYG_r', r'PuBu', r'PuBuGn', r'PuBuGn_r', r'PuBu_r', r'PuOr', r'PuOr_r', r'PuRd',
    r'PuRd_r', r'Purples', r'Purples_r', r'RdBu', r'RdBu_r', r'RdGy', r'RdGy_r', r'RdPu', r'RdPu_r', r'RdYlBu', r'RdYlBu_r',
    r'RdYlGn', r'RdYlGn_r', r'Reds', r'Reds_r', r'Set1', r'Set1_r', r'Set2', r'Set2_r', r'Set3', r'Set3_r', r'Spectral',
    r'Spectral_r', r'Wistia', r'Wistia_r', r'YlGn', r'YlGnBu', r'YlGnBu_r', r'YlGn_r', r'YlOrBr', r'YlOrBr_r', r'YlOrRd',
    r'YlOrRd_r', r'afmhot', r'afmhot_r', r'autumn', r'autumn_r', r'binary', r'binary_r', r'bone', r'bone_r', r'brg', r'brg_r',
    r'bwr', r'bwr_r', r'cividis', r'cividis_r', r'cool', r'cool_r', r'coolwarm', r'coolwarm_r', r'copper', r'copper_r', r'cubehelix',
    r'cubehelix_r', r'flag', r'flag_r', r'gist_earth', r'gist_earth_r', r'gist_gray', r'gist_gray_r', r'gist_heat', r'gist_heat_r',
    r'gist_ncar', r'gist_ncar_r', r'gist_rainbow', r'gist_rainbow_r', r'gist_stern', r'gist_stern_r', r'gist_yarg', r'gist_yarg_r',
    r'gnuplot', r'gnuplot2', r'gnuplot2_r', r'gnuplot_r', r'gray', r'gray_r', r'hot', r'hot_r', r'hsv', r'hsv_r', r'icefire',
    r'icefire_r', r'inferno', r'inferno_r', r'magma', r'magma_r', r'mako', r'mako_r', r'nipy_spectral', r'nipy_spectral_r',
    r'ocean', r'ocean_r', r'pink', r'pink_r', r'plasma', r'plasma_r', r'prism', r'prism_r', r'rainbow', r'rainbow_r', r'rocket', r'rocket_r',
    r'seismic', r'seismic_r', r'spring', r'spring_r', r'summer', r'summer_r', r'tab10', r'tab10_r', r'tab20', r'tab20_r', r'tab20b',
    r'tab20b_r', r'tab20c', r'tab20c_r', r'terrain', r'terrain_r', r'twilight', r'twilight_r', r'twilight_shifted',
    r'twilight_shifted_r', r'viridis', r'viridis_r', r'vlag', r'vlag_r', r'winter', r'winter_r'
    ]

global CM, COLORS
CM = {}
for key in CMapsNames:
    CM[key] = sns.color_palette(key, as_cmap=True)
del CMapsNames

# Cmaps for 2D spectra display
CM_2D = {key: CM[key] for key in ['Greys_r', 'Blues_r', 'Reds_r', 'Greens_r', 'Oranges_r', 'Purples_r', 'copper']}


# List of colors

colors = ['tab:blue', 'tab:red', 'tab:green', 'tab:orange', 'tab:cyan',
          'tab:purple', 'tab:pink', 'tab:gray', 'tab:brown', 'tab:olive',
          'salmon', 'indigo', 'm', 'c', 'g', 'r', 'b', 'k',
          ]

for w in range(10):
    colors += colors
COLORS = tuple(colors)

# Color_formatter text
if sys.version_info < (3, 13):      # python 3.9 - 3.12
    with files(__package__).joinpath('tables', 'xterm_colors').open('r', encoding='utf-8') as f:
        color_arr = loadtxt(f, dtype=str, comments='#', delimiter='&', converters=lambda w: w.strip(), skiprows=0, usecols=(0, 1), unpack=True)
else:                               # python 3.13 and above
    with open_text(__name__, Path('tables') / 'xterm_colors') as f:
        color_arr = loadtxt(f, dtype=str, comments='#', delimiter='&', converters=lambda w: w.strip(), skiprows=0, usecols=(0, 1), unpack=True)

dic_xcolors = {str(name): '\033[' + str(color) for name, color in zip(color_arr[0], color_arr[1])}
textcolor = XtermColors(dic_xcolors)
