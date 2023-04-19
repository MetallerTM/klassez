#! /usr/bin/env python3

import os
import sys
import numpy as np
from scipy import linalg, stats
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from pprint import pprint as Print 

from . import fit, misc, sim, figures, processing
from .Spectra import Spectrum_1D, pSpectrum_1D, Spectrum_2D, pSpectrum_2D, Pseudo_2D

__version__ = '0.2a.0'

# Use seaborn's colormaps and save it to a dictionary
from .config import CM, COLORS, cron

def open_doc():
    """ Open the documentation .pdf file in the browser. """
    import webbrowser
    webbrowser.open_new(__doc__)

__doc__ = f'{__path__[0]}/docs/klassez.pdf'
