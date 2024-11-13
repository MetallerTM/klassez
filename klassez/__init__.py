#! /usr/bin/env python3

import os
import sys
import numpy as np
from numpy import linalg
from scipy import stats
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from copy import deepcopy
from pprint import pprint

from . import fit, misc, sim, figures, processing
from .Spectra import Spectrum_1D, pSpectrum_1D, Spectrum_2D, pSpectrum_2D, Pseudo_2D

# Use seaborn's colormaps and save it to a dictionary
from .config import CM, CM_2D, COLORS, cron

__version__ = '0.4a.7'
