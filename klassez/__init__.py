#! /usr/bin/env python3

import os
import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from copy import deepcopy
from pprint import pprint

from . import fit, misc, sim, figures, processing, anal

from .Spectra import Spectrum_1D, pSpectrum_1D, Spectrum_2D, pSpectrum_2D, Pseudo_2D

from .config import CM, CM_2D, COLORS, cron

__version__ = "0.5a.2"
