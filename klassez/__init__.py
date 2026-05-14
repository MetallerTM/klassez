#! /usr/bin/env python3

import os
import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from copy import deepcopy
from pprint import pprint
from pathlib import Path

from . import fit, misc, sim, figures, processing, anal, gui

from .Spectra import *

from .config import CM, CM_2D, COLORS, cron, safe_kws, textcolor, _print, cprint

print = cprint

__version__ = "0.2.0"
