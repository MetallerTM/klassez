#! /usr/bin/env python3

from klassez import *

# Be aware that this is a BASIC processing 
# Read the documentation of the functions to see the full powers

if 1:
    # This example is for the simulated data
    s = Spectrum_1D('acqus_1D', isexp=False)
    s.to_vf()   # You can convert info on peaks to .ivf for fitting
else:
    # Use the following to read experimentals:
    spect = 'bruker', 'jeol', 'varian', 'magritek', 'oxford' # One of these
    s = Spectrum_1D(path_to_dataset, spect=spect)

# Setup the processing
#   Apodization
#       Follow the table in the user manual to see what reads what
s.procs['wf']['mode'] = 'em'
s.procs['wf']['lb'] = 5
#   Zero-filling
s.procs['zf'] = 2**14

#   Apply processing and do FT
s.process()
# Remove the digital filter
s.pknl()
# Phase correction
s.adjph()
# Plot the data
s.plot()



