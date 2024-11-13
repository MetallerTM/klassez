#! /usr/bin/env python3

from klassez import *

# Be aware that this is a BASIC processing 
# Read the documentation of the functions to see the full powers

if 1:
    # This example is for the simulated data
    s = Spectrum_2D('acqus_2D', isexp=False)
else:
    # For experimentals, at version 0.4a.7 klassez reads only 2D bruker
    s = Spectrum_2D(path_to_dataset)

# Setup the processing
#   Apodization
#       Follow the table in the user manual to see what reads what
#       REMEMBER: index 0 is F1, index 1 is F2, for procs
s.procs['wf'][1]['mode'] = 'em'
s.procs['wf'][1]['lb'] = 5
s.procs['wf'][0]['mode'] = 'qsin'
s.procs['wf'][0]['ssb'] = 2
#   Zero-filling
s.procs['zf'] = 512, 2048

#   Apply processing and do FT
s.process()
# Remove the digital filter
s.pknl()
# Phase correction
s.adjph()
# Plot the data
s.plot()

# Extract projections
ppm_f2 = 180
ppm_f1 = 10
s.projf1(ppm_f2)    # Extract F1 trace @ ppm_f2 ppm
f1 = s.Trf1[f'{ppm_f2:.2f}']    # Call it back: it is a Spectrum_1D object!
f1.plot()
s.projf2(ppm_f1)    # Extract F2 trace @ ppm_f1 ppm
f2 = s.Trf2[f'{ppm_f1:.2f}']    # Call it back: it is a Spectrum_1D object!
f2.plot()



