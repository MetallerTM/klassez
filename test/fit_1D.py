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



# s.F is a fit.Voigt_Fit object
filename = 'test_1D_fit'    # base filename for everything fit-related
# Compute the initial guess
auto = False        # True for peak-picker, False for manual
s.F.iguess(filename=filename, auto=auto)

if 0:   # Do the fit
    s.F.dofit(              # Parameters of the fitting
            u_lim=5,            # movement for chemical shift /ppm
            f_lim=50,           # movement for linewidth /Hz
            k_lim=(0, 3),       # limits for intensity
            vary_phase=True,    # optimize the phase of the peak 
            vary_b=True,        # optimize the lineshape (L/G ratio)
            method='leastsq',   # optimization method
            itermax=10000,      # max. number of iterations
            fit_tol=1e-10,      # arrest criterion threshold (see lmfit for details)
            filename=filename,  # filename for the .fvf file
            )
else:
    # Load an existing .fvf file
    s.F.load_fit(filename=filename) 

# Plot the results
s.F.plot(what='result',     # what='iguess' for initial guess
         show_total=True,   # Show the total trace or not
         show_res=True,     # Show the residuals
         res_offset=0.1,    # Displacement of the residuals (plots residuals - res_offset)
         labels=None,       # Labels for the peaks
         filename=filename, # Filename for the figures
         ext='png',         # format of the figure
         dpi=300,           # Resolution of the figure
         )

# Compute histogram of the residuals
s.F.res_histogram(what='result', 
              nbins=500,    # Number of bins of the histogram
              density=True, # Normalize them
              f_lims=None,  # Limits for x axis
              xlabel='Residuals',   # Guess what!
              x_symm=True,  # Symmetrize the x-scale
              barcolor='tab:green',     # Color of the bars
              fontsize=20,  # Guess what!
              filename=filename, ext='png', dpi=300)

# Convert the tables of numbers in arrays
peaks, total, limits = s.F.get_fit_lines(what='result') 
