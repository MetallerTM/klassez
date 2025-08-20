# Version 0.4a.11


# Version 0.4a.10

A couple of new processing functions.
Bug fixes.

## New functions:
- *processing.abc*
- *processing.abs*
- *processing.hilbert2*

## Modified functions:
- *processing.calibration*: now accepts a reference spectrum to be used as guideline
- *fit.lsp*: introduced option for weighted fit


# Version 0.4a.9

Changed way to compute the contours in the 2D interactive plots, in order to avoid the freezes when going close to the noise level.
The fitting procedure now features the use of a polynomial baseline. The input/output files (.ivf and .fvf) can be converted in input1 and input2 to be used by TrAGICo.
Approx. value 2.355 converted in 2 * (2 ln 2)^0.5
The function *cal* of the *Spectra* classes must now be called explicitely. New argument **from_procs**
Bug fixes.

## New functions:
- *misc.lenslice*
- *misc.sum_overlay*
- *fit.Voigt_Fit.to_tragico*

## Modified functions:
- *fit.Voigt_Fit.dofit*: added argument **basl_fit**, returns **lmfit_results**
- *fit.Voigt_Fit.get_fit_lines*: now returns also **whole_basl**
- *fit.Voigt_Fit.plot*: modified to use *fit.plot_fit*
- *fit.Voigt_Fit.res_hist*: the baseline is accounted for in the residuals computation
- *fit.make_iguess*: now allows to compute the baseline as well
- *fit.plot_fit*: new argument **dim**
- *fit.plot_fit*: new argument **show_basl**
- *fit.voigt_fit_indep*: new argument **basl_fit**, returns **lmfit_results**
- *fit.read_vf*: it reads the baseline coefficients and puts them in the key **bas_c**
- *fit.write_vf*: also writes the baseline coefficients
- *Pseudo_2D.cal*: new argument **from_procs**
- *Spectrum_1D.cal*: new argument **from_procs**
- *Spectrum_2D.cal*: new argument **from_procs**



# Version 0.4a.8

Fixed import statement in the *sim* module.
Added option to click with the mouse middle button as an alternative to left double click.
Now the rulers in the *plot* function of *Spectrum_1D* and *Spectrum_2D* are dragged with the right mouse button.
Phase adjustments are now possible with respect to another spectrum imported as reference.
The interactive plots are possible also as function of the frequency scales.

Bug fixed in the calibration functions.
Other bug fixes.

## New functions:
04/12/2024
- *processing.splitcomb* 
- *Spectrum_2D.splitcomb*

## Modified functions:
16/12/2024
- *Spectrum_1D.pknl*: now it works automatically either on FID or processed data by calling *processing.pknl*
xx/03/2025
- *Spectrum_nD.plot*: new argument **fqscale**
- *processing.ps*: new argument **reference**
- *Spectrum_nD.adjph*: uses the new *processing.ps*



# Version 0.4a.7

Changed *processing.gm* and *processing.gmb* implementation, and to use two different sets of **lb** and **gb** in the *Spectra.procs* classes.
*processing.fp* and *processing.interactive_fp* were also modified accordingly.
SIMPLISMA correction.

Corrected a few docstrings.
Remade user manual and added new test scripts.
Bug fixes.

## Modified functions:
- *misc.noise_snr*: corrected error in the formula, now also use *np.sum* instead of loops
- *processing.fp*: **gm** and **gmb** now use two different sets of parameters.
- *processing.gm*: now it works.
- *processing.gmb*: now it works.
- *processing.interactive_fp*: **gm** and **gmb** now use two different sets of parameters.
- *processing.interactive_phase_1D*: bug fix, removed sliders
- *processing.simplisma*: The first purest variable is computed after the correction for the first weight.
- *fit.histogram*: added option **fitG** to draw the gaussian or not
- *fit.fit_int*: change of behavior if one wants to compute intensity only, or also the offset
- *fit.make_iguess*: active peak is now drawn with a thicker line
- *fit.voigt_fit_indep*: changed <b>_tol</b> to <b>_lim</b>, added <b>k_lim</b>
- *fit.write_vf*: added **header** parameter to allow the writing of a separator.



# Version 0.4a.6

Added reading for Jeol. The package **jeol_parser** is now a required installation prerequisite.
New function for linear prediction. The previous function for backwards linear prediction was kept as a legacy option in *processing.blp_ng*
The **sim.gamma** dictionary now is read from a file in the new folder **klassez/tables**. There is another file there, that is read to generate the **sim.jeol_nuclei** dictionary, which is needed for the conversion from Jeol-style nuclei to *klassez* format.
The required minimum python version is then changed to 3.9.
All **x_g** labels were changed to **b**. Scripts written with previous versions of *klassez* may be corrected accordingly.
Improved fitting routines. Bug fixes.


## New functions: 
- *misc.makeacqus_1D_jeol*
- *processing.lp*
- *processing.blp_ng*

## Modified functions:
- *misc.pretty_scale*: now the scale is automatically recomputed. Additional options possible.
- *processing.blp*: now uses the new *processing.lp* function.
- *figures.dotmd*: all traces are deactivated by default
- *Spectrum_1D.__init__*: option **spect=jeol** added


# Version 0.4a.5

Added titles of all the figures.
Changed name of the following functions:
- Module **MISC**:
    - *SNR* &rarr; *snr*
    - *SNR_2D* &rarr; *snr_2D*
- Module **PROCESSING**
    - *Cadzow* &rarr; *cadzow*
    - *Cadzow_2D* &rarr; *cadzow_2D*
    - *EAE* &rarr; *eae*
    - *LRD* &rarr; *lrd*
    - *MCR* &rarr; *mcr*
    - *MCR_ALS* &rarr; *mcr_als*
    - *MCR_unpack* &rarr; *mcr_unpack*
    - *stack_MCR* &rarr; *mcr_stack*
    - *SIMPLISMA* &rarr; *simplisma*
- Module **FIT**
    - *LR* &rarr; *lr*
    - *LSP* &rarr; *lsp*
    - *SINC_phase* &rarr; *sinc_phase*

Several bug corrections.

## New functions:
- *misc.merge_dict*
- *misc.zero_crossing*
- *fit.make_iguess_auto*
- *Spectrum_1D.add_noise*
- *Spectrum_1D.scan*
- *Spectrum_2D.add_noise*
- *Spectrum_2D.scan*
- *Pseudo_2D.add_noise*
- *Pseudo_2D.scan*

## Modified functions
- *fit.Voigt_Fit.iguess*: added "auto" flag to use *fit.make_iguess_auto*
- *Spectrum_1D.plot*: added tracker for distance measurement
- *Spectrum_2D.plot*: added tracker for distance measurement

## Modified classes
- *fit.Voigt_Fit*: added attribute **SW**, computed on the ppm scale


# Version 0.4a.4

Several changes to allow compatibility with newer versions of **matplotlib**. 
Added the attribute **CM_2D** in *config*, a dictionary that collects only the colormaps to be used for displaying 2D spectra.
Modified dtype of empty arrays in *Spectra*.
Added new functions for convolution, and its reverse. The latter does not work yet.
Bug corrections. 

## New functions:
- *processing.convolve*
- *processing.inv_convolve* - DOES NOT WORK
- *fit.test_randomsign*
- *fit.test_correl*
- *fit.test_ks*

## Modified functions:
- *misc.select_traces*: changed identifier of mouse buttons
- *misc.show_cmap*: added parameter **filename** to save the figure, changed figure size
- *processing.MCR*: modified parameters to use the new *processing.stack_MCR* and *processing.MCR_unpack*
- *processing.MCR_unpack*: now uses positioning matrix
- *processing.stack_MCR*: now uses positioning matrix
- *figures.dotmd_2D*: now uses the new **CM_2D**
- *figures.redraw_contour*: change for compatibility with matplotlib>3.7
- *fit.test_residuals*: employs three new functions, improved performance
- *Spectrum_2D.__init__*: reshape of the fid is now made with last dimension implicit


# Version 0.4a.3

Added new functions to transform FID in audio files! 
New function for computing Hilbert transform.
Adaptation of some codes for the new versions of *numpy* and *scipy*.
Bug correction.

## New functions:
- *misc.data2wav*
- *processing.hilbert*
- *fit.Peak.get_fid*
- *Spectrum_1D.to_wav*
- *Spectrum_2D.to_wav*
- *Pseudo_2D.to_wav*

## Modified functions:
- *misc.binomial_triangle*: employs the function from the *math* library
- *misc.edit_checkboxes*: adapted for *matplotlib>=3.7*
- *processing.ft*, *processing.ift*: warnings for non-complex data are now suppressed
- *processing.qpol*: the polynomion is computed by solving the linear system with *fit.LSP*
- *sim.load_sim_1D*, *sim.load_sim_2D*: more intelligent handling of spacings, now writing input files is easier
- *sim.t_voigt*, *sim.t_2Dvoigt*: corrected the parameter **x_G** in order to preserve the linewidth of the signal
- *Pseudo_2D.mount*: the FIDs have now priority on the .npy file


## Modified classes:
- *fit.CostFunc*: all methods now compute the array of residuals


# Version 0.4a.2

Added new functions for fitting pseudo-2D experiments. Bug corrections.
Improved fitting routines.

## New functions:
22/01/2024
- *fit.make_iguess_P2D*
- *fit.plot_fit_P2D*
- *fit.read_vf_P2D*
- *fit.voigt_fit_P2D*
- *fit.write_vf_P2D*
- *fit.Voigt_Fit.res_histogram*
- *fit.Voigt_Fit_P2D.res_histogram*
- *Spectrum_1D.pknl*
- *Spectrum_2D.pknl*
- *Spectrum_1D.to_vf*

## Modified functions:
- *fit.histogram*: changed figure size to **figures.figsize_large**
- *<class>.write_ser* for all classes in *Spectra*: fixed positional arguments order

## New classes:
- *fit.Voigt_Fit_P2D*

## Modified classes:
- *fit.Peak*: instead of zero-filling, the acqusition timescale is extended to match the shape of the spectrum with *misc.extend_taq*
- *fit.Voigt_Fit*: acqusition timescale extended with *misc.extend_taq*
- *Pseudo_2D*: added new attribute **F** as a wrapper for a *fit.Voigt_Fit_P2D* object. The **self.F.S** attribute is overwritten every time **self.S** is changed by a function such as *adjph*.


# Version 0.4a.1

Bug corrections.
Added support for Topspin 4.3.0, i.e. a reshape for the FIDs.
Convdta seems to not work anymore, hence we introduced a way to remove the digital filter effects on the spectra by using the *pnkl* function.

## New functions: 
- *misc.load_ser*
- *processing.integrate*
- *processing.stack_fids*
- *fit.calc_R2*
- *Spectrum_2D.pknl*
- *Pseudo_2D.mount*
- *Pseudo_2D.cal*
- *Pseudo_2D.pknl*

## Modified functions:
- *misc.get_ylim*: changed method of calculations of plot edges
- *misc.set_ylim*: now uses modified *misc.get_ylim*
- *misc.polyn*: use the Vandermonde matrix to compute the polynomion instead of a for loop
- *fit.LR*: now treated as a polynomial fit, added option to force the intercept
- *fit.gen_iguess*: brand new implementation
- *fit.make_iguess*: added an interactive text to show the actual sensitivity value, and a tutorial for the keyboard shortcuts (thanks to Tino Golub for requesting for this)
- *Spectrum_1D.integrate*: corrected integral computation by using new function *processing.integrate*
- *Pseudo_2D.__init__*: self.fid = None by default, added acqus parameter "TD1", removed procs parameter "cal\_1", renamed procs parameter "cal\_2" into "cal" 

## Modified classes:
- *Spectrum_2D*: Added control for FnMODE="QF-nofreq"


# Version 0.4a.0

## Update notes:
It is now possible to work with Oxford Instrument NMR data, saved in .jdx files.
The class *fit.Voigt_Fit* was completely rewritten using new, better performant functions.
The default format for all figures became ".tiff".
Minor bugs corrected.

## New functions:

- *misc.makeacqus_1D_oxford*
- *processing.blp*
- *fit.plot_fit*
- *fit.voigt_fit_indep*
- *fit.write_vf*
- *fit.read_vf*
- *Spectrum_1D.blp*

## New classes:

- *fit.Peak*

## Modified functions:

- *figures.dotmd*: when the figure is closed, returns the scaling intensity factor for each spectrum
- *figures.dotmd_2D*: when the figure is closed, returns the contour level start for each spectrum.
- *fit.make_iguess*: infinite components!, new format for the output file

## Modified classes:
- *fit.Voigt_Fit*: completely rewritten using the new functions.


# Version 0.3a.1

## Update notes:
Extension of the Spectra classes for the management of other-than-Bruker data formats. 
Modifications to some attributes and methods to make everything more user-friendly.
See the dedicated paragraph for details.

The *linalg* package, previously imported from *scipy*, was now substituted with its *numpy* counterpart. Problems due to missing funcions were not tested, hence errors might arise.

New algorithms for phase correction, baseline computation, smoothing, linear least-squares fitting were also implemented.

All the functions that save figures, i.e. almost the whole *figures* package, have now two additional parameters: *ext*, which allows to save the figure in different formats than PNG, and *dpi*, to specify the desired resolution.

The user guide at the beginning of the documentation pdf file was updated to instruct the reader about the recent modifications of the package.

### General changes in the Spectra classes:
The attribute *datadir* has been split in two parts: *datadir*, which only contains the path to the directory that contains the file, and *filename*, which is actually the name of the file, without any extension. They are generated by functions in *os.path*, which should make the reading of the spectra cross-platform. 
The trimming of the extension is achieved by right-splitting at the last "." character of the filename, and retaining the left part.

Right after having read the FID and/or the processed data, the program looks for a file named *filename.procs* in the same directory of the input file, which contains the *procs* dictionary in a format that the **eval** function is able to interpret as a dictionary. If there is, this file is transformed in a dictionary and stored in the *procs* attribute; otherwise, the *procs* dictionary is initialized with default values and then saved in a file named *filename.procs* in the same directory of the input file.

The *processing* function, after the Fourier transform, applies the phase correction and calibrates the spectrum according to the values that are currently stored in the *procs* dictionary. With respect to the previous point, this means that you can store your phase angles to remember the processing.

All functions that change a processing parameter, such as *adjph* or the baseline computation, call for the method *write_procs*, which update the file *filename.procs* with the new values stored in the *procs* dictionary.
In order to comply with this new aspect, the methods *adjph* and *cal* have now a new parameter "update" which allows to disable the update of the *procs* dictionary with the new values. Its default value is "True", hence it behaves as it did previously unless it is explicitely declared.

More detailed comments added in the source code.

## Modified functions:
- *fit.fit_int*: now it computes both intensity and offset, with a slightly modified formula which takes into account also the less relevant term.
- *figures.ax2D*: corrected AssertionError when xscale and yscale have the same dimension
- *figures.figure2D*: corrected AssertionError when xscale and yscale have the same dimension
- *figures.figure2D_multi*: employs better colormaps. You can superimpose a maximum of 6 spectra (positive only) or 3 spectra (both positive and negative).
- *fit.get_region*: automatic zoom can be enabled/disabled by pressing "z" on the keyboard
- *fit.histogram*: now it calls for *fit.ax_histogram* to compute the histogram.
- *spectra.Pseudo_2D.plot_md*: default value of the "which" changed from "all" to None
- *spectra.Pseudo_2D.plot_stacked*: default value of the "which" changed from "all" to None
- *spectra.Spectrum_1D.__init__*: now the parameter "in\_file" can also be an acqus dictionary, in this case the function *sim.load_sim_1D* is not called
- *spectra.Spectrum_1D.__init__*: added new parameter "spect", that allows to read also Varian and Magritek datasets.
- *spectra.Spectrum_1D.basl* now subtracts *self.baseline* from *self.S*, then unpacks it into *self.r* and *self.i*. Its previous behaviour is copied in the new function *Spectrum_1D.baseline_correction*
- *spectra.Spectrum_1D.process* now reverses automatically non-Bruker data after FT for correct display of data
- *spectra.Spectrum_2D.eae*: the behaviour of the function is now related to the value of "self.eaeflag"
- *spectra.Spectrum_nD.save_ser* changed name into *.write_ser* for consistency with other functions. Also the definition of the destination pathway has changed.

## Added functions:
- *misc.makeacqus_1D_varian*
- *misc.makeacqus_1D_spinsolve*
- *processing.acme*
- *processing.whittaker_smoother*
- *processing.RPBC*
- *processing.align*
- *fit.LR*
- *fit.LSP*
- *fit.polyn_basl*
- *fit.SINC_phase*
- *spectra.Spectrum_nD.write_procs*
- *spectra.Spectrum_nD.read_procs*
- *spectra.Spectrum_1D.acme*
- *spectra.Spectrum_1D.baseline_correction*
- *spectra.Spectrum_1D.rpbc*

## Added classes:
- *fit.CostFunc*
- *fit.SINC_ObjFunc*

# Version 0.2a.0

## Update notes:
Serious bug found! The functions *processing.ft* and *processing.ift* did not return the correct frequency. They have been corrected by converting them to "normal" FT and iFT, but reversing the data manually.

## Modified functions:
- *processing.ft*: so to use *np.fft.fftshift(np.fft.fft(data))*
- *processing.ift*: so to use *np.fft.ifft(np.fft.ifft(data))*

# Version 0.1a.2

## Update notes:
The docstrings of the functions, as well as the comments and the Latex documentation, were corrected, expanded, and uniformed to a common standard.
This should make them more readable and useful.

## Added functions:
- *misc.in2px*
- *misc.px2in*
- *misc.trim_data_2D*
- *processing.interactive_echo_param*
- *fit.peak_pick*
- *fit.gen_iguess_2D*
- *fit.build_2D_sgn*
- *fit.voigt_fit_2D*
- *fit.Voigt_Fit_2D* (class)

## Modified functions:
- *processing.split_echo_train*: made more python-friendly and not limited to 1D and 2D data. It now treats the first decay separately from the true echoes.
- *processing.sum_echo_train*: now calls for *processing.split_echo_train* and sum on the first dimension
- *processing.ft*: removed the "Numpy" parameter.
- *processing.ift*: removed the "Numpy" parameter.

## Added features:
- replaced the "print" statement with "warnings.warn" in *processing.ft* and *processing.ift*
- decorator function *cron* added in *config* and imported by *__init__*

## Modified features:
- *Spectra*, all classes: the attributes *BYTORDA*, *DTYPA* and *grpdly* were removed. Three keys, *BYTORDA*, *DTYPA* and *GRPDLY*, were added to the attribute *acqus*
- *Spectra*, all classes: the method *write_ser* became a *@staticmethod*
- *Spectra.Pseudo_2D*: added method *write_integrals*

# Version 0.1a.1

First release of **KLASSEZ**.
