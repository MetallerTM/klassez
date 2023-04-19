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
