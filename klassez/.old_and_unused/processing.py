#! /usr/bin/env python3

import os
import nmrglue as ng
from datetime import datetime
import numpy as np
from numpy import linalg

from . import misc, processing, fit


def blp_ng(data, pred=1, order=8, N=2048):
    """
    Performs backwards linear prediction on data.
    This function calls ``nmrglue.process.proc_lp.lp`` with most of the parameters set automatically.
    The algorithm predicts ``pred`` points of the FID using ``order`` coefficient for the linear interpolation.
    Only the first ``N`` points of the FID are used in the LP equation, because the computational cost scales with :math:`N^2`,
    making the use of more than 8k points not effective: using more points brings negligible contiribution to the final result.

    For Oxford spectra, set ``pred`` to half the value written in "TDoff".

    .. note::
        Legacy function. Use :func:`klassez.processing.blp` instead

    Parameters
    ----------
    data : ndarray
        Data on which to perform the linear prediction. For 2D data, it is performed row-by-row
    pred : int
        Number of points to be predicted
    order : int
        Number of coefficients to be used for the prediction
    N : int
        Number of points of the FID to be used in the calculation

    Returns
    ----------
    datap : ndarray
        Data with the predicted points appended at the beginning

    .. seealso::

        :func:`klassez.processing.blp`

        :func:`klassez.processing.lp`
    """
    # Compute a slice to trim the data to decrease the computation time
    if data.shape[-1] >= N:     # Slice from 0 to N
        data_sl = slice(0, N)
    else:                       # Leave unchanged
        data_sl = None
    datap = ng.process.proc_lp.lp(
            data,
            pred=pred,          # Number of points to predict
            order=order,        # Number of coefficients to use
            slice=data_sl,      # Slicing
            mode='b',           # Backwards
            append='before',    # Append points before
            bad_roots='decr',   # Default option for mode='b'
            method='svd')       # Choice of method basically uninfluent
    return datap


def new_MCR_ALS(D, C, S, itermax=10000, tol=1e-5, reg_f=None, reg_fargs=[]):
    """
    Modified function to do ALS

    .. note::

        Work in progress!!! Does not work right now
    """

    itermax = int(itermax)

    start_time = datetime.now()
    print('\n-----------------------------------------------------\n')
    print('             MCR optimization running...             \n')

    convergence_flag = 0
    print('#   \tC convergence\tS convergence')
    reg_fargs.append(None)
    for kk in range(itermax):
        # Copy from previous cycle
        C0 = np.copy(C)
        S0 = np.copy(S)

        # Compute new C, S and E
        C = D @ linalg.pinv(S)

        # Regularization
        if reg_f is None:
            pass
        else:
            C, S, prev_param = reg_f(C, S, *reg_fargs, cycle=kk)
            reg_fargs[-1] = prev_param

        S = linalg.pinv(C) @ D

        # Compute the Frobenius norm of the difference matrices
        # between two subsequent cycles
        rC = linalg.norm(C - C0)
        rS = linalg.norm(S - S0)

        # Ongoing print of the residues
        print(str(kk+1)+' \t{:.5e}'.format(rC) + '\t'+'{:.5e}'.format(rS), end='\r')

        # Arrest criterion
        if (rC < tol) and (rS < tol):
            end_time = datetime.now()
            print('\n\n\tMCR converges in '+str(kk+1)+' steps.')
            convergence_flag = 1    # Set to 1 if the arrest criterion is reached
            break

    if not convergence_flag:
        print('\n\n\tMCR does not converge.')
    end_time = datetime.now()
    print('\tTotal runtime: {}'.format(end_time - start_time))

    return C, S


def new_MCR(input_data, nc, f=10, tol=1e-5, itermax=1e4, H=True, oncols=True, our_function=None, fargs=[], our_function2=None, f2args=[]):
    """
    This is an implementation of Multivariate Curve Resolution
    for the denoising of 2D NMR data. It requires:

    * input_data: a tensor containing the set of 2D NMR datasets to be coprocessed
      stacked along the first dimension;
    * nc      : number of purest components;
    * f       : percentage of allowed noise;
    * tol     : tolerance for the arrest criterion;
    * itermax : maximum number of allowed iterations, default 10000
    * H       : True for horizontal stacking of data (default), False for vertical;
    * oncols  : True to estimate S with purest components, False to estimate C

    This function returns the denoised datasets, 'CS', and the 'C' and 'S' matrices.

    .. note::
        Work in progress!!! Does not work right now.
    """

    # Get number of datasets (nds) from the shape of the input tensor
    if isinstance(input_data, list):
        nds = len(input_data)
    else:
        if len(input_data.shape) == 3:
            nds = input_data.shape[0]
        elif len(input_data.shape) == 2:
            nds = 1
            input_data = np.reshape(input_data, (1, input_data.shape[0], input_data.shape[1]))
        else:
            print('Input data is not a matrix!')
            exit()

    print('\n*****************************************************')
    print('*                                                   *')
    print('*           Multivariate Curve Resolution           *')
    print('*                                                   *')
    print('*****************************************************\n')

    D = processing.mcr_stack(input_data, H=H)           # Matrix augmentation

    # Get initial estimation of C, S and E
    if our_function is None:
        C0, S0 = processing.simplisma(D, nc, f, oncols=oncols)
    else:
        C0, S0, nc = our_function(D, *fargs)

    # Optimize C and S matrix through Alternating Least Squares
    if our_function2 is None:
        C, S = processing.mcr_als(D, C0, S0, itermax=itermax, tol=tol)
    else:
        C, S = processing.new_MCR_ALS(D, C0, S0, itermax, tol, our_function2, f2args)

    # Revert matrix augmentation
    C_f, S_f = processing.mcr_unpack(C, S, nds, H)

    # Obtain the denoised data of the same shape as the input
    if isinstance(input_data, list):
        CS_f = []
        for j in range(nds):
            CS_f.append(C_f[j] @ S_f[j])
    else:
        CS_f = np.zeros_like(input_data).astype(input_data.dtype)
        for j in range(nds):
            CS_f[j] = C_f[j] @ S_f[j]

    # Reshape if no matrix augmentation is performed
    if nds == 1:
        CS_f = CS_f[0]
        C_f = C_f[0]
        S_f = S_f[0]

    print('\n*****************************************************\n')

    return CS_f, C_f, S_f


def write_basl_info(f, limits, mode, data):
    """
    Writes the baseline parameters of a certain window in a file.

    Parameters
    ----------
    f : TextIO object
        File where to write the parameters
    limits : tuple
        Limits of the spectral window. (left, right)
    mode : str
        Baseline correction mode: ``'polynomion'`` or ``'spline'``
    data : float or 1darray
        It can be either the spline smoothing factor or the polynomion coefficients
    """
    f.write('***{:^54}***\n'.format('WINDOW LIMITS /PPM'))
    f.write('{: 8.3f}\t{: 8.3f}\n'.format(limits[0], limits[1]))
    f.write('***{:^54}***\n'.format('BASELINE CORRECTION MODE'))
    f.write('{}\n'.format(mode))
    f.write('***{:^54}***\n'.format('POLYNOMION COEFFICIENTS'))
    if mode == 'polynomion':
        N = len(data)
        for k, c in enumerate(data):
            if k < N - 1:
                f.write('{: 5.2e}\t'.format(c))
            else:
                f.write('{: 5.2e}\n'.format(c))
                break
    else:
        N = 5
        for k, c in enumerate(np.zeros(5)):
            if k < N - 1:
                f.write('{: 5.2e}\t'.format(c))
            else:
                f.write('{: 5.2e}\n'.format(c))
                break
    f.write('***{:^54}***\n'.format('SPLINE SMOOTHING FACTOR'))
    if mode == 'spline':
        f.write('{:5.3e}\n'.format(data))
    else:
        f.write('{:5.3e}\n'.format(0))
    f.write('***{:^54}***\n'.format('-'*50))


def baseline_correction(ppm, data, basl_file='spectrum.basl', winlim=None):
    """
    Interactively corrects the baseline of a given spectrum and saves the parameters in a file.
    The program starts with an interface to partition the spectrum in windows to correct separately.
    Then, for each window, an interactive panel opens to allow the user to compute the baseline.

    Parameters
    ----------
    ppm : 1darray
        PPM scale of the spectrum
    data : 1darray
        The spectrum of which to adjust the baseline
    basl_file : str
        Name for the baseline parameters file
    winlim : list or str or None
        List of the breakpoints for the window. If it is ``str``, indicates the location of a file to be read with ``np.loadtxt``. If it is None, the partitioning is done interactively.
    """

    # Check if winlim is passed as list
    if isinstance(winlim, list):
        coord = winlim
    elif isinstance(winlim, str):
        # It means it is a file. Try to read it
        if os.path.exists(winlim):
            coord = list(np.loadtxt(winlim))
        else:
            raise NameError('File {} not found.'.format(winlim))
    else:
        # Interactive partitioning
        coord = processing.interactive_basl_windows(ppm, data)

    # Clear the file
    if os.path.exists(basl_file):
        os.remove(basl_file)

    # Open the file
    F = open(basl_file, 'a')
    for i, _ in enumerate(coord):
        if i == len(coord) - 1:
            break       # Stop before it raises error
        limits = coord[i], coord[i+1]
        mode, C_f = processing.make_polynomion_baseline(ppm, data, limits)        # Interactive polynomion
        if isinstance(C_f, str):    # If you press "use spline" in the polynomion interactive figure
            # Get the limits
            lim1 = misc.ppmfind(ppm, limits[0])[0]
            lim2 = misc.ppmfind(ppm, limits[1])[0]
            lim1, lim2 = min(lim1, lim2), max(lim1, lim2)
            # trim ppm and data
            xdata, ydata = ppm[lim1:lim2], data[lim1:lim2]
            # Calculate the spline
            _, C_f = fit.interactive_smoothing(xdata, ydata)
        # Write the section in the file
        processing.write_basl_info(F, limits, mode, C_f)
    F.close()


def load_baseline(filename, ppm, data):
    """
    Read the baseline parameters from a file and builds the baseline itself.

    Parameters
    ----------
    filename : str
        Location of the baseline file
    ppm : 1darray
        PPM scale of the spectrum
    data : 1darray
        Spectrum of which to correct the baseline

    Returns
    ----------
    baseline : 1darray
        Computed baseline
    """

    # Opens the file
    f = open(filename, 'r')
    r = f.readlines()

    # Initialize the lists of the variables
    limits = []     # Window limits
    mode = []       # Baseline correction mode
    C = []          # Polynomion coefficients
    S = []          # Spline smoothing factor

    tmpmode = None      # Correction mode for the active section
    for k, line in enumerate(r):
        # Read the limits
        if 'WINDOW LIMITS /PPM' in line:
            Q = r[k+1]
            Q = Q.replace('\t', ', ')
            limits.append(eval(Q))
            continue
        # Read mode
        if 'BASELINE CORRECTION MODE' in line:
            tmpmode = r[k+1].strip()
            mode.append(tmpmode)
            continue
        # Read the polynomion coefficients
        if 'POLYNOMION COEFFICIENTS' in line:
            if tmpmode == 'polynomion':
                Q = r[k+1]
                Q = Q.replace('\t', ',')
                C.append(np.array(eval('['+Q+']')))
            else:
                C.append(np.zeros(5))
            continue
        # Read the spline smoothing factor
        if 'SPLINE SMOOTHING FACTOR' in line:
            if tmpmode == 'spline':
                Q = r[k+1]
                S.append(eval(Q))
            else:
                S.append(0)
            continue
        # Reset tmpmode
        if '-----' in line:
            tmpmode = None
            continue

    # Now, make the baseline

    # Initialize flat baseline
    baseline = np.zeros_like(ppm)
    n_w = len(limits)   # Number of windows

    for k in range(n_w):
        # Translate the limits in points
        lim1 = misc.ppmfind(ppm, limits[k][0])[0]
        lim2 = misc.ppmfind(ppm, limits[k][1])[0]
        lim1, lim2 = min(lim1, lim2), max(lim1, lim2)

        if mode[k] == 'polynomion':  # Compute polynomion in the active region
            x = np.linspace(0, 1, ppm[lim1:lim2].shape[-1])[::-1]
            tmpbasl = misc.polyn(x, C[k])
        elif mode[k] == 'spline':    # Fit the spectrum in the active region with a spline
            y = data[lim1:lim2]
            tmpbasl = fit.smooth_spl(y, S[k])
        # Put the just computed baseline in the corresponding region
        baseline[lim1:lim2] = tmpbasl

    return baseline
