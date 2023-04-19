#! /usr/bin/env python3

import os
import sys
import numpy as np
from scipy import linalg, stats
from scipy.spatial import ConvexHull
import random
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.widgets import Slider, Button, RadioButtons, TextBox, CheckButtons, Cursor, LassoSelector
from matplotlib.path import Path
import seaborn as sns
import nmrglue as ng
import lmfit as l
from datetime import datetime
import warnings

from . import fit, misc, sim, figures, processing
from .config import CM, COLORS, cron
#from .__init__ import CM

  
gamma = {   # gyromagnetic ratio of all NMR active nuclei in MHz/T
	'1H'	:	42.57748,
	'2H'	:	6.53564,
	'3H'	:	45.41461,
	'3He'	:	32.43467,
	'6Li'	:	6.2657,
	'7Li'	:	16.54646,
	'9Be'	:	-5.98384,
	'10B'	:	4.57538,
	'11B'	:	13.66056,
	'13C'	:	10.70611,
	'14N'	:	3.0758,
	'15N'	:	-4.31438,
	'17O'	:	-5.77223,
	'19F'	:	40.02581,
	'21Ne'	:	3.36107,
	'23Na'	:	11.26259,
	'25Mg'	:	2.60532,
	'27Al'	:	11.09441,
	'29Si'	:	-8.45802,
	'31P'	:	17.23579,
	'33S'	:	3.26527,
	'35Cl'	:	4.17174,
	'37Cl'	:	3.47219,
	'39K'	:	1.98667,
	'41K'	:	1.09041,
	'43Ca'	:	-2.86461,
	'45Sc'	:	10.34335,
	'47Ti'	:	-2.40009,
	'49Ti'	:	-2.40052,
	'50V'	:	4.24497,
	'51V'	:	11.19277,
	'53Cr'	:	2.40648,
	'55Mn'	:	10.45975,
	'57Fe'	:	1.37568,
	'59Co'	:	-10.05425,
	'61Ni'	:	-3.80472,
	'63Cu'	:	11.28559,
	'65Cu'	:	12.08945,
	'67Zn'	:	2.66322,
	'69Ga'	:	10.21987,
	'71Ga'	:	12.98443,
	'73Ge'	:	-1.4851,
	'75As'	:	7.29224,
	'77Se'	:	8.11825,
	'79Br'	:	8.53849,
	'81Br'	:	11.49847,
	'83Kr'	:	-1.63838,
	'85Rb'	:	4.11086,
	'87Rb'	:	4.11086,
	'87Sr'	:	13.93135,
	'89Y'	:	-2.08587,
	'91Zr'	:	-3.9729,
	'93Nb'	:	10.40764,
	'95Mo'	:	2.77392,
	'97Mo'	:	-2.83225,
	'99Ru'	:	-1.96452,
	'99Tc'	:	9.58334,
	'101Ru'	:	-2.20211,
	'103Rh'	:	1.33991,
	'105Pd'	:	-1.94835,
	'107Ag'	:	-1.72311,
	'109Ag'	:	-1.9807,
	'111Cd'	:	-9.02855,
	'113Cd'	:	-9.44496,
	'113In'	:	9.31042,
	'115In'	:	9.33043,
	'117Sn'	:	-15.16865,
	'119Sn'	:	-15.86948,
	'121Sb'	:	10.18922,
	'123Sb'	:	5.51762,
	'123Te'	:	-11.15871,
	'125Te'	:	-13.45065,
	'127I'	:	8.51848,
	'129Xe'	:	11.77736,
	'131Xe'	:	3.49093,
	'133Cs'	:	5.58489,
	'135Ba'	:	4.22965,
	'137Ba'	:	4.73164,
	'138La'	:	5.61725,
	'139La'	:	6.01449,
	'141Pr'	:	12.50032,
	'143Nd'	:	-2.31494,
	'145Nd'	:	-1.42209,
	'147Sm'	:	-1.75845,
	'149Sm'	:	-1.39995,
	'151Eu'	:	10.55921,
	'153Eu'	:	4.66266,
	'155Gd'	:	-1.61582,
	'157Gd'	:	-1.99986,
	'159Tb'	:	9.6604,
	'161Dy'	:	-1.4025,
	'163Dy'	:	1.99859,
	'165Ho'	:	8.73009,
	'167Er'	:	-1.23049,
	'169Tm'	:	-3.51988,
	'171Yb'	:	7.49917,
	'173Yb'	:	-2.06586,
	'175Lu'	:	4.85681,
	'177Hf'	:	1.3199,
	'177Lu'	:	1.10276,
	'179Hf'	:	-0.60002,
	'181Ta'	:	5.0961,
	'183W'	:	1.77165,
	'185Re'	:	9.58589,
	'187Os'	:	0.73542,
	'187Re'	:	9.68382,
	'189Os'	:	3.30359,
	'191Ir'	:	0.54893,
	'193Ir'	:	0.59747,
	'195Pt'	:	9.15331,
	'197Au'	:	0.54701,
	'199Hg'	:	7.59029,
	'201Hg'	:	-2.81011,
	'203Tl'	:	24.33303,
	'205Tl'	:	24.57061,
	'207Pb'	:	8.90806,
	'209Bi'	:	6.8422,
	'235U'	:	0.5716,
	}

def calc_splitting(u0, I0, m=1, J=0):
    """ 
    Calculate the frequency and the intensities of a NMR signal splitted by scalar coupling.
    -------
    Parameters:
    - u0: float
        Frequency of the non-splitted signal (Hz)
    - I0: float
        Total intensity of the non-splitted signal.
    - m: int
        Multiplicity, i.e. number of expected signals after the splitting
    - J: float
        Scalar coupling constant (Hz)
    ------
    Returns:
    - u_s: 1darray
        Frequencies of the splitted signal (Hz)
    - I_s: Intensities of the splitted signal 
    """
    # FREQUENCIES
    u_s = []
    # if m=2    =>  J_pattern = -0.5 +0.5
    # if m=3    =>  J_pattern = -1  0  1 
    J_pattern = np.arange(m) - np.mean(np.arange(m))
    for k in J_pattern:
        u_s.append(u0 + k*J)
    u_s = np.array(u_s)

    # INTENSITIES
    base_int = misc.binomial_triangle(m) / (2**(m-1))
    I_s = base_int * I0

    return u_s, I_s


def multiplet(u, I, m='s', J=[]):
    """ 
    Split a given signal according to a scalar coupling pattern.
    -------
    Parameters:
    - u: float
        Frequency of the non-splitted signal (Hz)
    - I: float
        Intensity of the non-splitted signal 
    - m: str
        Organic chemistry-like multiplet, i.e. s, d, dqt, etc.
    - J: float or list
        Scalar coupling constants. The number of constants should match the number of coupling branches
    --------
    Returns:
    - u_in: list
        List of the splitted frequencies (Hz)
    - I_in: list
        Intensities of the splitted signal
    """
    n_splitting = len(m)    # Number of splittings

    # Adjust the variables to make them fit in the loop
    if m=='s':      # Singlet: J is useless
        J_in = [0]
    elif isinstance(J, (list, tuple, np.ndarray)):
        J_in = J
    else:
        J_in = [J]

    u_in = [u]
    I_in = [I]
    for n in range(n_splitting):        # Loop in the number of splitting
        u_ret, I_ret = [], []           # Declare empty lists
        for k, _ in enumerate(u_in):    # u_in expands according to the splitting tree
            if m[n] == 'd':     # doublet
                mult = 2
            elif m[n] == 't':   # triplet
                mult = 3
            elif m[n] == 'q':   # quartet
                mult = 4
            else:               # anything else is useless
                mult = 1

            # Compute the splitting
            u_s, I_s = sim.calc_splitting(u_in[k], I_in[k], mult, J_in[n])

            for w, v in zip(u_s, I_s):  # Fill the _ret lists with the splitted signals
                u_ret.append(w)
                I_ret.append(v)

        # Replace the input variables with the splitted ones
        u_in = u_ret
        I_in = I_ret

    return u_in, I_in


def load_sim_1D(File):
    """
    Creates a dictionary from the spectral parameters listed in the input file.
    -------
    Parameters:
    - File: str
        Path to the input file location
    -------
    Returns:
    - dic: dict
        Dictionary of the parameters, ready to be read from the simulation functions.
    """
    
    inp = open(File, 'r').readlines()
    keys = []
    vals = []
    for i in range(len(inp)):
        if inp[i] == '\n' or inp[i][0] == '#':
            continue    # skip empty lines or comments
        line = inp[i].split('\t', 1)    # separate key from the rest
        if ' ' in line[0]:
            line = line[0].split(' ', 1)
        keys.append(line[0])

        rest = line[1].strip()
        rest = rest.split('\t')
        rest = rest[0].split('#')
        try:
            value = eval(rest[0])
        except:
            value = (f'{rest[0]}')
        vals.append(value)


    dic = {}
    for i, key in enumerate(keys):
        if 'nuc' in key:    # Remove unwanted spaces
            vals[i] = vals[i].replace(' ', '')
        dic[key] = vals[i]
    if 'phases' not in keys:
        dic['phases'] = tuple([0 for w in dic['shifts']])
    else:
        dic['phases'] = tuple([w * np.pi / 180 for w in dic['phases']])
    if 'mult' not in keys:      # Multiplicity
        dic['mult'] = tuple(['s' for w in dic['shifts']])
    else:
        dic['mult'] = tuple(dic['mult'].strip(',').replace(' ', '').split(','))
    if 'Jconst' not in keys:    # Coupling constants
        dic['Jconst'] = tuple([0 for w in dic['shifts']])

    dic['TD'] = int(dic['TD'])
    dic['SFO1'] = dic['B0'] * sim.gamma[dic['nuc']]
    dic['SW'] = dic['SWp'] * np.abs(dic['SFO1'])
    dic['dw'] = 1/dic['SW']
    dic['t1'] = np.linspace(0, dic['TD']*dic['dw'], dic['TD'])
    dic['AQ'] = dic['t1'][-1]
    dic['o1'] = dic['o1p'] * dic['SFO1']

    return dic

def sim_1D(File, pv=False):
    """
    Simulates a 1D NMR spectrum from the instructions written in File.
    -------
    Parameters:
    - File: str
        Path to the input file location
    - pv: bool
        True for pseudo-Voigt model, False for Voigt model.
    -------
    Returns:
    - fid: 1darray
        FID of the simulated spectrum.
    """
    if isinstance(File, str):
        in_file = load_sim_1D(File)
    elif isinstance(File, dict):
        in_file = File
    else:
        raise ValueError('Unknown file type, aborting...')
    
    TD = in_file['TD']     # Points of the FID

    shifts = np.array(in_file['shifts'])    # Chemical shift /ppm
    amplitudes = in_file['amplitudes']      # Relative intensity of the signals
    fwhm = np.array(in_file['fwhm'])        # Full width at half maximum of the signals
    x_g = in_file['x_g']                    # Fraction of gaussianity of the FID
    phases = in_file['phases']

    freq = misc.ppm2freq(shifts, B0=in_file['SFO1'], o1p=in_file['o1p'])     # peaks center frequency

    fid = np.zeros(TD, dtype='complex64')      # empty FID
    for j, _ in enumerate(freq):
        # Account for multiplicity
        u_split, A_split = multiplet(freq[j], amplitudes[j], m=in_file['mult'][j], J=in_file['Jconst'][j])
        for u, I in zip(u_split, A_split):
            sgn_par = dict(t=in_file['t1'], u=u, fwhm=2*np.pi*fwhm[j], x_g=x_g[j], A=I, phi=phases[j] )
            if pv:          # Generate pseudo-voigt signals
                fid += sim.t_pvoigt(**sgn_par)
            else:           # Make Voigt signals
                fid += sim.t_voigt(**sgn_par)
    return fid

def load_sim_2D(File, states=True):
    """
    Creates a dictionary from the spectral parameters listed in the input file.
    -------
    Parameters:
    - File: str
        Path to the input file location
    - states: bool
        If FnMODE is States or States-TPPI, set it to True to get the correct timescale.
    -------
    Returns:
    - dic: dict
        Dictionary of the parameters, ready to be read from the simulation functions.
    """
    inp = open(File, 'r').readlines()
    keys = []
    vals = []
    for i in range(len(inp)):
        if inp[i] == '\n' or inp[i][0] == '#':
            continue
        line = inp[i].split('\t', 1)
        if ' ' in line[0]:
            line = line[0].split(' ', 1)
        keys.append(line[0])

        rest = line[1].strip()
        rest = rest.split('\t')
        rest = rest[0].split('#')
        try:
            value = eval(rest[0])
        except:
            value = str(rest[0])
        vals.append(value)

    dic = {}
    for i, key in enumerate(keys):
        if 'nuc' in key:    # Remove unwanted spaces
            vals[i] = vals[i].replace(' ', '')
        dic[key] = vals[i]

    for key, value in dic.items():
        if 'TD' in key:
            dic[key] = int(value)
    dic['SFO1'] = dic['B0'] * sim.gamma[dic['nuc1']]       # Larmor frequency /MHz
    dic['SFO2'] = dic['B0'] * sim.gamma[dic['nuc2']]       # Larmor frequency /MHz
    dic['SW1'] = np.abs(dic['SW1p'] * dic['SFO1'])       # spectral width
    dic['SW2'] = np.abs(dic['SW2p'] * dic['SFO2'])       # spectral width
    dic['dw1'] = np.abs(1 / dic['SW1'])             # dwell time
    dic['dw2'] = np.abs(1 / dic['SW2'])             # dwell time
    dic['o1'] = dic['o1p'] * dic['SFO1']
    dic['o2'] = dic['o2p'] * dic['SFO2']
    if states: 
        dic['t1'] = np.linspace(0, dic['TD1']//2 * dic['dw1'], dic['TD1'])      # acquisition time scale
    else:
        dic['t1'] = np.linspace(0, dic['TD1'] * dic['dw1'], dic['TD1'])      # acquisition time scale
    dic['t2'] = np.linspace(0, dic['TD2'] * dic['dw2'], dic['TD2'])      # acquisition time scale
    dic['AQ1'] = dic['t1'][-1]
    dic['AQ2'] = dic['t2'][-1]

    return dic

def sim_2D(File, states=True, alt=True, pv=False):
    """
    Simulates a 2D NMR spectrum from the instructions written in File.
    The indirect dimension is sampled with states-TPPI as default.
    --------
    Parameters:
    - File: str
        Path to the input file location
    - states: bool
        Set it to True to allow for correct spectral arrangement in the indirect dimension.
    - alt: bool
        Set it to True to allow for correct spectral arrangement in the indirect dimension.
    - pv: bool
        True for pseudo-Voigt model, False for Voigt model.
    --------
    Returns:
    - fid: 2darray
        FID of the simulated spectrum.
    """
    
    # Generates a dictionary of parameters from an input file
    if isinstance(File, str):
        in_file = sim.load_sim_2D(File, states=states)
    elif isinstance(File, dict):
        in_file = File
    else:
        raise ValueError('Unknown file type, aborting...')
        

    # recall of timescales from in_file
    t1 = in_file['t1']
    t2 = in_file['t2']

    # recall of peaks parameters from in_file
    #   reshape is needed to allow for correct indexing
    shifts_f1 = np.array(in_file['shifts_f1']).reshape(-1)      # chemical shift in F1
    shifts_f2 = np.array(in_file['shifts_f2']).reshape(-1)      # chemical shift in F2
    fwhm_f1 = np.array(in_file['fwhm_f1']).reshape(-1)          # FWHM of peaks in F1
    fwhm_f2 = np.array(in_file['fwhm_f2']).reshape(-1)          # FWHM of peaks in F2
    amplitudes = np.array(in_file['amplitudes']).reshape(-1)    # relative intensity
    x_g = np.array(in_file['x_g']).reshape(-1)                  # fraction of gaussianity

    # conversion of FWHM from Hz to radians
    fwhm1 = 2 * np.pi * fwhm_f1
    fwhm2 = 2 * np.pi * fwhm_f2

    # calculation of stdev for gaussian peaks
    sigma1 = fwhm1 / 2.355
    sigma2 = fwhm2 / 2.355

    # conversion of chemical shift from ppm to rad/s
    freq1 = misc.ppm2freq(shifts_f1, B0=in_file['SFO1'], o1p=in_file['o1p'])     # peaks center frequency
    freq2 = misc.ppm2freq(shifts_f2, B0=in_file['SFO2'], o1p=in_file['o2p'])     # peaks center frequency

    # creation of empty FID
    fid = np.zeros((in_file['TD1'], in_file['TD2']), dtype='complex64')      # empty FID

    # The number of NMR signals is retrieved from the length of the amplitudes array. 
    # If there is only one peak, 'ns' (number of signals) is set to 1.
    try:
        ns = len(amplitudes)
    except:
        ns = 1
    # Creates a pseudo-voigt signal looping on the number of peaks

    for p in range(ns):
        if pv:              # Generate pseudo-Voigt signal
            fid += sim.t_2Dpvoigt(t1, t2, freq1[p], freq2[p], fwhm1[p], fwhm2[p], A=amplitudes[p], x_g=x_g[p], states=states, alt=alt)
        else:               # Generate Voigt signal
            fid += sim.t_2Dvoigt(t1, t2, freq1[p], freq2[p], fwhm1[p], fwhm2[p], A=amplitudes[p], x_g=x_g[p], states=states, alt=alt)
    return fid

def noisegen(size, o2, t2, s_n=1):
    """
    Simulates additive noise in the time domain.
    --------
    Parameters:
    - size: int or tuple
        Dimension of the noise matrix
    - o2: float
        Carrier frequency, in Hz.
    - t2: 1darray
        Time scale of the last temporal dimension.
    - s_n: float
        Standard deviation of the noise.
    --------
    Returns:
    - noise: 2darray
        Noise matrix, of dimensions size.
    """

    # correlated part of noise until ADC
    white_corr = np.random.normal(0, s_n, size)
    # white noise in FID has to be centered on the offset frequency
    noise_corr = white_corr * np.exp(1j* 2 * np.pi * o2 * t2)

    # uncorrelated part of noise: quadrature detection
    white_re = np.random.normal(0, s_n, size)
    white_im = np.random.normal(0, s_n, size)
    # cosine-modulated in the real channel and sine-modulated in the imaginary channel
    noise_re = white_re * np.cos( 2* np.pi * o2 * t2)
    noise_im = white_im * np.sin( 2* np.pi * o2 * t2)

    # final noise is sum of the two parts
    noise = noise_corr + (noise_re + 1j*noise_im)
    return noise


def mult_noise(data_size, mean, s_n):
    """ Multiplicative noise model. """
    N = data_size[0]

    white = np.random.lognormal(mean, s_n, N)
    #white = np.random.normal(0, s_n, N)

    #noisemat = np.diag(1 - 0.25 * white)
    noisemat = np.diag(white)
    return noisemat



def water7(N, t2, vW, fwhm=300, A=1, spread=701.125):
    """
    Simulates a feature like the water ridge in HSQC spectra, in the time domain.
    --------
    Parameters:
    - N: int
        Number of transients
    - t2: 1darray
        Time scale of the last temporal dimension.
    - vW: float
        Nominal peak position, in Hz.
    - fwhm: float
        Nominal full-width at half maximum of the peak, in rad/s.
    - A: float
        Signal intensity.
    - spread: float
        Standard deviation of the peak position distribution, in Hz.
    --------
    Returns:
    - ridge: 2darray
        Matrix of the ridge.
    """


    uW = np.random.normal(vW, spread, N)
    s = fwhm / 2.355        # conversion from fwhm to sigma
    ridge = np.zeros((N, len(t2)), dtype='complex64')
    for i in range(N):
        # each transient features a gaussian signal with the parameters specified above
        # but it is on phase in the even transients and 90 degree dephased in the odd ones
        ridge[i] = sim.t_gaussian(t2, uW[i], s, A=A, phi=np.pi/2*np.mod(i,2))
    return ridge 


def f_gaussian(x, u, s, A=1):
    """
    Gaussian function in the frequency domain:
    --------
    Parameters:
    - x: 1darray
        Independent variable
    - u: float
        Peak position
    - s: float
        Standard deviation
    - A: float
        Intensity
    --------
    Returns:
    - f: 1darray
        Gaussian function.
    """
    if s > 0:
        f = A/(np.sqrt(2 * np.pi)*s) * np.exp(-1/2*((x-u)/s)**2)
    else:
        f = np.zeros_like(x)
    return f

def f_lorentzian(x, u, fwhm, A=1):
    """
    Lorentzian function in the time domain:
    --------
    Parameters:
    - x: 1darray
        Independent variable
    - u: float
        Peak position
    - fwhm: float
        Full-width at half-maximum
    - A: float
        Intensity
    --------
    Returns:
    - f: 1darray
        Lorentzian function.
    """

    hwhm = fwhm/2   # half width at half maximum
    if hwhm > 0:
        f = A/(np.pi) * hwhm/((x-u)**2 + hwhm**2 )
    else:
        f = np.zeros_like(x)
    return f

def f_pvoigt(x, u, fwhm, A=1, x_g=0):
    """
    Pseudo-Voigt function in the frequency domain:
    --------
    Parameters:
    - x: 1darray
        Independent variable
    - u: float
        Peak position
    - fwhm: float
        Full-width at half-maximum
    - A: float
        Intensity
    - x_g: float
        Fraction of gaussianity
    --------
    Returns:
    - S: 1darray
        Pseudo-Voigt function.
    """
    s = fwhm / 2.355
    S = A* (sim.f_gaussian(x, u, s, A=x_g) + sim.f_lorentzian(x, u, fwhm, A=1-x_g))
    return S

def t_gaussian(t, u, s, A=1, phi=0):
    """
    Gaussian function in the time domain.
    --------
    Parameters:
    - t: 1darray
        Independent variable
    - u: float
        Peak position, in Hz
    - s: float
        Standard deviation, in rad/s
    - A: float
        Intensity
    - phi: float
        Phase, in radians
    --------
    Returns:
    - S: 1darray
        Gaussian function.
    """
    s = np.abs(s) # Avoid problems with s<0 
    if s >= 0:
        S = A * np.exp(1j*phi) * np.exp((1j*2*np.pi*u*t) - (t**2)*(s**2)/2)
    return S

def t_lorentzian(t, u, fwhm, A=1, phi=0):
    """
    Lorentzian function in the time domain.
    --------
    Parameters:
    - t: 1darray
        Independent variable
    - u: float
        Peak position, in Hz
    - fwhm: float
        Full-width at half-maximum, in rad/s
    - A: float
        Intensity
    - phi: float
        Phase, in radians
    --------
    Returns:
    - S: 1darray
        Lorentzian function.
    """
    hwhm = np.abs(fwhm) / 2       
    S = A * np.exp(1j*phi) * np.exp((1j *2*np.pi *u * t)-(t*hwhm))
    return S

def t_pvoigt(t, u, fwhm, A=1, x_g=0, phi=0):
    """
    Pseudo-Voigt function in the time domain:
    --------
    Parameters:
    - t: 1darray
        Independent variable
    - u: float
        Peak position, in Hz
    - fwhm: float
        Full-width at half-maximum, in rad/s
    - A: float
        Intensity
    - x_g: float
        Fraction of gaussianity
    - phi: float
        Phase, in radians
    --------
    Returns:
    - S: 1darray
        Pseudo-Voigt function.
    """

    s = fwhm / 2.355
    S = A * (sim.t_gaussian(t, u, s, A=x_g, phi=phi) + sim.t_lorentzian(t, u, fwhm, A=1-x_g, phi=phi))
    return S

def t_voigt(t, u, fwhm, A=1, x_g=0, phi=0):
    """
    Voigt function in the time domain. The parameter x_g affects the linewidth of the lorentzian and gaussian contributions.
    --------
    Parameters:
    - t: 1darray
        Independent variable
    - u: float
        Peak position, in Hz
    - fwhm: float
        Full-width at half-maximum, in rad/s
    - A: float
        Intensity
    - x_g: float
        Fraction of gaussianity
    - phi: float
        Phase, in radians
    --------
    Returns:
    - S: 1darray
        Voigt function.
    """

    s = fwhm / 2.355
    S = A * np.exp(1j*phi) * sim.t_gaussian(t, u/2, s*x_g) * sim.t_lorentzian(t, u/2, fwhm*(1-x_g))
    return S


def t_2Dgaussian(t1, t2, v1, v2, s1, s2, A=1, states=True, alt=True):
    """
    Bidimensional gaussian function.
    --------
    Parameters:
    - t1: 1darray
        Indirect evolution timescale
    - t2: 1darray
        Timescale of the direct dimension
    - v1: float
        Peak position in the indirect dimension, in Hz
    - v2: float
        Peak position in the direct dimension, in Hz
    - s1: float
        Standard deviation in the indirect dimension, in rad/s
    - s2: float
        Standard deviation in the direct dimension, in rad/s
    - A: float
        Intensity
    - states: bool
        Set to True for "FnMODE":"States-TPPI
    - alt: bool
        Set to True for "FnMODE":"States-TPPI
    --------
    Returns:
    - S: 2darray
        Gaussian function.
    """
    if states:
        # States acquires twice the same point of the indirect dimension time domain
        t1[1::2] = t1[::2]
    if alt:
        # TPPI cycles the receiver phase of 90 degrees at each transient acquisition
        freq_1 = np.zeros(len(t1), dtype='complex64')
        for k in range(4):
            t1t = t1[k::4]
            freq_1[k::4] = np.cos( (2 * np.pi * v1 * t1t) - (0.5 * np.pi * np.mod(k,4) ))
    else:
        freq_1 = np.exp(1j * 2 * np.pi * v1 * t1)
    # NMR signal in the direct dimension
    F2 = np.exp(1j*2*np.pi*v2*t2) * np.exp(-(s2**2 * t2**2)/2)
    # NMR signal in the indirect dimension
    F1 = freq_1 * np.exp(-(s1**2 * t1**2)/2)
    # The full FID is reconstructed by doing the external product between the two vectors
    S = A * F1.reshape(-1,1) @ F2.reshape(1,-1)
    return S

def t_2Dlorentzian(t1, t2, v1, v2, fwhm1, fwhm2, A=1, states=True, alt=True):
    """
    Bidimensional lorentzian function.
    --------
    Parameters:
    - t1: 1darray
        Indirect evolution timescale
    - t2: 1darray
        Timescale of the direct dimension
    - v1: float
        Peak position in the indirect dimension, in Hz
    - v2: float
        Peak position in the direct dimension, in Hz
    - fwhm1: float
        Full-width at half maximum in the indirect dimension, in rad/s
    - fwhm2: float
        Full-width at half maximum in the direct dimension, in rad/s
    - A: float
        Intensity
    - states: bool
        Set to True for "FnMODE":"States-TPPI
    - alt: bool
        Set to True for "FnMODE":"States-TPPI
    --------
    Returns:
    - S: 2darray
        Lorentzian function.
    """
    hwhm1 = fwhm1 / 2
    hwhm2 = fwhm2 / 2
    if states:
        # States acquires twice the same point of the indirect dimension time domain
        t1[1::2] = t1[::2]
    if alt:
        # TPPI cycles the receiver phase of 90 degrees at each transient acquisition
        freq_1 = np.zeros(len(t1), dtype='complex64')
        for k in range(4):
            t1t = t1[k::4]
            freq_1[k::4] = np.cos( (2 * np.pi * v1 * t1t) - (0.5 * np.pi * np.mod(k,4) ))
    else:
        freq_1 = np.exp(1j * 2 * np.pi * v1 * t1)
    # NMR signal in the direct dimension
    F2 = np.exp(1j*2*np.pi*v2*t2) * np.exp(-(hwhm2 * t2))
    # NMR signal in the indirect dimension
    F1 = freq_1 * np.exp(-(hwhm1 * t1))
    # The full FID is reconstructed by doing the external product between the two vectors
    S = A * F1.reshape(-1,1) @ F2.reshape(1,-1)
    return S

def t_2Dpvoigt(t1, t2, v1, v2, fwhm1, fwhm2, A=1, x_g=0.5, states=True, alt=True):
    """
    Generates a 2D pseudo-voigt signal in the time domain.
    x_g states for the fraction of gaussianity, whereas A defines the overall amplitude of the total peak.
    Indexes ’1’ and ’2’ on the variables stand for ’F1’ and ’F2’, respectively.
    --------
    Parameters:
    - t1: 1darray
        Indirect evolution timescale
    - t2: 1darray
        Timescale of the direct dimension
    - v1: float
        Peak position in the indirect dimension, in Hz
    - v2: float
        Peak position in the direct dimension, in Hz
    - fwhm1: float
        Full-width at half maximum in the indirect dimension, in rad/s
    - fwhm2: float
        Full-width at half maximum in the direct dimension, in rad/s
    - A: float
        Intensity
    - x_g: float
        Fraction of gaussianity
    - states: bool
        Set to True for "FnMODE":"States-TPPI
    - alt: bool
        Set to True for "FnMODE":"States-TPPI46
    --------
    Returns:
    - fid: 2darray
        Pseudo-Voigt function.
    """

    # stdev computed for the gaussian part.
    s1 = fwhm1 / 2.355
    s2 = fwhm2 / 2.355
    # Passing 's' to 'gaussian' and 'fwhm' to 'lorentzian' makes the two parts of the pseudo-voigt signal to have the same width and allow proper summation
    G = sim.t_2Dgaussian(t1, t2, v1, v2, s1, s2, A=x_g, states=states, alt=alt)
    L = sim.t_2Dlorentzian(t1, t2, v1, v2, fwhm1, fwhm2, A=(1-x_g), states=states, alt=alt)
    fid = A * (G + L)
    return fid

def t_2Dvoigt(t1, t2, v1, v2, fwhm1, fwhm2, A=1, x_g=0.5, states=True, alt=True):
    """
    Generates a 2D Voigt signal in the time domain.
    x_g states for the fraction of gaussianity, whereas A defines the overall amplitude of the total peak.
    Indexes ’1’ and ’2’ on the variables stand for ’F1’ and ’F2’, respectively.
    --------
    Parameters:
    - t1: 1darray
        Indirect evolution timescale
    - t2: 1darray
        Timescale of the direct dimension
    - v1: float
        Peak position in the indirect dimension, in Hz
    - v2: float
        Peak position in the direct dimension, in Hz
    - fwhm1: float
        Full-width at half maximum in the indirect dimension, in rad/s
    - fwhm2: float
        Full-width at half maximum in the direct dimension, in rad/s
    - A: float
        Intensity
    - x_g: float
        Fraction of gaussianity
    - states: bool
        Set to True for "FnMODE":"States-TPPI
    - alt: bool
        Set to True for "FnMODE":"States-TPPI
    --------
    Returns:
    - S: 2darray
        Voigt function.
    """
    # stdev computed for the gaussian part.
    s1 = fwhm1 / 2.355
    s2 = fwhm2 / 2.355
    # hwhm computed for the lorentzian part.
    hwhm1 = fwhm1 / 2
    hwhm2 = fwhm2 / 2
    if states:
        # States acquires twice the same point of the indirect dimension time domain
        t1[1::2] = t1[::2]

    # direct dimension
    #   frequency
    freq_2 = np.exp(1j * 2 * np.pi * v2 * t2) 

    #   Add line-broadening, fist lorentzian then gaussian, using:
    #   hwhm' = (1 - x_g) * hwhm        for L
    #   s' = x_g * s                    for G
    F2 = freq_2 * np.exp(-(1-x_g)*hwhm2 * t2) * np.exp(-((x_g*s2)**2 * t2**2)/2)

    # indirect dimension
    if alt:
        # Redfield cycles the receiver phase of 90 degrees at each transient acquisition
        freq_1 = np.zeros(len(t1), dtype='complex64')
        for k in range(4):
            t1t = t1[k::4]
            freq_1[k::4] = np.cos( (2 * np.pi * v1 * t1t) - (0.5 * np.pi * np.mod(k,4) ))
    else:
        freq_1 = np.exp(1j * 2 * np.pi * v1 * t1)
    #   Add line-broadening, fist lorentzian then gaussian, using:
    #   hwhm' = (1 - x_g) * hwhm        for L
    #   s' = x_g * s                    for G
    F1 = freq_1 * np.exp(-(1-x_g) * hwhm1 * t1) * np.exp(-((x_g*s1)**2 * t1**2)/2)

    # The full FID is reconstructed by doing the external product between the two vectors
    S = A * F1.reshape(-1,1) @ F2.reshape(1,-1)
    return S


def gaussian_filter(ppm, u, s):
    """ 
    Compute a gaussian filter to be used in order to suppress signals in the spectrum.
    ---------
    Parameters:
    - ppm: 1darray
        Scale on which to build the filter
    - u: float
        Position of the filter
    - s: float
        Width of the filter (standard deviation)
    --------
    Returns:
    - G: 1darray
        Computed gaussian filter
    """
    G = sim.f_gaussian(ppm, u, s)
    G /= max(G)     # Normalize to preserve intensities
    G = 1 - G
    return G

