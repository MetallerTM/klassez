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
#from .__init__ import CM
from .config import CM


warnings.filterwarnings(action='ignore', category=UserWarning)

"""
Classes for the management of NMR data.
"""
class Spectrum_1D:
    """
    Class: 1D NMR spectrum
    """
    def __str__(self):
        doc = '-'*64
        doc += '\nSpectrum_1D object.\n'
        if 'ngdic' in self.__dict__.keys():
            doc += f'Read from "{self.datadir}"\n'
        else:
            doc += f'Simulated from "{self.datadir}"\n'
        N = self.fid.shape[-1]
        doc += f'It is a {self.acqus["nuc"]} spectrum recorded over a\nsweep width of {self.acqus["SWp"]} ppm, centered at {self.acqus["o1p"]} ppm.\n'
        doc += f'The FID is {N} points long.\n'
        doc += '-'*64

        return doc

    def __len__(self):
        if 'S' in self.__dict__.keys():
            return self.S.shape[-1]
        else:
            return self.fid.shape[-1]

    def __init__(self, in_file, pv=False, isexp=True):
        """
        Initialize the class. 
        Simulation of the dataset (i.e. isexp=False) employs sim.sim_1D.
        -------
        Parameters:
        - in_file: str
            path to file to read, or to the folder of the spectrum
        - pv: bool
            True if you want to use pseudo-voigt lineshapes for simulation, False for Voigt
        - isexp: bool
            True if this is an experimental dataset, False if it is simulated
        """
        self.datadir = in_file
        if isexp is False:  # Simulate the dataset
            self.acqus = sim.load_sim_1D(in_file)
            self.fid = sim.sim_1D(in_file, pv=pv)
        else:
            warnings.filterwarnings("ignore")   # Suppress errors due to CONVDTA in TopSpin
            dic, data = ng.bruker.read(in_file, cplex=True)
            self.fid = data
            self.acqus = misc.makeacqus_1D(dic)
            self.BYTORDA = dic['acqus']['BYTORDA']
            self.DTYPA = dic['acqus']['DTYPA']
            self.ngdic = dic        # NMRGLUE dictionary of parameters
            del dic
            del data
        # Look for group delay points: if there is not, put it to 0
        try:
            self.grpdly = int(self.ngdic['acqus']['GRPDLY'])
        except:
            self.grpdly = 0

        # Initalize the procs dictionary with default values
        #       DEFAULT VALUES
        # -----------------------------------------------
        proc_keys_1D = ['wf', 'zf', 'fcor', 'tdeff']
        wf0 = {
                'mode':None,
                'ssb':2,
                'lb':5,
                'gb':10,
                'gc':0,
                'sw':None
                }
        proc_init_1D = (wf0, None, 0.5, 0)
        # -----------------------------------------------

        self.procs = {
                }
        for k, key in enumerate(proc_keys_1D):
            self.procs[key] = proc_init_1D[k]         # Processing parameters
        self.procs['wf']['sw'] = round(self.acqus['SW'], 4)
        #   Then, phases
        self.procs['p0'] = 0
        self.procs['p1'] = 0
        self.procs['pv'] = round(self.acqus['o1p'], 2)
        
    def convdta(self, scaling=1):
        """ Call processing.convdta using attribute self.grpdly """
        self.fid = processing.convdta(self.fid, self.grpdly, scaling)

    def process(self, interactive=False):
        """
        Performs the processing of the FID. The parameters are read from self.procs.
        Calls processing.interactive_fp or processing.fp using self.acqus and self.procs
        Writes the result is self.S, then unpacks it in self.r and self.i
        Calculates frequency and ppm scales.
        Initializes self.F with fit.Voigt_Fit class using the current parameters
        --------
        Parameters:
        - interactive: bool
            True if you want to open the interactive panel, False to read the parameters from self.procs.
        """
        if interactive is True:
            self.S, self.procs = processing.interactive_fp(self.fid, self.acqus, self.procs)
        else:
            self.S = processing.fp(self.fid, wf=self.procs['wf'], zf=self.procs['zf'], fcor=self.procs['fcor'], tdeff=self.procs['tdeff'])
        if self.acqus['SFO1'] < 0:
            self.S = self.S[::-1]
        self.r = self.S.real
        self.i = self.S.imag

        # Calculate frequency and ppm scales
        self.freq = processing.make_scale(self.r.shape[0], dw=self.acqus['dw'])
        if self.acqus['SFO1'] < 0:
            self.freq = self.freq[::-1]
        self.ppm = misc.freq2ppm(self.freq, B0=self.acqus['SFO1'], o1p=self.acqus['o1p'])

        # Initializes the F attribute
        self.F = fit.Voigt_Fit(self.ppm, self.S, self.acqus['t1'], self.acqus['SFO1'], self.acqus['o1p'], self.acqus['nuc'])
        self.baseline = np.zeros_like(self.ppm)
        self.integrals = {}

    def inv_process(self):
        """
        Performs the inverse processing of the spectrum according to the given parameters.
        Overwrites the S attribute!!
        Calls processing.inv_fp
        """
        if self.acqus['SFO1'] < 0:
            self.S = self.S[::-1]
        self.S = processing.inv_fp(self.S, wf=self.procs['wf'], size=self.acqus['TD'], fcor=self.procs['fcor'])

    def mc(self):
        """
        Calculates the magnitude of the spectrum and overwrites self.S, self.r, self.i
        """
        self.S = (self.S.real**2 + self.S.imag**2)**0.5
        self.r = self.S.real
        self.i = self.S.imag

        self.F.S = self.r

    def adjph(self, p0=None, p1=None, pv=None):
        """
        Adjusts the phases of the spectrum according to the given parameters, or interactively if they are left as default.
        Calls for processing.ps
        -------
        Parameters:
        - p0: float or None
            0-th order phase correction /°
        - p1: float or None
            1-st order phase correction /°
        - pv: float or None
            1-st order pivot /ppm
        """
        # Adjust the phases
        self.S, values = processing.ps(self.S, self.ppm, p0=p0, p1=p1, pivot=pv)
        self.r = self.S.real
        self.i = self.S.imag
        self.procs['p0'] += round(values[0], 2)
        self.procs['p1'] += round(values[1], 2)
        if values[2] is not None:
            self.procs['pv'] = round(values[2], 5)

        self.F.S = self.r

    def cal(self, offset=None, isHz=False):
        """
        Calibrates the ppm and frequency scale according to a given value, or interactively.
        Calls processing.calibration
        -------
        Parameters:
        - offset: float or None
            scale shift value
        - isHz: bool
            True if offset is in frequency units, False if offset is in ppm
        """
        in_ppm = np.copy(self.ppm)
        in_S = np.copy(self.r)
        if offset is None:
            offppm = processing.calibration(in_ppm, in_S)
            offhz = misc.ppm2freq(offppm, self.acqus['SFO1'], self.acqus['o1p'])
        else:
            if isHz:
                offhz = offset
                offppm = misc.freq2ppm(offhz, self.acqus['SFO1'], self.acqus['o1p'])
            else:
                offppm = offset
                offhz = misc.ppm2freq(offppm, self.acqus['SFO1'], self.acqus['o1p'])
        self.freq += offhz
        self.ppm += offppm

    def save_acqus(self, path='sim_in_1D'):
        """
        Write the acqus dictionary in a file.
        Calls misc.write_acqus_1D
        --------
        Parameters:
        - path: str
            Filename 
        """
        misc.write_acqus_1D(self.acqus, path=path)

    def write_ser(self, path=None):
        """
        Writes the FID in binary format.
        Calls misc.write_ser
        --------
        Parameters:
        - path: str or None
            Path where to save the binary file. If it is None, the original binary file is overwritten, so BE CAREFUL!!!
        """
        if path is None:
            path = self.datadir
        misc.write_ser(path, self.fid, self.BYTORDA, self.DTYPA)

    def plot(self):
        """
        Plots the real part of the spectrum.
        """
        n_xticks = 10

        # Make the figure
        fig = plt.figure(1)
        fig.set_size_inches(15,8)
        plt.subplots_adjust(left=0.10, bottom=0.15, right=0.95, top=0.90)    # Make room for the sliders
        ax = fig.add_subplot(1,1,1)
        # Auto-adjusts the limits for the y-axis
        misc.set_ylim(ax, self.r)
        # Make pretty x-scale
        xsx, xdx = max(self.ppm), min(self.ppm)
        misc.pretty_scale(ax, (xsx, xdx), axis='x', n_major_ticks=n_xticks)

        # Pretty y-axis numbers
        spect, = ax.plot(self.ppm, self.r, lw=0.8)
        # Create sliders for moving the borders

        X_label = '$\delta\ $'+misc.nuc_format(self.acqus['nuc'])+' /ppm'
        ax.set_xlabel(X_label)

        misc.mathformat(ax)
        misc.set_fontsizes(ax, 14)
        cursor = Cursor(ax, useblit=True, c='tab:red', lw=0.8, horizOn=False)


        plt.show()
        plt.close()

    def qfil(self, u=None, s=None):
        """
        Gaussian filter to suppress signals.
        Tries to read self.procs['qfil'], which is
            { 'u': u, 's': s }
        Calls processing.qfil
        ---------
        Parameters:
        - u: float
            Position /ppm
        - s: float
            Width (standard deviation) /ppm
        """
        if 'qfil' not in self.procs.keys():
            self.procs['qfil'] = {'u': u, 's': s}
        for key, value in self.procs['qfil'].items():
            if value is None:
                self.procs['qfil']['u'], self.procs['qfil']['s'] = processing.interactive_qfil(self.ppm, self.r)
                break
        self.S = processing.qfil(self.ppm, self.S, self.procs['qfil']['u'], self.procs['qfil']['s'])
        self.r = self.S.real
        self.i = self.S.imag


    def basl(self, basl_file='spectrum.basl', winlim=None):
        """
        Correct the baseline of the spectrum, according to a pre-existing file or interactively.
        Calls processing.baseline_correction or processing.load_baseline
        -------
        Parameters:
        - basl_file: str
            Path to the baseline file. If it already exists, the baseline will be built according to this file; otherwise this will be the destination file of the baseline.
        - winlim: tuple or None
            Limits of the baseline. If it is None, it will be interactively set. If basl_file exists, it will be read from there. Else, (ppm1, ppm2).
        """
        if not os.path.exists(basl_file):
            processing.baseline_correction(self.ppm, self.r, basl_file=basl_file, winlim=winlim)
        self.baseline = processing.load_baseline(basl_file, self.ppm, self.r)

    def integrate(self, lims=None):
        """
        Integrate the spectrum with a dedicated GUI.
        Calls fit.integrate and writes in self.integrals
        """
        X_label = '$\delta\,$'+misc.nuc_format(self.acqus['nuc'])+' /ppm'
        if lims is None:
            integrals = fit.integrate(self.ppm, self.r, X_label=X_label)
            for key, value in integrals.items():
                self.integrals[key] = value
        else:
            self.integrals[f'{lims[0]:.2f}:{lims[1]:.2f}'] = processing.integrate(self.r, self.ppm, lims)

    def write_integrals(self, filename='integrals.dat'):
        """
        Write the integrals in a file named filename.
        -------
        Parameters:
        - filename: str
            name of the file where to write the integrals.
        """
        f = open(filename, 'w')
        for key, value in self.integrals.items():
            if 'total' in key:
                f.write('{:12}\t\t{:.4e}\n'.format(key, value))
            elif 'ref' in key:
                if 'pos' in key:
                    f.write('{:12}\t\t{}\n'.format(key, value))
                elif 'int' in key:
                    f.write('{:12}\t\t{:.4e}\n'.format(key, value))
                elif 'val' in key:
                    f.write('{:12}\t\t{:.3f}\n'.format(key, value))
            else:
                f.write('{:12}\t{:.8f}\n'.format(key, value))
        f.close()

    
class pSpectrum_1D(Spectrum_1D):
    """
    Subclass of Spectrum_1D that allows to handle processed 1D NMR spectra.
    Useful when dealing with traces of 2D spectra.
    """
    def __init__(self, in_file, acqus=None, procs=None, istrace=False):
        """
        Initialize the class. 
        -------
        Parameters:
        - in_file: str or 1darray
            If istrace is True, in_file is the NMR spectrum, real part. If istrace is False, in_file is the directory of the processed data.
        - acqus: dict or None
            If istrace is True, you must pass the associated 'acqus' dictionary. If istrace is False, it is not necessary as it is read from the input directory
        - procs: dict or None
            You can pass the dictionary of processing parameters, if you want. Otherwise, it is initialized as empty.
        - istrace: bool
            Declare the object as trace extracted from a 2D (True) or true experimental spectrum (False)
        """
        if istrace is True:
            self.r = in_file
            self.S = self.r
            self.acqus = acqus

        else:
            warnings.filterwarnings("ignore")
            dic, data = ng.bruker.read_pdata(in_file)
            _, self.r = ng.bruker.read_pdata(in_file, bin_files=['1r'])
            _, self.i = ng.bruker.read_pdata(in_file, bin_files=['1i'])
            self.S = self.r + 1j * self.i
            self.acqus = misc.makeacqus_1D(dic)
            self.BYTORDA = dic['acqus']['BYTORDA']
            self.DTYPA = dic['acqus']['DTYPA']
            self.ngdic = dic
            del dic
            del data
        try:
            self.grpdly = int(self.ngdic['acqus']['GRPDLY'])
        except:
            self.grpdly = 0

        if procs is None:
            proc_keys_1D = ['wf', 'zf', 'fcor', 'tdeff']
            wf0 = {
                    'mode':None,
                    'ssb':2,
                    'lb':5,
                    'gb':10,
                    'gc':0,
                    'sw':None
                    }
            proc_init_1D = (wf0, None, 0.5, 0)

            self.procs = {
                    }
            for k, key in enumerate(proc_keys_1D):
                self.procs[key] = proc_init_1D[k]         # Processing parameters
            self.procs['wf']['sw'] = round(self.acqus['SW'], 4)
            # Then, phases
            self.procs['p0'] = 0
            self.procs['p1'] = 0
            self.procs['pv'] = self.acqus['o1p']
        else:
            self.procs = procs
        
        # Calculate frequency and ppm scales
        self.freq = processing.make_scale(self.r.shape[0], dw=self.acqus['dw'])
        if self.acqus['SFO1'] < 0:
            self.freq = self.freq[::-1]
        self.ppm = misc.freq2ppm(self.freq, B0=self.acqus['SFO1'], o1p=self.acqus['o1p'])

        self.F = fit.Voigt_Fit(self.ppm, self.S, self.acqus['t1'], self.acqus['SFO1'], self.acqus['o1p'], self.acqus['nuc'])

    def write_ser(self):
        """Overwrite the original function to prevent writing of the binary file. It does nothing!"""
        pass


class Spectrum_2D:
    """
    Class: 2D NMR spectrum
    """
    def __str__(self):
        doc = '-'*64
        doc += '\nSpectrum_2D object.\n'
        if 'ngdic' in self.__dict__.keys():
            doc += f'Read from "{self.datadir}"\n'
        else:
            doc += f'Simulated from "{self.datadir}"\n'
        N = self.fid.shape
        doc += f'It is a {self.acqus["nuc1"]}-{self.acqus["nuc2"]} spectrum recorded over a \nsweep width of \n{self.acqus["SW1p"]} ppm centered at {self.acqus["o1p"]} ppm in F1, and\n{self.acqus["SW2p"]} ppm centered at {self.acqus["o2p"]} ppm in F2.\n'
        doc += f'The FID is {N[0]}x{N[1]} points long.\n'
        doc += '-'*64

        return doc

    def __len__(self):
        if 'S' in self.__dict__.keys():
            return self.S.shape[-1]
        else:
            return self.fid.shape[-1]

    def __init__(self, in_file, pv=False, isexp=True, is_pseudo=False):
        """
        Initialize the class. 
        -------
        Parameters:
        - in_file: str
            path to file to read, or to the folder of the spectrum
        - isexp: bool
            True if this is an experimental dataset, False if it is simulated
        - pv: bool
            True if you want to use pseudo-voigt lineshapes for simulation, False for Voigt
        - is_pseudo: bool
            True if it is a pseudo-2D. 
        """
        self.datadir = in_file
        if isexp is False:
            self.acqus = sim.load_sim_2D(in_file)
            if is_pseudo:
                self.acqus['FnMODE'] = 'No'
            else:
                self.acqus['FnMODE'] = 'States-TPPI'
            self.fid = sim.sim_2D(in_file, pv=pv)
        else:
            warnings.filterwarnings("ignore")
            dic, data = ng.bruker.read(in_file, cplex=True)
            self.ngdic = dic
            self.fid = data
            self.acqus = misc.makeacqus_2D(dic)
            self.BYTORDA = dic['acqus']['BYTORDA']
            self.DTYPA = dic['acqus']['DTYPA']
            FnMODE_flag = dic['acqu2s']['FnMODE']
            FnMODEs = ['Undefined', 'QF', 'QSEC', 'TPPI', 'States', 'States-TPPI', 'Echo-Antiecho']
            self.acqus['FnMODE'] = FnMODEs[FnMODE_flag]
            # put a flag to say "shuffle"
            if self.acqus['FnMODE'] == 'Echo-Antiecho':
                self.eaeflag = 1
            else:
                self.eaeflag = 0
            del dic
            del data

        try:
            self.grpdly = int(self.ngdic['acqus']['GRPDLY'])
        except:
            self.grpdly = 0

        # initialize the procs dictionary with default values
        wf1 = {
                'mode':None,
                'ssb':2,
                'lb':5,
                'gb':10,
                'gc':0,
                'sw':None
                }
        wf2 = {
                'mode':None,
                'ssb':2,
                'lb':5,
                'gb':10,
                'gc':0,
                'sw':None
                }
        proc_init_2D = (
                [wf1, wf2],     # window function
                [None, None],   # zero-fill
                [0.5, 0.5],     # fcor
                [0,0]           # tdeff
                )

        proc_keys_1D = ['wf', 'zf', 'fcor', 'tdeff']
        self.procs = {}
        for k, key in enumerate(proc_keys_1D):
            self.procs[key] = proc_init_2D[k]         # Processing parameters
        self.procs['wf'][0]['sw'] = round(self.acqus['SW1'], 4)
        self.procs['wf'][1]['sw'] = round(self.acqus['SW2'], 4)

        # Then, phases
        self.procs['p0_1'] = 0
        self.procs['p1_1'] = 0
        self.procs['pv_1'] = round(self.acqus['o1p'], 2)
        self.procs['p0_2'] = 0
        self.procs['p1_2'] = 0
        self.procs['pv_2'] = round(self.acqus['o2p'], 2)

        # Create empty dictionary where to save the projections
        self.trf1 = {}
        self.trf2 = {}
        self.Trf1 = {}
        self.Trf2 = {}

    def convdta(self, scaling=1):
        """ Calls processing.convdta """
        self.fid = processing.convdta(self.fid, self.grpdly, scaling)

    def eae(self):
        """ Calls processing.EAE to shuffle the data. """
        self.fid = processing.EAE(self.fid)
        self.eaeflag = 0

    def xf2(self):
        """
        Process only the direct dimension.
        Calls processing.fp using procs[keys][1]
        freq_f1 and ppm_f1 are assigned with the indexes of the transients.
        """
        if self.procs['zf'][1] is None:
            self.S = np.zeros_like(self.fid)
        else:
            self.S = np.zeros((self.fid.shape[0], self.procs['zf'][1]))

        for k in range(self.fid.shape[0]):
            self.S[k] = processing.fp(self.fid[k], wf=self.procs['wf'][1], zf=self.procs['zf'][1], fcor=self.procs['fcor'][1], tdeff=self.procs['tdeff'][1])

        self.freq_f2 = processing.make_scale(self.S.shape[1], dw=self.acqus['dw2'])
        if self.acqus['SFO2'] < 0:
            self.freq_f2 = self.freq_f2[::-1]
        self.ppm_f2 = misc.freq2ppm(self.freq_f2, B0=self.acqus['SFO2'], o1p=self.acqus['o2p']) 

        if self.acqus['SFO2'] < 0:
            self.S = self.S[:,::-1]

        self.rr = self.S.real
        self.ii = self.S.imag

        self.freq_f1 = np.arange(self.S.shape[0])
        self.ppm_f1 = np.arange(self.S.shape[0])

    def xf1(self):
        """
        Process only the indirect dimension. 
        Transposes the spectrum in hypermode or normally if FnMODE != QF, then calls for processing.fp using self.procs[keys][0], then transposes it back.
        """
        if self.acqus['FnMODE']=='QF':
            self.fid = self.fid.T
        else:
            self.fid = processing.tp_hyper(self.fid)

        if self.procs['zf'][0] is None:
            self.S = np.zeros_like(self.fid)
        else:
            self.S = np.zeros((self.fid.shape[0], self.procs['zf'][0]))

        for k in range(self.fid.shape[0]):
            self.S[k] = processing.fp(self.fid[k], wf=self.procs['wf'][0], zf=self.procs['zf'][0], fcor=self.procs['fcor'][0], tdeff=self.procs['tdeff'][0])

        if self.acqus['FnMODE']=='QF':
            self.fid = self.fid.T
            self.S = self.S.T
        else:
            self.fid = processing.tp_hyper(self.fid)
            self.S = processing.tp_hyper(self.S)

        self.freq_f1 = processing.make_scale(self.S.shape[0], dw=self.acqus['dw1'])
        if self.acqus['SFO1'] < 0:
            self.freq_f1 = self.freq_f1[::-1]
        self.ppm_f1 = misc.freq2ppm(self.freq_f1, B0=self.acqus['SFO1'], o1p=self.acqus['o1p'])

        self.rr = np.copy(self.S.real)
        self.ppm_f2 = np.arange(self.S.shape[1])

    def process(self, interactive=False, **int_kwargs):
        """
        Performs the processing of the FID. The parameters are read from self.procs.
        If interactive is True, calls processing.interactive_xfb with **int_kwargs, else calls processing.xfb
        --------
        Parameters:
        - interactive: bool
            True if you want to open the interactive panel, False to read the parameters from self.procs.
        - int_kwargs: 
            - lvl0: float
                For interactive processing. Set the starting contour values.
            - show_cnt: bool
                For interactive processing. If it is True shows the contours of the spectrum, if it is False shows the heatmap.
        """

        # If Echo-Antiecho, pre-process the FID to get the correct spectral arrangement
        if self.acqus['FnMODE'] == 'Echo-Antiecho' and self.eaeflag == 1:
            self.fid = processing.EAE(self.fid)

        if interactive is True:
            self.S, self.procs = processing.interactive_xfb(self.fid, self.acqus, self.procs, **int_kwargs)
        else:
            self.S = processing.xfb(self.fid, wf=self.procs['wf'], zf=self.procs['zf'], fcor=self.procs['fcor'], tdeff=self.procs['tdeff'], FnMODE=self.acqus['FnMODE'], u=False)

        # For EAE, correct the 90° phase shift in F1
        if self.acqus['FnMODE'] == 'Echo-Antiecho':
            self.S = processing.tp_hyper(self.S)
            self.S = processing.ps(self.S, p0=-90)[0]
            self.S = processing.tp_hyper(self.S)

        if self.acqus['SFO2'] < 0:
            self.S = self.S[:,::-1]
        if self.acqus['SFO1'] < 0:
            # Reversing the spectrum in the indirect dimension causes a 90° dephasing
            self.S = self.S[::-1,:]
            if self.acqus['FnMODE'] == 'QF':
                self.S = self.S.T
            else:
                self.S = processing.tp_hyper(self.S)
            self.S = processing.ps(self.S, p0=-90)[0]   #...that has to be corrected
            if self.acqus['FnMODE'] == 'QF':
                self.S = self.S.T
            else:
                self.S = processing.tp_hyper(self.S)

        if self.acqus['FnMODE'] == 'QF':
            self.rr = self.S.real
            self.ii = self.S.imag
        else:
            rr, ir, ri, ii = processing.unpack_2D(self.S)
            self.rr = rr
            self.ri = ri
            self.ir = ir
            self.ii = ii

        # Calculates the frequency and ppm scales
        self.freq_f1 = processing.make_scale(self.rr.shape[0], dw=self.acqus['dw1'])
        if self.acqus['SFO1'] < 0:
            self.freq_f1 = self.freq_f1[::-1]
        self.ppm_f1 = misc.freq2ppm(self.freq_f1, B0=self.acqus['SFO1'], o1p=self.acqus['o1p'])
        self.freq_f2 = processing.make_scale(self.rr.shape[1], dw=self.acqus['dw2'])
        if self.acqus['SFO2'] < 0:
            self.freq_f2 = self.freq_f2[::-1]
        self.ppm_f2 = misc.freq2ppm(self.freq_f2, B0=self.acqus['SFO2'], o1p=self.acqus['o2p']) 


    def inv_process(self):
        """
        Performs the inverse processing of the spectrum according to the given parameters.
        Overwrites the S attribute!!
        Calls inv_xfb
        """

        # For EAE, correct the 90° phase shift in F1
        if self.acqus['FnMODE'] == 'Echo-Antiecho':
            self.S = processing.tp_hyper(self.S)
            self.S = processing.ps(self.S, p0=90)[0]
            self.S = processing.tp_hyper(self.S)

        if self.acqus['SFO2'] < 0:
            self.S = self.S[:,::-1]
        if self.acqus['SFO1'] < 0:
            self.S = self.S[::-1,:]
            self.S = processing.tp_hyper(self.S)
            self.S = processing.ps(self.S, p0=-90)[0]
            self.S = processing.tp_hyper(self.S)

        self.S = processing.inv_xfb(self.S, wf=self.procs['wf'], size=[self.acqus['TD1'], self.acqus['TD2']], fcor=self.procs['fcor'], FnMODE=self.acqus['FnMODE'])


    def mc(self):
        """ Compute the magnitude of the spectrum. """
        self.S = (self.S.real**2 + self.S.imag**2 )**0.5
        if self.acqus['FnMODE'] == 'QF':
            self.rr = self.S.real
            self.ii = self.S.imag
        else:
            rr, ir, ri, ii = processing.unpack_2D(self.S)
            self.rr = rr
            self.ri = ri
            self.ir = ir
            self.ii = ii

    def adjph(self, p01=None, p11=None, pv1=None, p02=None, p12=None, pv2=None):
        """
        Adjusts the phases of the spectrum according to the given parameters, or interactively if they are left as default.
        -------
        Parameters:
        - p01: float or None
            0-th order phase correction /° of the indirect dimension
        - p11: float or None
            1-st order phase correction /° of the indirect dimension
        - pv1: float or None
            1-st order pivot /ppm of the indirect dimension
        - p02: float or None
            0-th order phase correction /° of the direct dimension
        - p12: float or None
            1-st order phase correction /° of the direct dimension
        - pv2: float or None
            1-st order pivot /ppm of the direct dimension
        """
        interactive = True      # by default
        # Set pivot to carrier if not specified
        if pv1 is None:
            pv1 = self.acqus['o1p']
        if pv2 is None:
            pv2 = self.acqus['o2p']
        ph = [p01, p11, p02, p12]   # for easier handling
        for p in ph:
            # If a phase is specified, interactive is set to False...
            if p is not None:
                interactive = False
        if interactive is False:
            # ... and the not-set phases are put to 0
            for i, p in enumerate(ph):
                if p is None:
                    ph[i] = 0
            # Adjust the phases according to the given values

            self.S, values_f2 = processing.ps(self.S, self.ppm_f2, p0=ph[2], p1=ph[3], pivot=pv2)
            if self.acqus['FnMODE'] == 'No':
                pass
            elif self.acqus['FnMODE'] == 'QF':
                self.S = self.S.T
                self.S, values_f1 = processing.ps(self.S, self.ppm_f1, p0=ph[0], p1=ph[1], pivot=pv1)
                self.S = self.S.T
            else:
                self.S = processing.tp_hyper(self.S)
                self.S, values_f1 = processing.ps(self.S, self.ppm_f1, p0=ph[0], p1=ph[1], pivot=pv1)
                self.S = processing.tp_hyper(self.S)
        else:
            # Call interactive phase correction
            if self.acqus['FnMODE'] == 'QF' or self.acqus['FnMODE'] == 'No': 
                self.S, values_f1, values_f2 = processing.interactive_phase_2D(self.ppm_f1, self.ppm_f2, self.S, False)
            else:
                self.S, values_f1, values_f2 = processing.interactive_phase_2D(self.ppm_f1, self.ppm_f2, self.S)
        # Unpack the phased spectrum
        if self.acqus['FnMODE'] == 'QF' or self.acqus['FnMODE'] == 'No':
            self.rr = self.S.real
            self.ii = self.S.imag
        else:
            rr, ir, ri, ii = processing.unpack_2D(self.S)
            self.rr = rr
            self.ri = ri
            self.ir = ir
            self.ii = ii
        # update procs
        self.procs['p0_2'] += round(values_f2[0], 2)
        self.procs['p1_2'] += round(values_f2[1], 2)
        if values_f2[2] is not None:
            self.procs['pv_2'] = round(values_f2[2], 5)
        self.procs['p0_1'] += round(values_f1[0], 2)
        self.procs['p1_1'] += round(values_f1[1], 2)
        if values_f1[2] is not None:
            self.procs['pv_1'] = round(values_f1[2], 5)


    def qfil(self, which=None, u=None, s=None):
        """ 
        Suppress signals using qfil. 
        'Which' is the number of the trace to be used.
        Edits only 'rr' if FnMODE is phase-sensitive
        Calls processing.qfil
        """
        if 'qfil' not in self.procs.keys():
            self.procs['qfil'] = {'u': u, 's': s}
        if which is None:
            which_list = misc.select_traces(self.ppm_f1, self.ppm_f2, self.rr, Neg=False, grid=False)
            print(which_list)
            which, _ = misc.ppmfind(self.ppm_f1, which_list[0][1])
        print(which, self.ppm_f1[which])

        for key, value in self.procs['qfil'].items():
            if value is None:
                self.procs['qfil']['u'], self.procs['qfil']['s'] = processing.interactive_qfil(self.ppm_f2, self.rr[which])
                break
        self.S = processing.qfil(self.ppm_f2, self.S, self.procs['qfil']['u'], self.procs['qfil']['s'])
        if self.acqus['FnMODE'] == 'QF':
            self.rr = self.S.real
            self.ii = self.S.imag
        else:
            self.rr, self.ir, self.ri, self.ii = processing.unpack_2D(self.S)

    def cal(self, offset=[None,None], isHz=False):
        """
        Calibration of the ppm and frequency scales according to a given value, or interactively. In this latter case, a reference peak must be chosen.
        Calls processing.calibration
        --------
        Parameters:
        - offset: tuple
            (scale shift F1, scale shift F2)
        - isHz: tuple of bool
            True if offset is in frequency units, False if offset is in ppm
        """

        def _calibrate(ppm, trace, SFO1, o1p):
            offppm = processing.calibration(ppm, trace)
            offhz = misc.ppm2freq(offppm, SFO1, o1p)
            return offppm, offhz

        if offset[0] is None or offset[1] is None:
            coord = misc.select_traces(self.ppm_f1, self.ppm_f2, self.rr, Neg=False, grid=False)
            ix, iy = coord[0][0], coord[0][1]
            X = misc.get_trace(self.rr, self.ppm_f2, self.ppm_f1, iy, column=False)
            Y = misc.get_trace(self.rr, self.ppm_f2, self.ppm_f1, ix, column=True)

        if offset[1] is None:
            ppm_f2 = np.copy(self.ppm_f2)
            offp2, offh2 = _calibrate(ppm_f2, X, self.acqus['SFO2'], self.acqus['o2p'])
        else:
            if isHz:
                offh2 = offset[1]
                offp2 = misc.freq2ppm(offh2, self.acqus['SFO2'], self.acqus['o2p']) 
            else:
                offp2 = offset[1]
                offh2 = misc.ppm2freq(offp2, self.acqus['SFO2'], self.acqus['o2p']) 
            
        if offset[0] is None:
            ppm_f1 = np.copy(self.ppm_f1)
            offp1, offh1 = _calibrate(ppm_f1, Y, self.acqus['SFO1'], self.acqus['o1p'])
        else:
            if isHz:
                offh1 = offset[0]
                offp1 = misc.freq2ppm(offh1, self.acqus['SFO1'], self.acqus['o1p']) 
            else:
                offp1 = offset[0]
                offh1 = misc.ppm2freq(offp1, self.acqus['SFO1'], self.acqus['o1p']) 

        self.freq_f2 += offh2
        self.ppm_f2 += offp2
        self.freq_f1 += offh1
        self.ppm_f1 += offp1


    def calf2(self, value=None, isHz=False):
        """
        Calibrates the ppm and frequency scale of the direct dimension according to a given value, or interactively.
        Calls self.cal on F2 only
        -------
        Parameters:
        - value: float or None
            scale shift value
        - isHz: bool
            True if offset is in frequency units, False if offset is in ppm
        """
        offset = [0, value]
        self.cal(offset, isHz)

    def calf1(self, value=None, isHz=False):
        """
        Calibrates the ppm and frequency scale of the indirect dimension according to a given value, or interactively.
        Calls self.cal on F1 only.
        -------
        Parameters:
        - value: float or None
            scale shift value
        - isHz: bool
            True if offset is in frequency units, False if offset is in ppm
        """
        offset = [value, 0]
        self.cal(offset, isHz)



    def save_acqus(self, path='sim_in_2D'):
        """
        Write the acqus dictionary in a file.
        Calls misc.write_acqus_2D
        --------
        Parameters:
        - path: str
            Filename 
        """
        misc.write_acqus_2D(self.acqus, path=path)

    def write_ser(self, path=None):
        """
        Writes the FID in binary format.
        Calls misc.write_ser
        --------
        Parameters:
        - path: str or None
            Path where to save the binary file. If it is None, the original binary file is overwritten, so BE CAREFUL!!!
        """
        if path is None:
            path = self.datadir
        misc.write_ser(self.fid, path, self.BYTORDA, self.DTYPA)

    def projf1(self, a, b=None):
        """
        Calculates the sum trace of the indirect dimension, from a to b in F2.
        Store the trace in the dictionary trf1 and as 1D spectrum in Trf1. The key is 'a' or 'a:b'
        Calls misc.get_trace on self.rr with column=True
        -------
        Parameters:
        - a: float
            ppm F2 value where to extract the trace.
        - b: float or None.
            If it is None, extract the trace in a. Else, sum from a to b in F2.
        """
        # make dictionary label
        if b is None:
            label = str(a)
        else:
            label = str(a)+':'+str(b)
        f1 = misc.get_trace(self.rr, self.ppm_f2, self.ppm_f1, a, b, column=True)
        self.trf1[label] = f1
        self.Trf1[label] = pSpectrum_1D(f1, acqus=misc.split_acqus_2D(self.acqus)[0], procs=misc.split_procs_2D(self.procs)[0], istrace=True)

    def projf2(self, a, b=None):
        """
        Calculates the sum trace of the direct dimension, from a to b in F1.
        Store the trace in the dictionary trf2 and as 1D spectrum in Trf2. The key is 'a' or 'a:b'
        Calls misc.get_trace on self.rr with column=False
        -------
        Parameters:
        - a: float
            ppm F1 value where to extract the trace.
        - b: float or None.
            If it is None, extract the trace in a. Else, sum from a to b in F1.
        """
        # make dictionary label
        if b is None:
            label = str(a)
        else:
            label = str(a)+':'+str(b)
        f2 = misc.get_trace(self.rr, self.ppm_f2, self.ppm_f1, a, b, column=False)
        self.trf2[label] = f2
        self.Trf2[label] = pSpectrum_1D(f2, acqus=misc.split_acqus_2D(self.acqus)[1], procs=misc.split_procs_2D(self.procs)[1], istrace=True)

    def integrate(self, **kwargs):
        """
        Integrate the spectrum with a dedicated GUI.
        Calls fit.integrate_2D 
        """
        self.integrals = fit.integrate_2D(self.ppm_f1, self.ppm_f2, self.rr, self.acqus['SFO1'], self.acqus['SFO2'], **kwargs)

    def write_integrals(self, filename='integrals.dat'):
        """
        Write the integrals in a file named filename.
        -------
        Parameters:
        - filename: str
            name of the file where to write the integrals.
        """
        f = open(filename, 'w')
        f.write('{:12}\t{:12}\t\t{:20}\n'.format('ppm F2', 'ppm F1', 'Value'))
        f.write('-'*60+'\n')
        for key, value in self.integrals.items():
            ppm2, ppm1 = tuple(key.split(':'))
            f.write('{:12}\t{:12}\t\t{:20.5e}\n'.format(ppm2, ppm1, value))
        f.close()

    def plot(self, Neg=True, lvl0=0.2):
        """
        Plots the real part of the spectrum.
        -------
        Parameters:
        - Neg: bool
            Plot (True) or not (False) the negative contours.
        - lvl0: float
            Starting contour value.
        """
        warnings.filterwarnings("ignore", message="No contour levels were found within the data range.")
        # Plots data, set Neg=True to see negative contours
        S = self.rr
        n_xticks, n_yticks = 10, 10

        X_label = '$\delta\ $'+misc.nuc_format(self.acqus['nuc2'])+' /ppm'
        Y_label = '$\delta\ $'+misc.nuc_format(self.acqus['nuc1'])+' /ppm'

        cmaps = [cm.Blues_r, cm.Reds_r]

        # flags for the activation of scroll zoom
        lvlstep = 0.02

        # define boxes for sliders
        iz_box = plt.axes([0.925, 0.80, 0.05, 0.05])
        dz_box = plt.axes([0.925, 0.75, 0.05, 0.05])

        # Functions connected to the sliders
        def increase_zoom(event):
            nonlocal lvlstep
            lvlstep *= 2

        def decrease_zoom(event):
            nonlocal lvlstep
            lvlstep /= 2

        def on_scroll(event):
            nonlocal livello, cnt
            if Neg:
                nonlocal Ncnt
                
            if event.button == 'up':
                livello += lvlstep 
            elif event.button == 'down':
                livello += -lvlstep
            if livello <= 0:
                livello = 1e-6
            elif livello > 1:
                livello = 1

            if Neg:
                cnt, Ncnt = figures.redraw_contours(ax, self.ppm_f2, self.ppm_f1, S, lvl=livello, cnt=cnt, Neg=Neg, Ncnt=Ncnt, lw=0.5, cmap=[cmaps[0], cmaps[1]])
            else:
                cnt, _ = figures.redraw_contours(ax, self.ppm_f2, self.ppm_f1, S, lvl=livello, cnt=cnt, Neg=Neg, Ncnt=None, lw=0.5, cmap=[cmaps[0], cmaps[1]])

            misc.pretty_scale(ax, (max(self.ppm_f2), min(self.ppm_f2)), axis='x', n_major_ticks=n_xticks)
            misc.pretty_scale(ax, (max(self.ppm_f1), min(self.ppm_f1)), axis='y', n_major_ticks=n_yticks)
            ax.set_xlabel(X_label)
            ax.set_ylabel(Y_label)
            misc.set_fontsizes(ax, 14)
            print('{:.3f}'.format(livello), end='\r')
            fig.canvas.draw()

        # Make the figure
        fig = plt.figure(1)
        fig.set_size_inches(15,8)
        plt.subplots_adjust(left = 0.10, bottom=0.10, right=0.90, top=0.95)
        ax = fig.add_subplot(1,1,1)

        contour_num = 16
        contour_factor = 1.40

        livello = lvl0

        cnt = figures.ax2D(ax, self.ppm_f2, self.ppm_f1, S, lvl=livello, cmap=cmaps[0])
        if Neg:
            Ncnt = figures.ax2D(ax, self.ppm_f2, self.ppm_f1, -S, lvl=livello, cmap=cmaps[1])

        # Make pretty x-scale
        misc.pretty_scale(ax, (max(self.ppm_f2), min(self.ppm_f2)), axis='x', n_major_ticks=n_xticks)
        misc.pretty_scale(ax, (max(self.ppm_f1), min(self.ppm_f1)), axis='y', n_major_ticks=n_yticks)
        ax.set_xlabel(X_label)
        ax.set_ylabel(Y_label)

        scale_factor = 1

        # Create buttons
        iz_button = Button(iz_box, label='$\\uparrow$')
        dz_button = Button(dz_box, label='$\downarrow$')

        # Connect the widgets to functions
        scroll = fig.canvas.mpl_connect('scroll_event', on_scroll)
            
        iz_button.on_clicked(increase_zoom)
        dz_button.on_clicked(decrease_zoom)

        misc.set_fontsizes(ax, 14)

        cursor = Cursor(ax, useblit=True, c='tab:red', lw=0.8)

        plt.show()
        plt.close()


class pSpectrum_2D(Spectrum_2D):
    """
    Subclass of Spectrum_2D that allows to handle processed 2D NMR spectra.
    Reads the processed spectrum from Bruker.
    """

    def __init__(self, in_file):
        """
        Initialize the class. 
        -------
        Parameters:
        - in_file: str
            Path to the spectrum. Here, the 'pdata/#' folder must be specified.
        """
        if in_file[-1] != '/':
            in_file = in_file+'/'
        warnings.filterwarnings("ignore")
        dic, data = ng.bruker.read(in_file.split('pdata')[0], cplex=True)
        _, self.rr = ng.bruker.read_pdata(in_file, bin_files=['2rr'])
        _, self.ii = ng.bruker.read_pdata(in_file, bin_files=['2ii'])
        if os.path.exists(in_file+'2ir') and os.path.exists(in_file+'2ri'):
            _, self.ir = ng.bruker.read_pdata(in_file, bin_files=['2ir'])
            _, self.ri = ng.bruker.read_pdata(in_file, bin_files=['2ri'])
            self.S = processing.repack_2D(self.rr, self.ir, self.ri, self.ii)
        else:
            self.ir = np.array(np.copy(self.rr))
            self.ri = np.copy(self.ii)
            self.S = self.rr + 1j*self.ii

        self.acqus = misc.makeacqus_2D(dic)
        self.BYTORDA = dic['acqus']['BYTORDA']
        self.DTYPA = dic['acqus']['DTYPA']
        self.ngdic = dic
        del dic
        del data

        try:
            self.grpdly = int(self.ngdic['acqus']['GRPDLY'])
        except:
            self.grpdly = 0

        # initialize the procs dictionary with default values
        wf1 = {
                'mode':None,
                'ssb':2,
                'lb':5,
                'gb':10,
                'gc':0,
                'sw':None
                }
        wf2 = {
                'mode':None,
                'ssb':2,
                'lb':5,
                'gb':10,
                'gc':0,
                'sw':None
                }
        proc_init_2D = (
                [wf1, wf2],     # window function
                [None, None],   # zero-fill
                [0.5, 0.5],     # fcor
                [0,0]           # tdeff
                )

        proc_keys_1D = ['wf', 'zf', 'fcor', 'tdeff']
        self.procs = {}
        for k, key in enumerate(proc_keys_1D):
            self.procs[key] = proc_init_2D[k]         # Processing parameters
        self.procs['wf'][0]['sw'] = round(self.acqus['SW1'], 4)
        self.procs['wf'][1]['sw'] = round(self.acqus['SW2'], 4)

        # Then, phases
        self.procs['p0_1'] = 0
        self.procs['p1_1'] = 0
        self.procs['pv_1'] = round(self.acqus['o1p'], 2)
        self.procs['p0_2'] = 0
        self.procs['p1_2'] = 0
        self.procs['pv_2'] = round(self.acqus['o2p'], 2)
        
        # Calculates the frequency and ppm scales
        self.freq_f1 = processing.make_scale(self.rr.shape[0], dw=self.acqus['dw1'])
        if self.acqus['SFO1'] < 0:
            self.freq_f1 = self.freq_f1[::-1]
        self.ppm_f1 = misc.freq2ppm(self.freq_f1, B0=self.acqus['SFO1'], o1p=self.acqus['o1p'])

        self.freq_f2 = processing.make_scale(self.rr.shape[1], dw=self.acqus['dw2'])
        if self.acqus['SFO2'] < 0:
            self.freq_f2 = self.freq_f2[::-1]
        self.ppm_f2 = misc.freq2ppm(self.freq_f2, B0=self.acqus['SFO2'], o1p=self.acqus['o2p']) 

        # Create empty dictionary where to save the projections
        self.trf1 = {}
        self.trf2 = {}
        self.Trf1 = {}
        self.Trf2 = {}

    def write_ser(self):
        """Overwrite the original function to prevent writing of the binary file. It does nothing!"""
        pass

class Pseudo_2D(Spectrum_2D):
    """ Pseudo_2D experiment """

    def __str__(self):
        doc = '-'*64
        doc += '\nPseudo_2D object.\n'
        if 'ngdic' in self.__dict__.keys():
            doc += f'Read from "{self.datadir}"\n'
        else:
            doc += f'Simulated from "{self.datadir}"\n'
        doc += f'It is a {self.acqus["nuc"]} spectrum recorded over a\nsweep width of {self.acqus["SWp"]} ppm, centered at {self.acqus["o1p"]} ppm.\n'
        if self.fid is None:
            doc += 'The FID is not present yet.'
        else:
            N = self.fid.shape
            doc += f'The FID consists of {N[0]} experiments, each one is {N[1]} points long.\n'
        doc += '-'*64
        return doc

    def __init__(self, in_file, fid=None, pv=False, isexp=True):
        """
        Initialize the class. 
        -------
        Parameters:
        - in_file: str
            path to file to read, or to the folder of the spectrum
        - fid: 2darray or None
            Array that replaces self.fid.
        - isexp: bool
            True if this is an experimental dataset, False if it is simulated
        - pv: bool
            True if you want to use pseudo-voigt lineshapes for simulation, False for Voigt
        """
        self.datadir = in_file
        if isexp is False:
            self.acqus = sim.load_sim_1D(in_file)
            self.fid = fid
        else:
            dic, data = ng.bruker.read(in_file, cplex=True)
            self.fid = data
            self.acqus = misc.makeacqus_1D(dic)
            self.BYTORDA = dic['acqus']['BYTORDA']
            self.DTYPA = dic['acqus']['DTYPA']
            self.ngdic = dic
            del dic
            del data

        try:
            self.grpdly = int(self.ngdic['acqus']['GRPDLY'])
        except:
            self.grpdly = 0

        # Initalize the procs dictionary with default values
        proc_keys_1D = ['wf', 'zf', 'fcor', 'tdeff']
        wf0 = {
                'mode':None,
                'ssb':2,
                'lb':5,
                'gb':10,
                'gc':0,
                'sw':None
                }
        proc_init_1D = (wf0, None, 0.5, 0)

        self.procs = {
                }
        for k, key in enumerate(proc_keys_1D):
            self.procs[key] = proc_init_1D[k]         # Processing parameters
        self.procs['wf']['sw'] = round(self.acqus['SW'], 4)
        # Then, phases
        self.procs['p0'] = 0
        self.procs['p1'] = 0
        self.procs['pv'] = round(self.acqus['o1p'], 2)

    def convdta(self, scaling=1):
        """ Calls processing.convdta """
        self.fid = processing.convdta(self.fid, self.grpdly, scaling)
        
    def process(self):
        """
        Process only the direct dimension.
        Calls processing.fp on each transient
        """
        if self.procs['zf'] is None:
            self.S = np.zeros_like(self.fid)
        else:
            self.S = np.zeros((self.fid.shape[0], self.procs['zf'])).astype(self.fid.dtype)

        for k in range(self.fid.shape[0]):
            self.S[k] = processing.fp(self.fid[k], wf=self.procs['wf'], zf=self.procs['zf'], fcor=self.procs['fcor'], tdeff=self.procs['tdeff'])

        self.freq_f2 = processing.make_scale(self.S.shape[1], dw=self.acqus['dw'])
        if self.acqus['SFO1'] < 0:
            self.freq_f2 = self.freq_f2[::-1]
        self.ppm_f2 = misc.freq2ppm(self.freq_f2, B0=self.acqus['SFO1'], o1p=self.acqus['o1p']) 

        if self.acqus['SFO1'] < 0:
            self.S = self.S[:,::-1]

        self.rr = self.S.real
        self.ii = self.S.imag

        self.freq_f1 = np.arange(self.S.shape[0])
        self.ppm_f1 = np.arange(self.S.shape[0])

        self.integrals = {}
        
        # Create empty dictionary where to save the projections
        self.trf1 = {}
        self.trf2 = {}
        self.Trf1 = {}
        self.Trf2 = {}


    def adjph(self, expno=0, p0=None, p1=None, pv=None):
        """
        Adjusts the phases of the spectrum according to the given parameters, or interactively if they are left as default.
        -------
        Parameters:
        - expno: int
            Number of the experiment (python numbering) to use in the interactive panel
        - p0: float or None
            0-th order phase correction /°
        - p1: float or None
            1-st order phase correction /°
        - pv: float or None
            1-st order pivot /ppm
        """
        S = self.S[expno]
        # Adjust the phases
        _, values = processing.ps(S, self.ppm_f2, p0=p0, p1=p1, pivot=pv)
        self.S, _ = processing.ps(self.S, self.ppm_f2, *values)
        
        self.rr = self.S.real
        self.ii = self.S.imag

        self.procs['p0'] += round(values[0], 2)
        self.procs['p1'] += round(values[1], 2)
        if values[2] is not None:
            self.procs['pv'] = round(values[2], 5)


    def projf1(self, a, b=None):
        """
        Calculates the sum trace of the indirect dimension, from a to b in F2.
        Store the trace in the dictionary trf1 and as 1D spectrum in Trf1. The key is 'a' or 'a:b'
        -------
        Parameters:
        - a: float
            ppm F2 value where to extract the trace.
        - b: float or None.
            If it is None, extract the trace in a. Else, sum from a to b in F2.
        """
        # make dictionary label
        if b is None:
            label = str(a)
        else:
            label = str(a)+':'+str(b)
        f1 = misc.get_trace(self.rr, self.ppm_f2, self.ppm_f1, a, b, column=True)
        self.trf1[label] = f1
        self.Trf1[label] = pSpectrum_1D(f1, acqus=self.acqus, procs=self.procs, istrace=True)
        self.Trf1[label].freq = np.copy(self.freq_f1)
        self.Trf1[label].ppm = np.copy(self.ppm_f1)

    def projf2(self, a, b=None):
        """
        Calculates the sum trace of the direct dimension, from a to b in F1.
        Store the trace in the dictionary trf2 and as 1D spectrum in Trf2. The key is 'a' or 'a:b'
        -------
        Parameters:
        - a: float
            ppm F1 value where to extract the trace.
        - b: float or None.
            If it is None, extract the trace in a. Else, sum from a to b in F1.
        """
        # make dictionary label
        if b is None:
            label = str(a)
        else:
            label = str(a)+':'+str(b)
        f2 = misc.get_trace(self.rr, self.ppm_f2, self.ppm_f1, a, b, column=False)
        self.trf2[label] = f2
        self.Trf2[label] = pSpectrum_1D(f2, acqus=self.acqus, procs=self.procs, istrace=True)

    def plot(self, Neg=True, lvl0=0.2, Y_label=''):
        """
        Plots the real part of the spectrum.
        -------
        Parameters:
        - Neg: bool
            Plot (True) or not (False) the negative contours.
        - lvl0: float
            Starting contour value.
        """
        warnings.filterwarnings("ignore", message="No contour levels were found within the data range.")
        # Plots data, set Neg=True to see negative contours
        S = np.copy(self.rr)
        n_xticks, n_yticks = 10, 10

        X_label = '$\delta\ $'+misc.nuc_format(self.acqus['nuc'])+' /ppm'

        cmaps = [cm.Blues_r, cm.Reds_r]

        # flags for the activation of scroll zoom
        lvlstep = 0.02

        # define boxes for sliders
        iz_box = plt.axes([0.925, 0.80, 0.05, 0.05])
        dz_box = plt.axes([0.925, 0.75, 0.05, 0.05])

        # Functions connected to the sliders
        def increase_zoom(event):
            nonlocal lvlstep
            lvlstep *= 2

        def decrease_zoom(event):
            nonlocal lvlstep
            lvlstep /= 2

        def on_scroll(event):
            nonlocal livello, cnt
            if Neg:
                nonlocal Ncnt
                
            if event.button == 'up':
                livello += lvlstep 
            elif event.button == 'down':
                livello += -lvlstep
            if livello <= 0:
                livello = 1e-6
            elif livello > 1:
                livello = 1

            if Neg:
                cnt, Ncnt = figures.redraw_contours(ax, self.ppm_f2, self.ppm_f1, S, lvl=livello, cnt=cnt, Neg=Neg, Ncnt=Ncnt, lw=0.5, cmap=[cmaps[0], cmaps[1]])
            else:
                cnt, _ = figures.redraw_contours(ax, self.ppm_f2, self.ppm_f1, S, lvl=livello, cnt=cnt, Neg=Neg, Ncnt=None, lw=0.5, cmap=[cmaps[0], cmaps[1]])

            misc.pretty_scale(ax, (max(self.ppm_f2), min(self.ppm_f2)), axis='x', n_major_ticks=n_xticks)
            misc.pretty_scale(ax, (max(self.ppm_f1), min(self.ppm_f1)), axis='y', n_major_ticks=n_yticks)
            ax.set_xlabel(X_label)
            ax.set_ylabel(Y_label)
            misc.set_fontsizes(ax, 14)
            print('{:.3f}'.format(livello), end='\r')
            fig.canvas.draw()

        # Make the figure
        fig = plt.figure(1)
        fig.set_size_inches(15,8)
        plt.subplots_adjust(left = 0.10, bottom=0.10, right=0.90, top=0.95)
        ax = fig.add_subplot(1,1,1)

        contour_num = 16
        contour_factor = 1.40

        livello = lvl0

        cnt = figures.ax2D(ax, self.ppm_f2, self.ppm_f1, S, lvl=livello, cmap=cmaps[0])
        if Neg:
            Ncnt = figures.ax2D(ax, self.ppm_f2, self.ppm_f1, -S, lvl=livello, cmap=cmaps[1])

        # Make pretty x-scale
        misc.pretty_scale(ax, (max(self.ppm_f2), min(self.ppm_f2)), axis='x', n_major_ticks=n_xticks)
        misc.pretty_scale(ax, (max(self.ppm_f1), min(self.ppm_f1)), axis='y', n_major_ticks=n_yticks)
        ax.set_xlabel(X_label)
        ax.set_ylabel(Y_label)

        scale_factor = 1

        # Create buttons
        iz_button = Button(iz_box, label='$\\uparrow$')
        dz_button = Button(dz_box, label='$\downarrow$')

        # Connect the widgets to functions
        scroll = fig.canvas.mpl_connect('scroll_event', on_scroll)
            
        iz_button.on_clicked(increase_zoom)
        dz_button.on_clicked(decrease_zoom)

        misc.set_fontsizes(ax, 14)

        cursor = Cursor(ax, useblit=True, c='tab:red', lw=0.8)

        plt.show()
        plt.close()
    
    def plot_md(self, which='all', lims=None):
        """ 
        Plot a number of experiments, superimposed.
        --------
        Parameters:
        - which: str
            List of experiment indexes, so that eval(which) is meaningful
        - lims: tuple
            Region of the spectrum to show (ppm1, ppm2)
        """
        if 'all' in which:
            which_exp = np.arange(self.rr.shape[0])
        else:
            which_exp = eval(which)
        ppm = np.copy(self.ppm_f2)
        S = [np.copy(self.rr[w]) for w in which_exp]

        if lims is not None:
            for k, s in enumerate(S): 
                _, S[k] = misc.trim_data(ppm, s, *lims)
            ppm, _ = misc.trim_data(ppm, s, *lims)

        figures.dotmd(ppm, S, labels=[f'{w}' for w in which_exp])

    def plot_stacked(self, which='all', lims=None):
        """ 
        Plot a number of experiments, stacked.
        --------
        Parameters:
        - which: str
            List of experiment indexes, so that eval(which) is meaningful
        - lims: tuple
            Region of the spectrum to show (ppm1, ppm2)
        """
        if 'all' in which:
            which_exp = np.arange(self.rr.shape[0])
        else:
            which_exp = eval(which)
        ppm = np.copy(self.ppm_f2)
        S = [np.copy(self.rr[w]) for w in which_exp]

        if lims is not None:
            for k, s in enumerate(S): 
                _, S[k] = misc.trim_data(ppm, s, *lims)
            ppm, _ = misc.trim_data(ppm, s, *lims)

        X_label = '$\delta\ $'+misc.nuc_format(self.acqus['nuc'])+' /ppm'

        figures.stacked_plot(
                ppm, S, 
                X_label=X_label, Y_label='Normalized intensity /a.u.',
                labels=[f'{w}' for w in which_exp])


    def integrate(self, which=0, lims=None):
        """
        Integrate the spectrum with a dedicated GUI.
        Calls processing.integral on each experiment, then saves the results in self.integrals.
        If lims is not given, calls fit.integrate to select the regions to integrate.
        --------
        Parameters:
        - which: int
            Experiment index to show in interactive panel
        - lims: tuple
            Region of the spectrum to integrate (ppm1, ppm2)
        """
        if lims is None:
            X_label = '$\delta\,$'+misc.nuc_format(self.acqus['nuc'])+' /ppm'
            integrals = fit.integrate(self.ppm_f2, self.rr[which], X_label=X_label)
            for key, _ in integrals.items():
                if ':' in key:
                    lims = [eval(q) for q in key.split(':')] # trasforma stringa in float!!!
                    self.integrals[key] = [processing.integral(self.rr[k], self.ppm_f2, lims)[-1] for k in range(self.rr.shape[0])]
                else:
                    self.integrals[key] = np.array(integrals[key])

        else:
            self.integrals[f'{lims[0]:.2f}:{lims[1]:.2f}'] = np.array(processing.integral(self.rr, self.ppm_f2, lims)[...,-1])








