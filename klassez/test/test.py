#! /usr/bin/env python3

from klassez import *

# Test: import 1D spectrum
s1 = Spectrum_1D('acqus_1D', isexp=False)
s1.process()
s1.F.iguess()
s1.F.plot('iguess')
s1.F.dofit()
s1.F.plot('fit')

s2 = Spectrum_2D('acqus_2D', isexp=False)
s2.process()
