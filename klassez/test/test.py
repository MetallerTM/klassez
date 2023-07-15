#! /usr/bin/env python3

from klassez import *

sys.stdout = open('test.log', 'w')
sys.stderr = sys.stdout

print('TEST OF THE KLASSEZ PACKAGE')
print('This test is intended for internal use.')

print('\nBEGIN OF THE TEST')

print('Spectrum_1D object')
print('Reading')
s1 = Spectrum_1D('acqus_1D', isexp=False)
print('Processing')
s1.process()

print('Spectrum_2D object')
print('Reading')
s2 = Spectrum_2D('acqus_2D', isexp=False)
print('Processing')
s2.process()
print('Compute projection in f2')
s2.projf2(10)
print('Compute projection in f1')
s2.projf1(90)

print('END OF THE TEST')
