#! /usr/bin/env python3

from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

setup(
        name='klassez',
        version='0.4a.7',
        author='Francesco Bruno, Letizia Fiorucci',
        author_email='bruno@cerm.unifi.it',
        description='A collection of functions for NMR data handling.',
        url='https://github.com/MetallerTM/klassez',
        long_description=long_description,
        long_description_content_type = 'text/markdown',
        classifiers = [
            'Programming Language :: Python :: 3',
            'Operating System :: OS Independent',
            'License :: OSI Approved :: MIT License'
            ],
        license='LICENSE.txt',
        install_requires = ['numpy', 'scipy', 'lmfit', 'seaborn', 'nmrglue', 'matplotlib>=3.8', 'csaps', 'jeol_parser'],
        packages=['klassez'],
        include_package_data = True,
        python_requires = '>=3.9',
        )
