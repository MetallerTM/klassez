#! /usr/bin/env python3

from setuptools import setup, find_packages

try:
    with open('README.md', 'r', encoding='utf-8') as fh:
        long_description = fh.read()
except:
    long_description = 'LONG DESCRIPTION'

setup(
        name='klassez',
        version='0.1a.1',
        author='Francesco Bruno',
        author_email='bruno@cerm.unifi.it',
        description='A collection of functions for NMR data handling. Documentation: klassez.pdf in "docs" subfolder of your install dir.',
        url='https://test.pypi.org/legacy/klassez',
        long_description=long_description,
        long_description_content_type = 'text/markdown',
        classifiers = [
            'Programming Language :: Python :: 3',
            'Operating System :: OS Independent',
            'License :: OSI Approved :: MIT License'
            ],
        license='LICENSE.txt',
        install_requires = ['numpy', 'scipy', 'lmfit', 'seaborn', 'nmrglue', 'matplotlib', 'csaps'],
        packages=['klassez'],
        include_package_data = True,
        python_requires = '>=3.8',
        )
