#!/usr/bin/env python

################################################################################
# Copyright (C) 2011-2015 Jaakko Luttinen
#
# This file is licensed under the MIT License.
################################################################################


NAME         = 'bayespy'
DESCRIPTION  = 'Variational Bayesian inference tools for Python'
AUTHOR       = 'Jaakko Luttinen'
AUTHOR_EMAIL = 'jaakko.luttinen@iki.fi'
URL          = 'http://bayespy.org'
LICENSE      = 'MIT'
VERSION      = '0.3.6'

if __name__ == "__main__":

    import os

    # Utility function to read the README file.
    # Used for the long_description.  It's nice, because now 1) we have a top level
    # README file and 2) it's easier to type in the README file than to put a raw
    # string in below ...
    def read(fname):
        return open(os.path.join(os.path.dirname(__file__), fname)).read()

    from setuptools import setup, find_packages
    
    # Setup for BayesPy
    setup(
          install_requires = ['numpy>=1.8.0', # 1.8 implements broadcasting in numpy.linalg
                              'scipy>=0.13.0', # <0.13 have a bug in special.multigammaln
                              'matplotlib>=1.2.0',
                              'h5py'],
          
          packages         = find_packages(),
          package_data     = {NAME: ["tests/baseline_images/test_plot/*.png"]},
          name             = NAME,
          version          = VERSION,
          author           = AUTHOR,
          author_email     = AUTHOR_EMAIL,
          description      = DESCRIPTION,
          license          = LICENSE,
          url              = URL,
          long_description = read('README.rst'),
          keywords         =
            [
              'variational Bayes',
              'probabilistic programming',
              'Bayesian networks',
              'graphical models',
              'variational message passing'
            ],
          classifiers =
            [ 
              'Programming Language :: Python :: 3 :: Only',
              'Programming Language :: Python :: 3.3',
              'Programming Language :: Python :: 3.4',
              'Development Status :: 4 - Beta',
              'Environment :: Console',
              'Intended Audience :: Developers',
              'Intended Audience :: Science/Research',
              'License :: OSI Approved :: MIT License',
              'Operating System :: OS Independent',
              'Topic :: Scientific/Engineering',
              'Topic :: Scientific/Engineering :: Information Analysis'
            ]
          )

