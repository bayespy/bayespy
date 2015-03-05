#!/usr/bin/env python

######################################################################
# Copyright (C) 2011,2012,2014 Jaakko Luttinen
#
# This file is licensed under Version 3.0 of the GNU General Public
# License. See LICENSE for a text of the license.
######################################################################

######################################################################
# This file is part of BayesPy.
#
# BayesPy is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation.
#
# BayesPy is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with BayesPy.  If not, see <http://www.gnu.org/licenses/>.
######################################################################

LONG_DESCRIPTION    = \
"""
Bayesian inference tools for Python.

The package provides tools for building models and performing
posterior inference using variational Bayesian message passing.
"""

NAME         = 'bayespy'
DESCRIPTION  = 'Bayesian inference tools for Python'
AUTHOR       = 'Jaakko Luttinen'
AUTHOR_EMAIL = 'jaakko.luttinen@iki.fi'
URL          = 'http://bayespy.org'
LICENSE      = 'GPLv3'
VERSION      = '0.3'

if __name__ == "__main__":

    from setuptools import setup, find_packages
    
    # Setup for BayesPy
    setup(
          install_requires = ['numpy>=1.8.0', # 1.8 implements broadcasting in numpy.linalg
                              'scipy>=0.13.0', # <0.13 have a bug in special.multigammaln
                              'matplotlib>=1.2.0',
                              'h5py'],
          
          packages         = find_packages(),
          name             = NAME,
          version          = VERSION,
          author           = AUTHOR,
          author_email     = AUTHOR_EMAIL,
          description      = DESCRIPTION,
          license          = LICENSE,
          url              = URL,
          long_description = LONG_DESCRIPTION,
          classifiers =
            [ 
              'Programming Language :: Python :: 3',
              'Development Status :: 3 - Alpha',
              'Environment :: Console',
              'Intended Audience :: Developers',
              'Intended Audience :: Science/Research',
              'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
              'Operating System :: OS Independent',
              'Topic :: Scientific/Engineering',
              'Topic :: Scientific/Engineering :: Information Analysis'
            ]
          )

