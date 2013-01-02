#!/usr/bin/env python

######################################################################
# Copyright (C) 2011,2012 Jaakko Luttinen
#
# This file is licensed under Version 3.0 of the GNU General Public
# License. See LICENSE for a text of the license.
######################################################################

######################################################################
# This file is part of BayesPy.
#
# BayesPy is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
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
Bayesian inference tools.  The package provides tools for building
models and performing posterior inference.
"""

NAME                = 'BayesPy'
DESCRIPTION         = 'Bayesian inference tools for Python'
MAINTAINER          = 'Jaakko Luttinen',
MAINTAINER_EMAIL    = 'jaakko.luttinen@aalto.fi',
URL                 = 'https://github.com/jluttine/bayespy'
LICENSE             = 'GPL'
VERSION             = '0.0.0-dev'

if __name__ == "__main__":

    from distutils.core import setup, Extension
    from Cython.Distutils import build_ext
    import numpy as np
    
    # Sparse distance extension.
    # Use numpy.get_include() in order to use the correct NumPy for building.
    sparse_distance = Extension('bayespy.utils.covfunc.distance',
                                sources=['bayespy/utils/covfunc/distance.pyx',
                                         'bayespy/utils/covfunc/sparse_distance/sparse_distance.c'],
                                         include_dirs=['bayespy/utils/covfunc/sparse_distance',
                                                       np.get_include()])

    # Setup for BayesPy
    setup(requires = ['numpy (>=1.7.0)',
                      'scipy (>=0.11.0)',
                      'scikits.sparse (>=0.1)',
                      'matplotlib (>=1.2.0)'],
          packages = ['bayespy',
                      'bayespy.demos',
                      'bayespy.inference',
                      'bayespy.inference.vmp',
                      'bayespy.inference.vmp.nodes',
                      'bayespy.nodes',
                      'bayespy.nodes.variables',
                      'bayespy.plot',
                      'bayespy.utils',
                      'bayespy.utils.covfunc'],
          ## packages = ['bayespy',
          ##             'bayespy.demos',
          ##             'bayespy.nodes',
          ##             'bayespy.nodes.variables'],
          name = NAME,
          version = VERSION,
          maintainer = MAINTAINER,
          maintainer_email = MAINTAINER_EMAIL,
          description = DESCRIPTION,
          license = LICENSE,
          url = URL,
          long_description = LONG_DESCRIPTION,
          cmdclass = {'build_ext': build_ext},
          ext_modules = [sparse_distance],
          )
