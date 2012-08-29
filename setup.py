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

from distutils.core import setup

LONG_DESCRIPTION    = """Bayesian inference tools.  The package provides tools for building
models and performing posterior inference.
"""

NAME                = 'BayesPy'
DESCRIPTION         = 'Bayesian inference tools for Python'
MAINTAINER          = 'Jaakko Luttinen',
MAINTAINER_EMAIL    = 'jaakko.luttinen@aalto.fi',
URL                 = 'https://github.com/jluttine/bayespy'
LICENSE             = 'GPL'
VERSION             = '0.1.0'

if __name__ == "__main__":
    setup(requires = ['numpy', 'scipy'],
          packages = ['bayespy'],
          name = NAME,
          version = VERSION,
          maintainer = MAINTAINER,
          maintainer_email = MAINTAINER_EMAIL,
          description = DESCRIPTION,
          license = LICENSE,
          url = URL,
          long_description = LONG_DESCRIPTION,
          )
