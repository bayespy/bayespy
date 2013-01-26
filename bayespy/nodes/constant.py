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

import numpy as np

from .node import Node

def Constant(distribution):
    class _Constant(Node):

        @staticmethod
        def compute_fixed_moments(x):
            """ Compute u(x) for given x. """
            return distribution.compute_fixed_moments(x)
        
        @staticmethod
        def compute_fixed_u_and_f(x):
            """ Compute u(x) and f(x) for given x. """
            return distribution.compute_fixed_u_and_f(x)
        
        def __init__(self, x, **kwargs):
            # Compute moments
            self.u = distribution.compute_fixed_moments(x)
            # Dimensions of the moments
            dims = distribution.compute_dims_from_values(x)
            # Number of plate axes
            plates_ndim = np.ndim(x) - distribution.ndim_observations
            plates = np.shape(x)[:plates_ndim]
            # Parent constructor
            super().__init__(dims=dims, plates=plates, **kwargs)
            
        def get_moments(self):
            return self.u
        
    return _Constant
        
