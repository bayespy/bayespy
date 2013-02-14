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

import numpy as np

from .node import Node


class ConstantNumeric(Node):

    def __init__(self, x, ndim, **kwargs):
        # Compute moments
        self.u = [np.asarray(x)]
        # Dimensions and plates of the moments
        ind_dim = np.ndim(x) - ndim
        dims = np.shape(x)[ind_dim:]
        plates = np.shape(x)[:ind_dim]
        # Parent constructor
        super().__init__(dims=dims, plates=plates, **kwargs)

    def get_moments(self):
        return self.u

def Constant(distribution):

    class _Constant(Node):

        def __init__(self, x, **kwargs):
            x = np.asanyarray(x)
            # Compute moments
            self.u = distribution.compute_fixed_moments(x)
            # Dimensions of the moments
            dims = distribution.compute_dims_from_values(x)
            # Number of plate axes
            plates_ndim = np.ndim(x) - distribution.ndim_observations
            plates = np.shape(x)[:plates_ndim]
            # Parent constructor
            super().__init__(dims=dims, plates=plates, **kwargs)

        @staticmethod
        def compute_fixed_moments(x):
            """ Compute u(x) for given x. """
            return distribution.compute_fixed_moments(x)

        @staticmethod
        def compute_fixed_u_and_f(x):
            """ Compute u(x) and f(x) for given x. """
            raise Exception("I think constants should NOT need this function?")
            return distribution.compute_fixed_u_and_f(x)

        def get_moments(self):
            return self.u
        
    return _Constant
    

