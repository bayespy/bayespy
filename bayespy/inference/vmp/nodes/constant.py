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

import numpy as np

from .node import Node

class Constant(Node):
    r"""
    Node for presenting constant values.

    The node wraps arrays into proper node type.
    """

    def __init__(self, moments, x, **kwargs):
        self._moments = moments
        x = np.asanyarray(x)
        # Compute moments
        self.u = self._moments.compute_fixed_moments(x)
        # Dimensions of the moments
        dims = self._moments.compute_dims_from_values(x)
        # Resolve plates
        D = len(dims[0])
        if D > 0:
            plates = np.shape(self.u[0])[:-D]
        else:
            plates = np.shape(self.u[0])
        # Parent constructor
        super().__init__(dims=dims, plates=plates, **kwargs)

    def get_moments(self):
        return self.u
