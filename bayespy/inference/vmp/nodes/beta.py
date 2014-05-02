######################################################################
# Copyright (C) 2014 Jaakko Luttinen
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

"""
A module for the beta distribution node
"""

import numpy as np
import scipy.special as special

from .dirichlet import (DirichletMoments,
                        DirichletDistribution,
                        Dirichlet)

from .node import Moments, ensureparents


class BetaMoments(DirichletMoments):
    """
    Class for the moments of beta variables.
    """

    # Realizations are scalars
    ndim_observations = 0
    

    def compute_fixed_moments(self, p):
        """
        Compute the moments for a fixed value
        """
        p = np.asanyarray(p)[...,None] * [1,-1] + [0,1]
        return super().compute_fixed_moments(p)

    
    def compute_dims_from_values(self, p):
        """
        Return the shape of the moments for a fixed value.
        """
        return ( (2,), )


class BetaDistribution(DirichletDistribution):
    """
    Class for the VMP formulas of beta variables.

    Although the realizations are scalars (probability p), the moments is a
    two-dimensional vector: [log(p), log(1-p)].
    """

    # Moments is a vector
    ndims = (1,)
    # Parent's moments is a vector
    ndims_parents = ( (1,), )


    def compute_message_to_parent(self, parent, index, u_self, u_alpha):
        """
        Compute the message to a parent node.
        """
        return super().compute_message_to_parent(parent, index, u_self, u_alpha)

    
    def compute_phi_from_parents(self, u_alpha, mask=True):
        """
        Compute the natural parameter vector given parent moments.
        """
        return super().compute_phi_from_parents(u_alpha, mask=mask)

    
    def compute_moments_and_cgf(self, phi, mask=True):
        """
        Compute the moments and :math:`g(\phi)`.
        """
        return super().compute_moments_and_cgf(phi, mask)

    
    def compute_cgf_from_parents(self, u_alpha):
        """
        Compute :math:`\mathrm{E}_{q(p)}[g(p)]`
        """
        return super().compute_cgf_from_parents(u_alpha)

    
    def compute_fixed_moments_and_f(self, p, mask=True):
        """
        Compute the moments and :math:`f(x)` for a fixed value.
        """
        p = np.asanyarray(p)[...,None] * [1,-1] + [0,1]
        return super().compute_fixed_moments_and_f(p, mask=mask)


class Beta(Dirichlet):
    """
    Node for beta random variables.
    """

    _moments = BetaMoments()
    _distribution = BetaDistribution()


    @classmethod
    @ensureparents
    def _constructor(cls, alpha, **kwargs):
        """
        Constructs distribution and moments objects.
        """

        D = alpha.dims[0][0]
        if D != 2:
            raise ValueError("Parent has wrong dimensionality. Must be a "
                             "two-dimensional vector.")

        return super()._constructor(alpha, **kwargs)

    
    def show(self):
        """
        Print the distribution using standard parameterization.
        """
        a = self.phi[0][...,0]
        b = self.phi[0][...,1]
        print("%s ~ Beta(a, b)" % self.name)
        print("  a = ")
        print(a)
        print("  b = ")
        print(b)
