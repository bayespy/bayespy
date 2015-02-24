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


    def random(self, *phi, plates=None):
        """
        Draw a random sample from the distribution.
        """
        p = super().random(*phi, plates=plates)
        return p[...,0]
        

class Beta(Dirichlet):
    r"""
    Node for beta random variables.

    The node models a probability variable :math:`p \in [0,1]` as

    .. math::

        p \sim \mathrm{Beta}(a, b)

    where :math:`a` and :math:`b` are prior counts for success and failure,
    respectively.

    Parameters
    ----------
    
    alpha : (...,2)-shaped array
    
        Two-element vector containing :math:`a` and :math:`b`

    Examples
    --------

    .. code-block:: python

        from bayespy.nodes import Bernoulli, Beta
        p = Beta([1e-3, 1e-3])
        z = Bernoulli(p, plates=(10,))
        z.observe([0, 1, 1, 1, 0, 1, 1, 1, 0, 1])
        p.update()
        import bayespy.plot as bpplt
        import numpy as np
        bpplt.pdf(p, np.linspace(0, 1, num=100))
    """

    _moments = BetaMoments()
    _distribution = BetaDistribution()


    def __init__(self, alpha, **kwargs):
        """
        Create beta node
        """
        super().__init__(alpha, **kwargs)


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

    
    def __str__(self):
        """
        Print the distribution using standard parameterization.
        """
        a = self.phi[0][...,0]
        b = self.phi[0][...,1]
        return ("%s ~ Beta(a, b)\n"
                "  a = \n"
                "%s\n"
                "  b = \n"
                "%s\n"
                % (self.name, a, b))
