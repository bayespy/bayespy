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

"""
Module for the Dirichlet distribution node.
"""

import numpy as np
import scipy.special as special

from bayespy.utils import random

from .expfamily import ExponentialFamily, ExponentialFamilyDistribution
from .constant import Constant
from .node import Node, Moments, ensureparents


class DirichletPriorMoments(Moments):
    """
    Class for the moments of Dirichlet conjugate-prior variables.
    """

    
    def compute_fixed_moments(self, alpha):
        """
        Compute the moments for a fixed value
        """

        alpha = np.asanyarray(alpha)
        if np.ndim(alpha) < 1:
            raise ValueError("The prior sample sizes must be a vector")
        if np.any(alpha < 0):
            raise ValueError("The prior sample sizes must be non-negative")
        
        gammaln_sum = special.gammaln(np.sum(alpha, axis=-1))
        sum_gammaln = np.sum(special.gammaln(alpha), axis=-1)
        z = gammaln_sum - sum_gammaln
        return [alpha, z]

    
    def compute_dims_from_values(self, alpha):
        """
        Return the shape of the moments for a fixed value.
        """
        if np.ndim(alpha) < 1:
            raise ValueError("The array must be at least 1-dimensional array.")
        d = np.shape(alpha)[-1]
        return [(d,), ()]

    
class DirichletMoments(Moments):
    """
    Class for the moments of Dirichlet variables.
    """

    
    def compute_fixed_moments(self, p):
        """
        Compute the moments for a fixed value
        """
        # Check that probabilities are non-negative
        p = np.asanyarray(p)
        if np.ndim(p) < 1:
            raise ValueError("Probabilities must be given as a vector")
        if np.any(p < 0) or np.any(p > 1):
            raise ValueError("Probabilities must be in range [0,1]")
        if not np.allclose(np.sum(p, axis=-1), 1.0):
            raise ValueError("Probabilities must sum to one")
        # Normalize probabilities
        p = p / np.sum(p, axis=-1, keepdims=True)
        # Message is log-probabilities
        logp = np.log(p)
        u = [logp]
        return u

    
    def compute_dims_from_values(self, x):
        """
        Return the shape of the moments for a fixed value.
        """
        if np.ndim(x) < 1:
            raise ValueError("Probabilities must be given as a vector")
        D = np.shape(x)[-1]
        return ( (D,), )


class DirichletDistribution(ExponentialFamilyDistribution):
    """
    Class for the VMP formulas of Dirichlet variables.
    """

    
    def compute_message_to_parent(self, parent, index, u_self, u_alpha):
        r"""
        Compute the message to a parent node.
        """
        raise NotImplementedError()

    
    def compute_phi_from_parents(self, u_alpha, mask=True):
        r"""
        Compute the natural parameter vector given parent moments.
        """
        return [u_alpha[0]]

    
    def compute_moments_and_cgf(self, phi, mask=True):
        r"""
        Compute the moments and :math:`g(\phi)`.

        .. math::

           \overline{\mathbf{u}}  (\boldsymbol{\phi})
           &=
           \begin{bmatrix}
             \psi(\phi_1) - \psi(\sum_d \phi_{1,d})
           \end{bmatrix}
           \\
           g_{\boldsymbol{\phi}} (\boldsymbol{\phi})
           &=
           TODO
        """
        sum_gammaln = np.sum(special.gammaln(phi[0]), axis=-1)
        gammaln_sum = special.gammaln(np.sum(phi[0], axis=-1))
        psi_sum = special.psi(np.sum(phi[0], axis=-1, keepdims=True))
        
        # Moments <log x>
        u0 = special.psi(phi[0]) - psi_sum
        u = [u0]
        # G
        g = gammaln_sum - sum_gammaln

        return (u, g)

    
    def compute_cgf_from_parents(self, u_alpha):
        r"""
        Compute :math:`\mathrm{E}_{q(p)}[g(p)]`
        """
        return u_alpha[1]

    
    def compute_fixed_moments_and_f(self, x, mask=True):
        r"""
        Compute the moments and :math:`f(x)` for a fixed value.
        """
        # Check that probabilities are non-negative
        p = np.asanyarray(p)
        if np.ndim(p) < 1:
            raise ValueError("Probabilities must be given as a vector")
        if np.any(p < 0) or np.any(p > 1):
            raise ValueError("Probabilities must be in range [0,1]")
        if not np.allclose(np.sum(p, axis=-1), 1.0):
            raise ValueError("Probabilities must sum to one")
        # Normalize probabilities
        p = p / np.sum(p, axis=-1, keepdims=True)
        # Message is log-probabilities
        logp = np.log(p)
        u = [logp]
        f = np.nan
        raise NotImplementedError("Check formula for f")
        return (u, f)

    
    def random(self, *phi, plates=None):
        r"""
        Draw a random sample from the distribution.
        """
        return random.dirichlet(phi[0], size=plates)
        

    def compute_gradient(self, g, u, phi):
        r"""
        Compute the moments and :math:`g(\phi)`.

             \psi(\phi_1) - \psi(\sum_d \phi_{1,d})

        Standard gradient given the gradient with respect to the moments, that
        is, given the Riemannian gradient :math:`\tilde{\nabla}`:

        .. math::

           \nabla &=
           \begin{bmatrix}
             (\psi^{(1)}(\phi_1) - \psi^{(1)}(\sum_d \phi_{1,d}) \nabla_1
           \end{bmatrix}
        """
        sum_phi = np.sum(phi[0], axis=-1, keepdims=True)
        d0 = g[0] * (special.polygamma(1, phi[0]) - special.polygamma(1, sum_phi))
        return [d0]


class Dirichlet(ExponentialFamily):
    r"""
    Node for Dirichlet random variables.

    The node models a set of probabilities :math:`\{\pi_0, \ldots, \pi_{K-1}\}`
    which satisfy :math:`\sum_{k=0}^{K-1} \pi_k = 1` and :math:`\pi_k \in [0,1]
    \ \forall k=0,\ldots,K-1`.

    .. math::

        p(\pi_0, \ldots, \pi_{K-1}) = \mathrm{Dirichlet}(\alpha_0, \ldots,
        \alpha_{K-1})

    where :math:`\alpha_k` are concentration parameters.

    The posterior approximation has the same functional form but with different
    concentration parameters.

    Parameters
    ----------
    
    alpha : (...,K)-shaped array
    
        Prior counts :math:`\alpha_k`

    See also
    --------
    
    Beta, Categorical, Multinomial, CategoricalMarkovChain
    """

    _moments = DirichletMoments()
    _parent_moments = (DirichletPriorMoments(),)
    _distribution = DirichletDistribution()
    

    @classmethod
    @ensureparents
    def _constructor(cls, alpha, **kwargs):
        """
        Constructs distribution and moments objects.
        """
        # Number of categories
        D = alpha.dims[0][0]

        parents = [alpha]
        
        return ( parents,
                 kwargs,
                 ( (D,), ),
                 cls._total_plates(kwargs.get('plates'), alpha.plates),
                 cls._distribution, 
                 cls._moments, 
                 cls._parent_moments)

                 
    def __str__(self):
        """
        Show distribution as a string
        """
        alpha = self.phi[0]
        return ("%s ~ Dirichlet(alpha)\n"
                "  alpha =\n"
                "%s" % (self.name, alpha))
