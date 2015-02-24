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
Module for the gamma distribution node.
"""

import numpy as np
import scipy.special as special

from .node import Node
from .deterministic import Deterministic
from .expfamily import ExponentialFamily, ExponentialFamilyDistribution
from .constant import Constant
from .node import Moments
from .wishart import WishartMoments

from bayespy.utils import misc


def diagonal(alpha):
    """
    Create a diagonal Wishart node from a Gamma node.
    """
    return _GammaToDiagonalWishart(alpha,
                                   name=alpha.name + " as Wishart")


class GammaPriorMoments(Moments):
    """
    Class for the moments of the shape parameter in gamma distributions.
    """
    
    
    def compute_fixed_moments(self, a):
        """
        Compute the moments for a fixed value
        """
        u0 = a
        u1 = special.gammaln(a)
        return [u0, u1]
    

    def compute_dims_from_values(self, a):
        """
        Return the shape of the moments for a fixed value.
        """
        return ( (), () )
    

class GammaMoments(Moments):
    """
    Class for the moments of gamma variables.
    """
    
    
    def compute_fixed_moments(self, x):
        """
        Compute the moments for a fixed value
        """
        if np.any(x < 0):
            raise ValueError("Values must be positive")
        u0 = x
        u1 = np.log(x)
        return [u0, u1]


    def compute_dims_from_values(self, x):
        """
        Return the shape of the moments for a fixed value.
        """
        return ( (), () )
    

class GammaDistribution(ExponentialFamilyDistribution):
    """
    Class for the VMP formulas of gamma variables.
    """    
    

    def compute_message_to_parent(self, parent, index, u_self, *u_parents):
        r"""
        Compute the message to a parent node.
        """
        if index == 0:
            raise Exception("No analytic solution exists")
        elif index == 1:
            return [-u[0],
                    u_parents[0][0]]
        else:
            raise ValueError("Index out of bounds")


    def compute_phi_from_parents(self, *u_parents, mask=True):
        r"""
        Compute the natural parameter vector given parent moments.
        """
        return [-u_parents[1][0],
                1*u_parents[0][0]]
    

    def compute_moments_and_cgf(self, phi, mask=True):
        r"""
        Compute the moments and :math:`g(\phi)`.

        .. math::

           \overline{\mathbf{u}}  (\boldsymbol{\phi})
           &=
           \begin{bmatrix}
             - \frac{\phi_2} {\phi_1}
             \\
             \psi(\phi_2) - \log(-\phi_1)
           \end{bmatrix}
           \\
           g_{\boldsymbol{\phi}} (\boldsymbol{\phi})
           &=
           TODO
        """
        with np.errstate(invalid='raise', divide='raise'):
            log_b = np.log(-phi[0])
            u0 = phi[1] / (-phi[0])
        u1 = special.digamma(phi[1]) - log_b
        u = [u0, u1]
        g = phi[1] * log_b - special.gammaln(phi[1])
        return (u, g)


    def compute_cgf_from_parents(self, *u_parents):
        r"""
        Compute :math:`\mathrm{E}_{q(p)}[g(p)]`
        """
        a = u_parents[0][0]
        gammaln_a = u_parents[0][1] #special.gammaln(a)
        b = u_parents[1][0]
        log_b = u_parents[1][1]
        g = a * log_b - gammaln_a
        return g

            
    def compute_fixed_moments_and_f(self, x, mask=True):
        r"""
        Compute the moments and :math:`f(x)` for a fixed value.
        """
        if np.any(x < 0):
            raise ValueError("Values must be positive")
        logx = np.log(x)
        u = [x, logx]
        f = -logx
        return (u, f)


    def random(self, *phi, plates=None):
        r"""
        Draw a random sample from the distribution.
        """
        return np.random.gamma(phi[1],
                               -1/phi[0],
                               size=plates)

    
    def compute_gradient(self, g, u, phi):
        r"""
        Compute the moments and :math:`g(\phi)`.

        .. math::

           \mathrm{d}\overline{\mathbf{u}} &=
           \begin{bmatrix}
             - \frac{\mathrm{d}\phi_2} {phi_1} + \frac{\phi_2}{\phi_1^2} \mathrm{d}\phi_1
             \\
             \psi^{(1)}(\phi_2) \mathrm{d}\phi_2 - \frac{1}{\phi_1} \mathrm{d}\phi_1
           \end{bmatrix}


        Standard gradient given the gradient with respect to the moments, that
        is, given the Riemannian gradient :math:`\tilde{\nabla}`:

        .. math::

           \nabla =
           \begin{bmatrix}
             \nabla_1 \frac{\phi_2}{\phi_1^2} - \nabla_2 \frac{1}{\phi_1}
             \\
             \nabla_2 \psi^{(1)}(\phi_2) - \nabla_1 \frac {1} {\phi_1}
           \end{bmatrix}
        """
        d0 = g[0] * phi[1] / phi[0]**2 - g[1] / phi[0]
        d1 = g[1] * special.polygamma(1, phi[1]) - g[0] / phi[0]
        return [d0, d1]


class Gamma(ExponentialFamily):
    """
    Node for gamma random variables.

    Parameters
    ----------
    
    a : scalar or array
    
        Shape parameter
        
    b : gamma-like node or scalar or array
    
        Rate parameter
    """

    dims = ( (), () )
    _distribution = GammaDistribution()
    _moments = GammaMoments()
    _parent_moments = (GammaPriorMoments(),
                       GammaMoments())

            
    def __init__(self, a, b, **kwargs):
        """
        Create gamma random variable node
        """
        super().__init__(a, b, **kwargs)


    def __str__(self):
        """
        Print the distribution using standard parameterization.
        """
        a = self.phi[1]
        b = -self.phi[0]
        return ("%s ~ Gamma(a, b)\n"
                "  a =\n"
                "%s\n"
                "  b =\n"
                "%s\n"
                % (self.name, a, b))


    def as_diagonal_wishart(self):
        return _GammaToDiagonalWishart(self,
                                       name=self.name + " as Wishart")


class _GammaToDiagonalWishart(Deterministic):
    """
    Transform a set of gamma scalars into a diagonal Wishart matrix.

    The last plate is used as the diagonal dimension.
    """

    _moments = WishartMoments()
    _parent_moments = [GammaMoments()]
    
    
    def __init__(self, alpha, **kwargs):

        # Check for constant
        if misc.is_numeric(alpha):
            alpha = Constant(Gamma)(alpha)

        # Remove the last plate...
        #plates = alpha.plates[:-1]
        # ... and use it as the dimensionality of the Wishart
        # distribution
        if len(alpha.plates) == 0:
            raise Exception("Gamma variable needs to have plates in "
                            "order to be used as a diagonal Wishart.")
        D = alpha.plates[-1]
        dims = ( (D,D), () )

        # Construct the node
        super().__init__(alpha,
        #plates=plates,
                         dims=dims,
                         **kwargs)
        
    def _plates_to_parent(self, index):
        D = self.dims[0][0]
        return self.plates + (D,)

    def _plates_from_parent(self, index):
        return self.parents[index].plates[:-1]

    @staticmethod
    def _compute_mask_to_parent(index, mask):
        return mask[..., np.newaxis]

    def get_moments(self):
        u = self.parents[0].get_moments()

        # Form a diagonal matrix from the gamma variables
        return [np.identity(self.dims[0][0]) * u[0][...,np.newaxis],
                np.sum(u[1], axis=(-1))]

    @staticmethod
    def _compute_message_to_parent(index, m_children, *u_parents):

        # Take the diagonal
        m0 = np.einsum('...ii->...i', m_children[0])
        m1 = np.reshape(m_children[1], np.shape(m_children[1]) + (1,))

        return [m0, m1]
        

    ## def _compute_message_and_mask_to_parent(self, index, m_children, *u_parents):
        
    ##     m = self._message_from_children()
    ##     #(m, mask) = self.message_from_children()

    ##     # Take the diagonal
    ##     m[0] = np.einsum('...ii->...i', m[0])
    ##     # TODO/FIXME: I think m[1] is wrong..
    ##     m[1] = np.reshape(m[1], np.shape(m[1]) + (1,))
    ##     # m[1] is ok

    ##     mask = self._compute_mask_to_parent(index, self.mask)
    ##     #mask = mask[...,np.newaxis]

    ##     return (m, mask)
    ##     #return m
        


