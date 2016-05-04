################################################################################
# Copyright (C) 2011-2012,2014 Jaakko Luttinen
#
# This file is licensed under the MIT License.
################################################################################


"""
Module for the gamma distribution node.
"""

import numpy as np
import scipy.special as special

from .node import Node, Moments, ensureparents
from .deterministic import Deterministic
from .stochastic import Stochastic
from .expfamily import ExponentialFamily, ExponentialFamilyDistribution
from .constant import Constant

from bayespy.utils import misc
from bayespy.utils import random


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


    dims = ( (), () )


    def compute_fixed_moments(self, a):
        """
        Compute the moments for a fixed value
        """
        a = np.asanyarray(a)
        if np.any(a <= 0):
            raise ValueError("Shape parameter must be positive")
        u0 = a
        u1 = special.gammaln(a)
        return [u0, u1]


    @classmethod
    def from_values(cls, a):
        """
        Return the shape of the moments for a fixed value.
        """
        return cls()


class GammaMoments(Moments):
    """
    Class for the moments of gamma variables.
    """

    dims = ( (), () )


    def compute_fixed_moments(self, x):
        """
        Compute the moments for a fixed value
        """
        if np.any(x < 0):
            raise ValueError("Values must be positive")
        u0 = x
        u1 = np.log(x)
        return [u0, u1]


    @classmethod
    def from_values(cls, x):
        """
        Return the shape of the moments for a fixed value.
        """
        return cls()


class GammaDistribution(ExponentialFamilyDistribution):
    """
    Class for the VMP formulas of gamma variables.
    """    
    

    def compute_message_to_parent(self, parent, index, u_self, u_a, u_b):
        r"""
        Compute the message to a parent node.
        """
        x = u_self[0]
        logx = u_self[1]

        if index == 0:
            b = u_b[0]
            logb = u_b[1]
            return [logx + logb,
                    -1]
        elif index == 1:
            a = u_a[0]
            return [-x,
                    a]
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
        x = np.asanyarray(x)
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
        return random.gamma(phi[1], -1/phi[0], size=plates)

    
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


    def diag(self):
        return self.as_diagonal_wishart()


class GammaShape(Stochastic):
    """
    ML point estimator for the shape parameter of the gamma distribution
    """

    dims = ( (), () )
    _moments = GammaPriorMoments()
    _parent_moments = ()


    def __init__(self, **kwargs):
        """
        Create gamma random variable node
        """
        super().__init__(dims=self.dims, initialize=False, **kwargs)
        self.u = self._moments.compute_fixed_moments(1)
        return


    def _update_distribution_and_lowerbound(self, m):
        r"""
        Find maximum likelihood estimate for the shape parameter

        Messages from children appear in the lower bound as

        .. math::

           m_0 \cdot x +  m_1 \cdot \log(\Gamma(x))

        Take derivative, put it zero and solve:

        .. math::

           m_0 + m_1 \cdot d\log(\Gamma(x)) &= 0
           \\
           m_0 + m_1 \cdot \psi(x) &= 0
           \\
           x &= \psi^{-1}(-\frac{m_0}{m_1})

        where :math:`\psi^{-1}` is the inverse digamma function.
        """

        # Maximum likelihood estimate
        x = misc.invpsi(-m[0]/m[1])

        # Compute moments
        self.u = self._moments.compute_fixed_moments(x)

        return


    def initialize_from_value(self, x):
        self.u = self._moments.compute_fixed_moments(x)
        return


    def lower_bound_contribution(self):
        return 0


class _GammaToDiagonalWishart(Deterministic):
    """
    Transform a set of gamma scalars into a diagonal Wishart matrix.

    The last plate is used as the diagonal dimension.
    """


    _parent_moments = [GammaMoments()]


    @ensureparents
    def __init__(self, alpha, **kwargs):

        # Check for constant
        if misc.is_numeric(alpha):
            alpha = Constant(Gamma)(alpha)

        if len(alpha.plates) == 0:
            raise Exception("Gamma variable needs to have plates in "
                            "order to be used as a diagonal Wishart.")
        D = alpha.plates[-1]

        # FIXME: Put import here to avoid circular dependency import
        from .wishart import WishartMoments
        self._moments = WishartMoments((D,))
        dims = ( (D,D), () )

        # Construct the node
        super().__init__(alpha,
                         dims=self._moments.dims,
                         **kwargs)


    def _plates_to_parent(self, index):
        D = self.dims[0][0]
        return self.plates + (D,)

    def _plates_from_parent(self, index):
        return self.parents[index].plates[:-1]

    @staticmethod
    def _compute_weights_to_parent(index, weights):
        return weights[..., np.newaxis]

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
