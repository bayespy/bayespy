################################################################################
# Copyright (C) 2011-2012,2014 Jaakko Luttinen
#
# This file is licensed under the MIT License.
################################################################################


"""
Module for the Dirichlet distribution node.
"""

import numpy as np
from scipy import special

from bayespy.utils import random
from bayespy.utils import misc
from bayespy.utils import linalg

from .stochastic import Stochastic
from .expfamily import ExponentialFamily, ExponentialFamilyDistribution
from .constant import Constant
from .node import Node, Moments, ensureparents


class ConcentrationMoments(Moments):
    """
    Class for the moments of Dirichlet conjugate-prior variables.
    """


    def __init__(self, categories):
        self.categories = categories
        self.dims = ( (categories,), () )
        return


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


    @classmethod
    def from_values(cls, alpha):
        """
        Return the shape of the moments for a fixed value.
        """
        if np.ndim(alpha) < 1:
            raise ValueError("The array must be at least 1-dimensional array.")
        categories = np.shape(alpha)[-1]
        return cls(categories)


class DirichletMoments(Moments):
    """
    Class for the moments of Dirichlet variables.
    """


    def __init__(self, categories):
        self.categories = categories
        self.dims = ( (categories,), )


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


    @classmethod
    def from_values(cls, x):
        """
        Return the shape of the moments for a fixed value.
        """
        if np.ndim(x) < 1:
            raise ValueError("Probabilities must be given as a vector")
        categories = np.shape(x)[-1]
        return cls(categories)


class DirichletDistribution(ExponentialFamilyDistribution):
    """
    Class for the VMP formulas of Dirichlet variables.
    """


    def compute_message_to_parent(self, parent, index, u_self, u_alpha):
        r"""
        Compute the message to a parent node.
        """
        logp = u_self[0]
        m0 = logp
        m1 = 1
        return [m0, m1]


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

        if np.any(np.asanyarray(phi) <= 0):
            raise ValueError("Natural parameters should be positive")

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

    
    def compute_fixed_moments_and_f(self, p, mask=True):
        r"""
        Compute the moments and :math:`f(x)` for a fixed value.

        .. math::

           u(p) =
           \begin{bmatrix}
             \log(p_1)
             \\
             \vdots
             \\
             \log(p_D)
           \end{bmatrix}

        .. math::

           f(p) = - \sum_d \log(p_d)
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
        f = - np.sum(logp, axis=-1)
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


class Concentration(Stochastic):


    _parent_moments = ()


    def __init__(self, D, regularization=True, **kwargs):
        """
        ML estimation node for concentration parameters.

        Parameters
        ----------

        D : int
            Number of categories

        regularization : 2-tuple of arrays (optional)
            "Prior" log-probability and "prior" sample number
        """
        self.D = D
        self.dims = ( (D,), () )
        self._moments = ConcentrationMoments(D)
        super().__init__(dims=self.dims, initialize=False, **kwargs)
        self.u = self._moments.compute_fixed_moments(np.ones(D))
        if regularization is None or regularization is False:
            regularization = [0, 0]
        elif regularization is True:
            # Decent default regularization?
            regularization = [np.log(1/D), 1]
        self.regularization = regularization
        return


    @property
    def regularization(self):
        return self.__regularization


    @regularization.setter
    def regularization(self, regularization):
        if len(regularization) != 2:
            raise ValueError("Regularization must 2-tuple")
        if not misc.is_shape_subset(np.shape(regularization[0]), self.get_shape(0)):
            raise ValueError("Wrong shape")
        if not misc.is_shape_subset(np.shape(regularization[1]), self.get_shape(1)):
            raise ValueError("Wrong shape")
        self.__regularization = regularization
        return


    def _update_distribution_and_lowerbound(self, m):
        r"""
        Find maximum likelihood estimate for the concentration parameter
        """

        a = np.ones(self.D)
        da = np.inf
        logp = m[0] + self.regularization[0]
        N = m[1] + self.regularization[1]

        # Compute sufficient statistic
        mean_logp = logp / N[...,None]

        # It is difficult to estimate values lower than 0.02 because the
        # Dirichlet distributed probability vector starts to give numerically
        # zero random samples for lower values.
        if np.any(np.isinf(mean_logp)):
            raise ValueError(
                "Cannot estimate DirichletConcentration because of infs. This "
                "means that there are numerically zero probabilities in the "
                "child Dirichlet node."
            )

        # Fixed-point iteration
        while np.any(np.abs(da / a) > 1e-5):
            a_new = misc.invpsi(
                special.psi(np.sum(a, axis=-1, keepdims=True))
                + mean_logp
            )
            da = a_new - a
            a = a_new

        self.u = self._moments.compute_fixed_moments(a)

        return


    def initialize_from_value(self, x):
        self.u = self._moments.compute_fixed_moments(x)
        return


    def lower_bound_contribution(self):
        return (
            linalg.inner(self.u[0], self.regularization[0], ndim=1)
            + self.u[1] * self.regularization[1]
        )


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

    _distribution = DirichletDistribution()


    @classmethod
    def _constructor(cls, alpha, **kwargs):
        """
        Constructs distribution and moments objects.
        """
        # Number of categories
        alpha = cls._ensure_moments(alpha, ConcentrationMoments)
        parent_moments = (alpha._moments,)

        parents = [alpha]

        categories = alpha.dims[0][0]
        moments = DirichletMoments(categories)

        return (
            parents,
            kwargs,
            moments.dims,
            cls._total_plates(kwargs.get('plates'), alpha.plates),
            cls._distribution,
            moments,
            parent_moments
        )


    def __str__(self):
        """
        Show distribution as a string
        """
        alpha = self.phi[0]
        return ("%s ~ Dirichlet(alpha)\n"
                "  alpha =\n"
                "%s" % (self.name, alpha))
