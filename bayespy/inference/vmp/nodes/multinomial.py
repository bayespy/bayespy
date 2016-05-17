################################################################################
# Copyright (C) 2014 Jaakko Luttinen
#
# This file is licensed under the MIT License.
################################################################################


"""
Module for the multinomial distribution node.
"""

import numpy as np
from scipy import special

from .expfamily import ExponentialFamily
from .expfamily import ExponentialFamilyDistribution
from .expfamily import useconstructor
from .dirichlet import Dirichlet, DirichletMoments
from .node import Moments, ensureparents

from bayespy.utils import random
from bayespy.utils import misc
from bayespy.utils import linalg


class MultinomialMoments(Moments):
    """
    Class for the moments of multinomial variables.
    """


    def __init__(self, categories):
        self.categories = categories
        self.dims = ( (categories,), )


    def compute_fixed_moments(self, x):
        """
        Compute the moments for a fixed value

        `x` must be a vector of counts.
        """

        # Check that counts are valid
        x = np.asanyarray(x)
        if not misc.isinteger(x):
            raise ValueError("Counts must be integer")
        if np.any(x < 0):
            raise ValueError("Counts must be non-negative")

        # Moments is just the counts vector
        u0 = x.copy()
        return [u0]


    @classmethod
    def from_values(cls, x):
        D = np.shape(x)[-1]
        return cls( (D,) )


class MultinomialDistribution(ExponentialFamilyDistribution):
    """
    Class for the VMP formulas of multinomial variables.
    """


    def __init__(self, trials):
        """
        Create VMP formula node for a multinomial variable

        `trials` is the total number of trials.
        """
        trials = np.asanyarray(trials)
        if not misc.isinteger(trials):
            raise ValueError("Number of trials must be integer")
        if np.any(trials < 0):
            raise ValueError("Number of trials must be non-negative")
        self.N = trials
        super().__init__()


    def compute_message_to_parent(self, parent, index, u, u_p):
        """
        Compute the message to a parent node.
        """
        if index == 0:
            return [ u[0].copy() ]
        else:
            raise ValueError("Index out of bounds")


    def compute_phi_from_parents(self, u_p, mask=True):
        """
        Compute the natural parameter vector given parent moments.
        """
        logp = u_p[0]
        return [logp]


    def compute_moments_and_cgf(self, phi, mask=True):
        r"""
        Compute the moments and :math:`g(\phi)`.

        .. math::

           \overline{\mathbf{u}}
           = \mathrm{E}[x]
           = N \cdot \begin{bmatrix}
             \frac{e^{\phi_1}}{\sum_i e^{\phi_i}}
             & \cdots &
             \frac{e^{\phi_D}}{\sum_i e^{\phi_i}} \end{bmatrix}
        """
        # Compute the normalized probabilities in a numerically stable way
        (p, logsum_p) = misc.normalized_exp(phi[0])
        N = np.expand_dims(self.N, -1)
        u0 = N * p
        u = [u0]
        g = -np.squeeze(N * logsum_p, axis=-1)
        return (u, g)


    def compute_cgf_from_parents(self, u_p):
        r"""
        Compute :math:`\mathrm{E}_{q(p)}[g(p)]`
        """
        return 0


    def compute_fixed_moments_and_f(self, x, mask=True):
        r"""
        Compute the moments and :math:`f(x)` for a fixed value.
        """

        # Check that counts are valid
        x = np.asanyarray(x)
        if not misc.isinteger(x):
            raise ValueError("Counts must be integers")
        if np.any(x < 0):
            raise ValueError("Counts must be non-negative")
        if np.any(np.sum(x, axis=-1) != self.N):
            raise ValueError("Counts must sum to the number of trials")

        # Moments is just the counts vector
        u0 = x.copy()
        u = [u0]

        f = special.gammaln(self.N+1) - np.sum(special.gammaln(x+1), axis=-1)

        return (u, f)


    def random(self, *phi, plates=None):
        r"""
        Draw a random sample from the distribution.
        """
        (p, _) = misc.normalized_exp(phi[0])
        return random.multinomial(self.N, p, size=plates)


    def compute_gradient(self, g, u, phi):
        r"""
        Compute the Euclidean gradient.

        In order to compute the Euclidean gradient, we first need to derive the
        gradient of the moments with respect to the variational parameters:

        .. math::

           \mathrm{d}\overline{u}_i
           = N \cdot \frac {e^{\phi_i} \mathrm{d}\phi_i \sum_j e^{\phi_j}}
                           {(\sum_k e^{\phi_k})^2}
             - N \cdot \frac {e^{\phi_i} \sum_j e^\phi_j \mathrm{d}\phi_j}
                             {(\sum_k e^{\phi_k})^2}
           = \overline{u}_i \mathrm{d}\phi_i
             - \overline{u}_i \sum_j \frac{\overline{u}_j}{N} \mathrm{d}\phi_j


        Now we can make use of the chain rule. Given the Riemannian gradient
        :math:`\tilde{\nabla}` of the variational lower bound
        :math:`\mathcal{L}` with respect to the variational parameters
        :math:`\phi`, put the above result to the derivative term and
        re-organize the terms to get the Euclidean gradient :math:`\nabla`:

        .. math::

           \mathrm{d}\mathcal{L}
           = \tilde{\nabla}^T \mathrm{d}\overline{\mathbf{u}}
           = \sum_i \tilde{\nabla}_i \mathrm{d}\overline{u}_i
           = \sum_i \tilde{\nabla}_i (
                 \overline{u}_i \mathrm{d}\phi_i
                 - \overline{u}_i \sum_j \frac {\overline{u}_j} {N} \mathrm{d}\phi_j
             )
           = \sum_i \left(\tilde{\nabla}_i \overline{u}_i \mathrm{d}\phi_i
             - \frac{\overline{u}_i}{N} \mathrm{d}\phi_i \sum_j \tilde{\nabla}_j \overline{u}_j \right)
           \equiv \nabla^T \mathrm{d}\phi

        Thus, the Euclidean gradient is:

        .. math::

           \nabla_i = \tilde{\nabla}_i \overline{u}_i - \frac{\overline{u}_i}{N}
                      \sum_j \tilde{\nabla}_j \overline{u}_j

        See also
        --------

        compute_moments_and_cgf : Computes the moments
            :math:`\overline{\mathbf{u}}` given the variational parameters
            :math:`\phi`.

        """
        return u[0] * (g - linalg.inner(g, u[0])[...,None] / self.N)


class Multinomial(ExponentialFamily):
    r"""
    Node for multinomial random variables.

    Assume there are :math:`K` categories and :math:`N` trials each of which
    leads a success for exactly one of the categories.  Given the probabilities
    :math:`p_0,\ldots,p_{K-1}` for the categories, multinomial distribution is
    gives the probability of any combination of numbers of successes for the
    categories.
    
    The node models the number of successes :math:`x_k \in \{0, \ldots, n\}` in
    :math:`n` trials with probability :math:`p_k` for success in :math:`K`
    categories.

    .. math::

        \mathrm{Multinomial}(\mathbf{x}| N, \mathbf{p}) = \frac{N!}{x_0!\cdots
        x_{K-1}!} p_0^{x_0} \cdots p_{K-1}^{x_{K-1}}

    Parameters
    ----------

    n : scalar or array
        :math:`N`, number of trials
    p : Dirichlet-like node or (...,K)-array
        :math:`\mathbf{p}`, probabilities of successes for the categories

    See also
    --------

    Dirichlet, Binomial, Categorical
    """


    def __init__(self, n, p, **kwargs):
        """
        Create Multinomial node.
        """
        super().__init__(n, p, **kwargs)


    @classmethod
    def _constructor(cls, n, p, **kwargs):
        """
        Constructs distribution and moments objects.

        This method is called if useconstructor decorator is used for __init__.

        Becase the distribution and moments object depend on the number of
        categories, that is, they depend on the parent node, this method can be
        used to construct those objects.
        """

        # Get the number of categories
        p = cls._ensure_moments(p, DirichletMoments)
        D = p.dims[0][0]

        moments = MultinomialMoments(D)
        parent_moments = (p._moments,)

        parents = [p]

        distribution = MultinomialDistribution(n)

        return (parents,
                kwargs,
                moments.dims,
                cls._total_plates(kwargs.get('plates'),
                                  distribution.plates_from_parent(0, p.plates),
                                  np.shape(n)),
                distribution,
                moments,
                parent_moments)


    def __str__(self):
        """
        Print the distribution using standard parameterization.
        """
        logsum_p = misc.logsumexp(self.phi[0], axis=-1, keepdims=True)
        p = np.exp(self.phi[0] - logsum_p)
        p /= np.sum(p, axis=-1, keepdims=True)
        return ("%s ~ Multinomial(p)\n"
                "  p = \n"
                "%s\n"
                % (self.name, p))
