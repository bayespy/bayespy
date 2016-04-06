################################################################################
# Copyright (C) 2014 Jaakko Luttinen
#
# This file is licensed under the MIT License.
################################################################################


"""
A module for the beta distribution node
"""

import numpy as np
import scipy.special as special

from .deterministic import Deterministic
from .dirichlet import (DirichletMoments,
                        DirichletDistribution,
                        Dirichlet)

from .node import Moments, ensureparents


class BetaMoments(DirichletMoments):
    """
    Class for the moments of beta variables.
    """


    def __init__(self):
        super().__init__(2)


    def compute_fixed_moments(self, p):
        """
        Compute the moments for a fixed value
        """
        p = np.asanyarray(p)[...,None] * [1,-1] + [0,1]
        self.dims = ( (2,), )
        return super().compute_fixed_moments(p)


    @classmethod
    def from_values(cls, p):
        """
        Return the shape of the moments for a fixed value.
        """
        return cls()


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

    >>> import warnings
    >>> warnings.filterwarnings('ignore', category=RuntimeWarning)
    >>> from bayespy.nodes import Bernoulli, Beta
    >>> p = Beta([1e-3, 1e-3])
    >>> z = Bernoulli(p, plates=(10,))
    >>> z.observe([0, 1, 1, 1, 0, 1, 1, 1, 0, 1])
    >>> p.update()
    >>> import bayespy.plot as bpplt
    >>> import numpy as np
    >>> bpplt.pdf(p, np.linspace(0, 1, num=100))
    [<matplotlib.lines.Line2D object at 0x...>]
    """

    _moments = BetaMoments()
    _distribution = BetaDistribution()


    def __init__(self, alpha, **kwargs):
        """
        Create beta node
        """
        super().__init__(alpha, **kwargs)


    @classmethod
    def _constructor(cls, alpha, **kwargs):
        """
        Constructs distribution and moments objects.
        """

        retval = super()._constructor(alpha, **kwargs)

        if retval[2] != cls._moments.dims:
            raise ValueError("Parent has wrong dimensionality. Must be a "
                             "two-dimensional vector.")

        return (
            retval[0],
            retval[1],
            retval[2],
            retval[3],
            cls._distribution,
            cls._moments,
            retval[6]
        )


    def complement(self):
        return Complement(self)


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


class Complement(Deterministic):
    """
    Perform 1-p where p is a Beta node.
    """


    _moments = BetaMoments()
    _parent_moments = (BetaMoments(),)


    def __init__(self, p, **kwargs):
        super().__init__(p, dims=p.dims, **kwargs)


    def _compute_message_to_parent(self, index, m, u_p):
        if index != 0:
            raise IndexError()
        m0 = m[0][...,-1::-1]
        return [m0]


    def _compute_moments(self, u_p):
        u0 = u_p[0][...,-1::-1]
        return [u0]
