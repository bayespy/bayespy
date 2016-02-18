################################################################################
# Copyright (C) 2014 Jaakko Luttinen
#
# This file is licensed under the MIT License.
################################################################################


"""
Module for the Poisson distribution node.
"""

import numpy as np
from scipy import special

from .expfamily import ExponentialFamily
from .expfamily import ExponentialFamilyDistribution

from .node import Moments
from .gamma import GammaMoments

from bayespy.utils import misc


class PoissonMoments(Moments):
    """
    Class for the moments of Poisson variables
    """


    dims = ( (), )


    def compute_fixed_moments(self, x):
        """
        Compute the moments for a fixed value
        """
        # Make sure the values are integers in valid range
        x = np.asanyarray(x)
        if not misc.isinteger(x):
            raise ValueError("Count not integer")
        # Now, the moments are just the counts
        return [x]


    @classmethod
    def from_values(cls, x):
        """
        Return the shape of the moments for a fixed value.

        The realizations are scalars, thus the shape of the moment is ().
        """
        return cls()


class PoissonDistribution(ExponentialFamilyDistribution):
    """
    Class for the VMP formulas of Poisson variables.
    """


    def compute_message_to_parent(self, parent, index, u, u_lambda):
        """
        Compute the message to a parent node.
        """
        if index == 0:
            m0 = -1
            m1 = np.copy(u[0])
            return [m0, m1]
        else:
            raise ValueError("Index out of bounds")


    def compute_phi_from_parents(self, u_lambda, mask=True):
        """
        Compute the natural parameter vector given parent moments.
        """
        l = u_lambda[0]
        logl = u_lambda[1]
        phi0 = logl
        return [phi0]


    def compute_moments_and_cgf(self, phi, mask=True):
        """
        Compute the moments and :math:`g(\phi)`.
        """
        u0 = np.exp(phi[0])
        u = [u0]
        g = -u0
        return (u, g)

        
    def compute_cgf_from_parents(self, u_lambda):
        """
        Compute :math:`\mathrm{E}_{q(p)}[g(p)]`
        """
        l = u_lambda[0]
        g = -l
        return g
    

    def compute_fixed_moments_and_f(self, x, mask=True):
        """
        Compute the moments and :math:`f(x)` for a fixed value.
        """

        # Check the validity of x
        x = np.asanyarray(x)
        if not misc.isinteger(x):
            raise ValueError("Values must be integers")
        if np.any(x < 0):
            raise ValueError("Values must be positive")

        # Compute moments
        u0 = np.copy(x)
        u = [u0]

        # Compute f(x)
        f = -special.gammaln(x+1)

        return (u, f)

    
    def random(self, *phi):
        """
        Draw a random sample from the distribution.
        """
        raise NotImplementedError()

    
class Poisson(ExponentialFamily):
    """
    Node for Poisson random variables.

    The node uses Poisson distribution:

    .. math::

        p(x) = \mathrm{Poisson}(x|\lambda)

    where :math:`\lambda` is the rate parameter.

    Parameters
    ----------

    l : gamma-like node or scalar or array

        :math:`\lambda`, rate parameter

    See also
    --------

    Gamma, Exponential
    """

    dims = ( (), )
    _moments = PoissonMoments()
    _parent_moments = [GammaMoments()]
    _distribution = PoissonDistribution()


    def __init__(self, l, **kwargs):
        """
        Create Poisson random variable node
        """
        super().__init__(l, **kwargs)

        
    def __str__(self):
        """
        Print the distribution using standard parameterization.
        """
        l = self.u[0]
        return ("%s ~ Categorical(lambda)\n"
                "  lambda =\n"
                "%s\n"
                % (self.name, l))
