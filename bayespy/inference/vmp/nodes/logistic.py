######################################################################
# Copyright (C) 2015 Jaakko Luttinen
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
Module for Bernoulli using the logistic function for Gaussian
"""

import numpy as np

from .node import ensureparents
from .expfamily import (ExponentialFamily,
                        useconstructor)
from .multinomial import (MultinomialMoments,
                          MultinomialDistribution,
                          Multinomial)
from .dirichlet import DirichletMoments
from .gaussian import GaussianMoments

from bayespy.utils import random
from bayespy.utils import misc


class CategoricalDistribution(MultinomialDistribution):
    """
    Class for the VMP formulas of categorical variables.
    """    

    def __init__(self, categories):
        """
        Create VMP formula node for a categorical variable

        `categories` is the total number of categories.
        """
        if not isinstance(categories, int):
            raise ValueError("Number of categories must be integer")
        if categories < 0:
            raise ValueError("Number of categoriess must be non-negative")
        self.D = categories
        super().__init__(1)
        

    def compute_message_to_parent(self, parent, index, u, u_p):
        """
        Compute the message to a parent node.
        """
        return super().compute_message_to_parent(parent, index, u, u_p)


    def compute_phi_from_parents(self, u_p, mask=True):
        """
        Compute the natural parameter vector given parent moments.
        """
        return super().compute_phi_from_parents(u_p, mask=mask)


    def compute_moments_and_cgf(self, phi, mask=True):
        """
        Compute the moments and :math:`g(\phi)`.
        """
        return super().compute_moments_and_cgf(phi, mask=mask)

        
    def compute_cgf_from_parents(self, u_p):
        """
        Compute :math:`\mathrm{E}_{q(p)}[g(p)]`
        """
        return super().compute_cgf_from_parents(u_p)
    

    def compute_fixed_moments_and_f(self, x, mask=True):
        """
        Compute the moments and :math:`f(x)` for a fixed value.
        """

        # Check the validity of x
        x = np.asanyarray(x)
        if not misc.isinteger(x):
            raise ValueError("Values must be integers")
        if np.any(x < 0) or np.any(x >= self.D):
            raise ValueError("Invalid category index")

        # Form a binary matrix with only one non-zero (1) in the last axis
        u0 = np.zeros((np.size(x), self.D))
        u0[[np.arange(np.size(x)), np.ravel(x)]] = 1
        u0 = np.reshape(u0, np.shape(x) + (self.D,))
        u = [u0]

        # f(x) is zero
        f = 0

        return (u, f)

    
    def random(self, *phi, plates=None):
        """
        Draw a random sample from the distribution.
        """
        logp = phi[0]
        logp -= np.amax(logp, axis=-1, keepdims=True)
        p = np.exp(logp)
        return random.categorical(p, size=plates)

    
class Logistic(ExponentialFamily):
    r"""

    :cite:`Jaakkola:2000`

    The true probability density function:

    .. math::

       p(z=1|x) = g(x)
       \\
       p(z=0|x) = g(-x)

    which can be written as:

    .. math::

       p(z|x) = g(H_z)

    where :math:`H_z=(2z-1)x` and :math:`g(x)` is the logistic function:

    .. math::

       g(x) = \frac{1} {1 + e^{-x}}

    The log of the logistic function:

    .. math::
    
       \log g(x) = -\log(1 + e^{-x}) = \frac{x}{2} - \log(e^{x/2} + e^{-x/2})

    The latter term:

    .. math::

       f(x) = -\log(e^{x/2} + e^{-x/2})

    This is a convex function in the variable :math:`x^2`, thus it can be
    bounded globally with a first order Taylor expansion in the variable x^2:
    
    .. math::

       f(x) &\geq f(\xi) + \frac {\partial f(\xi)}{\partial(\xi^2)} (x^2 -
       \xi^2)
       \\
       &= -\frac{\xi}{2} + \log g(\xi) + \frac{1}{4\xi}\tanh(\frac{\xi}{2}) (x^2
       - \xi^2)

    Thus, the variational lower bound for the probability density function is:

    .. math::

       p(z|x) \geq g(xi) \exp( \frac{H_z-\xi}{2} - \lambda(\xi) (H_z^2 - \xi^2))

    and in log form:

    .. math::

       \log p(z|x) \geq \log g(xi) + ( \frac{H_z-\xi}{2} - \lambda(\xi) (H_z^2 -
       \xi^2) )

    where

    .. math::

       \lambda(\xi) = \frac {\tanh(\xi/2)} {4\xi}

    Now, this log lower bound is quadratic with respect to :math:`H_z`, thus it
    is quadratic with respect to :math:`x` and it is conjugate with the Gaussian
    distribution.  Re-organize terms:

    .. math::

       \log p(z|x) &\geq -\lambda(\xi)(2z-1)^2 x^2 + zx - \frac{1}{2}x -
       \frac{1}{2}\xi + \lambda(\xi) \xi^2 + \log g(\xi)
       \\
       &= -\lambda(\xi)(2z+1) x^2 + zx - \frac{1}{2}x -
       \frac{1}{2}\xi + \lambda(\xi) \xi^2 + \log g(\xi)
       \\
       &= z (-2\lambda(\xi) x^2 + x) - \lambda(\xi) x^2 - \frac{1}{2}x -
       \frac{1}{2}\xi + \lambda(\xi) \xi^2 + \log g(\xi)

    where we have used :math:`z^2=z`.

    See also
    --------
    Bernoulli, GaussianARD
    """


    _parent_moments = (
        GaussianMoments(()),
    )


    def __init__(self, x, **kwargs):
        """
        """
        super().__init__(x, **kwargs)


    @classmethod
    @ensureparents
    def _constructor(cls, x, **kwargs):
        """
        Constructs distribution and moments objects.
        """

        # Get the number of categories
        D = p.dims[0][0]

        parents = [p]
        moments = CategoricalMoments(D)
        distribution = CategoricalDistribution(D)

        return (parents,
                kwargs,
                ( (D,), ),
                cls._total_plates(kwargs.get('plates'),
                                  distribution.plates_from_parent(0, p.plates)),
                distribution,
                moments,
                cls._parent_moments)


    def __str__(self):
        """
        Print the distribution using standard parameterization.
        """
        raise NotImplementedError()
