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
Module for the categorical distribution node.
"""

import numpy as np

from .node import ensureparents
from .expfamily import (ExponentialFamily,
                        useconstructor)
from .multinomial import (MultinomialMoments,
                          MultinomialDistribution,
                          Multinomial)
from .dirichlet import DirichletMoments

from bayespy.utils import random
from bayespy.utils import misc


class CategoricalMoments(MultinomialMoments):
    """
    Class for the moments of categorical variables.
    """

    
    def __init__(self, categories):
        """
        Create moments object for categorical variables
        """
        self.D = categories
        super().__init__()

        
    def compute_fixed_moments(self, x):
        """
        Compute the moments for a fixed value
        """

        # Check that x is valid
        x = np.asanyarray(x)
        if not misc.isinteger(x):
            raise ValueError("Values must be integers")
        if np.any(x < 0) or np.any(x >= self.D):
            raise ValueError("Invalid category index")

        u0 = np.zeros((np.size(x), self.D))
        u0[[np.arange(np.size(x)), np.ravel(x)]] = 1
        u0 = np.reshape(u0, np.shape(x) + (self.D,))

        return [u0]
    

    def compute_dims_from_values(self, x):
        """
        Return the shape of the moments for a fixed value.

        The observations are scalar.
        """
        return ( (self.D,), )


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

    
class Categorical(ExponentialFamily):
    r"""
    Node for categorical random variables.
    
    The node models a categorical random variable :math:`x \in \{0,\ldots,K-1\}`
    with prior probabilities :math:`\{p_0, \ldots, p_{K-1}\}` for each category:
    
    .. math::

        p(x=k) = p_k \quad \text{for } k\in \{0,\ldots,K-1\}.

    Parameters
    ----------
    
    p : Dirichlet-like node or (...,K)-array
    
        Probabilities for each category

    See also
    --------
    Bernoulli, Multinomial, Dirichlet
    """


    _parent_moments = [DirichletMoments()]


    def __init__(self, p, **kwargs):
        """
        Create Categorical node.
        """
        super().__init__(p, **kwargs)


    @classmethod
    @ensureparents
    def _constructor(cls, p, **kwargs):
        """
        Constructs distribution and moments objects.

        This method is called if useconstructor decorator is used for __init__.
        
        Becase the distribution and moments object depend on the number of
        categories, that is, they depend on the parent node, this method can be
        used to construct those objects.
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
        p = self.u[0]
        return ("%s ~ Categorical(p)\n"
                "  p = \n"
                "%s\n"
                % (self.name, p))
