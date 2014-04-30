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
from bayespy.utils import utils

class MultinomialMoments(Moments):
    """
    Class for the moments of multinomial variables.
    """
    
    ndim_observations = 1

    
    def compute_fixed_moments(self, x):
        """
        Compute the moments for a fixed value

        `x` must be a vector of counts.
        """

        # Check that counts are valid
        x = np.asanyarray(x)
        if not utils.isinteger(x):
            raise ValueError("Counts must be integer")
        if np.any(x < 0):
            raise ValueError("Counts must be non-negative")

        # Moments is just the counts vector
        u0 = x.copy()
        return [u0]
    

    def compute_dims_from_values(self, x):
        """
        Return the shape of the moments for a fixed value.
        """
        D = np.shape(x)[-1]
        return ( (D,), )


class MultinomialDistribution(ExponentialFamilyDistribution):
    """
    Class for the VMP formulas of multinomial variables.
    """    

    ndims = (1,)
    ndims_parents = ( (1,), )


    def __init__(self, trials):
        """
        Create VMP formula node for a multinomial variable

        `trials` is the total number of trials.
        """
        if not utils.isinteger(trials):
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
        """
        Compute the moments and :math:`g(\phi)`.
        """
        # Compute the normalized probabilities in a numerically stable way
        logsum_p = utils.logsumexp(phi[0], axis=-1, keepdims=True)
        logp = phi[0] - logsum_p
        p = np.exp(logp)
        # Because of small numerical inaccuracy, normalize the probabilities
        # again for more accurate results
        N = np.expand_dims(self.N, -1)
        u0 = N * p / np.sum(p, axis=-1, keepdims=True)
        u = [u0]
        g = -N * np.squeeze(logsum_p, axis=-1)
        return (u, g)

    
    def compute_cgf_from_parents(self, u_p):
        """
        Compute :math:`\mathrm{E}_{q(p)}[g(p)]`
        """
        return 0

    
    def compute_fixed_moments_and_f(self, x, mask=True):
        """
        Compute the moments and :math:`f(x)` for a fixed value.
        """

        # Check that counts are valid
        x = np.asanyarray(x)
        if not utils.isinteger(x):
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

    
    def shape_of_value(self, dims):
        """
        Return the shape of realizations
        """
        D = dims[0][0]
        return (D,)


class Multinomial(ExponentialFamily):
    """
    Node for multinomial random variables.
    """
    
    _moments = MultinomialMoments()
    _parent_moments = (DirichletMoments(),)

    @useconstructor
    def __init__(self, p, n=None, **kwargs):
        super().__init__(p, **kwargs)


    @classmethod
    @ensureparents
    def _constructor(cls, p, plates=None, n=None, **kwargs):
        """
        Constructs distribution and moments objects.

        This method is called if useconstructor decorator is used for __init__.
        
        Becase the distribution and moments object depend on the number of
        categories, that is, they depend on the parent node, this method can be
        used to construct those objects.
        """

        # Get the number of categories
        D = p.dims[0][0]

        distribution = MultinomialDistribution(n)

        return (( (D,), ),
                cls._total_plates(plates, 
                                  distribution.plates_from_parent(0, p.plates),
                                  np.shape(n)),
                distribution, 
                cls._moments, 
                cls._parent_moments)

    def random(self):
        """
        Draw a random sample from the distribution.
        """
        raise NotImplementedError()

    def show(self):
        """
        Print the distribution using standard parameterization.
        """
        logsum_p = utils.logsumexp(self.phi[0], axis=-1, keepdims=True)
        p = np.exp(self.phi[0] - logsum_p)
        p /= np.sum(p, axis=-1, keepdims=True)
        print("%s ~ Multinomial(p)" % self.name)
        print("  p = ")
        print(p)
        return
