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
A module for the binomial distribution node
"""

import numpy as np
import scipy.special as special

from .expfamily import (ExponentialFamily,
                        ExponentialFamilyDistribution,
                        useconstructor)

from .beta import BetaMoments

from .poisson import PoissonMoments

from .node import (Moments,
                   ensureparents)

from bayespy.utils import misc, random


class BinomialMoments(PoissonMoments):
    """
    Class for the moments of binomial variables
    """

    
    def __init__(self, N):
        self.N = N
        super().__init__()

    
    def compute_fixed_moments(self, x):
        """
        Compute the moments for a fixed value
        """
        # Make sure the values are integers in valid range
        x = np.asanyarray(x)
        if np.any(x > self.N):
            raise ValueError("Invalid count")
        return super().compute_fixed_moments()

    
    def compute_dims_from_values(self, x):
        """
        Return the shape of the moments for a fixed value.

        The realizations are scalars, thus the shape of the moment is ().
        """
        return super().compute_dims_from_values()


class BinomialDistribution(ExponentialFamilyDistribution):
    """
    Class for the VMP formulas of binomial variables.
    """

    
    def __init__(self, N):
        N = np.asanyarray(N)
        if not misc.isinteger(N):
            raise ValueError("Number of trials must be integer")
        if np.any(N < 0):
            raise ValueError("Number of trials must be non-negative")
        self.N = np.asanyarray(N)
        super().__init__()


    def compute_message_to_parent(self, parent, index, u_self, u_p):
        """
        Compute the message to a parent node.
        """
        if index == 0:
            x = u_self[0][...,None]
            n = self.N[...,None]
            m0 = x*[1, -1] + n*[0, 1]
            m = [m0]
            return m
        else:
            raise ValueError("Incorrect parent index")

    
    def compute_phi_from_parents(self, u_p, mask=True):
        """
        Compute the natural parameter vector given parent moments.
        """
        logp0 = u_p[0][...,0]
        logp1 = u_p[0][...,1]
        phi0 = logp0 - logp1
        return [phi0]

    
    def compute_moments_and_cgf(self, phi, mask=True):
        """
        Compute the moments and :math:`g(\phi)`.
        """
        u0 = self.N / (1 + np.exp(-phi[0]))
        g = -self.N * np.log1p(np.exp(phi[0]))
        return ( [u0], g )

        
    def compute_cgf_from_parents(self, u_p):
        """
        Compute :math:`\mathrm{E}_{q(p)}[g(p)]`
        """
        logp0 = u_p[0][...,0]
        logp1 = u_p[0][...,1]
        return self.N * logp1

    
    def compute_fixed_moments_and_f(self, x, mask=True):
        """
        Compute the moments and :math:`f(x)` for a fixed value.
        """
        # Make sure the values are integers in valid range
        x = np.asanyarray(x)
        if not misc.isinteger(x):
            raise ValueError("Counts must be integer")
        if np.any(x < 0) or np.any(x > self.N):
            raise ValueError("Invalid count")
        # Now, the moments are just the counts
        u = [x]
        f = (special.gammaln(self.N+1) -
             special.gammaln(x+1) -
             special.gammaln(self.N-x+1))
        return (u, f)

    
    def random(self, *phi, plates=None):
        """
        Draw a random sample from the distribution.
        """
        p = random.logodds_to_probability(phi[0])
        return np.random.binomial(self.N, p, size=plates)

    
class Binomial(ExponentialFamily):
    r"""
    Node for binomial random variables.

    The node models the number of successes :math:`x \in \{0, \ldots, n\}` in
    :math:`n` trials with probability :math:`p` for success:

    .. math::

        x \sim \mathrm{Binomial}(n, p).

    Parameters
    ----------

    n : scalar or array
    
        Number of trials
        
    p : beta-like node or scalar or array
    
        Probability of a success in a trial

    Examples
    --------

    .. code-block:: python

        from bayespy.nodes import Binomial, Beta
        p = Beta([1e-3, 1e-3])
        x = Binomial(10, p)
        x.observe(7)
        p.update()
        import bayespy.plot as bpplt
        import numpy as np
        bpplt.pdf(p, np.linspace(0, 1, num=100))

    See also
    --------

    Bernoulli, Multinomial, Beta
    """

    
    _parent_moments = (BetaMoments(),)


    def __init__(self, n, p, **kwargs):
        """
        Create binomial node
        """
        super().__init__(n, p, **kwargs)


    @classmethod
    def _constructor(cls, n, p, **kwargs):
        """
        Constructs distribution and moments objects.
        """
        p = cls._ensure_moments(p, cls._parent_moments[0])
        parents = [p]
        moments = BinomialMoments(n)
        distribution = BinomialDistribution(n)
        return ( parents,
                 kwargs,
                 ( (), ),
                 cls._total_plates(kwargs.get('plates'),
                                   distribution.plates_from_parent(0, p.plates),
                                   np.shape(n)),
                 distribution, 
                 moments, 
                 cls._parent_moments)

    
    def __str__(self):
        """
        Print the distribution using standard parameterization.
        """
        p = 1 / (1 + np.exp(-self.phi[0]))
        n = self._distribution.N
        return ("%s ~ Binomial(n, p)\n"
                "  n = \n"
                "%s\n"
                "  p = \n"
                "%s\n"
                % (self.name, n, p))
