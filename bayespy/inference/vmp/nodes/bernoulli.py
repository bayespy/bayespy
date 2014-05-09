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
A module for the Bernoulli distribution node
"""

import numpy as np

from .binomial import (BinomialMoments,
                       BinomialDistribution,
                       Binomial)


class BernoulliMoments(BinomialMoments):
    """
    Class for the moments of Bernoulli variables.
    """


    def __init__(self):
        super().__init__(1)


class BernoulliDistribution(BinomialDistribution):
    """
    Class for the VMP formulas of Bernoulli variables.
    """


    def __init__(self):
        super().__init__(1)


class Bernoulli(Binomial):
    """
    Node for Bernoulli random variables.
    """

    _moments = BernoulliMoments()
    _distribution = BernoulliDistribution()

    
    @classmethod
    def _constructor(cls, p, **kwargs):
        """
        Constructs distribution and moments objects.
        """
        p = cls._ensure_moments(p, cls._parent_moments[0])
        parents = [p]
        return ( parents,
                 kwargs,
                 ( (), ),
                 cls._total_plates(kwargs.get('plates'),
                                   cls._distribution.plates_from_parent(0, p.plates)),
                 cls._distribution, 
                 cls._moments, 
                 cls._parent_moments)


    def show(self):
        """
        Print the distribution using standard parameterization.
        """
        p = 1 / (1 + np.exp(-self.phi[0]))
        print("%s ~ Bernoulli(p)" % self.name)
        print("  p = ")
        print(p)
