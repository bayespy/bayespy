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

import numpy as np

from .expfamily import ExponentialFamily
from .categorical import CategoricalStatistics
from .dirichlet import Dirichlet, \
                       DirichletStatistics
from .node import Statistics

class CategoricalMarkovChainStatistics(Statistics):

    ndim_observations = 0

    def __init__(self, categories):
        self.D = categories

    def converter(self, statistics_class):
        """
        Returns a node class which converts the node's statistics to another
        """
        if statistics_class is CategoricalStatistics:
            raise NotImplementedError()
        return super().converter(statistics_class)

    def compute_fixed_moments(self, x):
        raise NotImplementedError("compute_fixed_moments not implemented for "
                                  "%s" 
                                  % (self.__class__.__name__))
    
    def compute_dims_from_values(self, x):
        raise NotImplementedError("compute_dims_from_values not implemented "
                                  "for %s"
                                  % (self.__class__.__name__))


class CategoricalMarkovChainDistribution():

    ndims = (1, 3)
    ndims_parents = None

    def __init__(self, categories, states):
        self.categories = categories
        self.states = states
        return

    #
    # The following methods are for ExponentialFamily distributions
    #

    def compute_message_to_parent(self, index, u_self, u_p0, u_P):
        raise NotImplementedError()

    def compute_phi_from_parents(self, u_p0, u_P, mask=True):
        phi0 = u_p0[0]
        phi1 = utils.atleast_nd(u_P[0], 3)
        if np.shape(phi1)[-3] != self.states:
            if np.shape(phi1)[-3] != 1:
                raise ValueError("Moments of parent P have wrong shape")
            phi1 = np.repeat(phi1, self.states, axis=-3)
        return [phi0, phi1]

    def compute_moments_and_cgf(self, phi, mask=True):
        raise NotImplementedError()

    #
    # The following methods are for Mixture class
    #

    def compute_cgf_from_parents(self, ):
        raise NotImplementedError()
        
    def compute_fixed_moments_and_f(self, x, mask=True):
        raise NotImplementedError()

    def compute_dims(self):
        raise NotImplementedError()

class CategoricalMarkovChain(ExponentialFamily):
    
    _parent_statistics = (DirichletStatistics(),
                          DirichletStatistics())

    ndims = (1, 3)

    @useconstructor
    def __init__(self, p0, P, **kwargs):

        super().__init__(p0, P, **kwargs)

        return

    @classmethod
    def _constructor(cls, p0, P, **kwargs):
        p0 = cls._ensure_statistics(p0, cls._parent_statistics[0])
        P = cls._ensure_statistics(P, cls._parent_statistics[1])

        # Number of categories
        D = p0.dims[0][0]
        # Number of states
        if len(P.plates) < 2:
            if states is None:
                raise ValueError("Could not infer the length of the Markov "
                                 "chain")
            N = int(states)
        else:
            if P.plates[-2] == 1:
                if states is None:
                    N = 1
                else:
                    N = int(states)
            else:
                if states is not None and P.plates[-2] != states:
                    raise ValueError("Given length of the Markov chain is "
                                     "inconsistent with the transition "
                                     "probability matrix")
                N = P.plates[-2]


        distribution = CategoricalMarkovChainDistribution(D, N)
        
        statistics = CategoricalMarkovChainStatistics(D)

        parent_statistics = cls._parent_statistics

        return (distribution, statistics, parent_statistics)
