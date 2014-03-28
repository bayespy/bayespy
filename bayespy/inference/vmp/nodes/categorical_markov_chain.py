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

from .deterministic import Deterministic
from .expfamily import ExponentialFamily, \
                       ExponentialFamilyDistribution, \
                       useconstructor
from .node import Statistics, \
                  ensureparents
from .categorical import CategoricalStatistics
from .dirichlet import Dirichlet, \
                       DirichletStatistics

from bayespy.utils import utils

class CategoricalMarkovChainStatistics(Statistics):

    ndim_observations = 0

    def __init__(self, categories):
        self.D = categories

    def converter(self, statistics_class):
        """
        Returns a node class which converts the node's statistics to another
        """
        if statistics_class is CategoricalStatistics:
            return CategoricalMarkovChainToCategorical
        return super().converter(statistics_class)

    def compute_fixed_moments(self, x):
        raise NotImplementedError("compute_fixed_moments not implemented for "
                                  "%s" 
                                  % (self.__class__.__name__))
    
    def compute_dims_from_values(self, x):
        raise NotImplementedError("compute_dims_from_values not implemented "
                                  "for %s"
                                  % (self.__class__.__name__))


class CategoricalMarkovChainDistribution(ExponentialFamilyDistribution):

    ndims = (1, 3)
    ndims_parents = ( (1,), (1,) )

    def __init__(self, categories, states):
        self.K = categories
        self.N = states

    def compute_message_to_parent(self, parent, index, u, u_p0, u_P):
        if index == 0:
            return [ u[0] ]
        elif index == 1:
            return [ u[1] ]
        else:
            raise ValueError("Parent index out of bounds")

    def compute_mask_to_parent(self, index, mask):
        if index == 0:
            return mask
        elif index == 1:
            # Add plate axis for the time axis and row axis of the transition
            # matrix
            return np.asanyarray(mask)[...,None,None]
        else:
            raise ValueError("Parent index out of bounds")

    def compute_phi_from_parents(self, u_p0, u_P, mask=True):
        phi0 = u_p0[0]
        phi1 = u_P[0] * np.ones((self.N-1,self.K,self.K))
        return [phi0, phi1]

    def compute_moments_and_cgf(self, phi, mask=True):
        logp0 = phi[0]
        logP = phi[1]
        (z0, zz, cgf) = utils.alpha_beta_recursion(logp0, logP)
        u = [z0, zz]
        return (u, cgf)

    def compute_cgf_from_parents(self, u_p0, u_P):
        return 0
        
    def compute_fixed_moments_and_f(self, x, mask=True):
        raise NotImplementedError()

    def plates_to_parent(self, index, plates):
        if index == 0:
            return plates
        elif index == 1:
            return plates + (self.N-1, self.K)
        else:
            raise ValueError("Parent index out of bounds")
        
    def plates_from_parent(self, index, plates):
        if index == 0:
            return plates
        elif index == 1:
            return plates[:-2]
        else:
            raise ValueError("Parent index out of bounds")
        
class CategoricalMarkovChain(ExponentialFamily):
    
    _parent_statistics = (DirichletStatistics(),
                          DirichletStatistics())

    @useconstructor
    def __init__(self, p0, P, states=None, **kwargs):
        super().__init__(p0, P, **kwargs)

    @classmethod
    @ensureparents
    def _constructor(cls, p0, P, states=None, plates=None, **kwargs):

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
                    N = 2
                else:
                    N = int(states)
            else:
                if states is not None and P.plates[-2]+1 != states:
                    raise ValueError("Given length of the Markov chain is "
                                     "inconsistent with the transition "
                                     "probability matrix")
                N = P.plates[-2] + 1

        if p0.dims != P.dims:
            raise ValueError("Initial state probability vector and state "
                             "transition probability matrix have different "
                             "size")

        if len(P.plates) < 1 or P.plates[-1] != D:
            raise ValueError("Transition probability matrix is not square")

        dims = ( (D,), (N-1,D,D) )

        distribution = CategoricalMarkovChainDistribution(D, N)
        statistics = CategoricalMarkovChainStatistics(D)
        parent_statistics = cls._parent_statistics

        return (dims, 
                cls._total_plates(plates,
                                  distribution.plates_from_parent(0, p0.plates),
                                  distribution.plates_from_parent(1, P.plates)),
                distribution, 
                statistics, 
                parent_statistics)

        
class CategoricalMarkovChainToCategorical(Deterministic):
    
    def __init__(self, Z, **kwargs):
        # Convert parent to proper type. Z must be a node.
        Z = Z._convert(CategoricalMarkovChainStatistics)
        K = Z.dims[0][-1]
        dims = ( (K,), )
        self._statistics = CategoricalStatistics(K)
        self._parent_statistics = (CategoricalMarkovChainStatistics(K),)
        super().__init__(Z, dims=dims, **kwargs)
        
    def _compute_moments(self, u_Z):
        # Add time axis to p0
        p0 = u_Z[0][...,None,:]
        # Sum joint probability arrays to marginal probability vectors
        zz = u_Z[1]
        p = np.sum(zz, axis=-2)

        # Broadcast p0 and p to same shape, except the time axis
        plates_p0 = np.shape(p0)[:-2]
        plates_p = np.shape(p)[:-2]
        shape = utils.broadcasted_shape(plates_p0, plates_p) + (1,1)
        p0 = p0 * np.ones(shape)
        p = p * np.ones(shape)

        # Concatenate
        P = np.concatenate((p0,p), axis=-2)

        return [P]

    def _compute_message_to_parent(self, index, m, u_Z):
        m0 = m[0][...,0,:]
        m1 = m[0][...,1:,None,:]
        return [m0, m1]
    
    def _compute_mask_to_parent(self, index, mask):
        if index == 0:
            return np.any(mask, axis=-1)
        else:
            raise ValueError("Parent index out of bounds")
    
    def _plates_to_parent(self, index):
        if index == 0:
            return self.plates[:-1]
        else:
            raise ValueError("Parent index out of bounds")
    
    def _plates_from_parent(self, index):
        if index == 0:
            N = self.parents[0].dims[1][0]
            return self.parents[0].plates + (N+1,)
        else:
            raise ValueError("Parent index out of bounds")
