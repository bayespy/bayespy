######################################################################
# Copyright (C) 2011,2012 Jaakko Luttinen
#
# This file is licensed under Version 3.0 of the GNU General Public
# License. See LICENSE for a text of the license.
######################################################################

######################################################################
# This file is part of BayesPy.
#
# BayesPy is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
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
import scipy.special as special

from .node import Node
from .deterministic import Deterministic
from .expfamily import ExponentialFamily
from .constant import Constant

from bayespy.utils import utils

def diagonal(alpha):
    """
    Create a diagonal Wishart node from a Gamma node.
    """
    return _GammaToDiagonalWishart(alpha,
                                   name=alpha.name + " as Wishart")


class GammaPrior(Node):

    """ Conjugate prior node for the shape of the gamma
    distribution. This isn't a distribution but can be used to
    compute the correct fixed moments, e.g., for Constant node."""

    # Number of trailing axes for variable dimensions in
    # observations. The other axes correspond to plates.
    ndim_observations = 0

    @staticmethod
    def compute_fixed_moments(a):
        """ Compute moments for fixed x. """
        u0 = a
        u1 = special.gammaln(a)
        return [u0, u1]

    @staticmethod
    def compute_dims_from_values(a):
        """ Compute the dimensions of phi or u. """
        return [(), ()]


class Gamma(ExponentialFamily):

    ndims = (0, 0)

    # Observations/values are scalars (0-dimensional)
    ndim_observations = 0

    
    def __init__(self, a, b, **kwargs):

        self.parameter_distributions = (GammaPrior, Gamma)
        
        # TODO: USE asarray(a)

        # Check for constant a
        if utils.is_numeric(a):
            a = Constant(GammaPrior)(a)

        # Check for constant b
        if utils.is_numeric(b):
            b = Constant(Gamma)(b)
            #b = NodeConstant([b, np.log(b)], plates=np.shape(b), dims=[(),()])

        # Construct
        super().__init__(a, b, **kwargs)

    @staticmethod
    def compute_fixed_moments(x):
        """ Compute moments for fixed x. """
        u0 = x
        # TODO/FIXME: Shouldn't this be just log?
        u1 = special.gammaln(x)
        return [u0, u1]

    @staticmethod
    def _compute_phi_from_parents(*u_parents):
        return [-u_parents[1][0],
                1*u_parents[0][0]]

    @staticmethod
    def _compute_cgf_from_parents(*u_parents):
        a = u_parents[0][0]
        gammaln_a = u_parents[0][1] #special.gammaln(a)
        b = u_parents[1][0]
        log_b = u_parents[1][1]
        g = a * log_b - gammaln_a
        return g

    @staticmethod
    def _compute_moments_and_cgf(phi, mask=True):
        log_b = np.log(-phi[0])
        u0 = phi[1] / (-phi[0])
        u1 = special.digamma(phi[1]) - log_b
        u = [u0, u1]
        g = phi[1] * log_b - special.gammaln(phi[1])
        return (u, g)
        

    @staticmethod
    def _compute_fixed_moments_and_f(x, mask=True):
        """ Compute u(x) and f(x) for given x. """
        u = [x, np.log(x)]
        f = 0
        return (u, f)

    @staticmethod
    def _compute_message_to_parent(parent, index, u, *u_parents):
        """ . """
        if index == 0:
            raise Exception("No analytic solution exists")
        elif index == 1:
            return [-u[0],
                    u_parents[0][0]]

    @staticmethod
    def compute_dims(*parents):
        """ Compute the dimensions of phi/u. """
        return ( (), () )

    @staticmethod
    def compute_dims_from_values(x):
        """ Compute the dimensions of phi and u. """
        return ( (), () )

    def get_shape_of_value(self):
        # Dimensionality of a realization
        return ()
    
    def show(self):
        a = self.phi[1]
        b = -self.phi[0]
        print("%s ~ Gamma(a, b)" % self.name)
        print("  a =", a)
        print("  b =", b)

    def as_diagonal_wishart(self):
        return _GammaToDiagonalWishart(self,
                                       name=self.name + " as Wishart")

class _GammaToDiagonalWishart(Deterministic):
    """
    Transform a set of gamma scalars into a diagonal Wishart matrix.

    The last plate is used as the diagonal dimension.
    """
    
    ndims = (2, 0)
    ndims_parents = [None, (2, 0)]

    # Observations/values are 2-D matrices
    ndim_observations = 2

    def __init__(self, alpha, **kwargs):

        # Check for constant
        if utils.is_numeric(alpha):
            alpha = Constant(Gamma)(alpha)

        # Remove the last plate...
        #plates = alpha.plates[:-1]
        # ... and use it as the dimensionality of the Wishart
        # distribution
        if len(alpha.plates) == 0:
            raise Exception("Gamma variable needs to have plates in "
                            "order to be used as a diagonal Wishart.")
        D = alpha.plates[-1]
        dims = ( (D,D), () )

        # Construct the node
        super().__init__(alpha,
        #plates=plates,
                         dims=dims,
                         **kwargs)
        
    def _plates_to_parent(self, index):
        D = self.dims[0][0]
        return self.plates + (D,)

    def _plates_from_parent(self, index):
        return self.parents[index].plates[:-1]

    @staticmethod
    def _compute_mask_to_parent(index, mask):
        return mask[..., np.newaxis]

    def get_moments(self):
        u = self.parents[0].get_moments()

        # Form a diagonal matrix from the gamma variables
        return [np.identity(self.dims[0][0]) * u[0][...,np.newaxis],
                np.sum(u[1], axis=(-1))]

    @staticmethod
    def _compute_message_to_parent(index, m_children, *u_parents):

        # Take the diagonal
        m0 = np.einsum('...ii->...i', m_children[0])
        m1 = np.reshape(m_children[1], np.shape(m_children[1]) + (1,))

        return [m0, m1]
        

    ## def _compute_message_and_mask_to_parent(self, index, m_children, *u_parents):
        
    ##     m = self._message_from_children()
    ##     #(m, mask) = self.message_from_children()

    ##     # Take the diagonal
    ##     m[0] = np.einsum('...ii->...i', m[0])
    ##     # TODO/FIXME: I think m[1] is wrong..
    ##     m[1] = np.reshape(m[1], np.shape(m[1]) + (1,))
    ##     # m[1] is ok

    ##     mask = self._compute_mask_to_parent(index, self.mask)
    ##     #mask = mask[...,np.newaxis]

    ##     return (m, mask)
    ##     #return m
        


