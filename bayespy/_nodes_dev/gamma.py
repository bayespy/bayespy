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
import scipy.special as special

from .node import Node
from .variable import Variable
from .constant import Constant

class GammaPrior:

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


class Gamma(Variable):

    ndims = (0, 0)

    # Observations/values are scalars (0-dimensional)
    ndim_observations = 0

    
    @staticmethod
    def compute_fixed_moments(x):
        """ Compute moments for fixed x. """
        u0 = x
        u1 = special.gammaln(x)
        return [u0, u1]

    @staticmethod
    def compute_phi_from_parents(u_parents):
        return [-u_parents[1][0],
                1*u_parents[0][0]]

    @staticmethod
    def compute_g_from_parents(u_parents):
        a = u_parents[0][0]
        gammaln_a = u_parents[0][1] #special.gammaln(a)
        b = u_parents[1][0]
        log_b = u_parents[1][1]
        g = a * log_b - gammaln_a
        return g

    @staticmethod
    def compute_u_and_g(phi, mask=True):
        log_b = np.log(-phi[0])
        u0 = phi[1] / (-phi[0])
        u1 = special.digamma(phi[1]) - log_b
        u = [u0, u1]
        g = phi[1] * log_b - special.gammaln(phi[1])
        return (u, g)
        

    @staticmethod
    def compute_message(index, u, u_parents):
        """ . """
        if index == 0:
            raise Exception("No analytic solution exists")
        elif index == 1:
            return [-u[0],
                    u_parents[0][0]]

    @staticmethod
    def compute_dims(*parents):
        """ Compute the dimensions of phi/u. """
        return [(), ()]

    @staticmethod
    def compute_dims_from_values(x):
        """ Compute the dimensions of phi and u. """
        return [(), ()]

    def __init__(self, a, b, plates=(), **kwargs):

        self.parameter_distributions = (GammaPrior, Gamma)
        
        # TODO: USE asarray(a)

        # Check for constant a
        if np.isscalar(a) or isinstance(a, np.ndarray):
            a = Constant(GammaPrior)(a)

        # Check for constant b
        if np.isscalar(b) or isinstance(b, np.ndarray):
            b = Constant(Gamma)(b)
            #b = NodeConstant([b, np.log(b)], plates=np.shape(b), dims=[(),()])

        # Construct
        super().__init__(a, b, plates=plates, **kwargs)

    def show(self):
        a = self.phi[1]
        b = -self.phi[0]
        print("%s ~ Gamma(a, b)" % self.name)
        print("  a =", a)
        #print(a)
        print("  b =", b)
        #print(b)

class GammaToDiagonalWishart(Node):
    
    ndims = (2, 0)
    ndims_parents = [None, (2, 0)]

    # Observations/values are 2-D matrices
    ndim_observations = 2

    ## @staticmethod
    ## def compute_fixed_moments(Lambda):
    ##     """ Compute moments for fixed x. """
    ##     return Wishart.compute_fixed_moments(Lambda)

    ## @staticmethod
    ## def compute_g_from_parents(u_parents):
    ##     n = u_parents[0][0]
    ##     gammaln_n = u_parents[0][1]
    ##     V = u_parents[1][0]
    ##     logdet_V = u_parents[1][1]
    ##     k = np.shape(V)[-1]
    ##     #k = self.dims[0][0]
    ##     # TODO: Check whether this is correct:
    ##     #g = 0.5*n*logdet_V - special.multigammaln(n/2, k)
    ##     g = 0.5*n*logdet_V - 0.5*k*n*np.log(2) - gammaln_n #special.multigammaln(n/2, k)
    ##     return g

    ## @staticmethod
    ## def compute_phi_from_parents(u_parents):
    ##     return [-0.5 * u_parents[1][0],
    ##             0.5 * u_parents[0][0]]

    ## @staticmethod
    ## def compute_u_and_g(phi, mask=True):
    ##     U = utils.m_chol(-phi[0])
    ##     k = np.shape(phi[0])[-1]
    ##     #k = self.dims[0][0]
    ##     logdet_phi0 = utils.m_chol_logdet(U)
    ##     u0 = phi[1][...,np.newaxis,np.newaxis] * utils.m_chol_inv(U)
    ##     u1 = -logdet_phi0 + utils.m_digamma(phi[1], k)
    ##     u = [u0, u1]
    ##     g = phi[1] * logdet_phi0 - special.multigammaln(phi[1], k)
    ##     return (u, g)

    ## @staticmethod
    ## def compute_fixed_u_and_f(Lambda):
    ##     """ Compute u(x) and f(x) for given x. """
    ##     k = np.shape(Lambda)[-1]
    ##     ldet = utils.m_chol_logdet(utils.m_chol(Lambda))
    ##     u = [Lambda,
    ##          ldet]
    ##     f = -(k+1)/2 * ldet
    ##     return (u, f)

    ## @staticmethod
    ## def message(index, u, u_parents):
    ##     if index == 0:
    ##         raise Exception("No analytic solution exists")
    ##     elif index == 1:
    ##         return (-0.5 * u[0],
    ##                 0.5 * u_parents[0][0])

    ## @staticmethod
    ## def compute_dims(*parents):
    ##     """ Compute the dimensions of phi/u. """
    ##     # Has the same dimensionality as the second parent.
    ##     return parents[1].dims

    ## @staticmethod
    ## def compute_dims_from_values(x):
    ##     """ Compute the dimensions of phi and u. """
    ##     d = np.shape(x)[-1]
    ##     return [(d,d), ()]

    def __init__(self, alpha, **kwargs):

        # Check for constant n
        if np.isscalar(alpha) or isinstance(alpha, np.ndarray):            
            alpha = Constant(Gamma)(alpha)

        #ExponentialFamily.__init__(self, n, V, plates=plates, dims=V.dims, **kwargs)
        k = alpha.plates[-1]
        super().__init__(alpha,
                         plates=alpha.plates[:-1],
                         dims=[(k,k),()],
                         **kwargs)
        
    def get_moments(self):
        u = self.parents[0].message_to_child()

        # Form a diagonal matrix from the gamma variables
        return [np.identity(self.dims[0][0]) * u[0][...,np.newaxis],
                np.sum(u[1], axis=(-1))]

    def get_message(self, index, u_parents):
        
        (m, mask) = self.message_from_children()

        # Take the diagonal
        m[0] = np.einsum('...ii->...i', m[0])
        # TODO/FIXME: I think m[1] is wrong..
        m[1] = np.reshape(m[1], np.shape(m[1]) + (1,))
        # m[1] is ok

        mask = mask[...,np.newaxis]

        return (m, mask)
        


