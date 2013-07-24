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

from bayespy.utils import utils

from .variable import Variable
from .constant import Constant

def WishartPrior(k):
    class _WishartPrior:

        """ Conjugate prior node for the degrees of freedom of the
        Wishart distribution. This isn't a distribution but can be
        used to compute the correct fixed moments, e.g., for Constant
        node."""


        # Number of trailing axes for variable dimensions in
        # observations. The other axes correspond to plates.
        ndim_observations = 0

        @staticmethod
        def compute_fixed_moments(n):
            """ Compute moments for fixed x. """
            u0 = n
            u1 = special.multigammaln(0.5*n, k)
            return [u0, u1]

        @staticmethod
        def compute_dims_from_values(n):
            """ Compute the dimensions of phi or u. """
            return [(), ()]
        
    return _WishartPrior

class Wishart(Variable):


    ndims = (2, 0)
    ndims_parents = [None, (2, 0)]

    # Observations/values are 2-D matrices
    ndim_observations = 2

    #parameter_distributions = (WishartPrior, Wishart)
    
    ## @staticmethod
    ## def compute_fixed_parameter_moments(*args):
    ##     """ Compute the moments of the distribution parameters for
    ##     fixed values."""
    ##     n = args[0]
    ##     V = args[1]
    ##     k = np.shape(V)[-1]
    ##     u_n = WishartPrior(k).compute_fixed_moments(n)
    ##     u_V = Wishart.compute_fixed_moments(V)
    ##     return (u_n, u_V)

    @staticmethod
    def compute_fixed_moments(Lambda):
        """ Compute moments for fixed x. """
        ldet = utils.m_chol_logdet(utils.m_chol(Lambda))
        u = [Lambda,
             ldet]
        return u

    @staticmethod
    def compute_g_from_parents(u_parents):
        n = u_parents[0][0]
        gammaln_n = u_parents[0][1]
        V = u_parents[1][0]
        logdet_V = u_parents[1][1]
        k = np.shape(V)[-1]
        #k = self.dims[0][0]
        # TODO: Check whether this is correct:
        #g = 0.5*n*logdet_V - special.multigammaln(n/2, k)
        g = 0.5*n*logdet_V - 0.5*k*n*np.log(2) - gammaln_n #special.multigammaln(n/2, k)
        return g

    @staticmethod
    def compute_phi_from_parents(u_parents):
        return [-0.5 * u_parents[1][0],
                0.5 * u_parents[0][0]]

    @staticmethod
    def compute_u_and_g(phi, mask=True):
        U = utils.m_chol(-phi[0])
        k = np.shape(phi[0])[-1]
        #k = self.dims[0][0]
        logdet_phi0 = utils.m_chol_logdet(U)
        u0 = phi[1][...,np.newaxis,np.newaxis] * utils.m_chol_inv(U)
        u1 = -logdet_phi0 + utils.m_digamma(phi[1], k)
        u = [u0, u1]
        g = phi[1] * logdet_phi0 - special.multigammaln(phi[1], k)
        return (u, g)

    @staticmethod
    def compute_fixed_u_and_f(Lambda):
        """ Compute u(x) and f(x) for given x. """
        k = np.shape(Lambda)[-1]
        ldet = utils.m_chol_logdet(utils.m_chol(Lambda))
        u = [Lambda,
             ldet]
        f = -(k+1)/2 * ldet
        return (u, f)

    @staticmethod
    def message(index, u, u_parents):
        if index == 0:
            raise Exception("No analytic solution exists")
        elif index == 1:
            return (-0.5 * u[0],
                    0.5 * u_parents[0][0])

    @staticmethod
    def compute_dims(*parents):
        """ Compute the dimensions of phi/u. """
        # Has the same dimensionality as the second parent.
        return parents[1].dims

    @staticmethod
    def compute_dims_from_values(x):
        """ Compute the dimensions of phi and u. """
        d = np.shape(x)[-1]
        return [(d,d), ()]

    # Wishart(n, inv(V))

    def __init__(self, n, V, plates=(), **kwargs):

        # Check for constant V
        if np.isscalar(V) or isinstance(V, np.ndarray):
            V = Constant(Wishart)(V)

        k = V.dims[0][-1]
        
        # Check for constant n
        if np.isscalar(n) or isinstance(n, np.ndarray):
            n = Constant(WishartPrior(k))(n)
            #n = NodeConstantScalar(n)

        self.parameter_distributions = (WishartPrior(k), Wishart)
        
        super().__init__(n, V, plates=plates, **kwargs)
        
    def show(self):
        print("%s ~ Wishart(n, A)" % self.name)
        print("  n =")
        print(2*self.phi[1])
        print("  A =")
        print(0.5 * self.u[0] / self.phi[1][...,np.newaxis,np.newaxis])

