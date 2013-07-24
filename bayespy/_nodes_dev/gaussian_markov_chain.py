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

import utils

from .variable import Variable
from .constant import Constant
from .wishart import Wishart

class GaussianMarkovChain(Variable):

    ndims = (1, 2)
    ndims_parents = [(1, 2), (2, 0)]
    # Observations are vectors (1-D):
    ndim_observations = 1

    
    ## @staticmethod
    ## def compute_fixed_parameter_moments(*args):
    ##     """ Compute the moments of the distribution parameters for
    ##     fixed values."""
    ##     mu = args[0]
    ##     Lambda = args[1]
    ##     u_mu = Gaussian.compute_fixed_moments(mu)
    ##     u_Lambda = Wishart.compute_fixed_moments(Lambda)
    ##     return (u_mu, u_Lambda)

    @staticmethod
    def compute_fixed_moments(x):
        """ Compute moments for fixed x. """
        return [x, utils.m_outer(x,x)]

    @staticmethod
    def compute_phi_from_parents(u_parents):
        ## print('in Gaussian.compute_phi_from_parents')
        ## print(u_parents)
        ## print(np.shape(u_parents[1][0]))
        ## print(np.shape(u_parents[0][0]))
        return [utils.m_dot(u_parents[1][0], u_parents[0][0]),
                -0.5 * u_parents[1][0]]

    @staticmethod
    def compute_g_from_parents(u_parents):
        mu = u_parents[0][0]
        mumu = u_parents[0][1]
        Lambda = u_parents[1][0]
        logdet_Lambda = u_parents[1][1]
        g = (-0.5 * np.einsum('...ij,...ij',mumu,Lambda)
             + 0.5 * logdet_Lambda)
        ## g = (-0.5 * np.einsum('...ij,...ij',mumu,Lambda)
        ##      + 0.5 * np.sum(logdet_Lambda))
        return g

    @staticmethod
    def compute_u_and_g(phi, mask=True):
        #print(-phi[1])
        L = utils.m_chol(-phi[1])
        k = np.shape(phi[0])[-1]
        # Moments
        u0 = utils.m_chol_solve(L, 0.5*phi[0])
        u1 = utils.m_outer(u0, u0) + 0.5 * utils.m_chol_inv(L)
        u = [u0, u1]
        # G
        g = (-0.5 * np.einsum('...i,...i', u[0], phi[0])
             + 0.5 * utils.m_chol_logdet(L)
             + 0.5 * np.log(2) * k)
             #+ 0.5 * np.log(2) * self.dims[0][0])
        return (u, g)

    @staticmethod
    def compute_fixed_u_and_f(x):
        """ Compute u(x) and f(x) for given x. """
        k = np.shape(x)[-1]
        u = [x, utils.m_outer(x,x)]
        f = -k/2*np.log(2*np.pi)
        return (u, f)

    @staticmethod
    def compute_message(index, u, u_parents):
        """ . """
        if index == 0:
            return [utils.m_dot(u_parents[1][0], u[0]),
                    -0.5 * u_parents[1][0]]
        elif index == 1:
            xmu = utils.m_outer(u[0], u_parents[0][0])
            return [-0.5 * (u[1] - xmu - xmu.swapaxes(-1,-2) + u_parents[0][1]),
                    0.5]

    @staticmethod
    def compute_dims(*parents):
        """ Compute the dimensions of phi and u. """
        # Has the same dimensionality as the first parent.
        ## print('in gaussian compute dims: parent.dims:', parents[0].dims)
        ## print('in gaussian compute dims: parent.u:', parents[0].u)
        return parents[0].dims

    @staticmethod
    def compute_dims_from_values(x):
        """ Compute the dimensions of phi and u. """
        d = np.shape(x)[-1]
        return [(d,), (d,d)]

    # Gaussian(mu, inv(Lambda))

    def __init__(self, mu, Lambda, plates=(), **kwargs):

        self.parameter_distributions = (Gaussian, Wishart)
        
        # Check for constant mu
        if np.isscalar(mu) or isinstance(mu, np.ndarray):
            mu = Constant(Gaussian)(mu)

        # Check for constant Lambda
        if np.isscalar(Lambda) or isinstance(Lambda, np.ndarray):
            Lambda = Constant(Wishart)(Lambda)

        # You could check whether the dimensions of mu and Lambda
        # match (and Lambda is square)
        if Lambda.dims[0][-1] != mu.dims[0][-1]:
            raise Exception("Dimensionalities of mu and Lambda do not match.")

        # Construct
        super().__init__(mu, Lambda,
                         plates=plates,
                         **kwargs)

    def random(self):
        # TODO/FIXME: You shouldn't draw random values for
        # observed/fixed elements!

        # Note that phi[1] is -0.5*inv(Cov)
        U = utils.m_chol(-2*self.phi[1])
        mu = self.u[0]
        z = np.random.normal(0, 1, self.get_shape(0))
        # Compute mu + U'*z
        #return mu + np.einsum('...ij,...i->...j', U, z)
        #scipy.linalg.solve_triangular(a, b, trans=0, lower=False, unit_diagonal=False, overwrite_b=False, debug=False)
        #print('gaussian.random', np.shape(mu), np.shape(z))
        z = utils.m_solve_triangular(U, z, trans='T', lower=False)
        return mu + z
        #return self.u[0] + utils.m_chol_solve(U, z)

    ## def initialize_random_mean(self):
    ##     # First, initialize the distribution from prior?
    ##     self.initialize_from_prior()
        
    ##     if not np.all(self.observed):
    ##         # Draw a random sample
    ##         x = self.random()

    ##         # Update parameter for the mean using the sample
    ##         self.phi[0] = -2*utils.m_dot(self.phi[1], x)

    ##         # Update moments
    ##         (u, g) = self.compute_u_and_g(self.phi, mask=True)
    ##         self.update_u_and_g(u, g, mask=True)
            

    def show(self):
        mu = self.u[0]
        Cov = self.u[1] - utils.m_outer(mu, mu)
        print("%s ~ Gaussian(mu, Cov)" % self.name)
        print("  mu = ")
        print(mu)
        print("  Cov = ")
        print(str(Cov))

    ## def observe(self, x):
    ##     self.fix_moments([x, utils.m_outer(x,x)])
