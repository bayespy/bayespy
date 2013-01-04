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

'''
The VB inference class for Gaussian variable.
'''

import numpy as np

from bayespy.utils import utils

from .variable import Variable
from .constant import Constant
from .wishart import Wishart

class Gaussian(Variable):

    ndims = (1, 2)
    ndims_parents = [(1, 2), (2, 0)]
    # Observations are vectors (1-D):
    ndim_observations = 1

    
    @staticmethod
    def compute_fixed_moments(x):
        """ Compute moments for fixed x. """
        return [x, utils.m_outer(x,x)]

    @staticmethod
    def compute_phi_from_parents(u_parents):
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
        z = utils.m_solve_triangular(U, z, trans='T', lower=False)
        return mu + z
            

    def show(self):
        mu = self.u[0]
        Cov = self.u[1] - utils.m_outer(mu, mu)
        print("%s ~ Gaussian(mu, Cov)" % self.name)
        print("  mu = ")
        print(mu)
        print("  Cov = ")
        print(str(Cov))

    ## @staticmethod
    ## def compute_rotated_moments(R, x, xx):
    ##     x = np.einsum('...ij,...j->...i', R, x)
    ##     xx = np.einsum('...ij,...jk,...lk->...il', R, xx, R)
    ##     return [x, xx]

    ## @staticmethod
    ## def rotation_entropy(U,s,V, n=1, gradient=False):
    ##     # Entropy
    ##     e = n*np.sum(np.log(S))
    ##     if gradient:
    ##         # Derivative w.r.t. rotation matrix R=U*S*V is inv(R).T
    ##         dR = n*np.dot(V.T, np.dot(np.diag(1/s), U.T))
    ##         return (e, dR)
    ##     else:
    ##         return e


    ## def start_rotation(self):

    ##     R = None
    ##     U = None
    ##     s = None
    ##     V = None

    ##     # There should not be any observed/fixed values.

    ##     u = self.u
    ##     u0 = self.u

    ##     self.u = None

    ##     dR = None
        

    ##     def transform_rotation(A, svd=None):
    ##         # Rotation matrix
    ##         R = np.atleast_2d(A)
    ##         if svd is None:
    ##             (U,s,V) = np.svd(R)
    ##         else:
    ##             (U,s,V) = svd
                
    ##         # Transform moments
    ##         u = self.compute_rotated_moments(R, *u0)

    ##         # Put gradient to zero
    ##         dR = np.zeros(np.shape(R))

    ##         # Return transformed moments
    ##         return u
            

    ##     def cost_rotation(gradient=True):
    ##         # Compute E[phi] over the parents' distribution
    ##         phi_p_X = self.phi_from_parents(gradient=gradient)
            
    ##         # Compute the cost

    ##         # Entropy term
    ##         #log_qh_X = N_X * np.sum(np.log(S))
    ##         log_qh_X = self.rotation_entropy(U, s, V,
    ##                                          n=N_X,
    ##                                          gradient=gradient)

    ##         # Prior term
    ##         log_ph_X = X.compute_logpdf(u,
    ##                                     phi_p_X,
    ##                                     0,
    ##                                     0,
    ##                                     gradient=gradient)
    ##         # Total cost
    ##         l = log_qh_X + log_ph_X
    ##         return l

    ##     def gradient_rotation():
    ##         return dR

    ##     #def stop_rotation():
    ##     #    self.u = 

    ##     return (transform_rotation, cost_rotation, gradient_rotation)
        

