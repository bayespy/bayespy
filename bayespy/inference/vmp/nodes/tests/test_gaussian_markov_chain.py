######################################################################
# Copyright (C) 2013 Jaakko Luttinen
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
Unit tests for gaussian_markov_chain module.
"""

import unittest

import numpy as np

from numpy import testing

from ..gaussian_markov_chain import GaussianMarkovChain
from ..gaussian import Gaussian
from ..wishart import Wishart
from ..gamma import Gamma

from bayespy import utils

class TestGaussianMarkovChain(unittest.TestCase):

    def create_model(self, N, D):

        # Construct the model
        Mu = Gaussian(np.random.randn(D),
                      np.identity(D))
        Lambda = Wishart(D,
                         utils.random.covariance(D))
        A = Gaussian(np.random.randn(D,D),
                     np.identity(D))
        V = Gamma(D,
                  np.random.rand(D))
        X = GaussianMarkovChain(Mu, Lambda, A, V, n=N)
        Y = Gaussian(X.as_gaussian(), np.identity(D))

        return (Y, X, Mu, Lambda, A, V)
        

    def test_plates(self):
        """
        Test that plates are handled correctly.
        """

    def test_message_to_mu0(self):
        pass

    def test_message_to_Lambda0(self):
        pass

    def test_message_to_A(self):
        pass

    def test_message_to_v(self):
        pass

    def test_message_to_child(self):
        pass

    def test_moments(self):
        """
        Test the updating of GaussianMarkovChain.

        Check that the moments and the lower bound contribution are computed
        correctly.
        """

        # TODO: Add plates and missing values!

        # Dimensionalities
        D = 3
        N = 5
        (Y, X, Mu, Lambda, A, V) = self.create_model(N, D)

        # Inference with arbitrary observations
        y = np.random.randn(N,D)
        Y.observe(y)
        X.update()
        (x_vb, xnxn_vb, xpxn_vb) = X.get_moments()

        # Get parameter moments
        (mu0, mumu0) = Mu.get_moments()
        (icov0, logdet0) = Lambda.get_moments()
        (a, aa) = A.get_moments()
        (icov_x, logdetx) = V.get_moments()
        icov_x = np.diag(icov_x)
        # Prior precision
        Z = np.einsum('...kij,...kk->...ij', aa, icov_x)
        U_diag = [icov0+Z] + (N-2)*[icov_x+Z] + [icov_x]
        U_super = (N-1) * [-np.dot(a.T, icov_x)]
        U = utils.utils.block_banded(U_diag, U_super)
        # Prior mean
        mu_prior = np.zeros(D*N)
        mu_prior[:D] = np.dot(icov0,mu0)
        # Data 
        Cov = np.linalg.inv(U + np.identity(D*N))
        mu = np.dot(Cov, mu_prior + y.flatten())
        # Moments
        xx = mu[:,np.newaxis]*mu[np.newaxis,:] + Cov
        mu = np.reshape(mu, (N,D))
        xx = np.reshape(xx, (N,D,N,D))

        # Check results
        testing.assert_allclose(x_vb, mu,
                                err_msg="Incorrect mean")
        for n in range(N):
            testing.assert_allclose(xnxn_vb[n,:,:], xx[n,:,n,:],
                                    err_msg="Incorrect second moment")
        for n in range(N-1):
            testing.assert_allclose(xpxn_vb[n,:,:], xx[n,:,n+1,:],
                                    err_msg="Incorrect lagged second moment")


        # Compute the entropy H(X)
        ldet = utils.linalg.logdet_cov(Cov)
        H = utils.random.gaussian_entropy(-ldet, N*D)
        # Compute <log p(X|...)>
        xx = np.reshape(xx, (N*D, N*D))
        mu = np.reshape(mu, (N*D,))
        ldet = -logdet0 - np.sum(np.ones((N-1,D))*logdetx)
        P = utils.random.gaussian_logpdf(np.einsum('...ij,...ij', 
                                                   xx, 
                                                   U),
                                         np.einsum('...i,...i', 
                                                   mu, 
                                                   mu_prior),
                                         np.einsum('...ij,...ij', 
                                                   mumu0,
                                                   icov0),
                                         -ldet,
                                         N*D)
                                                   
        # The VB bound from the net
        l = X.lower_bound_contribution()

        testing.assert_allclose(l, H+P)
                                                   

        # Compute the true bound <log p(X|...)> + H(X)
        
        

    def test_smoothing(self):
        """
        Test the posterior estimation of GaussianMarkovChain.

        Create time-variant dynamics and compare the results of BayesPy VB
        inference and standard Kalman filtering & smoothing.

        This is not that useful anymore, because the moments are checked much
        better in another test method.
        """

        #
        # Set up an artificial system
        #

        # Dimensions
        N = 500
        D = 2
        # Dynamics (time varying)
        A0 = np.array([[.9, -.4], [.4, .9]])
        A1 = np.array([[.98, -.1], [.1, .98]])
        l = np.linspace(0, 1, N-1).reshape((-1,1,1))
        A = (1-l)*A0 + l*A1
        # Innovation covariance matrix (time varying)
        v = np.random.rand(D)
        V = np.diag(v)
        # Observation noise covariance matrix
        C = np.identity(D)

        #
        # Simulate data
        #
        
        X = np.empty((N,D))
        Y = np.empty((N,D))

        x = np.array([0.5, -0.5])
        X[0,:] = x
        Y[0,:] = x + np.random.multivariate_normal(np.zeros(D), C)
        for n in range(N-1):
            x = np.dot(A[n,:,:],x) + np.random.multivariate_normal(np.zeros(D), V)
            X[n+1,:] = x
            Y[n+1,:] = x + np.random.multivariate_normal(np.zeros(D), C)

        #
        # BayesPy inference
        #

        # Construct VB model
        Xh = GaussianMarkovChain(np.zeros(D), np.identity(D), A, 1/v, n=N)
        Yh = Gaussian(Xh.as_gaussian(), np.identity(D), plates=(N,))
        # Put data 
        Yh.observe(Y)
        # Run inference
        Xh.update()
        # Store results
        Xh_vb = Xh.u[0]
        CovXh_vb = Xh.u[1] - Xh_vb[...,np.newaxis,:] * Xh_vb[...,:,np.newaxis]

        #
        # "The ground truth" using standard Kalman filter and RTS smoother
        #
        V = N*(V,)
        UY = Y
        U = N*(C,)
        (Xh, CovXh) = utils.utils.kalman_filter(UY, U, A, V, np.zeros(D), np.identity(D))
        (Xh, CovXh) = utils.utils.rts_smoother(Xh, CovXh, A, V)

        #
        # Check results
        #
        self.assertTrue(np.allclose(Xh_vb, Xh))
        self.assertTrue(np.allclose(CovXh_vb, CovXh))
        
## if __name__ == '__main__':
##     unittest.main()
