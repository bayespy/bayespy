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

"""
Unit tests for bayespy.utils.utils module.
"""

import unittest

import numpy as np

from bayespy.inference.vmp.nodes.gaussian_markov_chain import GaussianMarkovChain
from bayespy.inference.vmp.nodes.gaussian import Gaussian

from bayespy.utils import utils

class TestGaussianMarkovChain(unittest.TestCase):

    def test_smoothing(self):
        """
        Test the posterior estimation of GaussianMarkovChain.

        Create time-variant dynamics and compare the results of
        BayesPy VB inference and standard Kalman filtering &
        smoothing.
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
        Xh = GaussianMarkovChain(np.zeros(D), np.identity(D), A, 1/v, N=N)
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
        (Xh, CovXh) = utils.kalman_filter(UY, U, A, V, np.zeros(D), np.identity(D))
        (Xh, CovXh) = utils.rts_smoother(Xh, CovXh, A, V)

        #
        # Check results
        #
        self.assertTrue(np.allclose(Xh_vb, Xh))
        self.assertTrue(np.allclose(CovXh_vb, CovXh))
        
if __name__ == '__main__':
    unittest.main()
