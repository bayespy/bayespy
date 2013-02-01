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
Unit tests for `dot` module.
"""

import unittest


import numpy as np
import scipy

from numpy import testing

from ..dot import Dot, MatrixDot
from ..gaussian import Gaussian

from ...vmp import VB

from bayespy.utils import utils
from bayespy.utils import linalg

class TestMatrixDot(unittest.TestCase):

    def test_parent_A(self):
        """
        Test messages between MatrixDot and the first parent A.

        Compare the results of A to the analytic solution, which can be obtained
        because only one node is unknown.
        """

        # Dimensionalities
        M = 3
        N = 4

        # Plates
        # Y: D4,D3,D2,D1
        # A:  1,D3,D2, 1
        # X:  1,D3, 1,D1
        D4 = 5#5
        D3 = 7#7
        D2 = 6#6
        D1 = 1#2

        # Generate data
        a = np.random.randn(D3,D2,1,M,N)
        x = np.random.randn(D3,1,D1,N)
        f = np.einsum('...ij,...j->...i', a, x)
        v = np.random.randn(M,M)
        vv = np.dot(v.T, v)
        cov = np.linalg.inv(vv)
        l_cov = scipy.linalg.cholesky(cov, lower=True)
        y = f + np.einsum('...ij,...j->...i', 
                          l_cov,
                          np.random.randn(D4, D3, D2, D1, M))

        # Construct the model
        A = Gaussian(np.zeros(M*N), 
                     np.identity(M*N),
                     plates=(D3,D2,1),
                     name='A')
        # TODO: Distribution for X (but fixed)
        X = x
        AX = MatrixDot(A, X, name='AX')
        Lambda = vv
        Y = Gaussian(AX,
                     Lambda,
                     plates=(D4,D3,D2,D1),
                     name='Y')

        # TODO: Add missing values
        # Put in data
        Y.observe(y)

        # Run inference
        Q = VB(Y, A)
        Q.update(A, repeat=1)
        u = A.get_moments()

        # Compute true solution
        xx = x[...,:,np.newaxis] * x[...,np.newaxis,:]
        Cov_A = np.kron(Lambda, xx) # from data
        Cov_A = np.sum(Cov_A, axis=2, keepdims=True)
        Cov_A = Cov_A + np.identity(M*N) # add prior
        Cov_A = linalg.chol_inv(linalg.chol(Cov_A))
        mu_A = np.einsum('...ij,...j,...k->...ik', Lambda, y, x)
        mu_A = np.sum(mu_A, axis=(0,3))
        mu_A = np.reshape(mu_A, (D3,D2,1,M*N))
        mu_A = np.einsum('...ij,...j->...i', Cov_A, mu_A)

        print(np.shape(u[0]))
        print(np.shape(mu_A))

        # Compare VB results to the analytic solution:
        testing.assert_allclose(u[0], mu_A)
        
