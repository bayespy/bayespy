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
from bayespy.utils import random

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
        D4 = 5
        D3 = 7
        D2 = 6
        D1 = 2

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

        Lambda = vv
        for mask in [np.array([True]), random.mask(D4,D3,D2,D1,p=0.5)]:
            # Construct the model (use non-constants for parents and children)
            A = Gaussian(np.zeros(M*N), 
                         np.identity(M*N),
                         plates=(D3,D2,1),
                         name='A')
            X = Gaussian(x, np.identity(N), name='X')
            AX = MatrixDot(A, X, name='AX')
            Y = Gaussian(AX,
                         Lambda,
                         plates=(D4,D3,D2,D1),
                         name='Y')

            # Put in data
            Y.observe(y, mask=mask)

            # Run inference
            Q = VB(Y, A)
            Q.update(A, repeat=1)
            u = A.get_moments()

            # Compute true solution
            vv = np.ones((D4,D3,D2,D1,1,1)) * mask[...,np.newaxis,np.newaxis] * vv
            xx = (x[...,:,np.newaxis] * x[...,np.newaxis,:]
                  + np.identity(N))
            Cov_A = (vv[...,:,np.newaxis,:,np.newaxis] * 
                     xx[...,np.newaxis,:,np.newaxis,:]) # from data
            Cov_A = np.sum(Cov_A, axis=(0,3), keepdims=True)
            Cov_A = np.reshape(Cov_A, (D3,D2,1,M*N,M*N))
            Cov_A = Cov_A + np.identity(M*N) # add prior
            Cov_A = linalg.chol_inv(linalg.chol(Cov_A))
            mu_A = np.einsum('...ij,...j,...k->...ik', vv, y, x)
            mu_A = np.sum(mu_A, axis=(0,3))
            mu_A = np.reshape(mu_A, (D3,D2,1,M*N))
            mu_A = np.einsum('...ij,...j->...i', Cov_A, mu_A)


            # Compare VB results to the analytic solution:
            Cov_vb = u[1] - u[0][...,np.newaxis,:]*u[0][...,:,np.newaxis]
            testing.assert_allclose(Cov_vb, Cov_A,
                                    err_msg="Incorrect second moment.")
            testing.assert_allclose(u[0], mu_A,
                                    err_msg="Incorrect first moment.")
        
