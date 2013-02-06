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

    def setUp(self):
        """
        Generate artificial data for the tests.
        """
        
        # Generate data
        self.a = np.random.randn(self.D3,self.D2,1,self.M,self.N)
        self.x = np.random.randn(self.D3,1,self.D1,self.N)
        f = np.einsum('...ij,...j->...i', self.a, self.x)
        v = np.random.randn(self.M,self.M)
        self.vv = np.dot(v.T, v)
        self.cov = np.linalg.inv(self.vv)
        l_cov = scipy.linalg.cholesky(self.cov, lower=True)
        self.y = f + np.einsum('...ij,...j->...i', 
                               l_cov,
                               np.random.randn(self.D4, 
                                               self.D3,
                                               self.D2,
                                               self.D1,
                                               self.M))

        self.Lambda = self.vv
        
        super().setUp()

    def check_A(self, mask):
        """
        Check update equations of parent A.

        Use a simple model that has parent nodes and a child node around
        MatrixDot.
        """

        # Parent nodes
        A = Gaussian(np.zeros(self.M*self.N), 
                     np.identity(self.M*self.N),
                     plates=(self.D3,self.D2,1),
                     name='A')
        X = Gaussian(self.x, np.identity(self.N), name='X')
        # Node itself
        AX = MatrixDot(A, X, name='AX')
        # Child node
        Y = Gaussian(AX,
                     self.Lambda,
                     plates=(self.D4,self.D3,self.D2,self.D1),
                     name='Y')
        
        # Put in data
        Y.observe(self.y, mask=mask)

        # VB model
        Q = VB(Y, A)
        Q.update(A, repeat=1)
        u = A.get_moments()

        # Compute true solution
        mask = np.ones((self.D4,self.D3,self.D2,self.D1)) * mask
        vv = mask[...,np.newaxis,np.newaxis] * self.vv
        xx = (self.x[...,:,np.newaxis] * self.x[...,np.newaxis,:]
              + np.identity(self.N))
        Cov_A = (vv[...,:,np.newaxis,:,np.newaxis] * 
                 xx[...,np.newaxis,:,np.newaxis,:]) # from data
        Cov_A = np.sum(Cov_A, axis=(0,3), keepdims=True)
        Cov_A = np.reshape(Cov_A, 
                           (self.D3,self.D2,1,self.M*self.N,self.M*self.N))
        Cov_A = Cov_A + np.identity(self.M*self.N) # add prior
        Cov_A = linalg.chol_inv(linalg.chol(Cov_A))
        mu_A = np.einsum('...ij,...j,...k->...ik', vv, self.y, self.x)
        mu_A = np.sum(mu_A, axis=(0,3))
        mu_A = np.reshape(mu_A, (self.D3,self.D2,1,self.M*self.N))
        mu_A = np.einsum('...ij,...j->...i', Cov_A, mu_A)


        # Compare VB results to the analytic solution:
        Cov_vb = u[1] - u[0][...,np.newaxis,:]*u[0][...,:,np.newaxis]
        testing.assert_allclose(Cov_vb, Cov_A,
                                err_msg="Incorrect second moment.")
        testing.assert_allclose(u[0], mu_A,
                                err_msg="Incorrect first moment.")

    def test_parent_A(self):
        """
        Test messages from MatrixDot to A without missing values.

        Compare the results of A to the analytic solution, which can be obtained
        because only one node is unknown.
        """

        self.check_A(np.array([True]))

    def test_parent_A_mv(self):
        """
        Test messages from MatrixDot to A with missing values.
        """

        self.check_A(random.mask(self.D4,self.D3,self.D2,self.D1,p=0.5))
        
    def check_X(self, mask):
        """
        Check update equations of parent X.

        Use a simple model that has parent nodes and a child node around
        MatrixDot.
        """

        # Parent nodes
        A = Gaussian(self.a.reshape((self.D3,self.D2,1,self.M*self.N)),
                     np.identity(self.M*self.N),
                     name='A')
        X = Gaussian(np.zeros(self.N),
                     np.identity(self.N),
                     plates=(self.D3,1,self.D1),
                     name='X')
        # Node itself
        AX = MatrixDot(A, X, name='AX')
        # Child node
        Y = Gaussian(AX,
                     self.Lambda,
                     plates=(self.D4,self.D3,self.D2,self.D1),
                     name='Y')
        
        # Put in data
        Y.observe(self.y, mask=mask)

        # VB model
        Q = VB(Y, X)
        Q.update(X, repeat=1)
        u = X.get_moments()

        # Compute true solution
        mask = np.ones((self.D4,self.D3,self.D2,self.D1)) * mask
        vv = mask[...,np.newaxis,np.newaxis] * self.vv
        aa = np.einsum('...ki,...kl,...lj->...ij', self.a, vv, self.a)
        aa = aa + np.einsum('...ij,...kl,...kl->...ij', 
                            np.identity(self.N),
                            np.identity(self.M),
                            vv)
        aa = np.sum(aa, axis=2, keepdims=True)
        aa = np.sum(aa, axis=0)
        Cov_X = linalg.chol_inv(linalg.chol(aa + np.identity(self.N)))
        mu_X = np.einsum('...kj,...kl,...l->...j', 
                         self.a,
                         vv,
                         self.y)
        mu_X = np.sum(mu_X, axis=2, keepdims=True)
        mu_X = np.sum(mu_X, axis=0)
        mu_X = np.einsum('...ij,...j->...i', Cov_X, mu_X)

        # Compare VB results to the analytic solution:
        Cov_vb = u[1] - u[0][...,np.newaxis,:]*u[0][...,:,np.newaxis]
        testing.assert_allclose(Cov_vb, Cov_X,
                                err_msg="Incorrect second moment.")
        testing.assert_allclose(u[0], mu_X,
                                err_msg="Incorrect first moment.")

    def test_parent_X(self):
        """
        Test messages from MatrixDot to X without missing values.

        Compare the results of X to the analytic VB solution.
        """

        self.check_X(np.array([True]))

        
    def test_parent_X_mv(self):
        """
        Test messages from MatrixDot to X with missing values.

        Compare the results of X to the analytic VB solution.
        """

        self.check_X(random.mask(self.D4,self.D3,self.D2,self.D1,p=0.5))
        

    def test_child(self):
        """
        Test messages from MatrixDot to children.
        """
        # Parent nodes
        A = Gaussian(self.a.reshape((self.D3,self.D2,1,self.M*self.N)),
                     np.identity(self.M*self.N),
                     name='A')
        X = Gaussian(self.x,
                     np.identity(self.N),
                     plates=(self.D3,1,self.D1),
                     name='X')
        # Node itself
        AX = MatrixDot(A, X, name='AX')

        # Moments from the model
        (u0, u1) = AX.get_moments()

        # Compute the true moments
        ax = np.einsum('...ij,...j->...i', self.a, self.x)

        aa = (np.einsum('...ij,...kl->...ijkl', self.a, self.a)
              + np.identity(self.N*self.M).reshape((self.M,
                                                    self.N,
                                                    self.M,
                                                    self.N)))
        xx = (np.einsum('...i,...j->...ij', self.x, self.x)
              + np.identity(self.N))
        axxa = np.einsum('...ikjl,...kl->...ij', aa, xx)

        # Check the moments
        testing.assert_allclose(u0, ax,
                                err_msg="Incorrect first moment")
        testing.assert_allclose(u1, axxa,
                                err_msg="Incorrect second moment")
