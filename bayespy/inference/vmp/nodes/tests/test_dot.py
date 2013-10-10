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

from ..dot import Dot, MatrixDot, SumMultiply
from ..gaussian import Gaussian, GaussianArrayARD
from ..normal import Normal

from ...vmp import VB

from bayespy.utils import utils
from bayespy.utils import linalg
from bayespy.utils import random

from bayespy.utils.utils import TestCase

class TestSumMultiply(TestCase):

    def test_parent_validity(self):
        """
        Test that the parent nodes are validated properly in SumMultiply
        """
        V = Normal(1, 1)
        X = Gaussian(np.ones(1), np.identity(1))
        Y = Gaussian(np.ones(3), np.identity(3))
        Z = Gaussian(np.ones(5), np.identity(5))

        A = SumMultiply(X, ['i'])
        self.assertEqual(A.dims, ((), ()))
        A = SumMultiply('i', X)
        self.assertEqual(A.dims, ((), ()))
        
        A = SumMultiply(X, ['i'], ['i'])
        self.assertEqual(A.dims, ((1,), (1,1)))
        A = SumMultiply('i->i', X)
        self.assertEqual(A.dims, ((1,), (1,1)))
        
        A = SumMultiply(X, ['i'], Y, ['j'], ['i','j'])
        self.assertEqual(A.dims, ((1,3), (1,3,1,3)))
        A = SumMultiply('i,j->ij', X, Y)
        self.assertEqual(A.dims, ((1,3), (1,3,1,3)))
        
        A = SumMultiply(V, [], X, ['i'], Y, ['i'], [])
        self.assertEqual(A.dims, ((), ()))
        A = SumMultiply(',i,i->', V, X, Y)
        self.assertEqual(A.dims, ((), ()))

        # Error: not enough inputs
        self.assertRaises(ValueError,
                          SumMultiply)
        self.assertRaises(ValueError,
                          SumMultiply,
                          X)
        # Error: too many keys
        self.assertRaises(ValueError,
                          SumMultiply,
                          Y, 
                          ['i', 'j'])
        self.assertRaises(ValueError,
                          SumMultiply,
                          'ij',
                          Y)
        # Error: not broadcastable
        self.assertRaises(ValueError,
                          SumMultiply,
                          Y,
                          ['i'],
                          Z,
                          ['i'])
        self.assertRaises(ValueError,
                          SumMultiply,
                          'i,i',
                          Y,
                          Z)
        # Error: output key not in inputs
        self.assertRaises(ValueError,
                          SumMultiply,
                          X,
                          ['i'],
                          ['j'])
        self.assertRaises(ValueError,
                          SumMultiply,
                          'i->j',
                          X)
        # Error: non-unique input keys
        self.assertRaises(ValueError,
                          SumMultiply,
                          X,
                          ['i','i'])
        self.assertRaises(ValueError,
                          SumMultiply,
                          'ii',
                          X)
        # Error: non-unique output keys
        self.assertRaises(ValueError,
                          SumMultiply,
                          X,
                          ['i'],
                          ['i','i'])
        self.assertRaises(ValueError,
                          SumMultiply,
                          'i->ii',
                          X)                          
        # String has too many '->'
        self.assertRaises(ValueError,
                          SumMultiply,
                          'i->i->i',
                          X)
        # String has too many input nodes
        self.assertRaises(ValueError,
                          SumMultiply,
                          'i,i->i',
                          X)

    def compute_moments(self, string, einsum_mean, einsum_cov, *shapes):
        Xs = [GaussianArrayARD(np.random.randn(*(plates+dims)),
                               np.random.rand(*(plates+dims)),
                               plates=plates,
                               shape=dims)
              for (plates, dims) in shapes]
        Y = SumMultiply(string, *Xs)
        u_Y = Y.get_moments()
        u0 = [X.get_moments()[0] for X in Xs]
        u1 = [X.get_moments()[1] for X in Xs]
        # Check mean
        self.assertAllClose(u_Y[0],
                            np.einsum(einsum_mean, *u0))
        self.assertAllClose(u_Y[1],
                            np.einsum(einsum_cov, *u1))

    def compare_moments(self, u0, u1, *args):
        Y = SumMultiply(*args)
        u_Y = Y.get_moments()
        self.assertAllClose(u_Y[0], u0)
        self.assertAllClose(u_Y[1], u1)
        
    def test_moments(self):
        # (3,3,3) x 0-D Gaussian
        V1 = GaussianArrayARD(np.random.randn(3,3,3),
                              np.random.rand(3,3,3),
                              plates=(3,3,3),
                              shape=())
        v1 = V1.get_moments()
        # 1-D Gaussians
        X1 = GaussianArrayARD(np.random.randn(3),
                              np.random.rand(3),
                              plates=(),
                              shape=(3,))
        x1 = X1.get_moments()
        X2 = GaussianArrayARD(np.random.randn(3,1,3),
                              np.random.rand(3,1,3),
                              plates=(3,1),
                              shape=(3,))
        x2 = X2.get_moments()
        X3 = GaussianArrayARD(np.random.randn(3,3,3,3),
                              np.random.rand(3,3,3,3),
                              plates=(3,3,3),
                              shape=(3,))
        x3 = X3.get_moments()

        # 2-D Gaussians
        Y1 = GaussianArrayARD(np.random.randn(3,3),
                              np.random.rand(3,3),
                              plates=(),
                              shape=(3,3))
        y1 = Y1.get_moments()
        Y2 = GaussianArrayARD(np.random.randn(3,3,3),
                              np.random.rand(3,3,3),
                              plates=(3,),
                              shape=(3,3))
        y2 = Y2.get_moments()

        # 3-D Gaussians
        Z1 = GaussianArrayARD(np.random.randn(3,3,3),
                              np.random.rand(3,3,3),
                              plates=(),
                              shape=(3,3,3))
        z1 = Z1.get_moments()

        # Do nothing for 2-D array
        self.compare_moments(y2[0],
                             y2[1],
                             'ij->ij',
                             Y2)
        self.compare_moments(y2[0],
                             y2[1],
                             Y2,
                             [0,1],
                             [0,1])

        # Sum over the rows of a matrix
        mu = np.einsum('...ij->...j', y2[0])
        cov = np.einsum('...ijkl->...jl', y2[1])
        self.compare_moments(mu,
                             cov,
                             'ij->j',
                             Y2)
        self.compare_moments(mu,
                             cov,
                             Y2,
                             [0,1],
                             [1])

        # Inner product of three vectors
        mu = np.einsum('...i,...i,...i->...', x1[0], x2[0], x3[0])
        cov = np.einsum('...ij,...ij,...ij->...', x1[1], x2[1], x3[1])
        self.compare_moments(mu,
                             cov,
                             'i,i,i',
                             X1,
                             X2,
                             X3)
        self.compare_moments(mu,
                             cov,
                             'i,i,i->',
                             X1,
                             X2,
                             X3)
        self.compare_moments(mu,
                             cov,
                             X1,
                             [9],
                             X2,
                             [9],
                             X3,
                             [9])
        self.compare_moments(mu,
                             cov,
                             X1,
                             [9],
                             X2,
                             [9],
                             X3,
                             [9],
                             [])
                            

        # Outer product of two vectors
        mu = np.einsum('...i,...j->...ij', x2[0], x3[0])
        cov = np.einsum('...ik,...jl->...ijkl', x2[1], x3[1])
        self.compare_moments(mu,
                             cov,
                             'i,j->ij',
                             X2,
                             X3)
        self.compare_moments(mu,
                             cov,
                             X2,
                             [9],
                             X3,
                             [7],
                             [9,7])

        # Matrix product
        mu = np.einsum('...ik,...kj->...ij', y1[0], y2[0])
        cov = np.einsum('...ikjl,...kmln->...imjn', y1[1], y2[1])
        self.compare_moments(mu,
                             cov,
                             'ik,kj->ij',
                             Y1,
                             Y2)
        self.compare_moments(mu,
                             cov,
                             Y1,
                             ['i','k'],
                             Y2,
                             ['k','j'],
                             ['i','j'])

        # Trace of a matrix product
        mu = np.einsum('...ij,...ji->...', y1[0], y2[0])
        cov = np.einsum('...ikjl,...kilj->...', y1[1], y2[1])
        self.compare_moments(mu,
                             cov,
                             'ij,ji',
                             Y1,
                             Y2)
        self.compare_moments(mu,
                             cov,
                             'ij,ji->',
                             Y1,
                             Y2)
        self.compare_moments(mu,
                             cov,
                             Y1,
                             ['i','j'],
                             Y2,
                             ['j','i'])
        self.compare_moments(mu,
                             cov,
                             Y1,
                             ['i','j'],
                             Y2,
                             ['j','i'],
                             [])

        # Vector-matrix-vector product
        mu = np.einsum('...i,...ij,...j->...', x1[0], y2[0], x2[0])
        cov = np.einsum('...ia,...ijab,...jb->...', x1[1], y2[1], x2[1])
        self.compare_moments(mu,
                             cov,
                             'i,ij,j',
                             X1,
                             Y2,
                             X2)
        self.compare_moments(mu,
                             cov,
                             X1,
                             [1],
                             Y2,
                             [1,2],
                             X2,
                             [2])
        
        # Complex sum-product of 0-D, 1-D, 2-D and 3-D arrays
        mu = np.einsum('...,...i,...kj,...jik->...k', v1[0], x3[0], y2[0], z1[0])
        cov = np.einsum('...,...ia,...kjcb,...jikbac->...kc', v1[1], x3[1], y2[1], z1[1])
        self.compare_moments(mu,
                             cov,
                             ',i,kj,jik->k',
                             V1,
                             X3,
                             Y2,
                             Z1)
        self.compare_moments(mu,
                             cov,
                             V1,
                             [],
                             X3,
                             ['i'],
                             Y2,
                             ['k','j'],
                             Z1,
                             ['j','i','k'],
                             ['k'])


        
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
