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
Unit tests for `gaussian` module.
"""

import unittest


import numpy as np
import scipy

from numpy import testing

from .. import gaussian
from ..gaussian import Gaussian, GaussianARD
from ..gamma import Gamma

from ...vmp import VB

from bayespy.utils import utils
from bayespy.utils import linalg
from bayespy.utils import random

from bayespy.utils.utils import TestCase

class TestGaussianFunctions(TestCase):

    def test_rotate_covariance(self):
        """
        Test the Gaussian array covariance rotation.
        """
        # Check matrix
        R = np.random.randn(2,2)
        Cov = np.random.randn(2,2)
        self.assertAllClose(gaussian.rotate_covariance(Cov, R),
                            np.einsum('ik,kl,lj', R, Cov, R.T))

        # Check matrix with plates
        R = np.random.randn(2,2)
        Cov = np.random.randn(4,3,2,2)
        self.assertAllClose(gaussian.rotate_covariance(Cov, R),
                            np.einsum('...ik,...kl,...lj', R, Cov, R.T))
        
        # Check array, first axis
        R = np.random.randn(2,2)
        Cov = np.random.randn(2,3,3,2,3,3)
        self.assertAllClose(gaussian.rotate_covariance(Cov, R,
                                                       ndim=3,
                                                       axis=-3),
                            np.einsum('...ik,...kablcd,...lj->...iabjcd', 
                                      R, 
                                      Cov,
                                      R.T))
        self.assertAllClose(gaussian.rotate_covariance(Cov, R,
                                                       ndim=3,
                                                       axis=0),
                            np.einsum('...ik,...kablcd,...lj->...iabjcd', 
                                      R, 
                                      Cov,
                                      R.T))
        
        # Check array, middle axis
        R = np.random.randn(2,2)
        Cov = np.random.randn(3,2,3,3,2,3)
        self.assertAllClose(gaussian.rotate_covariance(Cov, R,
                                                       ndim=3,
                                                       axis=-2),
                            np.einsum('...ik,...akbcld,...lj->...aibcjd', 
                                      R, 
                                      Cov,
                                      R.T))
        self.assertAllClose(gaussian.rotate_covariance(Cov, R,
                                                       ndim=3,
                                                       axis=1),
                            np.einsum('...ik,...akbcld,...lj->...aibcjd', 
                                      R, 
                                      Cov,
                                      R.T))

        # Check array, last axis
        R = np.random.randn(2,2)
        Cov = np.random.randn(3,3,2,3,3,2)
        self.assertAllClose(gaussian.rotate_covariance(Cov, R,
                                                       ndim=3,
                                                       axis=-1),
                            np.einsum('...ik,...abkcdl,...lj->...abicdj', 
                                      R, 
                                      Cov,
                                      R.T))
        self.assertAllClose(gaussian.rotate_covariance(Cov, R,
                                                       ndim=3,
                                                       axis=2),
                            np.einsum('...ik,...abkcdl,...lj->...abicdj', 
                                      R, 
                                      Cov,
                                      R.T))

        # Check array, middle axis with plates
        R = np.random.randn(2,2)
        Cov = np.random.randn(4,4,3,2,3,3,2,3)
        self.assertAllClose(gaussian.rotate_covariance(Cov, R,
                                                       ndim=3,
                                                       axis=-2),
                            np.einsum('...ik,...akbcld,...lj->...aibcjd', 
                                      R, 
                                      Cov,
                                      R.T))
        self.assertAllClose(gaussian.rotate_covariance(Cov, R,
                                                       ndim=3,
                                                       axis=1),
                            np.einsum('...ik,...akbcld,...lj->...aibcjd', 
                                      R, 
                                      Cov,
                                      R.T))

        pass

    
class TestGaussianARD(TestCase):

    def test_init(self):
        """
        Test the constructor
        """
        
        def check_init(true_plates, true_shape, mu, alpha, **kwargs):
            X = GaussianARD(mu, alpha, **kwargs)
            self.assertEqual(X.dims, (true_shape, true_shape+true_shape),
                             msg="Constructed incorrect dimensionality")
            self.assertEqual(X.plates, true_plates,
                             msg="Constructed incorrect plates")

        #
        # Create from constant parents
        #

        # Take the broadcasted shape of the parents
        check_init((), 
                   (), 
                   0, 
                   1)
        check_init((),
                   (3,2),
                   np.zeros((3,2,)),
                   np.ones((2,)))
        check_init((),
                   (4,2,2,3),
                   np.zeros((2,1,3,)),
                   np.ones((4,1,2,3)))
        # Use ndim
        check_init((4,2),
                   (2,3),
                   np.zeros((2,1,3,)),
                   np.ones((4,1,2,3)),
                   ndim=2)
        # Use shape
        check_init((4,2),
                   (2,3),
                   np.zeros((2,1,3,)),
                   np.ones((4,1,2,3)),
                   shape=(2,3))
        # Use ndim and shape
        check_init((4,2),
                   (2,3),
                   np.zeros((2,1,3,)),
                   np.ones((4,1,2,3)),
                   ndim=2,
                   shape=(2,3))

        #
        # Create from node parents
        #

        # Take broadcasted shape
        check_init((),
                   (4,2,2,3),
                   GaussianARD(np.zeros((2,1,3)),
                               np.ones((2,1,3))),
                   Gamma(np.ones((4,1,2,3)),
                         np.ones((4,1,2,3))))
        # Use ndim
        check_init((4,),
                   (2,2,3),
                   GaussianARD(np.zeros((4,2,3)),
                               np.ones((4,2,3)),
                               ndim=2),
                   Gamma(np.ones((4,2,1,3)),
                         np.ones((4,2,1,3))),
                   ndim=3)
        # Use shape
        check_init((4,),
                   (2,2,3),
                   GaussianARD(np.zeros((4,2,3)),
                               np.ones((4,2,3)),
                               ndim=2),
                   Gamma(np.ones((4,2,1,3)),
                         np.ones((4,2,1,3))),
                   shape=(2,2,3))
        # Use ndim and shape
        check_init((4,2),
                   (2,3),
                   GaussianARD(np.zeros((2,1,3)),
                               np.ones((2,1,3)),
                               ndim=2),
                   Gamma(np.ones((4,1,2,3)),
                         np.ones((4,1,2,3))),
                   ndim=2,
                   shape=(2,3))

        #
        # Errors
        #

        # Inconsistent dims of mu and alpha
        self.assertRaises(ValueError,
                          GaussianARD,
                          np.zeros((2,3)),
                          np.ones((2,)))
        # Inconsistent plates of mu and alpha
        self.assertRaises(ValueError,
                          GaussianARD,
                          GaussianARD(np.zeros((4,2,3)),
                                      np.ones((4,2,3)),
                                      ndim=2),
                          np.ones((3,4,2,3)),
                          ndim=3)
        # Inconsistent ndim and shape
        self.assertRaises(ValueError,
                          GaussianARD,
                          np.zeros((2,3)),
                          np.ones((2,)),
                          shape=(2,3),
                          ndim=1)
        # Parent mu has more axes
        self.assertRaises(ValueError,
                          GaussianARD,
                          GaussianARD(np.zeros((2,3)),
                                      np.ones((2,3))),
                          np.ones((2,3)),
                          ndim=1)
        # Incorrect shape
        self.assertRaises(ValueError,
                          GaussianARD,
                          GaussianARD(np.zeros((2,3)),
                                      np.ones((2,3)),
                                      ndim=2),
                          np.ones((2,3)),
                          shape=(2,2))
                          
        pass

    def test_message_to_child(self):
        """
        Test that GaussianARD computes the message to children correctly.
        """

        # Check that moments have full shape when broadcasting
        X = GaussianARD(np.zeros((2,)),
                        np.ones((3,2)),
                        shape=(4,3,2))
        (u0, u1) = X._message_to_child()
        self.assertEqual(np.shape(u0),
                         (4,3,2))
        self.assertEqual(np.shape(u1),
                         (4,3,2,4,3,2))

        # Check the formula
        X = GaussianARD(2, 3)
        (u0, u1) = X._message_to_child()
        self.assertAllClose(u0, 2)
        self.assertAllClose(u1, 2**2 + 1/3)

        # Check the formula for multidimensional arrays
        X = GaussianARD(2*np.ones((2,1,4)),
                        3*np.ones((2,3,1)),
                        ndim=3)
        (u0, u1) = X._message_to_child()
        self.assertAllClose(u0, 2*np.ones((2,3,4)))
        self.assertAllClose(u1, 
                            2**2 * np.ones((2,3,4,2,3,4))
                            + 1/3 * utils.identity(2,3,4))
                            

        # Check the formula for dim-broadcasted mu
        X = GaussianARD(2*np.ones((3,1)),
                        3*np.ones((2,3,4)),
                        ndim=3)
        (u0, u1) = X._message_to_child()
        self.assertAllClose(u0, 2*np.ones((2,3,4)))
        self.assertAllClose(u1, 
                            2**2 * np.ones((2,3,4,2,3,4))
                            + 1/3 * utils.identity(2,3,4))
                            
        # Check the formula for dim-broadcasted alpha
        X = GaussianARD(2*np.ones((2,3,4)),
                        3*np.ones((3,1)),
                        ndim=3)
        (u0, u1) = X._message_to_child()
        self.assertAllClose(u0, 2*np.ones((2,3,4)))
        self.assertAllClose(u1, 
                            2**2 * np.ones((2,3,4,2,3,4))
                            + 1/3 * utils.identity(2,3,4))
                            
        # Check the formula for dim-broadcasted mu and alpha
        X = GaussianARD(2*np.ones((3,1)),
                        3*np.ones((3,1)),
                        shape=(2,3,4))
        (u0, u1) = X._message_to_child()
        self.assertAllClose(u0, 2*np.ones((2,3,4)))
        self.assertAllClose(u1, 
                            2**2 * np.ones((2,3,4,2,3,4))
                            + 1/3 * utils.identity(2,3,4))
                            
        # Check the formula for dim-broadcasted mu with plates
        mu = GaussianARD(2*np.ones((5,3,4)),
                         np.ones((5,3,4)),
                         shape=(3,4),
                         plates=(5,))
        X = GaussianARD(mu,
                        3*np.ones((5,2,3,4)),
                        shape=(2,3,4),
                        plates=(5,))
        (u0, u1) = X._message_to_child()
        self.assertAllClose(u0, 2*np.ones((5,2,3,4)))
        self.assertAllClose(u1, 
                            2**2 * np.ones((5,2,3,4,2,3,4))
                            + 1/3 * utils.identity(2,3,4))

        # Check posterior
        X = GaussianARD(2, 3)
        Y = GaussianARD(X, 1)
        Y.observe(10)
        X.update()
        (u0, u1) = X._message_to_child()
        self.assertAllClose(u0,
                            1/(3+1) * (3*2 + 1*10))
        self.assertAllClose(u1,
                            (1/(3+1) * (3*2 + 1*10))**2 + 1/(3+1))
        
        pass

    def test_message_to_parent_mu(self):
        """
        Test that GaussianARD computes the message to the 1st parent correctly.
        """

        # Check formula with uncertain parent alpha
        alpha = Gamma(2,1)
        X = GaussianARD(0,
                        alpha)
        X.observe(3)
        (m0, m1) = X._message_to_parent(0)
        self.assertAllClose(m0,
                            2*3)
        self.assertAllClose(m1,
                            -0.5*2)

        # Check formula with uncertain node
        X = GaussianARD(1, 2)
        Y = GaussianARD(X, 1)
        Y.observe(5)
        X.update()
        (m0, m1) = X._message_to_parent(0)
        self.assertAllClose(m0,
                            2 * 1/(2+1)*(2*1+1*5))
        self.assertAllClose(m1,
                            -0.5*2)

        # Check alpha larger than mu
        X = GaussianARD(np.zeros((2,3)),
                        2*np.ones((3,2,3)))
        X.observe(3*np.ones((3,2,3)))
        (m0, m1) = X._message_to_parent(0)
        self.assertAllClose(m0,
                            2*3 * 3 * np.ones((2,3)))
        self.assertAllClose(m1,
                            -0.5 * 3 * 2*utils.identity(2,3))

        # Check mu larger than alpha
        X = GaussianARD(np.zeros((3,2,3)),
                        2*np.ones((2,3)))
        X.observe(3*np.ones((3,2,3)))
        (m0, m1) = X._message_to_parent(0)
        self.assertAllClose(m0,
                            2 * 3 * np.ones((3,2,3)))
        self.assertAllClose(m1,
                            -0.5 * 2*utils.identity(3,2,3))

        # Check node larger than mu and alpha
        X = GaussianARD(np.zeros((2,3)),
                        2*np.ones((3,)),
                        shape=(3,2,3))
        X.observe(3*np.ones((3,2,3)))
        (m0, m1) = X._message_to_parent(0)
        self.assertAllClose(m0,
                            2*3 * 3*np.ones((2,3)))
        self.assertAllClose(m1,
                            -0.5 * 2 * 3*utils.identity(2,3))

        # Check broadcasting of dimensions
        X = GaussianARD(np.zeros((2,1)),
                        2*np.ones((2,3)),
                        shape=(2,3))
        X.observe(3*np.ones((2,3)))
        (m0, m1) = X._message_to_parent(0)
        self.assertAllClose(m0,
                            2*3 * 3*np.ones((2,1)))
        self.assertAllClose(m1,
                            -0.5 * 2 * 3*utils.identity(2,1))

        # Check plates for smaller mu than node
        X = GaussianARD(GaussianARD(0,1, 
                                    shape=(3,),
                                    plates=(4,1)),
                        2*np.ones((3,)),
                        shape=(2,3),
                        plates=(4,5))
        X.observe(3*np.ones((4,5,2,3)))
        (m0, m1) = X._message_to_parent(0)
        self.assertAllClose(m0,
                            2*3 * 5*2*np.ones((4,1,3)))
        self.assertAllClose(m1,
                            -0.5*2 * 5*2*utils.identity(3))

        # Check mask
        X = GaussianARD(np.zeros((2,1,3)),
                        2*np.ones((2,4,3)),
                        shape=(3,),
                        plates=(2,4,))
        X.observe(3*np.ones((2,4,3)), mask=[[True, True, True, False],
                                            [False, True, False, True]])
        (m0, m1) = X._message_to_parent(0)
        self.assertAllClose(m0,
                            (2*3 * np.ones((2,1,3)) 
                             * np.array([[[3]], [[2]]])))
        self.assertAllClose(m1,
                            (-0.5*2 * utils.identity(3)
                             * np.ones((2,1,1,1))
                             * np.array([[[[3]]], [[[2]]]])))

        # Check non-ARD Gaussian child
        mu = np.array([1,2])
        alpha = np.array([3,4])
        Lambda = np.array([[1, 0.5],
                          [0.5, 1]])
        X = GaussianARD(mu, alpha)
        Y = Gaussian(X, Lambda)
        y = np.array([5,6])
        Y.observe(y)
        X.update()
        (m0, m1) = X._message_to_parent(0)
        mean = np.dot(np.linalg.inv(np.diag(alpha)+Lambda),
                      np.dot(np.diag(alpha), mu)
                      + np.dot(Lambda, y))
        self.assertAllClose(m0,
                            np.dot(np.diag(alpha), mean))
        self.assertAllClose(m1,
                            -0.5*np.diag(alpha))

        pass
        
    def test_message_to_parent_alpha(self):
        """
        Test that GaussianARD computes the message to the 2nd parent correctly.
        """

        # Check formula with uncertain parent mu
        mu = GaussianARD(1,1)
        X = GaussianARD(mu,
                        0.5)
        X.observe(3)
        (m0, m1) = X._message_to_parent(1)
        self.assertAllClose(m0,
                            -0.5*(3**2 - 2*3*1 + 1**2+1))
        self.assertAllClose(m1,
                            0.5)

        # Check formula with uncertain node
        X = GaussianARD(2, 1)
        Y = GaussianARD(X, 1)
        Y.observe(5)
        X.update()
        (m0, m1) = X._message_to_parent(1)
        self.assertAllClose(m0,
                            -0.5*(1/(1+1)+3.5**2 - 2*3.5*2 + 2**2))
        self.assertAllClose(m1,
                            0.5)

        # Check alpha larger than mu
        X = GaussianARD(np.ones((2,3)),
                        np.ones((3,2,3)))
        X.observe(2*np.ones((3,2,3)))
        (m0, m1) = X._message_to_parent(1)
        self.assertAllClose(m0,
                            -0.5*(2**2 - 2*2*1 + 1**2) * np.ones((3,2,3)))
        self.assertAllClose(m1,
                            0.5)

        # Check mu larger than alpha
        X = GaussianARD(np.ones((3,2,3)),
                        np.ones((2,3)))
        X.observe(2*np.ones((3,2,3)))
        (m0, m1) = X._message_to_parent(1)
        self.assertAllClose(m0,
                            -0.5*(2**2 - 2*2*1 + 1**2) * 3 * np.ones((2,3)))
        self.assertAllClose(m1,
                            0.5 * 3)

        # Check node larger than mu and alpha
        X = GaussianARD(np.ones((2,3)),
                        np.ones((3,)),
                        shape=(3,2,3))
        X.observe(2*np.ones((3,2,3)))
        (m0, m1) = X._message_to_parent(1)
        self.assertAllClose(m0,
                            -0.5*(2**2 - 2*2*1 + 1**2) * 6 * np.ones((3,)))
        self.assertAllClose(m1,
                            0.5 * 6)

        # Check plates for smaller mu than node
        X = GaussianARD(GaussianARD(1, 1, 
                                    shape=(3,),
                                    plates=(4,1)),
                        np.ones((4,1,2,3)),
                        shape=(2,3),
                        plates=(4,5))
        X.observe(2*np.ones((4,5,2,3)))
        (m0, m1) = X._message_to_parent(1)
        self.assertAllClose(m0,
                            (-0.5 * (2**2 - 2*2*1 + 1**2+1)
                             * 5*np.ones((4,1,2,3))))
        self.assertAllClose(m1,
                            5*0.5)

        # Check mask
        X = GaussianARD(np.ones(3),
                        np.ones((4,3)),
                        shape=(3,),
                        plates=(2,4,))
        X.observe(2*np.ones((2,4,3)), mask=[[True, False, True, False],
                                            [False, True, True, False]])
        (m0, m1) = X._message_to_parent(1)
        self.assertAllClose(m0,
                            (-0.5 * (2**2 - 2*2*1 + 1**2) 
                             * np.ones((3,)) 
                             * np.array([[1], [1], [2], [0]])))
        self.assertAllClose(m1,
                            0.5 * np.array([[1], [1], [2], [0]]))
        
        # Check non-ARD Gaussian child
        mu = np.array([1,2])
        alpha = np.array([3,4])
        Lambda = np.array([[1, 0.5],
                          [0.5, 1]])
        X = GaussianARD(mu, alpha)
        Y = Gaussian(X, Lambda)
        y = np.array([5,6])
        Y.observe(y)
        X.update()
        (m0, m1) = X._message_to_parent(1)
        Cov = np.linalg.inv(np.diag(alpha)+Lambda)
        mean = np.dot(Cov, np.dot(np.diag(alpha), mu)
                           + np.dot(Lambda, y))
        self.assertAllClose(m0,
                            -0.5 * np.diag(
                                np.outer(mean, mean) + Cov
                                - np.outer(mean, mu)
                                - np.outer(mu, mean)
                                + np.outer(mu, mu)))
        self.assertAllClose(m1,
                            0.5)
        
        pass
        

    def test_lowerbound(self):
        """
        Test the variational Bayesian lower bound term for GaussianARD.
        """

        # Test vector formula with full noise covariance
        m = np.random.randn(2)
        alpha = np.random.rand(2)
        y = np.random.randn(2)
        X = GaussianARD(m, alpha)
        V = np.array([[3,1],[1,3]])
        Y = Gaussian(X, V)
        Y.observe(y)
        X.update()
        Cov = np.linalg.inv(np.diag(alpha) + V)
        mu = np.dot(Cov, np.dot(V, y) + alpha*m)
        x2 = np.outer(mu, mu) + Cov
        logH_X = (+ 2*0.5*(1+np.log(2*np.pi)) 
                  + 0.5*np.log(np.linalg.det(Cov)))
        logp_X = (- 2*0.5*np.log(2*np.pi) 
                  + 0.5*np.log(np.linalg.det(np.diag(alpha)))
                  - 0.5*np.sum(np.diag(alpha)
                               * (x2 
                                  - np.outer(mu,m) 
                                  - np.outer(m,mu) 
                                  + np.outer(m,m))))
        self.assertAllClose(logp_X + logH_X,
                            X.lower_bound_contribution())

        def check_lower_bound(shape_mu, shape_alpha, plates_mu=(), **kwargs):
            M = GaussianARD(np.ones(plates_mu + shape_mu),
                            np.ones(plates_mu + shape_mu),
                            shape=shape_mu,
                            plates=plates_mu)
            X = GaussianARD(M,
                            2*np.ones(shape_alpha),
                            **kwargs)
            Y = GaussianARD(X,
                            3*np.ones(X.get_shape(0)),
                            **kwargs)
            Y.observe(4*np.ones(Y.get_shape(0)))
            X.update()
            Cov = 1/(2+3)
            mu = Cov * (2*1 + 3*4)
            x2 = mu**2 + Cov
            logH_X = (+ 0.5*(1+np.log(2*np.pi)) 
                      + 0.5*np.log(Cov))
            logp_X = (- 0.5*np.log(2*np.pi) 
                      + 0.5*np.log(2) 
                      - 0.5*2*(x2 - 2*mu*1 + 1**2+1))
            r = np.prod(X.get_shape(0))
            self.assertAllClose(r * (logp_X + logH_X),
                                X.lower_bound_contribution())
            
        # Test scalar formula
        check_lower_bound((), ())

        # Test array formula
        check_lower_bound((2,3), (2,3))

        # Test dim-broadcasting of mu
        check_lower_bound((3,1), (2,3,4))

        # Test dim-broadcasting of alpha
        check_lower_bound((2,3,4), (3,1))

        # Test dim-broadcasting of mu and alpha
        check_lower_bound((3,1), (3,1),
                          shape=(2,3,4))

        # Test dim-broadcasting of mu with plates
        check_lower_bound((), (),
                          plates_mu=(),
                          shape=(),
                          plates=(5,))

        # BUG: Scalar parents for array variable caused einsum error
        check_lower_bound((), (),
                          shape=(3,))
        
        # BUG: Log-det was summed over plates
        check_lower_bound((), (),
                          shape=(3,),
                          plates=(4,))

        pass

    def test_rotate(self):
        """
        Test the rotation of Gaussian ARD arrays.
        """

        def check(shape, plates, einsum_x, einsum_xx, axis=-1):
            # TODO/FIXME: Improve by having non-diagonal precision/covariance
            # parameter for the Gaussian X
            D = shape[axis]
            X = GaussianARD(np.random.randn(*(plates+shape)),
                            np.random.rand(*(plates+shape)),
                            shape=shape,
                            plates=plates)
            (x, xx) = X.get_moments()
            R = np.random.randn(D,D)
            X.rotate(R, axis=axis)
            (rx, rxxr) = X.get_moments()
            self.assertAllClose(rx,
                                np.einsum(einsum_x, R, x))
            self.assertAllClose(rxxr,
                                np.einsum(einsum_xx, R, xx, R))
            pass

        # Rotate vector
        check((3,), (),    
              '...jk,...k->...j', 
              '...mk,...kl,...nl->...mn')
        check((3,), (2,4), 
              '...jk,...k->...j', 
              '...mk,...kl,...nl->...mn')

        # Rotate array
        check((2,3,4), (), 
              '...jc,...abc->...abj', 
              '...mc,...abcdef,...nf->...abmden',
              axis=-1)
        check((2,3,4), (5,6), 
              '...jc,...abc->...abj', 
              '...mc,...abcdef,...nf->...abmden',
              axis=-1)
        check((2,3,4), (), 
              '...jb,...abc->...ajc', 
              '...mb,...abcdef,...ne->...amcdnf',
              axis=-2)
        check((2,3,4), (5,6), 
              '...jb,...abc->...ajc', 
              '...mb,...abcdef,...ne->...amcdnf',
              axis=-2)
        check((2,3,4), (), 
              '...ja,...abc->...jbc', 
              '...ma,...abcdef,...nd->...mbcnef',
              axis=-3)
        check((2,3,4), (5,6), 
              '...ja,...abc->...jbc', 
              '...ma,...abcdef,...nd->...mbcnef',
              axis=-3)
        
        pass

    def test_rotate_plates(self):

        # Basic test for Gaussian vectors
        X = GaussianARD(np.random.randn(3,2),
                        np.random.rand(3,2),
                        shape=(2,),
                        plates=(3,))
        (u0, u1) = X.get_moments()
        Cov = u1 - linalg.outer(u0, u0, ndim=1)
        Q = np.random.randn(3,3)
        Qu0 = np.einsum('ik,kj->ij', Q, u0)
        QCov = np.einsum('k,kij->kij', np.sum(Q, axis=0)**2, Cov)
        Qu1 = QCov + linalg.outer(Qu0, Qu0, ndim=1)
        X.rotate_plates(Q, plate_axis=-1)
        (u0, u1) = X.get_moments()
        self.assertAllClose(u0, Qu0)
        self.assertAllClose(u1, Qu1)

        # Test full covariance, that is, with observations
        X = GaussianARD(np.random.randn(3,2),
                        np.random.rand(3,2),
                        shape=(2,),
                        plates=(3,))
        Y = Gaussian(X, [[2.0, 1.5], [1.5, 3.0]],
                     plates=(3,))
        Y.observe(np.random.randn(3,2))
        X.update()
        (u0, u1) = X.get_moments()
        Cov = u1 - linalg.outer(u0, u0, ndim=1)
        Q = np.random.randn(3,3)
        Qu0 = np.einsum('ik,kj->ij', Q, u0)
        QCov = np.einsum('k,kij->kij', np.sum(Q, axis=0)**2, Cov)
        Qu1 = QCov + linalg.outer(Qu0, Qu0, ndim=1)
        X.rotate_plates(Q, plate_axis=-1)
        (u0, u1) = X.get_moments()
        self.assertAllClose(u0, Qu0)
        self.assertAllClose(u1, Qu1)

        pass
        
