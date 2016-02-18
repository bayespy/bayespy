################################################################################
# Copyright (C) 2013-2015 Jaakko Luttinen
#
# This file is licensed under the MIT License.
################################################################################


"""
Unit tests for `gaussian` module.
"""

import numpy as np

from scipy import special

from numpy import testing

from .. import gaussian
from bayespy.nodes import (Gaussian, 
                           GaussianARD,
                           GaussianGamma,
                           Gamma,
                           Wishart)

from ...vmp import VB

from bayespy.utils import misc
from bayespy.utils import linalg
from bayespy.utils import random

from bayespy.utils.misc import TestCase

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
        Test the constructor of GaussianARD
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

        # Use ndim=0 for constant mu
        check_init((), 
                   (), 
                   0, 
                   1)
        check_init((3,2),
                   (),
                   np.zeros((3,2,)),
                   np.ones((2,)))
        check_init((4,2,2,3),
                   (),
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

        # ndim=0 by default
        check_init((3,),
                   (),
                   GaussianARD(0, 1,
                               plates=(3,)),
                   Gamma(1, 1,
                         plates=(3,)))

        check_init((4,2,2,3),
                   (),
                   GaussianARD(np.zeros((2,1,3)),
                               np.ones((2,1,3)),
                               ndim=3),
                   Gamma(np.ones((4,1,2,3)),
                         np.ones((4,1,2,3))))
        # Use ndim
        check_init((4,),
                   (2,2,3),
                   GaussianARD(np.zeros((4,1,2,3)),
                               np.ones((4,1,2,3)),
                               ndim=2),
                   Gamma(np.ones((4,2,1,3)),
                         np.ones((4,2,1,3))),
                   ndim=3)
        # Use shape
        check_init((4,),
                   (2,2,3),
                   GaussianARD(np.zeros((4,1,2,3)),
                               np.ones((4,1,2,3)),
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

        # Test for a found bug
        check_init((),
                   (3,),
                   np.ones(3),
                   1,
                   ndim=1)

        # Parent mu has more axes
        check_init(
            (2,),
            (3,),
            GaussianARD(np.zeros((2,3)),
                        np.ones((2,3)),
                        ndim=2),
            np.ones((2,3)),
            ndim=1
        )
        # DO NOT add axes if necessary
        self.assertRaises(
            ValueError,
            GaussianARD,
            GaussianARD(np.zeros((2,3)),
                        np.ones((2,3)),
                        ndim=2),
            1,
            ndim=3
        )

        #
        # Errors
        #

        # Inconsistent shapes
        self.assertRaises(ValueError,
                          GaussianARD,
                          GaussianARD(np.zeros((2,3)),
                                      np.ones((2,3)),
                                      ndim=1),
                          np.ones((4,3)),
                          ndim=2)

        # Inconsistent dims of mu and alpha
        self.assertRaises(ValueError,
                          GaussianARD,
                          np.zeros((2,3)),
                          np.ones((2,)))
        # Inconsistent plates of mu and alpha
        self.assertRaises(ValueError,
                          GaussianARD,
                          GaussianARD(np.zeros((3,2,3)),
                                      np.ones((3,2,3)),
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
        Test moments of GaussianARD.
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
                            + 1/3 * misc.identity(2,3,4))
                            

        # Check the formula for dim-broadcasted mu
        X = GaussianARD(2*np.ones((3,1)),
                        3*np.ones((2,3,4)),
                        ndim=3)
        (u0, u1) = X._message_to_child()
        self.assertAllClose(u0, 2*np.ones((2,3,4)))
        self.assertAllClose(u1, 
                            2**2 * np.ones((2,3,4,2,3,4))
                            + 1/3 * misc.identity(2,3,4))
                            
        # Check the formula for dim-broadcasted alpha
        X = GaussianARD(2*np.ones((2,3,4)),
                        3*np.ones((3,1)),
                        ndim=3)
        (u0, u1) = X._message_to_child()
        self.assertAllClose(u0, 2*np.ones((2,3,4)))
        self.assertAllClose(u1, 
                            2**2 * np.ones((2,3,4,2,3,4))
                            + 1/3 * misc.identity(2,3,4))
                            
        # Check the formula for dim-broadcasted mu and alpha
        X = GaussianARD(2*np.ones((3,1)),
                        3*np.ones((3,1)),
                        shape=(2,3,4))
        (u0, u1) = X._message_to_child()
        self.assertAllClose(u0, 2*np.ones((2,3,4)))
        self.assertAllClose(u1, 
                            2**2 * np.ones((2,3,4,2,3,4))
                            + 1/3 * misc.identity(2,3,4))
                            
        # Check the formula for dim-broadcasted mu with plates
        mu = GaussianARD(2*np.ones((5,1,3,4)),
                         np.ones((5,1,3,4)),
                         shape=(3,4),
                         plates=(5,1))
        X = GaussianARD(mu,
                        3*np.ones((5,2,3,4)),
                        shape=(2,3,4),
                        plates=(5,))
        (u0, u1) = X._message_to_child()
        self.assertAllClose(u0, 2*np.ones((5,2,3,4)))
        self.assertAllClose(u1, 
                            2**2 * np.ones((5,2,3,4,2,3,4))
                            + 1/3 * misc.identity(2,3,4))

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
        mu = GaussianARD(0, 1)
        alpha = Gamma(2,1)
        X = GaussianARD(mu,
                        alpha)
        X.observe(3)
        (m0, m1) = mu._message_from_children()
        #(m0, m1) = X._message_to_parent(0)
        self.assertAllClose(m0,
                            2*3)
        self.assertAllClose(m1,
                            -0.5*2)

        # Check formula with uncertain node
        mu = GaussianARD(1, 1e10)
        X = GaussianARD(mu, 2)
        Y = GaussianARD(X, 1)
        Y.observe(5)
        X.update()
        (m0, m1) = mu._message_from_children()
        self.assertAllClose(m0,
                            2 * 1/(2+1)*(2*1+1*5))
        self.assertAllClose(m1,
                            -0.5*2)

        # Check alpha larger than mu
        mu = GaussianARD(np.zeros((2,3)), 1e10, shape=(2,3))
        X = GaussianARD(mu,
                        2*np.ones((3,2,3)))
        X.observe(3*np.ones((3,2,3)))
        (m0, m1) = mu._message_from_children()
        self.assertAllClose(m0,
                            2*3 * 3 * np.ones((2,3)))
        self.assertAllClose(m1,
                            -0.5 * 3 * 2*misc.identity(2,3))

        # Check mu larger than alpha
        mu = GaussianARD(np.zeros((3,2,3)), 1e10, shape=(3,2,3))
        X = GaussianARD(mu,
                        2*np.ones((2,3)))
        X.observe(3*np.ones((3,2,3)))
        (m0, m1) = mu._message_from_children()
        self.assertAllClose(m0,
                            2 * 3 * np.ones((3,2,3)))
        self.assertAllClose(m1,
                            -0.5 * 2*misc.identity(3,2,3))

        # Check node larger than mu and alpha
        mu = GaussianARD(np.zeros((2,3)), 1e10, shape=(2,3))
        X = GaussianARD(mu,
                        2*np.ones((3,)),
                        shape=(3,2,3))
        X.observe(3*np.ones((3,2,3)))
        (m0, m1) = mu._message_from_children()
        self.assertAllClose(m0,
                            2*3 * 3*np.ones((2,3)))
        self.assertAllClose(m1,
                            -0.5 * 2 * 3*misc.identity(2,3))

        # Check broadcasting of dimensions
        mu = GaussianARD(np.zeros((2,1)), 1e10, shape=(2,1))
        X = GaussianARD(mu,
                        2*np.ones((2,3)),
                        shape=(2,3))
        X.observe(3*np.ones((2,3)))
        (m0, m1) = mu._message_from_children()
        self.assertAllClose(m0,
                            2*3 * 3*np.ones((2,1)))
        self.assertAllClose(m1,
                            -0.5 * 2 * 3*misc.identity(2,1))

        # Check plates for smaller mu than node
        mu = GaussianARD(0,1, 
                         shape=(3,),
                         plates=(4,1,1))
        X = GaussianARD(mu,
                        2*np.ones((3,)),
                        shape=(2,3),
                        plates=(4,5))
        X.observe(3*np.ones((4,5,2,3)))
        (m0, m1) = mu._message_from_children()
        self.assertAllClose(m0 * np.ones((4,1,1,3)),
                            2*3 * 5*2*np.ones((4,1,1,3)))
        self.assertAllClose(m1 * np.ones((4,1,1,3,3)),
                            -0.5*2 * 5*2*misc.identity(3) * np.ones((4,1,1,3,3)))

        # Check mask
        mu = GaussianARD(np.zeros((2,1,3)), 1e10, shape=(3,))
        X = GaussianARD(mu,
                        2*np.ones((2,4,3)),
                        shape=(3,),
                        plates=(2,4,))
        X.observe(3*np.ones((2,4,3)), mask=[[True, True, True, False],
                                            [False, True, False, True]])
        (m0, m1) = mu._message_from_children()
        self.assertAllClose(m0,
                            (2*3 * np.ones((2,1,3)) 
                             * np.array([[[3]], [[2]]])))
        self.assertAllClose(m1,
                            (-0.5*2 * misc.identity(3)
                             * np.ones((2,1,1,1))
                             * np.array([[[[3]]], [[[2]]]])))

        # Check mask with different shapes
        mu = GaussianARD(np.zeros((2,1,3)), 1e10, shape=())
        X = GaussianARD(mu,
                        2*np.ones((2,4,3)),
                        shape=(3,),
                        plates=(2,4,))
        mask = np.array([[True, True, True, False],
                         [False, True, False, True]])
        X.observe(3*np.ones((2,4,3)), mask=mask)
        (m0, m1) = mu._message_from_children()
        self.assertAllClose(m0,
                            2*3 * np.sum(np.ones((2,4,3))*mask[...,None], 
                                         axis=-2,
                                         keepdims=True))
        self.assertAllClose(m1,
                            (-0.5*2 * np.sum(np.ones((2,4,3))*mask[...,None],
                                             axis=-2,
                                             keepdims=True)))

        # Check non-ARD Gaussian child
        mu = np.array([1,2])
        Mu = GaussianARD(mu, 1e10, shape=(2,))
        alpha = np.array([3,4])
        Lambda = np.array([[1, 0.5],
                          [0.5, 1]])
        X = GaussianARD(Mu, alpha, ndim=1)
        Y = Gaussian(X, Lambda)
        y = np.array([5,6])
        Y.observe(y)
        X.update()
        (m0, m1) = Mu._message_from_children()
        mean = np.dot(np.linalg.inv(np.diag(alpha)+Lambda),
                      np.dot(np.diag(alpha), mu)
                      + np.dot(Lambda, y))
        self.assertAllClose(m0,
                            np.dot(np.diag(alpha), mean))
        self.assertAllClose(m1,
                            -0.5*np.diag(alpha))

        # Check broadcasted variable axes
        mu = GaussianARD(np.zeros(1), 1e10, shape=(1,))
        X = GaussianARD(mu,
                        2,
                        shape=(3,))
        X.observe(3*np.ones(3))
        (m0, m1) = mu._message_from_children()
        self.assertAllClose(m0,
                            2*3 * np.sum(np.ones(3), axis=-1, keepdims=True))
        self.assertAllClose(m1,
                            -0.5*2 * np.sum(np.identity(3), 
                                            axis=(-1,-2), 
                                            keepdims=True))

        pass
        
    def test_message_to_parent_alpha(self):
        """
        Test the message from GaussianARD the 2nd parent (alpha).
        """

        # Check formula with uncertain parent mu
        mu = GaussianARD(1,1)
        tau = Gamma(0.5*1e10, 1e10)
        X = GaussianARD(mu,
                        tau)
        X.observe(3)
        (m0, m1) = tau._message_from_children()
        self.assertAllClose(m0,
                            -0.5*(3**2 - 2*3*1 + 1**2+1))
        self.assertAllClose(m1,
                            0.5)

        # Check formula with uncertain node
        tau = Gamma(1e10, 1e10)
        X = GaussianARD(2, tau)
        Y = GaussianARD(X, 1)
        Y.observe(5)
        X.update()
        (m0, m1) = tau._message_from_children()
        self.assertAllClose(m0,
                            -0.5*(1/(1+1)+3.5**2 - 2*3.5*2 + 2**2))
        self.assertAllClose(m1,
                            0.5)

        # Check alpha larger than mu
        alpha = Gamma(np.ones((3,2,3))*1e10, 1e10)
        X = GaussianARD(np.ones((2,3)),
                        alpha,
                        ndim=3)
        X.observe(2*np.ones((3,2,3)))
        (m0, m1) = alpha._message_from_children()
        self.assertAllClose(m0 * np.ones((3,2,3)),
                            -0.5*(2**2 - 2*2*1 + 1**2) * np.ones((3,2,3)))
        self.assertAllClose(m1*np.ones((3,2,3)),
                            0.5*np.ones((3,2,3)))

        # Check mu larger than alpha
        tau = Gamma(np.ones((2,3))*1e10, 1e10)
        X = GaussianARD(np.ones((3,2,3)),
                        tau,
                        ndim=3)
        X.observe(2*np.ones((3,2,3)))
        (m0, m1) = tau._message_from_children()
        self.assertAllClose(m0,
                            -0.5*(2**2 - 2*2*1 + 1**2) * 3 * np.ones((2,3)))
        self.assertAllClose(m1 * np.ones((2,3)),
                            0.5 * 3 * np.ones((2,3)))

        # Check node larger than mu and alpha
        tau = Gamma(np.ones((3,))*1e10, 1e10)
        X = GaussianARD(np.ones((2,3)),
                        tau,
                        shape=(3,2,3))
        X.observe(2*np.ones((3,2,3)))
        (m0, m1) = tau._message_from_children()
        self.assertAllClose(m0 * np.ones(3),
                            -0.5*(2**2 - 2*2*1 + 1**2) * 6 * np.ones((3,)))
        self.assertAllClose(m1 * np.ones(3),
                            0.5 * 6 * np.ones(3))

        # Check plates for smaller mu than node
        tau = Gamma(np.ones((4,1,2,3))*1e10, 1e10)
        X = GaussianARD(GaussianARD(1, 1, 
                                    shape=(3,),
                                    plates=(4,1,1)),
                        tau,
                        shape=(2,3),
                        plates=(4,5))
        X.observe(2*np.ones((4,5,2,3)))
        (m0, m1) = tau._message_from_children()
        self.assertAllClose(m0 * np.ones((4,1,2,3)),
                            (-0.5 * (2**2 - 2*2*1 + 1**2+1)
                             * 5*np.ones((4,1,2,3))))
        self.assertAllClose(m1 * np.ones((4,1,2,3)),
                            5*0.5 * np.ones((4,1,2,3)))

        # Check mask
        tau = Gamma(np.ones((4,3))*1e10, 1e10)
        X = GaussianARD(np.ones(3),
                        tau,
                        shape=(3,),
                        plates=(2,4,))
        X.observe(2*np.ones((2,4,3)), mask=[[True, False, True, False],
                                            [False, True, True, False]])
        (m0, m1) = tau._message_from_children()
        self.assertAllClose(m0 * np.ones((4,3)),
                            (-0.5 * (2**2 - 2*2*1 + 1**2) 
                             * np.ones((4,3)) 
                             * np.array([[1], [1], [2], [0]])))
        self.assertAllClose(m1 * np.ones((4,3)),
                            0.5 * np.array([[1], [1], [2], [0]]) * np.ones((4,3)))
        
        # Check non-ARD Gaussian child
        mu = np.array([1,2])
        alpha = np.array([3,4])
        Alpha = Gamma(alpha*1e10, 1e10)
        Lambda = np.array([[1, 0.5],
                          [0.5, 1]])
        X = GaussianARD(mu, Alpha, ndim=1)
        Y = Gaussian(X, Lambda)
        y = np.array([5,6])
        Y.observe(y)
        X.update()
        (m0, m1) = Alpha._message_from_children()
        Cov = np.linalg.inv(np.diag(alpha)+Lambda)
        mean = np.dot(Cov, np.dot(np.diag(alpha), mu)
                           + np.dot(Lambda, y))
        self.assertAllClose(m0 * np.ones(2),
                            -0.5 * np.diag(
                                np.outer(mean, mean) + Cov
                                - np.outer(mean, mu)
                                - np.outer(mu, mean)
                                + np.outer(mu, mu)))
        self.assertAllClose(m1 * np.ones(2),
                            0.5 * np.ones(2))
        
        pass
        

    def test_lowerbound(self):
        """
        Test the variational Bayesian lower bound term for GaussianARD.
        """

        # Test vector formula with full noise covariance
        m = np.random.randn(2)
        alpha = np.random.rand(2)
        y = np.random.randn(2)
        X = GaussianARD(m, alpha, ndim=1)
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
            if not ('ndim' in kwargs or 'shape' in kwargs):
                kwargs['ndim'] = len(shape_mu)
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


    def test_initialization(self):
        """
        Test initialization methods of GaussianARD
        """

        X = GaussianARD(1, 2, shape=(2,), plates=(3,))

        # Prior initialization
        mu = 1 * np.ones((3, 2))
        alpha = 2 * np.ones((3, 2))
        X.initialize_from_prior()
        u = X._message_to_child()
        self.assertAllClose(u[0]*np.ones((3,2)), 
                            mu)
        self.assertAllClose(u[1]*np.ones((3,2,2)), 
                            linalg.outer(mu, mu, ndim=1) + 
                            misc.diag(1/alpha, ndim=1))

        # Parameter initialization
        mu = np.random.randn(3, 2)
        alpha = np.random.rand(3, 2)
        X.initialize_from_parameters(mu, alpha)
        u = X._message_to_child()
        self.assertAllClose(u[0], mu)
        self.assertAllClose(u[1], linalg.outer(mu, mu, ndim=1) + 
                                  misc.diag(1/alpha, ndim=1))

        # Value initialization
        x = np.random.randn(3, 2)
        X.initialize_from_value(x)
        u = X._message_to_child()
        self.assertAllClose(u[0], x)
        self.assertAllClose(u[1], linalg.outer(x, x, ndim=1))

        # Random initialization
        X.initialize_from_random()

        pass
        

class TestGaussianGamma(TestCase):
    """
    Unit tests for GaussianGamma node.
    """
    

    def test_init(self):
        """
        Test the creation of GaussianGamma node
        """

        # Simple construction
        X_alpha = GaussianGamma([1,2,3], np.identity(3), 2, 10)
        self.assertEqual(X_alpha.plates, ())
        self.assertEqual(X_alpha.dims, ( (3,), (3,3), (), () ))

        # Plates
        X_alpha = GaussianGamma([1,2,3], np.identity(3), 2, 10, plates=(4,))
        self.assertEqual(X_alpha.plates, (4,))
        self.assertEqual(X_alpha.dims, ( (3,), (3,3), (), () ))

        # Plates in mu
        X_alpha = GaussianGamma(np.ones((4,3)), np.identity(3), 2, 10)
        self.assertEqual(X_alpha.plates, (4,))
        self.assertEqual(X_alpha.dims, ( (3,), (3,3), (), () ))
        
        # Plates in Lambda
        X_alpha = GaussianGamma(np.ones(3), np.ones((4,3,3))*np.identity(3), 2, 10)
        self.assertEqual(X_alpha.plates, (4,))
        self.assertEqual(X_alpha.dims, ( (3,), (3,3), (), () ))
        
        # Plates in a
        X_alpha = GaussianGamma(np.ones(3), np.identity(3), np.ones(4), 10)
        self.assertEqual(X_alpha.plates, (4,))
        self.assertEqual(X_alpha.dims, ( (3,), (3,3), (), () ))
        
        # Plates in Lambda
        X_alpha = GaussianGamma(np.ones(3), np.identity(3), 2, np.ones(4))
        self.assertEqual(X_alpha.plates, (4,))
        self.assertEqual(X_alpha.dims, ( (3,), (3,3), (), () ))

        # Inconsistent plates
        self.assertRaises(ValueError,
                          GaussianGamma,
                          np.ones((4,3)),
                          np.identity(3), 
                          2,
                          10,
                          plates=())
        
        # Inconsistent plates
        self.assertRaises(ValueError,
                          GaussianGamma,
                          np.ones((4,3)),
                          np.identity(3), 
                          2,
                          10,
                          plates=(5,))

        # Unknown parameters
        mu = Gaussian(np.zeros(3), np.identity(3))
        Lambda = Wishart(10, np.identity(3))
        b = Gamma(1, 1)
        X_alpha = GaussianGamma(mu, Lambda, 2, b)
        self.assertEqual(X_alpha.plates, ())
        self.assertEqual(X_alpha.dims, ( (3,), (3,3), (), () ))

        # mu is Gaussian-gamma
        mu_tau = GaussianGamma(np.ones(3), np.identity(3), 5, 5)
        X_alpha = GaussianGamma(mu_tau, np.identity(3), 5, 5)
        self.assertEqual(X_alpha.plates, ())
        self.assertEqual(X_alpha.dims, ( (3,), (3,3), (), () ))
        
        pass
        

    def test_message_to_child(self):
        """
        Test the message to child of GaussianGamma node.
        """

        # Simple test
        mu = np.array([1,2,3])
        Lambda = np.identity(3)
        a = 2
        b = 10
        X_alpha = GaussianGamma(mu, Lambda, a, b)
        u = X_alpha._message_to_child()
        self.assertEqual(len(u), 4)
        tau = np.array(a/b)
        self.assertAllClose(u[0],
                            tau[...,None] * mu)
        self.assertAllClose(u[1],
                            (linalg.inv(Lambda) 
                             + tau[...,None,None] * linalg.outer(mu, mu)))
        self.assertAllClose(u[2],
                            tau)
        self.assertAllClose(u[3],
                            -np.log(b) + special.psi(a))

        # Test with unknown parents
        mu = Gaussian(np.arange(3), 10*np.identity(3))
        Lambda = Wishart(10, np.identity(3))
        a = 2
        b = Gamma(3, 15)
        X_alpha = GaussianGamma(mu, Lambda, a, b)
        u = X_alpha._message_to_child()
        (mu, mumu) = mu._message_to_child()
        Cov_mu = mumu - linalg.outer(mu, mu)
        (Lambda, _) = Lambda._message_to_child()
        (b, _) = b._message_to_child()
        (tau, logtau) = Gamma(a, b + 0.5*np.sum(Lambda*Cov_mu))._message_to_child()
        self.assertAllClose(u[0],
                            tau[...,None] * mu)
        self.assertAllClose(u[1],
                            (linalg.inv(Lambda)
                             + tau[...,None,None] * linalg.outer(mu, mu)))
        self.assertAllClose(u[2],
                            tau)
        self.assertAllClose(u[3],
                            logtau)

        # Test with plates
        mu = Gaussian(np.reshape(np.arange(3*4), (4,3)),
                      10*np.identity(3),
                      plates=(4,))
        Lambda = Wishart(10, np.identity(3))
        a = 2
        b = Gamma(3, 15)
        X_alpha = GaussianGamma(mu, Lambda, a, b, plates=(4,))
        u = X_alpha._message_to_child()
        (mu, mumu) = mu._message_to_child()
        Cov_mu = mumu - linalg.outer(mu, mu)
        (Lambda, _) = Lambda._message_to_child()
        (b, _) = b._message_to_child()
        (tau, logtau) = Gamma(a, 
                              b + 0.5*np.sum(Lambda*Cov_mu, 
                                             axis=(-1,-2)))._message_to_child()
        self.assertAllClose(u[0] * np.ones((4,1)),
                            np.ones((4,1)) * tau[...,None] * mu)
        self.assertAllClose(u[1] * np.ones((4,1,1)),
                            np.ones((4,1,1)) * (linalg.inv(Lambda)
                                                + tau[...,None,None] * linalg.outer(mu, mu)))
        self.assertAllClose(u[2] * np.ones(4),
                            np.ones(4) * tau)
        self.assertAllClose(u[3] * np.ones(4),
                            np.ones(4) * logtau)
        
        pass


    def test_mask_to_parent(self):
        """
        Test the mask handling in GaussianGamma node
        """

        pass


class TestGaussianGradient(TestCase):
    """Numerically check Riemannian gradient of several nodes.
    
    Using VB-EM update equations will take a unit length step to the
    Riemannian gradient direction.  Thus, the change caused by a VB-EM
    update and the Riemannian gradient should be equal.
    """


    def test_riemannian_gradient(self):
        """Test Riemannian gradient of a Gaussian node."""
        D = 3

        #
        # Without observations
        #
        
        # Construct model
        mu = np.random.randn(D)
        Lambda = random.covariance(D)
        X = Gaussian(mu, Lambda)
        # Random initialization
        mu0 = np.random.randn(D)
        Lambda0 = random.covariance(D)
        X.initialize_from_parameters(mu0, Lambda0)
        # Initial parameters 
        phi0 = X.phi
        # Gradient
        g = X.get_riemannian_gradient()
        # Parameters after VB-EM update
        X.update()
        phi1 = X.phi
        # Check
        self.assertAllClose(g[0],
                            phi1[0] - phi0[0])
        self.assertAllClose(g[1],
                            phi1[1] - phi0[1])

        # TODO/FIXME: Actually, gradient should be zero because cost function
        # is zero without observations! Use the mask!

        #
        # With observations
        #
        
        # Construct model
        mu = np.random.randn(D)
        Lambda = random.covariance(D)
        X = Gaussian(mu, Lambda)
        V = random.covariance(D)
        Y = Gaussian(X, V)
        Y.observe(np.random.randn(D))
        # Random initialization
        mu0 = np.random.randn(D)
        Lambda0 = random.covariance(D)
        X.initialize_from_parameters(mu0, Lambda0)
        # Initial parameters 
        phi0 = X.phi
        # Gradient
        g = X.get_riemannian_gradient()
        # Parameters after VB-EM update
        X.update()
        phi1 = X.phi
        # Check
        self.assertAllClose(g[0],
                            phi1[0] - phi0[0])
        self.assertAllClose(g[1],
                            phi1[1] - phi0[1])

        pass
        

    def test_gradient(self):
        """Test standard gradient of a Gaussian node."""
        D = 3

        np.random.seed(42)

        #
        # Without observations
        #
        
        # Construct model
        mu = np.random.randn(D)
        Lambda = random.covariance(D)
        X = Gaussian(mu, Lambda)
        # Random initialization
        mu0 = np.random.randn(D)
        Lambda0 = random.covariance(D)
        X.initialize_from_parameters(mu0, Lambda0)
        Q = VB(X)
        # Initial parameters 
        phi0 = X.phi
        # Gradient
        rg = X.get_riemannian_gradient()
        g = X.get_gradient(rg)
        # Numerical gradient
        eps = 1e-6
        p0 = X.get_parameters()
        l0 = Q.compute_lowerbound(ignore_masked=False)
        g_num = [np.zeros(D), np.zeros((D,D))]
        for i in range(D):
            e = np.zeros(D)
            e[i] = eps
            p1 = p0[0] + e
            X.set_parameters([p1, p0[1]])
            l1 = Q.compute_lowerbound(ignore_masked=False)
            g_num[0][i] = (l1 - l0) / eps
        for i in range(D):
            for j in range(i+1):
                e = np.zeros((D,D))
                e[i,j] += eps
                e[j,i] += eps
                p1 = p0[1] + e
                X.set_parameters([p0[0], p1])
                l1 = Q.compute_lowerbound(ignore_masked=False)
                g_num[1][i,j] = (l1 - l0) / (2*eps)
                g_num[1][j,i] = (l1 - l0) / (2*eps)
                
        # Check
        self.assertAllClose(g[0],
                            g_num[0])
        self.assertAllClose(g[1],
                            g_num[1])

        #
        # With observations
        #
        
        # Construct model
        mu = np.random.randn(D)
        Lambda = random.covariance(D)
        X = Gaussian(mu, Lambda)
        # Random initialization
        mu0 = np.random.randn(D)
        Lambda0 = random.covariance(D)
        X.initialize_from_parameters(mu0, Lambda0)
        V = random.covariance(D)
        Y = Gaussian(X, V)
        Y.observe(np.random.randn(D))
        Q = VB(Y, X)
        # Initial parameters 
        phi0 = X.phi
        # Gradient
        rg = X.get_riemannian_gradient()
        g = X.get_gradient(rg)
        # Numerical gradient
        eps = 1e-6
        p0 = X.get_parameters()
        l0 = Q.compute_lowerbound()
        g_num = [np.zeros(D), np.zeros((D,D))]
        for i in range(D):
            e = np.zeros(D)
            e[i] = eps
            p1 = p0[0] + e
            X.set_parameters([p1, p0[1]])
            l1 = Q.compute_lowerbound()
            g_num[0][i] = (l1 - l0) / eps
        for i in range(D):
            for j in range(i+1):
                e = np.zeros((D,D))
                e[i,j] += eps
                e[j,i] += eps
                p1 = p0[1] + e
                X.set_parameters([p0[0], p1])
                l1 = Q.compute_lowerbound()
                g_num[1][i,j] = (l1 - l0) / (2*eps)
                g_num[1][j,i] = (l1 - l0) / (2*eps)
                
        # Check
        self.assertAllClose(g[0],
                            g_num[0])
        self.assertAllClose(g[1],
                            g_num[1])


        #
        # With plates
        #

        # Construct model
        K = D+1
        mu = np.random.randn(D)
        Lambda = random.covariance(D)
        X = Gaussian(mu, Lambda, plates=(K,))
        V = random.covariance(D, size=(K,))
        Y = Gaussian(X, V)
        Y.observe(np.random.randn(K,D))
        Q = VB(Y, X)
        # Random initialization
        mu0 = np.random.randn(*(X.get_shape(0)))
        Lambda0 = random.covariance(D, size=X.plates)
        X.initialize_from_parameters(mu0, Lambda0)
        # Initial parameters 
        phi0 = X.phi
        # Gradient
        rg = X.get_riemannian_gradient()
        g = X.get_gradient(rg)
        # Numerical gradient
        eps = 1e-6
        p0 = X.get_parameters()
        l0 = Q.compute_lowerbound()
        g_num = [np.zeros(X.get_shape(0)), np.zeros(X.get_shape(1))]
        for k in range(K):
            for i in range(D):
                e = np.zeros(X.get_shape(0))
                e[k,i] = eps
                p1 = p0[0] + e
                X.set_parameters([p1, p0[1]])
                l1 = Q.compute_lowerbound()
                g_num[0][k,i] = (l1 - l0) / eps
            for i in range(D):
                for j in range(i+1):
                    e = np.zeros(X.get_shape(1))
                    e[k,i,j] += eps
                    e[k,j,i] += eps
                    p1 = p0[1] + e
                    X.set_parameters([p0[0], p1])
                    l1 = Q.compute_lowerbound()
                    g_num[1][k,i,j] = (l1 - l0) / (2*eps)
                    g_num[1][k,j,i] = (l1 - l0) / (2*eps)
                
        # Check
        self.assertAllClose(g[0],
                            g_num[0])
        self.assertAllClose(g[1],
                            g_num[1])


        pass
        

