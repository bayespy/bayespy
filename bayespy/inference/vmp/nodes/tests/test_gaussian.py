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

from ..gaussian import Gaussian, GaussianArrayARD
from ..gamma import Gamma
#from ..normal import Normal

from ...vmp import VB

from bayespy.utils import utils
from bayespy.utils import linalg
from bayespy.utils import random

from bayespy.utils.utils import TestCase

class TestGaussianArrayARD(TestCase):

    def test_init(self):
        """
        Test the constructor
        """
        
        def check_init(true_plates, true_shape, mu, alpha, **kwargs):
            X = GaussianArrayARD(mu, alpha, **kwargs)
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
                   GaussianArrayARD(np.zeros((2,1,3)),
                                    np.ones((2,1,3))),
                   Gamma(np.ones((4,1,2,3)),
                         np.ones((4,1,2,3))))
        # Use ndim
        check_init((4,),
                   (2,2,3),
                   GaussianArrayARD(np.zeros((4,2,3)),
                                    np.ones((4,2,3)),
                                    ndim=2),
                   Gamma(np.ones((4,2,1,3)),
                         np.ones((4,2,1,3))),
                   ndim=3)
        # Use shape
        check_init((4,),
                   (2,2,3),
                   GaussianArrayARD(np.zeros((4,2,3)),
                                    np.ones((4,2,3)),
                                    ndim=2),
                   Gamma(np.ones((4,2,1,3)),
                         np.ones((4,2,1,3))),
                   shape=(2,2,3))
        # Use ndim and shape
        check_init((4,2),
                   (2,3),
                   GaussianArrayARD(np.zeros((2,1,3)),
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
                          GaussianArrayARD,
                          np.zeros((2,3)),
                          np.ones((2,)))
        # Inconsistent plates of mu and alpha
        self.assertRaises(ValueError,
                          GaussianArrayARD,
                          GaussianArrayARD(np.zeros((4,2,3)),
                                           np.ones((4,2,3)),
                                           ndim=2),
                          np.ones((3,4,2,3)),
                          ndim=3)
        # Inconsistent ndim and shape
        self.assertRaises(ValueError,
                          GaussianArrayARD,
                          np.zeros((2,3)),
                          np.ones((2,)),
                          shape=(2,3),
                          ndim=1)
        # Parent mu has more axes
        self.assertRaises(ValueError,
                          GaussianArrayARD,
                          GaussianArrayARD(np.zeros((2,3)),
                                           np.ones((2,3))),
                          np.ones((2,3)),
                          ndim=1)
        # Incorrect shape
        self.assertRaises(ValueError,
                          GaussianArrayARD,
                          GaussianArrayARD(np.zeros((2,3)),
                                           np.ones((2,3)),
                                           ndim=2),
                          np.ones((2,3)),
                          shape=(2,2))
                          
        pass

    def test_message_to_child(self):
        """
        Test that GaussianArrayARD computes the message to children correctly.
        """

        # Check that moments have full shape when broadcasting
        X = GaussianArrayARD(np.zeros((2,)),
                             np.ones((3,2)),
                             shape=(4,3,2))
        (u0, u1) = X._message_to_child()
        self.assertEqual(np.shape(u0),
                         (4,3,2))
        self.assertEqual(np.shape(u1),
                         (4,3,2,4,3,2))

        # Check the formula
        X = GaussianArrayARD(2, 3)
        (u0, u1) = X._message_to_child()
        self.assertAllClose(u0, 2)
        self.assertAllClose(u1, 2**2 + 1/3)

        # Check the formula for multidimensional arrays
        X = GaussianArrayARD(2*np.ones((2,1,4)),
                             3*np.ones((2,3,1)),
                             ndim=3)
        (u0, u1) = X._message_to_child()
        self.assertAllClose(u0, 2*np.ones((2,3,4)))
        self.assertAllClose(u1, 
                            2**2 * np.ones((2,3,4,2,3,4))
                            + 1/3 * utils.identity(2,3,4))
                            

        # Check posterior
        X = GaussianArrayARD(2, 3)
        Y = GaussianArrayARD(X, 1)
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
        X = GaussianArrayARD(0,
                             alpha)
        X.observe(3)
        (m0, m1) = X._message_to_parent(0)
        self.assertAllClose(m0,
                            2*3)
        self.assertAllClose(m1,
                            -0.5*2)

        # Check formula with uncertain node
        X = GaussianArrayARD(1, 2)
        Y = GaussianArrayARD(X, 1)
        Y.observe(5)
        X.update()
        (m0, m1) = X._message_to_parent(0)
        self.assertAllClose(m0,
                            2 * 1/(2+1)*(2*1+1*5))
        self.assertAllClose(m1,
                            -0.5*2)

        # Check alpha larger than mu
        X = GaussianArrayARD(np.zeros((2,3)),
                             2*np.ones((3,2,3)))
        X.observe(3*np.ones((3,2,3)))
        (m0, m1) = X._message_to_parent(0)
        self.assertAllClose(m0,
                            2*3 * 3 * np.ones((2,3)))
        self.assertAllClose(m1,
                            -0.5 * 3 * 2*utils.identity(2,3))

        # Check mu larger than alpha
        X = GaussianArrayARD(np.zeros((3,2,3)),
                             2*np.ones((2,3)))
        X.observe(3*np.ones((3,2,3)))
        (m0, m1) = X._message_to_parent(0)
        self.assertAllClose(m0,
                            2 * 3 * np.ones((3,2,3)))
        self.assertAllClose(m1,
                            -0.5 * 2*utils.identity(3,2,3))

        # Check node larger than mu and alpha
        X = GaussianArrayARD(np.zeros((2,3)),
                             2*np.ones((3,)),
                             shape=(3,2,3))
        X.observe(3*np.ones((3,2,3)))
        (m0, m1) = X._message_to_parent(0)
        self.assertAllClose(m0,
                            2*3 * 3*np.ones((2,3)))
        self.assertAllClose(m1,
                            -0.5 * 2 * 3*utils.identity(2,3))

        # Check broadcasting of dimensions
        X = GaussianArrayARD(np.zeros((2,1)),
                             2*np.ones((2,3)),
                             shape=(2,3))
        X.observe(3*np.ones((2,3)))
        (m0, m1) = X._message_to_parent(0)
        self.assertAllClose(m0,
                            2*3 * 3*np.ones((2,1)))
        self.assertAllClose(m1,
                            -0.5 * 2 * 3*utils.identity(2,1))

        # Check plates for smaller mu than node
        X = GaussianArrayARD(GaussianArrayARD(0,1, 
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
        X = GaussianArrayARD(np.zeros((2,1,3)),
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
        X = GaussianArrayARD(mu, alpha)
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
        mu = GaussianArrayARD(1,1)
        X = GaussianArrayARD(mu,
                             0.5)
        X.observe(3)
        (m0, m1) = X._message_to_parent(1)
        self.assertAllClose(m0,
                            -0.5*(3**2 - 2*3*1 + 1**2+1))
        self.assertAllClose(m1,
                            0.5)

        # Check formula with uncertain node
        X = GaussianArrayARD(2, 1)
        Y = GaussianArrayARD(X, 1)
        Y.observe(5)
        X.update()
        (m0, m1) = X._message_to_parent(1)
        self.assertAllClose(m0,
                            -0.5*(1/(1+1)+3.5**2 - 2*3.5*2 + 2**2))
        self.assertAllClose(m1,
                            0.5)

        # Check alpha larger than mu
        X = GaussianArrayARD(np.ones((2,3)),
                             np.ones((3,2,3)))
        X.observe(2*np.ones((3,2,3)))
        (m0, m1) = X._message_to_parent(1)
        self.assertAllClose(m0,
                            -0.5*(2**2 - 2*2*1 + 1**2) * np.ones((3,2,3)))
        self.assertAllClose(m1,
                            0.5)

        # Check mu larger than alpha
        X = GaussianArrayARD(np.ones((3,2,3)),
                             np.ones((2,3)))
        X.observe(2*np.ones((3,2,3)))
        (m0, m1) = X._message_to_parent(1)
        self.assertAllClose(m0,
                            -0.5*(2**2 - 2*2*1 + 1**2) * 3 * np.ones((2,3)))
        self.assertAllClose(m1,
                            0.5 * 3)

        # Check node larger than mu and alpha
        X = GaussianArrayARD(np.ones((2,3)),
                             np.ones((3,)),
                             shape=(3,2,3))
        X.observe(2*np.ones((3,2,3)))
        (m0, m1) = X._message_to_parent(1)
        self.assertAllClose(m0,
                            -0.5*(2**2 - 2*2*1 + 1**2) * 6 * np.ones((3,)))
        self.assertAllClose(m1,
                            0.5 * 6)

        # Check plates for smaller mu than node
        X = GaussianArrayARD(GaussianArrayARD(1, 1, 
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
        X = GaussianArrayARD(np.ones(3),
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
        X = GaussianArrayARD(mu, alpha)
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
        
