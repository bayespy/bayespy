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

from ..gaussian import GaussianArrayARD#, Gaussian
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
                          
