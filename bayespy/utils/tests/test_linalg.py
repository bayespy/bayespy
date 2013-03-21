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

#import unittest

import numpy as np

#from numpy import testing

from ...utils.utils import TestCase

from .. import linalg
#from bayespy.utils import utils

class TestDot(TestCase):

    def test_dot(self):
        """
        Test dot product multiple multi-dimensional arrays.
        """

        # If no arrays, return 0
        self.assertAllClose(linalg.dot(),
                            0)
        # If only one array, return itself
        self.assertAllClose(linalg.dot([[1,2,3],
                                        [4,5,6]]),
                            [[1,2,3],
                             [4,5,6]])
        # Basic test of two arrays: (2,3) * (3,2)
        self.assertAllClose(linalg.dot([[1,2,3],
                                        [4,5,6]],
                                       [[7,8],
                                        [9,1],
                                        [2,3]]),
                            [[31,19],
                             [85,55]])
        # Basic test of four arrays: (2,3) * (3,2) * (2,1) * (1,2)
        self.assertAllClose(linalg.dot([[1,2,3],
                                        [4,5,6]],
                                       [[7,8],
                                        [9,1],
                                        [2,3]],
                                       [[4],
                                        [5]],
                                       [[6,7]]),
                            [[1314,1533],
                             [3690,4305]])

        # Test broadcasting: (2,2,2) * (2,2,2,2)
        self.assertAllClose(linalg.dot([[[1,2],
                                         [3,4]],
                                        [[5,6],
                                         [7,8]]],
                                       [[[[1,2],
                                          [3,4]],
                                         [[5,6],
                                          [7,8]]],
                                        [[[9,1],
                                          [2,3]],
                                         [[4,5],
                                          [6,7]]]]),
                            [[[[  7,  10],
                               [ 15,  22]],

                              [[ 67,  78],
                               [ 91, 106]]],


                             [[[ 13,   7],
                               [ 35,  15]],

                              [[ 56,  67],
                               [ 76,  91]]]])

        # Inconsistent shapes: (2,3) * (2,3)
        self.assertRaises(ValueError,
                          linalg.dot,
                          [[1,2,3],
                           [4,5,6]],
                          [[1,2,3],
                           [4,5,6]])
        # Other axes do not broadcast: (2,2,2) * (3,2,2)
        self.assertRaises(ValueError,
                          linalg.dot,
                          [[[1,2],
                            [3,4]],
                           [[5,6],
                            [7,8]]],
                          [[[1,2],
                            [3,4]],
                           [[5,6],
                            [7,8]],
                           [[9,1],
                            [2,3]]])
        # Do not broadcast matrix axes: (2,1) * (3,2)
        self.assertRaises(ValueError,
                          linalg.dot,
                          [[1],
                           [2]],
                          [[1,2,3],
                           [4,5,6]])
        # Do not accept less than 2-D arrays: (2) * (2,2)
        self.assertRaises(ValueError,
                          linalg.dot,
                          [1,2],
                          [[1,2,3],
                           [4,5,6]])
