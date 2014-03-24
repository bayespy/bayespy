######################################################################
# Copyright (C) 2014 Jaakko Luttinen
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
Unit tests for bayespy.utils.random module.
"""

import numpy as np

from ..utils import TestCase
from .. import random

class TestCeilDiv(TestCase):

    def test_categorical(self):

        # Test dummy one category
        y = random.categorical([1])
        self.assertEqual(y, 0)

        # Test multiple categories
        y = random.categorical([1,0,0])
        self.assertEqual(y, 0)
        y = random.categorical([0,1,0])
        self.assertEqual(y, 1)
        y = random.categorical([0,0,1])
        self.assertEqual(y, 2)

        # Test un-normalized probabilities
        y = random.categorical([0,0.1234])
        self.assertEqual(y, 1)

        # Test multiple distributions
        y = random.categorical([ [1,0,0], [0,0,1], [0,1,0] ])
        self.assertArrayEqual(y, [0,2,1])

        # Test multiple samples
        y = random.categorical([0,1,0], size=(4,))
        self.assertArrayEqual(y, [1,1,1,1])

        #
        # ERRORS
        #
        
        # Negative probablities
        self.assertRaises(ValueError, 
                          random.categorical,
                          [0, -1])

        # Requested size and probability array size mismatch
        self.assertRaises(ValueError, 
                          random.categorical,
                          [[1,0],[0,1]], 
                          size=(3,))

        pass
        
