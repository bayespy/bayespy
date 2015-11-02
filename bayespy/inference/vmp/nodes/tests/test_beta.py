################################################################################
# Copyright (C) 2014 Jaakko Luttinen
#
# This file is licensed under the MIT License.
################################################################################


"""
Unit tests for `beta` module.
"""

import numpy as np
from scipy import special

from bayespy.nodes import Beta

from bayespy.utils import random

from bayespy.utils.misc import TestCase

class TestBeta(TestCase):
    """
    Unit tests for Beta node
    """

    
    def test_init(self):
        """
        Test the creation of beta nodes.
        """

        # Some simple initializations
        p = Beta([1.5, 4.2])

        # Check that plates are correct
        p = Beta([2, 3], plates=(4,3))
        self.assertEqual(p.plates,
                         (4,3))
        p = Beta(np.ones((4,3,2)))
        self.assertEqual(p.plates,
                         (4,3))

        # Parent not a vector
        self.assertRaises(ValueError,
                          Beta,
                          4)
        
        # Parent vector has wrong shape
        self.assertRaises(ValueError,
                          Beta,
                          [4])
        self.assertRaises(ValueError,
                          Beta,
                          [4,4,4])

        # Parent vector has invalid values
        self.assertRaises(ValueError,
                          Beta,
                          [-2,3])

        # Plates inconsistent
        self.assertRaises(ValueError,
                          Beta,
                          np.ones((4,2)),
                          plates=(3,))

        # Explicit plates too small
        self.assertRaises(ValueError,
                          Beta,
                          np.ones((4,2)),
                          plates=(1,))

        pass

    
    def test_moments(self):
        """
        Test the moments of beta nodes.
        """

        p = Beta([2, 3])
        u = p._message_to_child()
        self.assertAllClose(u[0],
                            special.psi([2,3]) - special.psi(2+3))
        
        pass

    
    def test_random(self):
        """
        Test random sampling of beta nodes.
        """

        p = Beta([1e20, 3e20])
        x = p.random()
        self.assertAllClose(x,
                            0.25)

        p = Beta([[1e20, 3e20],
                  [1e20, 1e20]])
        x = p.random()
        self.assertAllClose(x,
                            [0.25, 0.5])

        p = Beta([1e20, 3e20], plates=(3,))
        x = p.random()
        self.assertAllClose(x,
                            [0.25, 0.25, 0.25])

        pass
