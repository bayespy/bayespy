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
Unit tests for `poisson` module.
"""

import numpy as np
import scipy

from bayespy.nodes import Poisson
from bayespy.nodes import Gamma

from bayespy.utils import random

from bayespy.utils.misc import TestCase


class TestPoisson(TestCase):
    """
    Unit tests for Poisson node
    """

    
    def test_init(self):
        """
        Test the creation of Poisson nodes.
        """

        # Some simple initializations
        X = Poisson(12.8)
        X = Poisson(Gamma(43, 24))

        # Check that plates are correct
        X = Poisson(np.ones((2,3)))
        self.assertEqual(X.plates,
                         (2,3))
        X = Poisson(Gamma(1, 1, plates=(2,3)))
        self.assertEqual(X.plates,
                         (2,3))
        
        # Invalid rate
        self.assertRaises(ValueError,
                          Poisson,
                          -0.1)

        # Inconsistent plates
        self.assertRaises(ValueError,
                          Poisson,
                          np.ones(3),
                          plates=(2,))

        # Explicit plates too small
        self.assertRaises(ValueError,
                          Poisson,
                          np.ones(3),
                          plates=(1,))

        pass

    
    def test_moments(self):
        """
        Test the moments of Poisson nodes.
        """

        # Simple test
        X = Poisson(12.8)
        u = X._message_to_child()
        self.assertEqual(len(u),
                         1)
        self.assertAllClose(u[0],
                            12.8)

        # Test plates in rate
        X = Poisson(12.8*np.ones((2,3)))
        u = X._message_to_child()
        self.assertAllClose(u[0],
                            12.8*np.ones((2,3)))

        # Test with gamma prior
        alpha = Gamma(5, 2)
        r = np.exp(alpha._message_to_child()[1])
        X = Poisson(alpha)
        u = X._message_to_child()
        self.assertAllClose(u[0],
                            r)

        # Test with broadcasted plates in parents
        X = Poisson(Gamma(5, 2, plates=(2,3)))
        u = X._message_to_child()
        self.assertAllClose(u[0]*np.ones((2,3)),
                            r*np.ones((2,3)))

        pass
