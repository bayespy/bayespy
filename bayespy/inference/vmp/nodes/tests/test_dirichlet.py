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
Unit tests for `dirichlet` module.
"""

import numpy as np
from scipy import special

from bayespy.nodes import Dirichlet

from bayespy.utils import random

from bayespy.utils.misc import TestCase

class TestDirichlet(TestCase):
    """
    Unit tests for Dirichlet node
    """

    
    def test_init(self):
        """
        Test the creation of Dirichlet nodes.
        """

        # Some simple initializations
        p = Dirichlet([1.5, 4.2, 3.5])

        # Check that plates are correct
        p = Dirichlet([2, 3, 4], plates=(4,3))
        self.assertEqual(p.plates,
                         (4,3))
        p = Dirichlet(np.ones((4,3,5)))
        self.assertEqual(p.plates,
                         (4,3))

        # Parent not a vector
        self.assertRaises(ValueError,
                          Dirichlet,
                          4)
        
        # Parent vector has invalid values
        self.assertRaises(ValueError,
                          Dirichlet,
                          [-2,3,1])

        # Plates inconsistent
        self.assertRaises(ValueError,
                          Dirichlet,
                          np.ones((4,3)),
                          plates=(3,))

        # Explicit plates too small
        self.assertRaises(ValueError,
                          Dirichlet,
                          np.ones((4,3)),
                          plates=(1,))

        pass

    
    def test_moments(self):
        """
        Test the moments of Dirichlet nodes.
        """

        p = Dirichlet([2, 3, 4])
        u = p._message_to_child()
        self.assertAllClose(u[0],
                            special.psi([2,3,4]) - special.psi(2+3+4))
        
        pass


    def test_constant(self):
        """
        Test the constant moments of Dirichlet nodes.
        """

        p = Dirichlet([1, 1, 1])
        p.initialize_from_value([0.5, 0.4, 0.1])
        u = p._message_to_child()
        self.assertAllClose(u[0],
                            np.log([0.5, 0.4, 0.1]))

        pass
