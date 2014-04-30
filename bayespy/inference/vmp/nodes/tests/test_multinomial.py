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
Unit tests for `multinomial` module.
"""

import numpy as np
import scipy

from bayespy.nodes import Multinomial
from bayespy.nodes import Dirichlet

from bayespy.utils import utils
from bayespy.utils import random

from bayespy.utils.utils import TestCase


class TestMultinomial(TestCase):
    """
    Unit tests for Multinomial node
    """

    
    def test_init(self):
        """
        Test the creation of multinomial nodes.
        """

        # Some simple initializations
        X = Multinomial([0.1, 0.3, 0.6], n=10)
        X = Multinomial(Dirichlet([5,4,3]), n=10)

        # Check that plates are correct
        X = Multinomial([0.1, 0.3, 0.6], n=10, plates=(3,4))
        self.assertEqual(X.plates,
                         (3,4))
        X = Multinomial(0.25*np.ones((2,3,4)), n=10)
        self.assertEqual(X.plates,
                         (2,3))
        n = 10 * np.ones((3,4), dtype=np.int)
        X = Multinomial([0.1, 0.3, 0.6],
                        n=n)
        self.assertEqual(X.plates,
                         (3,4))
        X = Multinomial(Dirichlet([2,1,9], plates=(3,4)),
                        n=10)
        self.assertEqual(X.plates,
                         (3,4))
        

        # Missing the number of trials
        self.assertRaises(ValueError,
                          Multinomial,
                          [0.1, 0.3, 0.6])

        # Probabilities not a vector
        self.assertRaises(ValueError,
                          Multinomial,
                          0.5)

        # Invalid probability
        self.assertRaises(ValueError,
                          Multinomial,
                          [-0.5, 1.5],
                          n=10)
        self.assertRaises(ValueError,
                          Multinomial,
                          [0.5, 1.5],
                          n=10)

        # Invalid number of trials
        self.assertRaises(ValueError,
                          Multinomial,
                          [0.5, 0.5],
                          n=-1)
        self.assertRaises(ValueError,
                          Multinomial,
                          [0.5, 0.5],
                          n=8.5)

        # Inconsistent plates
        self.assertRaises(ValueError,
                          Multinomial,
                          0.25*np.ones((2,4)),
                          plates=(3,),
                          n=10)

        # Explicit plates too small
        self.assertRaises(ValueError,
                          Multinomial,
                          0.25*np.ones((2,4)),
                          plates=(1,),
                          n=10)

        pass

    
    def test_moments(self):
        """
        Test the moments of multinomial nodes.
        """

        # Simple test
        X = Multinomial([0.7,0.2,0.1], n=1)
        u = X._message_to_child()
        self.assertEqual(len(u), 1)
        self.assertAllClose(u[0],
                            [0.7,0.2,0.1])

        # Test n
        X = Multinomial([0.7,0.2,0.1], n=10)
        u = X._message_to_child()
        self.assertAllClose(u[0],
                            [7,2,1])

        # Test plates in p
        n = np.random.randint(1, 10)
        p = np.random.dirichlet([1,1], size=3)
        X = Multinomial(p, n=n)
        u = X._message_to_child()
        self.assertAllClose(u[0],
                            p*n)
        
        # Test plates in n
        n = np.random.randint(1, 10, size=(3,))
        p = np.random.dirichlet([1,1,1,1])
        X = Multinomial(p, n=n)
        u = X._message_to_child()
        self.assertAllClose(u[0],
                            p*n[:,None])

        # Test plates in p and n
        n = np.random.randint(1, 10, size=(4,1))
        p = np.random.dirichlet([1,1], size=3)
        X = Multinomial(p, n=n)
        u = X._message_to_child()
        self.assertAllClose(u[0],
                            p*n[...,None])

        # Test with Dirichlet prior
        P = Dirichlet([7, 3])
        logp = P._message_to_child()[0]
        p0 = np.exp(logp[0]) / (np.exp(logp[0]) + np.exp(logp[1]))
        p1 = np.exp(logp[1]) / (np.exp(logp[0]) + np.exp(logp[1]))
        X = Multinomial(P, n=1)
        u = X._message_to_child()
        p = np.array([p0, p1])
        self.assertAllClose(u[0],
                            p)

        # Test with broadcasted plates
        P = Dirichlet([7, 3], plates=(10,))
        X = Multinomial(P, n=5)
        u = X._message_to_child()
        self.assertAllClose(u[0] * np.ones(X.get_shape(0)),
                            5*p*np.ones((10,1)))

        pass

    

