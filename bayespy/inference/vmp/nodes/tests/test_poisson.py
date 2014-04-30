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

from bayespy.utils import utils
from bayespy.utils import random

from bayespy.utils.utils import TestCase


class TestPoisson(TestCase):
    """
    Unit tests for Poisson node
    """

    
    def test_init(self):
        """
        Test the creation of Poisson nodes.
        """

        # Some simple initializations

        # Check that plates are correct
        
        # Invalid rate

        # Inconsistent plates

        # Explicit plates too small

        pass

    
    def test_moments(self):
        """
        Test the moments of Poisson nodes.
        """

        # Simple test

        # Test plates in rate

        # Test with gamma prior

        # Test with broadcasted plates

        pass
