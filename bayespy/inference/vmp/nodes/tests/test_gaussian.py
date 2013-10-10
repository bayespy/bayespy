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
#from ..normal import Normal

from ...vmp import VB

from bayespy.utils import utils
from bayespy.utils import linalg
from bayespy.utils import random

from bayespy.utils.utils import TestCase

class TestGaussianArrayARD(TestCase):

    def test_parent_validity(self):
        """
        Test that the parent nodes are validated properly in the constructor
        """
        # Create from constant parents
        GaussianArrayARD(0,
                         1)
        GaussianArrayARD(np.ones((2,)),
                         np.ones((2,)))

