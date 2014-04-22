######################################################################
# Copyright (C) 2012-2013 Jaakko Luttinen
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
Unit tests for distance module.
"""

import unittest

import numpy as np
import scipy.sparse as sp
import scipy.spatial.distance as dist
#from .. import distance as spdist
#import scikits.sparse.distance as spdist
import numpy.random as rand

## class TestDistance(unittest.TestCase):
##     def test_pdist(self):
##         N = 50
##         D = 3
##         x = rand.uniform(size=(N,D))
##         threshold = 0.1

##         i = np.arange(N)[:,np.newaxis]
##         j = np.arange(N)[np.newaxis,:]

##         # Compute full&dense distance matrix
##         Dd = dist.squareform(dist.pdist(x, metric="sqeuclidean"))

##         for form in ["lower", "strictly_lower", "upper", "strictly_upper", "full"]:
##             Ds = spdist.sparse_pdist(x, threshold, form=form)
##             self.assertTrue(sp.issparse(Ds))
##             Ds = Ds.tocsr()
##             Ds.sort_indices()
##             if form == "lower":
##                 D0 = Dd[i>=j]
##             elif form == "strictly_lower":
##                 D0 = Dd[i>j]
##             elif form == "upper":
##                 D0 = Dd[i<=j]
##             elif form == "strictly_upper":
##                 D0 = Dd[i<j]
##             else:
##                 D0 = Dd.ravel()

##             D0 = D0[D0<=threshold]
##             D1 = Ds.data
##             self.assertTrue(np.allclose(D0, D1))

##     def test_cdist(self):
##         N1 = 50
##         N2 = 80
##         D = 3
##         x1 = rand.uniform(size=(N1,D))
##         x2 = rand.uniform(size=(N2,D))
##         threshold = 0.1

##         # Compute full&dense distance matrix
##         Dd = dist.cdist(x1, x2, metric="sqeuclidean")
##         Dd = Dd[Dd<=threshold]

##         # Compute sparse distance matrix
##         Ds = spdist.sparse_cdist(x1, x2, threshold)
##         Ds = Ds.tocsr()
##         Ds.sort_indices()

##         self.assertTrue(np.allclose(Ds.data, Dd))

