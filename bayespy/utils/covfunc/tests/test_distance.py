################################################################################
# Copyright (C) 2012-2013 Jaakko Luttinen
#
# This file is licensed under the MIT License.
################################################################################


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

