# Copyright (C) 2012 Jaakko Luttinen, Aalto University, <jaakko.luttinen@aalto.fi>
# Released under the terms of the GNU GPL v3, or, at your option, any
# later version.

import numpy as np
import scipy.sparse as sp
import scipy.spatial.distance as dist
import scikits.sparse.distance as spdist
import numpy.random as rand

def test_pdist():
    N = 50
    D = 3
    x = rand.uniform(size=(N,D))
    threshold = 0.1

    i = np.arange(N)[:,np.newaxis]
    j = np.arange(N)[np.newaxis,:]

    # Compute full&dense distance matrix
    Dd = dist.squareform(dist.pdist(x, metric="sqeuclidean"))

    for form in ["lower", "strictly_lower", "upper", "strictly_upper", "full"]:
        Ds = spdist.pdist(x, threshold=threshold, form=form)
        assert(sp.issparse(Ds))
        Ds = Ds.tocsr()
        Ds.sort_indices()
        if form == "lower":
            D0 = Dd[i>=j]
        elif form == "strictly_lower":
            D0 = Dd[i>j]
        elif form == "upper":
            D0 = Dd[i<=j]
        elif form == "strictly_upper":
            D0 = Dd[i<j]
        else:
            D0 = Dd.ravel()
            
        D0 = D0[D0<=threshold]
        D1 = Ds.data
        assert(np.allclose(D0, D1))

def test_cdist():
    N1 = 50
    N2 = 80
    D = 3
    x1 = rand.uniform(size=(N1,D))
    x2 = rand.uniform(size=(N2,D))
    threshold = 0.1

    # Compute full&dense distance matrix
    Dd = dist.cdist(x1, x2, metric="sqeuclidean")
    Dd = Dd[Dd<=threshold]

    # Compute sparse distance matrix
    Ds = spdist.cdist(x1, x2, threshold=threshold)
    Ds = Ds.tocsr()
    Ds.sort_indices()

    assert(np.allclose(Ds.data, Dd))

