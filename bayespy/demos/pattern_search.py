################################################################################
# Copyright (C) 2015 Jaakko Luttinen
#
# This file is licensed under the MIT License.
################################################################################


"""
Demonstration of the pattern search method for PCA.

The pattern searches are compared to standard VB-EM algorithm in CPU
time.  For more info on the pattern search method, see
:cite:`Honkela:2002`.
"""


import numpy as np
import scipy

import matplotlib.pyplot as plt
import bayespy.plot as myplt

from bayespy.utils import misc
from bayespy.utils import random
from bayespy import nodes

from bayespy.inference.vmp.vmp import VB
from bayespy.inference.vmp import transformations

import bayespy.plot as bpplt

from bayespy.demos import pca


def run(M=40, N=100, D_y=6, D=8, seed=42, rotate=False, maxiter=1000, debug=False, plot=True):
    """
    Run pattern search demo for PCA.
    """

    if seed is not None:
        np.random.seed(seed)
    
    # Generate data
    w = np.random.normal(0, 1, size=(M,1,D_y))
    x = np.random.normal(0, 1, size=(1,N,D_y))
    f = misc.sum_product(w, x, axes_to_sum=[-1])
    y = f + np.random.normal(0, 0.2, size=(M,N))

    # Construct model
    Q = pca.model(M, N, D)

    # Data with missing values
    mask = random.mask(M, N, p=0.5) # randomly missing
    y[~mask] = np.nan
    Q['Y'].observe(y, mask=mask)

    # Initialize some nodes randomly
    Q['X'].initialize_from_random()
    Q['W'].initialize_from_random()

    # Use a few VB-EM updates at the beginning
    Q.update(repeat=10)
    Q.save()

    # Standard VB-EM as a baseline
    Q.update(repeat=maxiter)
    if plot:
        bpplt.pyplot.plot(np.cumsum(Q.cputime), Q.L, 'k-')

    # Restore initial state
    Q.load()

    # Pattern search method for comparison
    for n in range(maxiter):

        Q.pattern_search('W', 'tau', maxiter=3, collapsed=['X', 'alpha'])
        Q.update(repeat=20)

        if Q.has_converged():
            break

    if plot:
        bpplt.pyplot.plot(np.cumsum(Q.cputime), Q.L, 'r:')

        bpplt.pyplot.xlabel('CPU time (in seconds)')
        bpplt.pyplot.ylabel('VB lower bound')
        bpplt.pyplot.legend(['VB-EM', 'Pattern search'], loc='lower right')


if __name__ == '__main__':
    import sys, getopt, os
    try:
        opts, args = getopt.getopt(sys.argv[1:],
                                   "",
                                   ["m=",
                                    "n=",
                                    "d=",
                                    "k=",
                                    "seed=",
                                    "maxiter=",
                                    "debug"])
    except getopt.GetoptError:
        print('python demo_pca.py <options>')
        print('--m=<INT>        Dimensionality of data vectors')
        print('--n=<INT>        Number of data vectors')
        print('--d=<INT>        Dimensionality of the latent vectors in the model')
        print('--k=<INT>        Dimensionality of the true latent vectors')
        print('--maxiter=<INT>  Maximum number of VB iterations')
        print('--seed=<INT>     Seed (integer) for the random number generator')
        print('--debug          Check that the rotations are implemented correctly')
        sys.exit(2)

    kwargs = {}
    for opt, arg in opts:
        if opt == "--rotate":
            kwargs["rotate"] = True
        elif opt == "--maxiter":
            kwargs["maxiter"] = int(arg)
        elif opt == "--debug":
            kwargs["debug"] = True
        elif opt == "--seed":
            kwargs["seed"] = int(arg)
        elif opt in ("--m",):
            kwargs["M"] = int(arg)
        elif opt in ("--n",):
            kwargs["N"] = int(arg)
        elif opt in ("--d",):
            kwargs["D"] = int(arg)
        elif opt in ("--k",):
            kwargs["D_y"] = int(arg)

    run(**kwargs)
    plt.show()

