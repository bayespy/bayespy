################################################################################
# Copyright (C) 2015 Jaakko Luttinen
#
# This file is licensed under the MIT License.
################################################################################


"""
Black-box variational inference
"""


import numpy as np
import scipy

import matplotlib.pyplot as plt
import bayespy.plot as myplt

from bayespy.utils import misc
from bayespy.utils import random
from bayespy.nodes import GaussianARD, LogPDF, Dot

from bayespy.inference.vmp.vmp import VB
from bayespy.inference.vmp import transformations

import bayespy.plot as bpplt

from bayespy.demos import pca


def run(M=10, N=100, D=5, seed=42, maxiter=100, plot=True):
    """
    Run deterministic annealing demo for 1-D Gaussian mixture.
    """

    raise NotImplementedError("Black box variational inference not yet implemented, sorry")

    if seed is not None:
        np.random.seed(seed)

    # Generate data
    data = np.dot(np.random.randn(M,D),
                  np.random.randn(D,N))

    # Construct model
    C = GaussianARD(0, 1, shape=(2,), plates=(M,1), name='C')
    X = GaussianARD(0, 1, shape=(2,), plates=(1,N), name='X')
    F = Dot(C, X)

    # Some arbitrary log likelihood
    def logpdf(y, f):
        """
        exp(f) / (1 + exp(f)) = 1/(1+exp(-f))

        -log(1+exp(-f)) = -log(exp(0)+exp(-f))

        also:
        1 - exp(f) / (1 + exp(f)) = (1 + exp(f) - exp(f)) / (1 + exp(f))
        = 1 / (1 + exp(f))
        = -log(1+exp(f)) = -log(exp(0)+exp(f))
        """
        return -np.logaddexp(0, -f * np.where(y, -1, +1))
        
    Y = LogPDF(logpdf, F, samples=10, shape=())
    #Y = GaussianARD(F, 1)

    Y.observe(data)

    Q = VB(Y, C, X)
    Q.ignore_bound_checks = True

    delay = 1
    forgetting_rate = 0.7
    for n in range(maxiter):

        # Observe a mini-batch
        #subset = np.random.choice(N, N_batch)
        #Y.observe(data[subset,:])

        # Learn intermediate variables
        #Q.update(Z)

        # Set step length
        step = (n + delay) ** (-forgetting_rate)

        # Stochastic gradient for the global variables
        Q.gradient_step(C, X, scale=step)
    
    if plot:
        bpplt.pyplot.plot(np.cumsum(Q.cputime), Q.L, 'r:')
        bpplt.pyplot.xlabel('CPU time (in seconds)')
        bpplt.pyplot.ylabel('VB lower bound')

    return


if __name__ == '__main__':
    import sys, getopt, os
    try:
        opts, args = getopt.getopt(sys.argv[1:],
                                   "",
                                   ["n=",
                                    "batch=",
                                    "seed=",
                                    "maxiter="])
    except getopt.GetoptError:
        print('python stochastic_inference.py <options>')
        print('--n=<INT>        Number of data points')
        print('--batch=<INT>    Mini-batch size')
        print('--maxiter=<INT>  Maximum number of VB iterations')
        print('--seed=<INT>     Seed (integer) for the random number generator')
        sys.exit(2)

    kwargs = {}
    for opt, arg in opts:
        if opt == "--maxiter":
            kwargs["maxiter"] = int(arg)
        elif opt == "--seed":
            kwargs["seed"] = int(arg)
        elif opt in ("--n",):
            kwargs["N"] = int(arg)
        elif opt in ("--batch",):
            kwargs["N_batch"] = int(arg)

    run(**kwargs)

    plt.show()

