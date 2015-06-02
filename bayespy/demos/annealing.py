################################################################################
# Copyright (C) 2015 Jaakko Luttinen
#
# This file is licensed under the MIT License.
################################################################################


"""
Demonstration of deterministic annealing.

Deterministic annealing aims at avoiding convergence to local optima and
finding the global optimum :cite:`Katahira:2008`.
"""


import numpy as np
import scipy

import matplotlib.pyplot as plt
import bayespy.plot as myplt

from bayespy.utils import misc
from bayespy.utils import random
from bayespy.nodes import GaussianARD, Categorical, Mixture

from bayespy.inference.vmp.vmp import VB
from bayespy.inference.vmp import transformations

import bayespy.plot as bpplt

from bayespy.demos import pca


def run(N=500, seed=42, maxiter=100, plot=True):
    """
    Run deterministic annealing demo for 1-D Gaussian mixture.
    """

    if seed is not None:
        np.random.seed(seed)

    mu = GaussianARD(0, 1,
                     plates=(2,),
                     name='means')
    Z = Categorical([0.3, 0.7],
                    plates=(N,),
                    name='classes')
    Y = Mixture(Z, GaussianARD, mu, 1,
                name='observations')

    # Generate data
    z = Z.random()
    data = np.empty(N)
    for n in range(N):
        data[n] = [4, -4][z[n]]

    Y.observe(data)

    # Initialize means closer to the inferior local optimum in which the
    # cluster means are swapped
    mu.initialize_from_value([0, 6])

    Q = VB(Y, Z, mu)
    Q.save()

    #
    # Standard VB-EM algorithm
    #
    Q.update(repeat=maxiter)

    mu_vbem = mu.u[0].copy()
    L_vbem = Q.compute_lowerbound()

    #
    # VB-EM with deterministic annealing
    #
    Q.load()
    beta = 0.01
    while beta < 1.0:
        beta = min(beta*1.2, 1.0)
        print("Set annealing to %.2f" % beta)
        Q.set_annealing(beta)
        Q.update(repeat=maxiter, tol=1e-4)

    mu_anneal = mu.u[0].copy()
    L_anneal = Q.compute_lowerbound()

    print("==============================")
    print("RESULTS FOR VB-EM vs ANNEALING")
    print("Fixed component probabilities:", np.array([0.3, 0.7]))
    print("True component means:", np.array([4, -4]))
    print("VB-EM component means:", mu_vbem)
    print("VB-EM lower bound:", L_vbem)
    print("Annealed VB-EM component means:", mu_anneal)
    print("Annealed VB-EM lower bound:", L_anneal)
    
    return


if __name__ == '__main__':
    import sys, getopt, os
    try:
        opts, args = getopt.getopt(sys.argv[1:],
                                   "",
                                   ["n=",
                                    "seed=",
                                    "maxiter="])
    except getopt.GetoptError:
        print('python annealing.py <options>')
        print('--n=<INT>        Number of data points')
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

    run(**kwargs)

    plt.show()

