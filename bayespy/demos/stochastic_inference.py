################################################################################
# Copyright (C) 2015 Jaakko Luttinen
#
# This file is licensed under the MIT License.
################################################################################


"""
Stochastic variational inference on mixture of Gaussians

Stochastic variational inference is a scalable variational Bayesian
learning method which utilizes stochastic gradient.  For details, see
:cite:`Hoffman:2013`.
"""


import numpy as np
import scipy

import matplotlib.pyplot as plt
import bayespy.plot as myplt

from bayespy.utils import misc
from bayespy.utils import random
from bayespy.nodes import Gaussian, Categorical, Mixture, Dirichlet

from bayespy.inference.vmp.vmp import VB
from bayespy.inference.vmp import transformations

import bayespy.plot as bpplt

from bayespy.demos import pca


def run(N=100000, N_batch=50, seed=42, maxiter=100, plot=True):
    """
    Run deterministic annealing demo for 1-D Gaussian mixture.
    """

    if seed is not None:
        np.random.seed(seed)

    # Number of clusters in the model
    K = 20

    # Dimensionality of the data
    D = 5

    # Generate data
    K_true = 10
    spread = 5
    means = spread * np.random.randn(K_true, D)
    z = random.categorical(np.ones(K_true), size=N)
    data = np.empty((N,D))
    for n in range(N):
        data[n] = means[z[n]] + np.random.randn(D)

    #
    # Standard VB-EM algorithm
    #

    # Full model
    mu = Gaussian(np.zeros(D), np.identity(D),
                  plates=(K,),
                  name='means')
    alpha = Dirichlet(np.ones(K),
                      name='class probabilities')
    Z = Categorical(alpha,
                    plates=(N,),
                    name='classes')
    Y = Mixture(Z, Gaussian, mu, np.identity(D),
                name='observations')

    # Break symmetry with random initialization of the means
    mu.initialize_from_random()

    # Put the data in
    Y.observe(data)

    # Run inference
    Q = VB(Y, Z, mu, alpha)
    Q.save(mu)
    Q.update(repeat=maxiter)
    if plot:
        bpplt.pyplot.plot(np.cumsum(Q.cputime), Q.L, 'k-')
    max_cputime = np.sum(Q.cputime[~np.isnan(Q.cputime)])


    #
    # Stochastic variational inference
    #

    # Construct smaller model (size of the mini-batch)
    mu = Gaussian(np.zeros(D), np.identity(D),
                  plates=(K,),
                  name='means')
    alpha = Dirichlet(np.ones(K),
                      name='class probabilities')
    Z = Categorical(alpha,
                    plates=(N_batch,),
                    plates_multiplier=(N/N_batch,),
                    name='classes')
    Y = Mixture(Z, Gaussian, mu, np.identity(D),
                name='observations')

    # Break symmetry with random initialization of the means
    mu.initialize_from_random()

    # Inference engine
    Q = VB(Y, Z, mu, alpha, autosave_filename=Q.autosave_filename)
    Q.load(mu)

    # Because using mini-batches, messages need to be multiplied appropriately
    print("Stochastic variational inference...")
    Q.ignore_bound_checks = True

    maxiter *= int(N/N_batch)
    delay = 1
    forgetting_rate = 0.7
    for n in range(maxiter):

        # Observe a mini-batch
        subset = np.random.choice(N, N_batch)
        Y.observe(data[subset,:])

        # Learn intermediate variables
        Q.update(Z)

        # Set step length
        step = (n + delay) ** (-forgetting_rate)

        # Stochastic gradient for the global variables
        Q.gradient_step(mu, alpha, scale=step)

        if np.sum(Q.cputime[:n]) > max_cputime:
            break
    
    if plot:
        bpplt.pyplot.plot(np.cumsum(Q.cputime), Q.L, 'r:')

        bpplt.pyplot.xlabel('CPU time (in seconds)')
        bpplt.pyplot.ylabel('VB lower bound')
        bpplt.pyplot.legend(['VB-EM', 'Stochastic inference'], loc='lower right')
        bpplt.pyplot.title('VB for Gaussian mixture model')

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

