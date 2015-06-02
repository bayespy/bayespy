################################################################################
# Copyright (C) 2014-2015 Jaakko Luttinen
#
# This file is licensed under the MIT License.
################################################################################


"""
Demonstrate Riemannian conjugate gradient
"""

import numpy as np

from bayespy.nodes import (GaussianARD,
                           Gamma,
                           SumMultiply)

from bayespy.utils import random

from bayespy.inference.vmp.vmp import VB

import bayespy.plot as bpplt

from bayespy.demos import mog


def pca():

    np.random.seed(41)

    M = 10
    N = 3000
    D = 5

    # Construct the PCA model
    alpha = Gamma(1e-3, 1e-3, plates=(D,), name='alpha')
    W = GaussianARD(0, alpha, plates=(M,1), shape=(D,), name='W')
    X = GaussianARD(0, 1, plates=(1,N), shape=(D,), name='X')
    tau = Gamma(1e-3, 1e-3, name='tau')
    W.initialize_from_random()
    F = SumMultiply('d,d->', W, X)
    Y = GaussianARD(F, tau, name='Y')

    # Observe data
    data = np.sum(np.random.randn(M,1,D-1) * np.random.randn(1,N,D-1), axis=-1) + 1e-1 * np.random.randn(M,N)
    Y.observe(data)

    # Initialize VB engine
    Q = VB(Y, X, W, alpha, tau)

    # Take one update step (so phi is ok)
    Q.update(repeat=1)
    Q.save()

    # Run VB-EM
    Q.update(repeat=200)
    bpplt.pyplot.plot(np.cumsum(Q.cputime), Q.L, 'k-')

    # Restore the state
    Q.load()

    # Run Riemannian conjugate gradient
    #Q.optimize(X, alpha, maxiter=100, collapsed=[W, tau])
    Q.optimize(W, tau, maxiter=100, collapsed=[X, alpha])
    bpplt.pyplot.plot(np.cumsum(Q.cputime), Q.L, 'r:')

    bpplt.pyplot.show()


def mixture_of_gaussians():
    """Collapsed Riemannian conjugate gradient demo

    This is similar although not exactly identical to an experiment in
    (Hensman et al 2012).
    """

    np.random.seed(41)

    # Number of samples
    N = 1000
    # Number of clusters in the model (five in the data)
    K = 10

    # Overlap parameter of clusters
    R = 2

    # Construct the model
    Q = mog.gaussianmix_model(N, K, 2, covariance='diagonal')

    # Generate data from five Gaussian clusters
    mu = np.array([[0, 0],
                   [R, R],
                   [-R, R],
                   [R, -R],
                   [-R, -R]])
    Z = random.categorical(np.ones(5), size=N)
    data = np.empty((N, 2))
    for n in range(N):
        data[n,:] = mu[Z[n]] + np.random.randn(2)
    Q['Y'].observe(data)

    # Take one update step (so phi is ok)
    Q.update(repeat=1)
    Q.save()

    # Run standard VB-EM
    Q.update(repeat=1000, tol=0)
    bpplt.pyplot.plot(np.cumsum(Q.cputime), Q.L, 'k-')

    # Restore the initial state
    Q.load()

    # Run Riemannian conjugate gradient
    Q.optimize('alpha', 'X', 'Lambda', collapsed=['z'], maxiter=300, tol=0)
    bpplt.pyplot.plot(np.cumsum(Q.cputime), Q.L, 'r:')

    bpplt.pyplot.xlabel('CPU time (in seconds)')
    bpplt.pyplot.ylabel('VB lower bound')
    bpplt.pyplot.legend(['VB-EM', 'Collapsed Riemannian CG'], loc='lower right')

    ## bpplt.pyplot.figure()
    ## bpplt.pyplot.plot(data[:,0], data[:,1], 'rx')
    ## bpplt.pyplot.title('Data')

    bpplt.pyplot.show()
    

if __name__ == "__main__":
    #pca()
    mixture_of_gaussians()
