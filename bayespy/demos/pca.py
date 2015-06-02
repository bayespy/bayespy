################################################################################
# Copyright (C) 2011-2014 Jaakko Luttinen
#
# This file is licensed under the MIT License.
################################################################################


import numpy as np

import matplotlib.pyplot as plt
import bayespy.plot as myplt

from bayespy.utils import misc
from bayespy.utils import random
from bayespy import nodes

from bayespy.inference.vmp.vmp import VB
from bayespy.inference.vmp import transformations

import bayespy.plot as bpplt

def model(M, N, D):
    # Construct the PCA model with ARD

    # ARD
    alpha = nodes.Gamma(1e-2,
                        1e-2,
                        plates=(D,),
                        name='alpha')

    # Loadings
    W = nodes.GaussianARD(0,
                          alpha,
                          shape=(D,),
                          plates=(M,1),
                          name='W')

    # States
    X = nodes.GaussianARD(0,
                          1,
                          shape=(D,),
                          plates=(1,N),
                          name='X')

    # PCA
    F = nodes.SumMultiply('i,i', W, X,
                          name='F')

    # Noise
    tau = nodes.Gamma(1e-2, 1e-2,
                      name='tau')

    # Noisy observations
    Y = nodes.GaussianARD(F, tau,
                          name='Y')

    # Initialize some nodes randomly
    X.initialize_from_random()
    W.initialize_from_random()

    return VB(Y, F, W, X, tau, alpha)


@bpplt.interactive
def run(M=10, N=100, D_y=3, D=5, seed=42, rotate=False, maxiter=1000, debug=False, plot=True):

    if seed is not None:
        np.random.seed(seed)
    
    # Generate data
    w = np.random.normal(0, 1, size=(M,1,D_y))
    x = np.random.normal(0, 1, size=(1,N,D_y))
    f = misc.sum_product(w, x, axes_to_sum=[-1])
    y = f + np.random.normal(0, 0.1, size=(M,N))

    # Construct model
    Q = model(M, N, D)

    # Data with missing values
    mask = random.mask(M, N, p=0.5) # randomly missing
    y[~mask] = np.nan
    Q['Y'].observe(y, mask=mask)

    # Run inference algorithm
    if rotate:
        # Use rotations to speed up learning
        rotW = transformations.RotateGaussianARD(Q['W'], Q['alpha'])
        rotX = transformations.RotateGaussianARD(Q['X'])
        R = transformations.RotationOptimizer(rotW, rotX, D)
        if debug:
            Q.callback = lambda : R.rotate(check_bound=True,
                                           check_gradient=True)
        else:
            Q.callback = R.rotate

    # Use standard VB-EM alone
    Q.update(repeat=maxiter)

    # Plot results
    if plot:
        plt.figure()
        bpplt.timeseries_normal(Q['F'], scale=2)
        bpplt.timeseries(f, color='g', linestyle='-')
        bpplt.timeseries(y, color='r', linestyle='None', marker='+')


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
                                    "debug",
                                    "rotate"])
    except getopt.GetoptError:
        print('python demo_pca.py <options>')
        print('--m=<INT>        Dimensionality of data vectors')
        print('--n=<INT>        Number of data vectors')
        print('--d=<INT>        Dimensionality of the latent vectors in the model')
        print('--k=<INT>        Dimensionality of the true latent vectors')
        print('--rotate         Apply speed-up rotations')
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

