######################################################################
# Copyright (C) 2011-2014 Jaakko Luttinen
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

    return (Y, F, W, X, tau, alpha)


@bpplt.interactive
def run(M=10, N=100, D_y=3, D=5, seed=42, rotate=False, maxiter=100, debug=False, plot=True):

    if seed is not None:
        np.random.seed(seed)
    
    # Generate data
    w = np.random.normal(0, 1, size=(M,1,D_y))
    x = np.random.normal(0, 1, size=(1,N,D_y))
    f = misc.sum_product(w, x, axes_to_sum=[-1])
    y = f + np.random.normal(0, 0.2, size=(M,N))

    # Construct model
    (Y, F, W, X, tau, alpha) = model(M, N, D)

    # Data with missing values
    mask = random.mask(M, N, p=0.5) # randomly missing
    y[~mask] = np.nan
    Y.observe(y, mask=mask)

    # Construct inference machine
    Q = VB(Y, W, X, tau, alpha)

    # Initialize some nodes randomly
    X.initialize_from_random()
    W.initialize_from_random()

    # Run inference algorithm
    if rotate:
        # Use rotations to speed up learning
        rotW = transformations.RotateGaussianARD(W, alpha)
        rotX = transformations.RotateGaussianARD(X)
        R = transformations.RotationOptimizer(rotW, rotX, D)
        for ind in range(maxiter):
            Q.update()
            if debug:
                R.rotate(check_bound=True,
                         check_gradient=True)
            else:
                R.rotate()
            
    else:
        # Use standard VB-EM alone
        Q.update(repeat=maxiter)

    # Plot results
    if plot:
        plt.figure()
        bpplt.timeseries_normal(F, scale=2)
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

