######################################################################
# Copyright (C) 2013-2014 Jaakko Luttinen
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
Demonstrate linear Gaussian state-space model.
"""

import numpy as np
import scipy
import matplotlib.pyplot as plt

from bayespy.nodes import GaussianMarkovChain
from bayespy.nodes import Gaussian, GaussianArrayARD
from bayespy.nodes import Gamma
from bayespy.nodes import Normal
from bayespy.nodes import SumMultiply
from bayespy.inference.vmp.nodes.gamma import diagonal

from bayespy.utils import utils
from bayespy.utils import random

from bayespy.inference.vmp.vmp import VB
from bayespy.inference.vmp import transformations

import bayespy.plot.plotting as bpplt

def linear_state_space_model(D=3, N=100, M=10):

    # Dynamics matrix with ARD
    alpha = Gamma(1e-5,
                  1e-5,
                  plates=(D,),
                  name='alpha')
    A = GaussianArrayARD(0,
                         alpha,
                         shape=(D,),
                         plates=(D,),
                         name='A')

    # Latent states with dynamics
    X = GaussianMarkovChain(np.zeros(D),         # mean of x0
                            1e-3*np.identity(D), # prec of x0
                            A,                   # dynamics
                            np.ones(D),          # innovation
                            n=N,                 # time instances
                            name='X')

    # Mixing matrix from latent space to observation space using ARD
    gamma = Gamma(1e-5,
                  1e-5,
                  plates=(D,),
                  name='gamma')
    C = GaussianArrayARD(0,
                         gamma,
                         shape=(D,),
                         plates=(M,1),
                         name='C')

    # Observation noise
    tau = Gamma(1e-5,
                1e-5,
                name='tau')

    # Underlying noiseless function
    F = SumMultiply('i,i', 
                    C, 
                    X.as_gaussian())
    
    # Noisy observations
    Y = GaussianArrayARD(F,
                         tau,
                         name='Y')

    return (Y, F, X, tau, C, gamma, A, alpha)

def run(M=6, N=200, D=3, maxiter=100, debug=False, seed=42, rotate=False, precompute=False):

    # Use deterministic random numbers
    if seed is not None:
        np.random.seed(seed)

    # Simulate some data
    K = 3
    c = np.random.randn(M,K)
    w = 0.3
    a = np.array([[np.cos(w), -np.sin(w), 0], 
                  [np.sin(w), np.cos(w),  0], 
                  [0,         0,          1]])
    x = np.empty((N,K))
    f = np.empty((M,N))
    y = np.empty((M,N))
    x[0] = 10*np.random.randn(K)
    f[:,0] = np.dot(c,x[0])
    y[:,0] = f[:,0] + 3*np.random.randn(M)
    for n in range(N-1):
        x[n+1] = np.dot(a,x[n]) + np.random.randn(K)
        f[:,n+1] = np.dot(c,x[n+1])
        y[:,n+1] = f[:,n+1] + 3*np.random.randn(M)

    # Create the model
    (Y, CX, X, tau, C, gamma, A, alpha) = linear_state_space_model(D=D, 
                                                                   N=N,
                                                                   M=M)
    
    # Add missing values randomly
    mask = random.mask(M, N, p=0.3)
    # Add missing values to a period of time
    mask[:,30:80] = False
    y[~mask] = np.nan # BayesPy doesn't require this. Just for plotting.
    # Observe the data
    Y.observe(y, mask=mask)

    # Initialize nodes (must use some randomness for C)
    C.initialize_from_random()

    # Run inference
    Q = VB(Y, X, C, gamma, A, alpha, tau)

    #
    # Run inference with rotations.
    #
    if rotate:
        rotA = transformations.RotateGaussianArrayARD(A, alpha, precompute=precompute)
        rotX = transformations.RotateGaussianMarkovChain(X, rotA)
        rotC = transformations.RotateGaussianArrayARD(C, gamma)
        R = transformations.RotationOptimizer(rotX, rotC, D)

        for ind in range(maxiter):
            Q.update()
            if debug:
                R.rotate(maxiter=10, 
                         check_gradient=True,
                         check_bound=True)
            else:
                R.rotate()

    else:
        Q.update(repeat=maxiter)
        
    # Show results
    plt.figure()
    bpplt.timeseries_normal(CX, scale=2)
    bpplt.timeseries(f, 'b-')
    bpplt.timeseries(y, 'r.')
    plt.show()
    

if __name__ == '__main__':
    import sys, getopt, os
    try:
        opts, args = getopt.getopt(sys.argv[1:],
                                   "",
                                   ["m=",
                                    "n=",
                                    "d=",
                                    "seed=",
                                    "maxiter=",
                                    "debug",
                                    "precompute",
                                    "rotate"])
    except getopt.GetoptError:
        print('python demo_lssm.py <options>')
        print('--m=<INT>        Dimensionality of data vectors')
        print('--n=<INT>        Number of data vectors')
        print('--d=<INT>        Dimensionality of the latent vectors in the model')
        print('--rotate         Apply speed-up rotations')
        print('--maxiter=<INT>  Maximum number of VB iterations')
        print('--seed=<INT>     Seed (integer) for the random number generator')
        print('--debug          Check that the rotations are implemented correctly')
        print('--precompute     Precompute some moments when rotating. May '
              'speed up or slow down.')
        sys.exit(2)

    kwargs = {}
    for opt, arg in opts:
        if opt == "--rotate":
            kwargs["rotate"] = True
        elif opt == "--maxiter":
            kwargs["maxiter"] = int(arg)
        elif opt == "--debug":
            kwargs["debug"] = True
        elif opt == "--precompute":
            kwargs["precompute"] = True
        elif opt == "--seed":
            kwargs["seed"] = int(arg)
        elif opt in ("--m",):
            kwargs["M"] = int(arg)
        elif opt in ("--n",):
            kwargs["N"] = int(arg)
        elif opt in ("--d",):
            kwargs["D"] = int(arg)

    run(**kwargs)
