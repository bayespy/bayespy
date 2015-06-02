################################################################################
# Copyright (C) 2013-2014 Jaakko Luttinen
#
# This file is licensed under the MIT License.
################################################################################


"""
Demonstrate linear Gaussian state-space model.

Some of the functions in this module are re-usable: 
  * ``model`` can be used to construct the classical linear state-space model.
  * ``infer`` can be used to apply linear state-space model to given data.
"""

import numpy as np
import scipy
import matplotlib.pyplot as plt

from bayespy.nodes import GaussianMarkovChain
from bayespy.nodes import Gaussian, GaussianARD
from bayespy.nodes import Gamma
from bayespy.nodes import SumMultiply
from bayespy.inference.vmp.nodes.gamma import diagonal

from bayespy.utils import random

from bayespy.inference.vmp.vmp import VB
from bayespy.inference.vmp import transformations

import bayespy.plot as bpplt


def model(M=10, N=100, D=3):
    """
    Construct linear state-space model.

    See, for instance, the following publication:
    "Fast variational Bayesian linear state-space model"
    Luttinen (ECML 2013)
    """

    # Dynamics matrix with ARD
    alpha = Gamma(1e-5,
                  1e-5,
                  plates=(D,),
                  name='alpha')
    A = GaussianARD(0,
                    alpha,
                    shape=(D,),
                    plates=(D,),
                    plotter=bpplt.GaussianHintonPlotter(rows=0, 
                                                        cols=1,
                                                        scale=0),
                    name='A')
    A.initialize_from_value(np.identity(D))

    # Latent states with dynamics
    X = GaussianMarkovChain(np.zeros(D),         # mean of x0
                            1e-3*np.identity(D), # prec of x0
                            A,                   # dynamics
                            np.ones(D),          # innovation
                            n=N,                 # time instances
                            plotter=bpplt.GaussianMarkovChainPlotter(scale=2),
                            name='X')
    X.initialize_from_value(np.random.randn(N,D))

    # Mixing matrix from latent space to observation space using ARD
    gamma = Gamma(1e-5,
                  1e-5,
                  plates=(D,),
                  name='gamma')
    gamma.initialize_from_value(1e-2*np.ones(D))
    C = GaussianARD(0,
                    gamma,
                    shape=(D,),
                    plates=(M,1),
                    plotter=bpplt.GaussianHintonPlotter(rows=0,
                                                        cols=2,
                                                        scale=0),
                    name='C')
    C.initialize_from_value(np.random.randn(M,1,D))

    # Observation noise
    tau = Gamma(1e-5,
                1e-5,
                name='tau')
    tau.initialize_from_value(1e2)

    # Underlying noiseless function
    F = SumMultiply('i,i', 
                    C, 
                    X,
                    name='F')
    
    # Noisy observations
    Y = GaussianARD(F,
                    tau,
                    name='Y')

    Q = VB(Y, F, C, gamma, X, A, alpha, tau, C)

    return Q


def infer(y, D, 
          mask=True, 
          maxiter=100,
          rotate=True,
          debug=False,
          precompute=False,
          update_hyper=0,
          start_rotating=0,
          plot_C=True,
          monitor=True,
          autosave=None):
    """
    Apply linear state-space model for the given data.
    """
        
    (M, N) = np.shape(y)

    # Construct the model
    Q = model(M, N, D)
    if not plot_C:
        Q['C'].set_plotter(None)
        
    if autosave is not None:
        Q.set_autosave(autosave, iterations=10)

    # Observe data
    Q['Y'].observe(y, mask=mask)

    # Set up rotation speed-up
    if rotate:
        
        # Initial rotate the D-dimensional state space (X, A, C)
        # Does not update hyperparameters
        rotA_init = transformations.RotateGaussianARD(Q['A'], 
                                                      axis=0,
                                                      precompute=precompute)
        rotX_init = transformations.RotateGaussianMarkovChain(Q['X'], 
                                                              rotA_init)
        rotC_init = transformations.RotateGaussianARD(Q['C'],
                                                      axis=0,
                                                      precompute=precompute)
        R_X_init = transformations.RotationOptimizer(rotX_init, rotC_init, D)

        # Rotate the D-dimensional state space (X, A, C)
        rotA = transformations.RotateGaussianARD(Q['A'], 
                                                 Q['alpha'],
                                                 axis=0,
                                                 precompute=precompute)
        rotX = transformations.RotateGaussianMarkovChain(Q['X'], 
                                                         rotA)
        rotC = transformations.RotateGaussianARD(Q['C'],
                                                 Q['gamma'],
                                                 axis=0,
                                                 precompute=precompute)
        R_X = transformations.RotationOptimizer(rotX, rotC, D)

        # Keyword arguments for the rotation
        if debug:
            rotate_kwargs = {'maxiter': 10,
                             'check_bound': True,
                             'check_gradient': True}
        else:
            rotate_kwargs = {'maxiter': 10}

    # Plot initial distributions
    if monitor:
        Q.plot()

    # Run inference using rotations
    for ind in range(maxiter):

        if ind < update_hyper:
            # It might be a good idea to learn the lower level nodes a bit
            # before starting to learn the upper level nodes.
            Q.update('X', 'C', 'A', 'tau', plot=monitor)
            if rotate and ind >= start_rotating:
                # Use the rotation which does not update alpha nor beta
                R_X_init.rotate(**rotate_kwargs)
        else:
            Q.update(plot=monitor)
            if rotate and ind >= start_rotating:
                # It might be a good idea to not rotate immediately because it
                # might lead to pruning out components too efficiently before
                # even estimating them roughly
                R_X.rotate(**rotate_kwargs)

    # Return the posterior approximation
    return Q


def simulate_data(M, N):
    """
    Generate a dataset using linear state-space model.

    The process has two latent oscillation components and one random walk
    component.
    """
    # Simulate some data
    D = 3
    c = np.random.randn(M, D)
    w = 0.3
    a = np.array([[np.cos(w), -np.sin(w), 0], 
                  [np.sin(w), np.cos(w),  0], 
                  [0,         0,          1]])
    x = np.empty((N,D))
    f = np.empty((M,N))
    y = np.empty((M,N))
    x[0] = 10*np.random.randn(D)
    f[:,0] = np.dot(c,x[0])
    y[:,0] = f[:,0] + 3*np.random.randn(M)
    for n in range(N-1):
        x[n+1] = np.dot(a,x[n]) + np.random.randn(D)
        f[:,n+1] = np.dot(c,x[n+1])
        y[:,n+1] = f[:,n+1] + 3*np.random.randn(M)

    return (y, f)


@bpplt.interactive
def demo(M=6, N=200, D=3, maxiter=100, debug=False, seed=42, rotate=True,
         precompute=False, plot=True, monitor=True):
    """
    Run the demo for linear state-space model.
    """

    # Use deterministic random numbers
    if seed is not None:
        np.random.seed(seed)

    # Get data
    (y, f) = simulate_data(M, N)

    # Add missing values randomly
    mask = random.mask(M, N, p=0.3)
    # Add missing values to a period of time
    mask[:,30:80] = False
    y[~mask] = np.nan # BayesPy doesn't require this. Just for plotting.

    # Run inference
    Q = infer(y, D,
              mask=mask,
              rotate=rotate,
              debug=debug,
              monitor=monitor,
              maxiter=maxiter)

    if plot:
        # Show results
        plt.figure()
        bpplt.timeseries_normal(Q['F'], scale=2)
        bpplt.timeseries(f, linestyle='-', color='b')
        bpplt.timeseries(y, linestyle='None', color='r', marker='.')


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
                                    "no-plot",
                                    "no-monitor",
                                    "no-rotation"])
    except getopt.GetoptError:
        print('python lssm.py <options>')
        print('--m=<INT>        Dimensionality of data vectors')
        print('--n=<INT>        Number of data vectors')
        print('--d=<INT>        Dimensionality of the latent vectors in the model')
        print('--no-rotation    Do not apply speed-up rotations')
        print('--maxiter=<INT>  Maximum number of VB iterations')
        print('--seed=<INT>     Seed (integer) for the random number generator')
        print('--debug          Check that the rotations are implemented correctly')
        print('--no-plot        Do not plot the results')
        print('--no-monitor     Do not plot distributions during learning')
        print('--precompute     Precompute some moments when rotating. May '
              'speed up or slow down.')
        sys.exit(2)

    kwargs = {}
    for opt, arg in opts:
        if opt == "--no-rotation":
            kwargs["rotate"] = False
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
        elif opt in ("--no-plot"):
            kwargs["plot"] = False
        elif opt in ("--no-monitor"):
            kwargs["monitor"] = False
        else:
            raise ValueError("Unhandled option given")

    demo(**kwargs)
    plt.show()
