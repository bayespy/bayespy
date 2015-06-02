################################################################################
# Copyright (C) 2013-2014 Jaakko Luttinen
#
# This file is licensed under the MIT License.
################################################################################


"""
Demonstrate the linear state-space model with time-varying dynamics.

The observation is 1-D signal with changing frequency. The frequency oscillates
so it can be learnt too. Missing values are used to create a few gaps in the
data so the task is to reconstruct the gaps.

For reference, see the following publication:
(TODO)

Some functions in this module are re-usable:
  * ``model`` can be used to construct the LSSM with switching dynamics.
  * ``infer`` can be used to apply the model to given data.
"""

import numpy as np
import scipy
import matplotlib.pyplot as plt

from bayespy.nodes import (GaussianMarkovChain,
                           VaryingGaussianMarkovChain,
                           GaussianARD,
                           Gamma,
                           SumMultiply)

from bayespy.utils import misc
from bayespy.utils import random

from bayespy.inference.vmp.vmp import VB
from bayespy.inference.vmp import transformations
from bayespy.inference.vmp.nodes.gaussian import GaussianMoments

import bayespy.plot as bpplt


def model(M, N, D, K):
    """
    Construct the linear state-space model with time-varying dynamics

    For reference, see the following publication:
    (TODO)
    """

    #
    # The model block for the latent mixing weight process
    #
    
    # Dynamics matrix with ARD
    # beta : (K) x ()
    beta = Gamma(1e-5,
                 1e-5,
                 plates=(K,),
                 name='beta')
    # B : (K) x (K)
    B = GaussianARD(np.identity(K),
                    beta,
                    shape=(K,),
                    plates=(K,),
                    name='B',
                    plotter=bpplt.GaussianHintonPlotter(rows=0, 
                                                        cols=1,
                                                        scale=0),
                    initialize=False)
    B.initialize_from_value(np.identity(K))

    # Mixing weight process, that is, the weights in the linear combination of
    # state dynamics matrices
    # S : () x (N,K)
    S = GaussianMarkovChain(np.ones(K),
                            1e-6*np.identity(K),
                            B,
                            np.ones(K),
                            n=N,
                            name='S',
                            plotter=bpplt.GaussianMarkovChainPlotter(scale=2),
                            initialize=False)
    s = 10*np.random.randn(N,K)
    s[:,0] = 10
    S.initialize_from_value(s)

    #
    # The model block for the latent states
    #
        
    # Projection matrix of the dynamics matrix
    # alpha : (K) x ()
    alpha = Gamma(1e-5,
                  1e-5,
                  plates=(D,K),
                  name='alpha')
    alpha.initialize_from_value(1*np.ones((D,K)))
    # A : (D) x (D,K)
    A = GaussianARD(0,
                    alpha,
                    shape=(D,K),
                    plates=(D,),
                    name='A',
                    plotter=bpplt.GaussianHintonPlotter(rows=0, 
                                                        cols=1,
                                                        scale=0),
                    initialize=False)

    # Initialize S and A such that A*S is almost an identity matrix
    a = np.zeros((D,D,K))
    a[np.arange(D),np.arange(D),np.zeros(D,dtype=int)] = 1
    a[:,:,0] = np.identity(D) / s[0,0]
    a[:,:,1:] = 0.1/s[0,0]*np.random.randn(D,D,K-1)
    A.initialize_from_value(a)

    # Latent states with dynamics
    # X : () x (N,D)
    X = VaryingGaussianMarkovChain(np.zeros(D),         # mean of x0
                                   1e-3*np.identity(D), # prec of x0
                                   A,                   # dynamics matrices
                                   S._convert(GaussianMoments)[1:], # temporal weights
                                   np.ones(D),          # innovation
                                   n=N,                 # time instances
                                   name='X',
                                   plotter=bpplt.GaussianMarkovChainPlotter(scale=2),
                                   initialize=False)
    X.initialize_from_value(np.random.randn(N,D))

    #
    # The model block for observations
    #

    # Mixing matrix from latent space to observation space using ARD
    # gamma : (D) x ()
    gamma = Gamma(1e-5,
                  1e-5,
                  plates=(D,),
                  name='gamma')
    gamma.initialize_from_value(1e-2*np.ones(D))
    # C : (M,1) x (D)
    C = GaussianARD(0,
                    gamma,
                    shape=(D,),
                    plates=(M,1),
                    name='C',
                    plotter=bpplt.GaussianHintonPlotter(rows=0,
                                                        cols=2,
                                                        scale=0))
    C.initialize_from_value(np.random.randn(M,1,D))

    # Noiseless process
    # F : (M,N) x ()
    F = SumMultiply('d,d',
                    C,
                    X,
                    name='F')
                  
    # Observation noise
    # tau : () x ()
    tau = Gamma(1e-5,
                1e-5,
                name='tau')
    tau.initialize_from_value(1e2)

    # Observations
    # Y: (M,N) x ()
    Y = GaussianARD(F,
                    tau,
                    name='Y')

    # Construct inference machine
    Q = VB(Y, F, C, gamma, X, A, alpha, tau, S, B, beta)

    return Q


def infer(y, D, K,
          mask=True, 
          maxiter=100,
          rotate=False, 
          debug=False, 
          precompute=False,
          update_hyper=0,
          start_rotating=0,
          start_rotating_weights=0,
          plot_C=True,
          monitor=True,
          autosave=None):
    
    """
    Run VB inference for linear state-space model with time-varying dynamics.
    """

    y = misc.atleast_nd(y, 2)
    (M, N) = np.shape(y)

    # Construct the model
    Q = model(M, N, D, K)
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
        rotX_init = transformations.RotateVaryingMarkovChain(Q['X'], 
                                                             Q['A'], 
                                                             Q['S']._convert(GaussianMoments)[...,1:,None], 
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
        rotX = transformations.RotateVaryingMarkovChain(Q['X'], 
                                                        Q['A'], 
                                                        Q['S']._convert(GaussianMoments)[...,1:,None], 
                                                        rotA)
        rotC = transformations.RotateGaussianARD(Q['C'],
                                                 Q['gamma'],
                                                 axis=0,
                                                 precompute=precompute)
        R_X = transformations.RotationOptimizer(rotX, rotC, D)

        # Rotate the K-dimensional latent dynamics space (S, A, C)
        rotB = transformations.RotateGaussianARD(Q['B'],
                                                 Q['beta'], 
                                                 precompute=precompute)
        rotS = transformations.RotateGaussianMarkovChain(Q['S'], rotB)
        rotA = transformations.RotateGaussianARD(Q['A'],
                                                 Q['alpha'],
                                                 axis=-1,
                                                 precompute=precompute)
        R_S = transformations.RotationOptimizer(rotS, rotA, K)
            
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
                if ind >= start_rotating_weights:
                    R_S.rotate(**rotate_kwargs)

    # Return the posterior approximation
    return Q


def simulate_data(N):
    """
    Generate a signal with changing frequency
    """

    t = np.arange(N)
    a = 0.1 * 2*np.pi  # base frequency
    b = 0.01 * 2*np.pi # frequency of the frequency change
    c = 8              # magnitude of the frequency change
    f = np.sin( a * (t + c*np.sin(b*t)) )
    y = f + 0.1*np.random.randn(N)

    return (y, f)


@bpplt.interactive
def demo(N=1000, D=5, K=4, seed=42, maxiter=200, rotate=True, debug=False,
         precompute=False, plot=True):

    # Seed for random number generator
    if seed is not None:
        np.random.seed(seed)

    # Create data
    (y, f) = simulate_data(N)

    # Create some gaps
    mask_gaps = misc.trues(N)
    for m in range(100, N, 140):
        start = m
        end = min(m+15, N-1)
        mask_gaps[start:end] = False
    # Randomly missing values
    mask_random = np.logical_or(random.mask(N, p=0.8),
                                np.logical_not(mask_gaps))
    # Remove the observations
    mask = np.logical_and(mask_gaps, mask_random)
    y[~mask] = np.nan # BayesPy doesn't require NaNs, they're just for plotting.

    # Add row axes
    y = y[None,...]
    f = f[None,...]
    mask = mask[None,...]
    mask_gaps = mask_gaps[None,...]
    mask_random = mask_random[None,...]
    
    # Run the method
    Q = infer(y, D, K,
              mask=mask, 
              maxiter=maxiter,
              rotate=rotate,
              debug=debug,
              precompute=precompute,
              update_hyper=10,
              start_rotating_weights=20,
              monitor=True)

    if plot:

        # Plot observations
        plt.figure()
        bpplt.timeseries_normal(Q['F'], scale=2)
        bpplt.timeseries(f, linestyle='-', color='b')
        bpplt.timeseries(y, linestyle='None', color='r', marker='.')
        plt.ylim([-2, 2])
    
        # Plot latent space
        Q.plot('X')
    
        # Plot mixing weight space
        Q.plot('S')

    # Compute RMSE
    rmse_random = misc.rmse(Q['Y'].get_moments()[0][~mask_random], 
                            f[~mask_random])
    rmse_gaps = misc.rmse(Q['Y'].get_moments()[0][~mask_gaps],
                          f[~mask_gaps])
    print("RMSE for randomly missing values: %f" % rmse_random)
    print("RMSE for gap values: %f" % rmse_gaps)


if __name__ == '__main__':
    import sys, getopt, os
    try:
        opts, args = getopt.getopt(sys.argv[1:],
                                   "",
                                   [
                                    "n=",
                                    "d=",
                                    "k=",
                                    "seed=",
                                    "maxiter=",
                                    "debug",
                                    "precompute",
                                    "no-plot",
                                    "no-rotation"])
    except getopt.GetoptError:
        print('python lssm_tvd.py <options>')
        print('--n=<INT>        Number of data vectors')
        print('--d=<INT>        Dimensionality of the latent vectors in the model')
        print('--k=<INT>        Dimensionality of the latent mixing weights')
        print('--no-rotation    Do not apply speed-up rotations')
        print('--maxiter=<INT>  Maximum number of VB iterations')
        print('--seed=<INT>     Seed (integer) for the random number generator')
        print('--debug          Check that the rotations are implemented correctly')
        print('--no-plot        Do not plot results')
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
        elif opt == "--n":
            kwargs["N"] = int(arg)
        elif opt == "--d":
            kwargs["D"] = int(arg)
        elif opt == "--k":
            if int(arg) == 0:
                kwargs["K"] = None
            else:
                kwargs["K"] = int(arg)
        elif opt == "--no-plot":
            kwargs["plot"] = False
        else:
            raise ValueError("Unhandled argument given")

    demo(**kwargs)
    plt.show()
