################################################################################
# Copyright (C) 2014 Jaakko Luttinen
#
# This file is licensed under the MIT License.
################################################################################


"""
Demonstrate the linear state-space model with switching dynamics.

The model differs from the classical linear state-space model in that it has a
set of state dynamics matrices of which one is used at each time instance.  A
hidden Markov model is used to select the dynamics matrix.

Some functions in this module are re-usable:
  * ``model`` can be used to construct the LSSM with switching dynamics.
  * ``infer`` can be used to apply the model to given data.
"""

import numpy as np
import matplotlib.pyplot as plt

from bayespy.nodes import (GaussianARD,
                           SwitchingGaussianMarkovChain,
                           CategoricalMarkovChain,
                           Dirichlet,
                           Mixture,
                           Gamma,
                           SumMultiply)

from bayespy.inference.vmp.vmp import VB
from bayespy.inference.vmp import transformations

import bayespy.plot as bpplt


def model(M=20, N=100, D=10, K=3):
    """
    Construct the linear state-space model with switching dynamics.
    """

    #
    # Switching dynamics (HMM)
    #

    # Prior for initial state probabilities
    rho = Dirichlet(1e-3*np.ones(K),
                    name='rho')

    # Prior for state transition probabilities
    V = Dirichlet(1e-3*np.ones(K),
                  plates=(K,),
                  name='V')
    v = 10*np.identity(K) + 1*np.ones((K,K))
    v /= np.sum(v, axis=-1, keepdims=True)
    V.initialize_from_value(v)

    # Hidden states (with unknown initial state probabilities and state
    # transition probabilities)
    Z = CategoricalMarkovChain(rho, V,
                               states=N-1,
                               name='Z',
                               plotter=bpplt.CategoricalMarkovChainPlotter(),
                               initialize=False)
    Z.u[0] = np.random.dirichlet(np.ones(K))
    Z.u[1] = np.reshape(np.random.dirichlet(0.5*np.ones(K*K), size=(N-2)),
                        (N-2, K, K))

    #
    # Linear state-space models
    #

    # Dynamics matrix with ARD
    # (K,D) x ()
    alpha = Gamma(1e-5,
                  1e-5,
                  plates=(K,1,D),
                  name='alpha')
    # (K,1,1,D) x (D)
    A = GaussianARD(0,
                    alpha,
                    shape=(D,),
                    plates=(K,D),
                    name='A',
                    plotter=bpplt.GaussianHintonPlotter())
    A.initialize_from_value(np.identity(D)*np.ones((K,D,D))
                            + 0.1*np.random.randn(K,D,D))

    # Latent states with dynamics
    # (K,1) x (N,D)
    X = SwitchingGaussianMarkovChain(np.zeros(D),         # mean of x0
                                     1e-3*np.identity(D), # prec of x0
                                     A,                   # dynamics
                                     Z,                   # dynamics selection
                                     np.ones(D),          # innovation
                                     n=N,                 # time instances
                                     name='X',
                                     plotter=bpplt.GaussianMarkovChainPlotter())
    X.initialize_from_value(10*np.random.randn(N,D))

    # Mixing matrix from latent space to observation space using ARD
    # (K,1,1,D) x ()
    gamma = Gamma(1e-5,
                  1e-5,
                  plates=(D,),
                  name='gamma')
    # (K,M,1) x (D)
    C = GaussianARD(0,
                    gamma,
                    shape=(D,),
                    plates=(M,1),
                    name='C',
                    plotter=bpplt.GaussianHintonPlotter(rows=-3,cols=-1))
    C.initialize_from_value(np.random.randn(M,1,D))

    # Underlying noiseless function
    # (K,M,N) x ()
    F = SumMultiply('i,i', 
                    C, 
                    X,
                    name='F')
    
    #
    # Mixing the models
    #

    # Observation noise
    tau = Gamma(1e-5,
                1e-5,
                name='tau')
    tau.initialize_from_value(1e2)

    # Emission/observation distribution
    Y = GaussianARD(F, tau,
                    name='Y')

    Q = VB(Y, F,
           Z, rho, V,
           C, gamma, X, A, alpha,
           tau)

    return Q


def infer(y, D, K, rotate=True, debug=False, maxiter=100, mask=True,
          plot_C=True, monitor=False, update_hyper=0, autosave=None):
    
    """
    Apply LSSM with switching dynamics to the given data.
    """
    
    (M, N) = np.shape(y)

    # Construct model
    Q = model(M=M, K=K, N=N, D=D)
    if not plot_C:
        Q['C'].set_plotter(None)

    if autosave is not None:
        Q.set_autosave(autosave, iterations=10)

    Q['Y'].observe(y, mask=mask)

    # Set up rotation speed-up
    if rotate:
        # Initial rotate the D-dimensional state space (X, A, C)
        # Do not update hyperparameters
        rotA_init = transformations.RotateGaussianARD(Q['A'])
        rotX_init = transformations.RotateSwitchingMarkovChain(Q['X'],
                                                               Q['A'],
                                                               Q['Z'],
                                                               rotA_init)
        rotC_init = transformations.RotateGaussianARD(Q['C'])
        R_init = transformations.RotationOptimizer(rotX_init, rotC_init, D)
        # Rotate the D-dimensional state space (X, A, C)
        rotA = transformations.RotateGaussianARD(Q['A'], 
                                                 Q['alpha'])
        rotX = transformations.RotateSwitchingMarkovChain(Q['X'], 
                                                          Q['A'],
                                                          Q['Z'],
                                                          rotA)
        rotC = transformations.RotateGaussianARD(Q['C'],
                                                 Q['gamma'])
        R = transformations.RotationOptimizer(rotX, rotC, D)
        if debug:
            rotate_kwargs = {'maxiter': 10,
                             'check_bound': True,
                             'check_gradient': True}
        else:
            rotate_kwargs = {'maxiter': 10}

    # Run inference
    if monitor:
        Q.plot()
    for n in range(maxiter):
        if n < update_hyper:
            Q.update('X', 'C', 'A', 'tau', 'Z', plot=monitor)
            if rotate:
                R_init.rotate(**rotate_kwargs)
        else:
            Q.update(plot=monitor)
            if rotate:
                R.rotate(**rotate_kwargs)

    return Q


def simulate_data(N):
    """
    Generate time-series data with switching dynamics.
    """

    # Two states: 1) oscillation, 2) random walk
    w1 = 0.02 * 2*np.pi
    A = [ [[np.cos(w1), -np.sin(w1)],
           [np.sin(w1),  np.cos(w1)]],
          [[        1.0,         0.0],
           [        0.0,         0.0]] ]
    C = [[1.0, 0.0]]

    # State switching probabilities
    q = 0.993        # probability to stay in the same state
    r = (1-q)/(2-1)  # probability to switch
    P = q*np.identity(2) + r*(np.ones((2,2))-np.identity(2))

    X = np.zeros((N, 2))
    Z = np.zeros(N)
    Y = np.zeros(N)
    F = np.zeros(N)
    z = np.random.randint(2)
    x = np.random.randn(2)
    Z[0] = z
    X[0,:] = x
    for n in range(1,N):
        x = np.dot(A[z], x) + np.random.randn(2)
        f = np.dot(C, x)
        y = f + 5*np.random.randn()
        z = np.random.choice(2, p=P[z])

        Z[n] = z
        X[n,:] = x
        Y[n] = y
        F[n] = f

    Y = Y[None,:]

    return (Y, F)
    

@bpplt.interactive
def demo(N=1000, maxiter=100, D=3, K=2, seed=42, plot=True, debug=False,
        rotate=True, monitor=True):
    """
    Run the demo for linear state-space model with switching dynamics.
    """

    # Use deterministic random numbers
    if seed is not None:
        np.random.seed(seed)

    # Generate data
    (Y, F) = simulate_data(N)

    # Plot observations
    if plot:
        plt.figure()
        bpplt.timeseries(F, linestyle='-', color='b')
        bpplt.timeseries(Y, linestyle='None', color='r', marker='x')

    # Apply the linear state-space model with switching dynamics
    Q = infer(Y, D, K, 
              debug=debug,
              maxiter=maxiter,
              monitor=monitor,
              rotate=rotate,
              update_hyper=5)

    # Show results
    if plot:
        Q.plot()

    return
    

if __name__ == '__main__':
    import sys, getopt, os
    try:
        opts, args = getopt.getopt(sys.argv[1:],
                                   "",
                                   ["n=",
                                    "d=",
                                    "k=",
                                    "seed=",
                                    "debug",
                                    "no-rotation",
                                    "no-monitor",
                                    "no-plot",
                                    "maxiter="])
    except getopt.GetoptError:
        print('python lssm_sd.py <options>')
        print('--n=<INT>        Number of data vectors')
        print('--d=<INT>        Latent space dimensionality')
        print('--k=<INT>        Number of mixed models')
        print('--maxiter=<INT>  Maximum number of VB iterations')
        print('--seed=<INT>     Seed (integer) for the random number generator')
        print('--no-rotation    Do not peform rotation speed ups')
        print('--no-plot        Do not plot results')
        print('--no-monitor     Do not plot distributions during VB learning')
        print('--debug          Check that the rotations are implemented correctly')
        sys.exit(2)

    kwargs = {}
    for opt, arg in opts:
        if opt == "--maxiter":
            kwargs["maxiter"] = int(arg)
        elif opt == "--d":
            kwargs["D"] = int(arg)
        elif opt == "--k":
            kwargs["K"] = int(arg)
        elif opt == "--seed":
            kwargs["seed"] = int(arg)
        elif opt == "--no-rotation":
            kwargs["rotate"] = False
        elif opt == "--no-monitor":
            kwargs["monitor"] = False
        elif opt == "--no-plot":
            kwargs["plot"] = False
        elif opt == "--debug":
            kwargs["debug"] = True
        elif opt in ("--n",):
            kwargs["N"] = int(arg)
        else:
            raise ValueError("Unhandled option given")

    demo(**kwargs)
    plt.show()
