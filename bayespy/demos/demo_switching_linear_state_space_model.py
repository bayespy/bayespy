######################################################################
# Copyright (C) 2014 Jaakko Luttinen
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
Demonstrate switching linear state-space model
"""

import numpy as np
import matplotlib.pyplot as plt

from bayespy.nodes import GaussianARD, \
                          GaussianMarkovChain, \
                          CategoricalMarkovChain, \
                          Dirichlet, \
                          Mixture, \
                          Gamma, \
                          SumMultiply

from bayespy.inference.vmp.vmp import VB

import bayespy.plot.plotting as bpplt

def switching_linear_state_space_model(K=3, D=10, N=100, M=20):

    #
    # Linear state-space models
    #

    # Dynamics matrix with ARD
    # (K,1,1,1,D) x ()
    alpha = Gamma(1e-5,
                  1e-5,
                  plates=(K,1,1,1,D),
                  name='alpha')
    # (K,1,1,D) x (D)
    A = GaussianARD(0,
                    alpha,
                    shape=(D,),
                    plates=(K,1,1,D),
                    name='A')

    # Latent states with dynamics
    # (K,1) x (N,D)
    X = GaussianMarkovChain(np.zeros(D),         # mean of x0
                            1e-3*np.identity(D), # prec of x0
                            A,                   # dynamics
                            np.ones(D),          # innovation
                            n=N,                 # time instances
                            name='X')

    # Mixing matrix from latent space to observation space using ARD
    # (K,1,1,D) x ()
    gamma = Gamma(1e-5,
                  1e-5,
                  plates=(K,1,1,D),
                  name='gamma')
    # (K,M,1) x (D)
    C = GaussianARD(0,
                    gamma,
                    shape=(D,),
                    plates=(K,M,1),
                    name='C')

    # Underlying noiseless function
    # (K,M,N) x ()
    F = SumMultiply('i,i', 
                    C, 
                    X)
    
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

    # Hidden states (with unknown initial state probabilities and state
    # transition probabilities)
    Z = CategoricalMarkovChain(rho, V,
                               states=N,
                               name='Z')

    #
    # Mixing the models
    #

    # Observation noise
    tau = Gamma(1e-5,
                1e-5,
                name='tau')

    # Emission/observation distribution
    Y = Mixture(Z, GaussianARD, F, tau,
                cluster_plate=-3,
                name='Y')

    Q = VB(Y, 
           Z, rho, V,
           C, gamma, X, A, alpha)

    return Q

def run(M=20, N=200, K=3, D=10, maxiter=10, seed=42, plot=True):

    # Use deterministic random numbers
    if seed is not None:
        np.random.seed(seed)

    #
    # Generate data
    #
    

    plt.figure()

    # Plot data

    #
    # Use switching linear state-space model
    #

    # Run VB inference for HMM
    Q = switching_linear_state_space_model(M=M, K=K, N=N, D=D)
    Q.update(repeat=maxiter)

    # Plot results

    plt.show()
    

if __name__ == '__main__':
    import sys, getopt, os
    try:
        opts, args = getopt.getopt(sys.argv[1:],
                                   "",
                                   ["n=",
                                    "k=",
                                    "seed=",
                                    "maxiter="])
    except getopt.GetoptError:
        print('python demo_lssm.py <options>')
        print('--n=<INT>        Number of data vectors')
        print('--d=<INT>        Latent space dimensionality')
        print('--k=<INT>        Number of mixed models')
        print('--maxiter=<INT>  Maximum number of VB iterations')
        print('--seed=<INT>     Seed (integer) for the random number generator')
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
        elif opt in ("--n",):
            kwargs["N"] = int(arg)
        else:
            raise ValueError("Unhandled option given")

    run(**kwargs)
