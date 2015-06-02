################################################################################
# Copyright (C) 2014 Jaakko Luttinen
#
# This file is licensed under the MIT License.
################################################################################


"""
Demonstrate categorical Markov chain with hidden Markov model (HMM)
"""

import numpy as np
import matplotlib.pyplot as plt

from bayespy.nodes import Gaussian, \
                          CategoricalMarkovChain, \
                          Dirichlet, \
                          Mixture, \
                          Categorical

from bayespy.inference.vmp.vmp import VB

import bayespy.plot as bpplt

def hidden_markov_model(distribution, *args, K=3, N=100):

    # Prior for initial state probabilities
    alpha = Dirichlet(1e-3*np.ones(K),
                      name='alpha')

    # Prior for state transition probabilities
    A = Dirichlet(1e-3*np.ones(K),
                  plates=(K,),
                  name='A')

    # Hidden states (with unknown initial state probabilities and state
    # transition probabilities)
    Z = CategoricalMarkovChain(alpha, A,
                               states=N,
                               name='Z')

    # Emission/observation distribution
    Y = Mixture(Z, distribution, *args,
                name='Y')

    Q = VB(Y, Z, alpha, A)

    return Q

def mixture_model(distribution, *args, K=3, N=100):

    # Prior for state probabilities
    alpha = Dirichlet(1e-3*np.ones(K),
                      name='alpha')

    # Cluster assignments
    Z = Categorical(alpha,
                    plates=(N,),
                    name='Z')

    # Observation distribution
    Y = Mixture(Z, distribution, *args,
                name='Y')

    Q = VB(Y, Z, alpha)

    return Q


@bpplt.interactive
def run(N=200, maxiter=10, seed=42, std=2.0, plot=True):

    # Use deterministic random numbers
    if seed is not None:
        np.random.seed(seed)

    #
    # Generate data
    #
    
    mu = np.array([ [0,0], [3,4], [6,0] ])

    K = 3
    p0 = np.ones(K) / K
    q = 0.9 # probability to stay in the same state
    r = (1-q)/(K-1)
    P = q*np.identity(K) + r*(np.ones((3,3))-np.identity(3))

    y = np.zeros((N,2))
    z = np.zeros(N)
    state = np.random.choice(K, p=p0)
    for n in range(N):
        z[n] = state
        y[n,:] = std*np.random.randn(2) + mu[state]
        state = np.random.choice(K, p=P[state])

    plt.figure()

    # Plot data
    plt.subplot(1,3,1)
    plt.axis('equal')
    plt.title('True classification')
    colors = [ [[1,0,0], [0,1,0], [0,0,1]][int(state)] for state in z ]
    plt.plot(y[:,0], y[:,1], 'k-', zorder=-10)
    plt.scatter(y[:,0], y[:,1], c=colors, s=40)

    #
    # Use HMM
    #

    # Run VB inference for HMM
    Q_hmm = hidden_markov_model(Gaussian, mu, K*[std**(-2)*np.identity(2)],
                                K=K, N=N)
    Q_hmm['Y'].observe(y)
    Q_hmm.update(repeat=maxiter)

    # Plot results
    plt.subplot(1,3,2)
    plt.axis('equal')
    plt.title('Classification with HMM')
    colors = Q_hmm['Y'].parents[0]._message_to_child()[0]
    plt.plot(y[:,0], y[:,1], 'k-', zorder=-10)
    plt.scatter(y[:,0], y[:,1], c=colors, s=40)

    #
    # Use mixture model
    #

    # For comparison, run VB for Gaussian mixture
    Q_mix = mixture_model(Gaussian, mu, K*[std**(-2)*np.identity(2)],
                          K=K, N=N)
    Q_mix['Y'].observe(y)
    Q_mix.update(repeat=maxiter)

    # Plot results
    plt.subplot(1,3,3)
    plt.axis('equal')
    plt.title('Classification with mixture')
    colors = Q_mix['Y'].parents[0]._message_to_child()[0]
    plt.plot(y[:,0], y[:,1], 'k-', zorder=-10)
    plt.scatter(y[:,0], y[:,1], c=colors, s=40)


if __name__ == '__main__':
    import sys, getopt, os
    try:
        opts, args = getopt.getopt(sys.argv[1:],
                                   "",
                                   ["n=",
                                    "seed=",
                                    "std=",
                                    "maxiter="])
    except getopt.GetoptError:
        print('python demo_lssm.py <options>')
        print('--n=<INT>        Number of data vectors')
        print('--std=<FLT>      Standard deviation of the Gaussians')
        print('--maxiter=<INT>  Maximum number of VB iterations')
        print('--seed=<INT>     Seed (integer) for the random number generator')
        sys.exit(2)

    kwargs = {}
    for opt, arg in opts:
        if opt == "--maxiter":
            kwargs["maxiter"] = int(arg)
        elif opt == "--std":
            kwargs["std"] = float(arg)
        elif opt == "--seed":
            kwargs["seed"] = int(arg)
        elif opt in ("--n",):
            kwargs["N"] = int(arg)
        else:
            raise ValueError("Unhandled option given")

    run(**kwargs)
    plt.show()
