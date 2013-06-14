######################################################################
# Copyright (C) 2013 Jaakko Luttinen
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
Demonstrate linear Gaussian state-space model with drifting dynamics.
"""

import numpy as np
import scipy
import matplotlib.pyplot as plt

from bayespy.inference.vmp.nodes.gaussian_markov_chain import GaussianMarkovChain
from bayespy.inference.vmp.nodes.gaussian import Gaussian
from bayespy.inference.vmp.nodes.gamma import Gamma, diagonal
from bayespy.inference.vmp.nodes.normal import Normal
from bayespy.inference.vmp.nodes.dot import Dot, MatrixDot
from bayespy.inference.vmp.nodes.deterministic import tile

from bayespy.utils import utils
from bayespy.utils import random

from bayespy.inference.vmp.vmp import VB
from bayespy.inference.vmp import transformations

import bayespy.plot.plotting as bpplt


def simulate_static_lssm(M, N):
    # Simulate some data
    D = 3
    c = np.random.randn(M,D)
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

def simulate_drifting_lssm(M, N):
    # Simulate some data
    D = 3
    c = np.random.randn(M,D)
    a = np.empty((N-1,D,D))
    n = 0
    for l in np.linspace(5, 1, num=N-1):
        w = 1/l
        a[n] = np.array([[np.cos(w), -np.sin(w), 0], 
                         [np.sin(w), np.cos(w),  0], 
                         [0,         0,          1]])
        n = n + 1
    x = np.empty((N,D))
    f = np.empty((M,N))
    y = np.empty((M,N))
    x[0] = 10*np.random.randn(D)
    f[:,0] = np.dot(c,x[0])
    y[:,0] = f[:,0] + 3*np.random.randn(M)
    for n in range(N-1):
        x[n+1] = np.dot(a[n],x[n]) + np.random.randn(D)
        f[:,n+1] = np.dot(c,x[n+1])
        y[:,n+1] = f[:,n+1] + 3*np.random.randn(M)
    return (y, f)

def run_dlssm(y, f, mask, D, K, maxiter):
    """
    Run VB inference for linear state space model with drifting dynamics.
    """
        
    (M, N) = np.shape(y)

    # Dynamics matrix with ARD
    # alpha : (D) x ()
    alpha = Gamma(1e-5,
                  1e-5,
                  plates=(K,),
                  name='alpha')
    # A : (K) x (K)
    A = Gaussian(np.zeros(K),
    #np.identity(K),
                 diagonal(alpha),
                 plates=(K,),
                 name='A_S')
    A.initialize_from_value(np.identity(K))

    # rho
    ## rho = Gamma(1e-5,
    ##             1e-5,
    ##             plates=(K,),
    ##             name="rho")

    # S : () x (N-1,K)
    S = GaussianMarkovChain(np.ones(K),
                            1e-6*np.identity(K),
                            A,
                            np.ones(K),
                            n=N-1,
                            name='S')
    S.initialize_from_value(1*np.ones((N-1,K)))

    # Projection matrix of the dynamics matrix
    # beta : (K) x ()
    beta = Gamma(1e-5,
                 1e-5,
                 plates=(K,),
                 name='beta')
    # B : (D) x (D*K)
    B = Gaussian(np.zeros(D*K),
                 diagonal(tile(beta, D)),
                 plates=(D,),
                 name='B')
    b = np.zeros((D,D,K))
    b[np.arange(D),np.arange(D),np.zeros(D,dtype=int)] = 1
    B.initialize_from_value(np.reshape(1*b, (D,D*K)))

    # A : (N-1,D) x (D)
    BS = MatrixDot(B, 
                   S.as_gaussian().add_plate_axis(-1), 
                   name='BS')

    # Latent states with dynamics
    # X : () x (N,D)
    X = GaussianMarkovChain(np.zeros(D),         # mean of x0
                            1e-3*np.identity(D), # prec of x0
                            BS,                   # dynamics
                            np.ones(D),          # innovation
                            n=N,                 # time instances
                            name='X',
                            initialize=False)
    X.initialize_from_value(np.random.randn(N,D))

    # Mixing matrix from latent space to observation space using ARD
    # gamma : (D) x ()
    gamma = Gamma(1e-5,
                  1e-5,
                  plates=(D,),
                  name='gamma')
    # C : (M,1) x (D)
    C = Gaussian(np.zeros(D),
                 diagonal(gamma),
                 plates=(M,1),
                 name='C')
    C.initialize_from_value(np.random.randn(M,1,D))

    # Observation noise
    # tau : () x ()
    tau = Gamma(1e-5,
                1e-5,
                name='tau')

    # Observations
    # Y : (M,N) x ()
    CX = Dot(C, X.as_gaussian())
    Y = Normal(CX,
               tau,
               name='Y')

    #
    # RUN INFERENCE
    #

    # Observe data
    Y.observe(y, mask=mask)
    # Construct inference machine
    Q = VB(Y, X, S, A, alpha, B, beta, C, gamma, tau)

    #
    # Run inference with rotations.
    #

    # Rotate the D-dimensional state space (C, X)
    rotB = transformations.RotateGaussianMatrixARD(B, beta, axis='rows')
    rotX = transformations.RotateDriftingMarkovChain(X, B, S, rotB)
    rotC = transformations.RotateGaussianARD(C, gamma)
    R_X = transformations.RotationOptimizer(rotX, rotC, D)

    # Rotate the K-dimensional latent dynamics space (B, S)
    rotA = transformations.RotateGaussianARD(A, alpha)
    rotS = transformations.RotateGaussianMarkovChain(S, A, rotA)
    rotB = transformations.RotateGaussianMatrixARD(B, beta, axis='cols')
    R_S = transformations.RotationOptimizer(rotS, rotB, K)

    # Iterate
    for ind in range(int(maxiter/5)):
        Q.update(repeat=5)
        #Q.update(X, S, A, alpha, rho, B, beta, C, gamma, tau, repeat=maxiter)
        R_X.rotate()
        R_S.rotate()
        ## R_X.rotate(
        ## check_bound=Q.compute_lowerbound,
        ## check_bound_terms=Q.compute_lowerbound_terms,
        ## check_gradient=True
        ##     )
        ## R_S.rotate(
        ## check_bound=Q.compute_lowerbound,
        ## check_bound_terms=Q.compute_lowerbound_terms,
        ## check_gradient=True
        ##     )

    #
    # SHOW RESULTS
    #

    # Mean and standard deviation of the posterior
    (f_mean, f_squared) = CX.get_moments()
    f_std = np.sqrt(f_squared - f_mean**2)

    # Plot observations space
    for m in range(M):
        plt.subplot(M,1,m+1)
        plt.plot(y[m,:], 'r.')
        plt.plot(f[m,:], 'b-')
        bpplt.errorplot(y=f_mean[m,:], error=2*f_std[m,:])
    

def run_lssm(y, f, mask, D, maxiter):
    """
    Run VB inference for linear state space model.
    """

    (M, N) = np.shape(y)

    #
    # CONSTRUCT THE MODEL
    #

    # Dynamic matrix
    # alpha: (D) x ()
    alpha = Gamma(1e-5,
                  1e-5,
                  plates=(D,),
                  name='alpha')
    # A : (D) x (D)
    A = Gaussian(np.zeros(D),
                 diagonal(alpha),
                 plates=(D,),
                 name='A')
    A.initialize_from_value(np.identity(D))

    # Latent states with dynamics
    # X : () x (N,D)
    X = GaussianMarkovChain(np.zeros(D),         # mean of x0
                            1e-3*np.identity(D), # prec of x0
                            A,                   # dynamics
                            np.ones(D),          # innovation
                            n=N,                 # time instances
                            name='X',
                            initialize=False)
    X.initialize_from_value(np.random.randn(N,D))

    # Mixing matrix from latent space to observation space using ARD
    # gamma : (D) x ()
    gamma = Gamma(1e-5,
                  1e-5,
                  plates=(D,),
                  name='gamma')
    # C : (M,1) x (D)
    C = Gaussian(np.zeros(D),
                 diagonal(gamma),
                 plates=(M,1),
                 name='C')
    C.initialize_from_value(np.random.randn(M,1,D))

    # Observation noise
    # tau : () x ()
    tau = Gamma(1e-5,
                1e-5,
                name='tau')

    # Observations
    # Y : (M,N) x ()
    CX = Dot(C, X.as_gaussian())
    Y = Normal(CX,
               tau,
               name='Y')

    #
    # RUN INFERENCE
    #

    # Observe data
    Y.observe(y, mask=mask)
    # Construct inference machine
    Q = VB(Y, X, A, alpha, C, gamma, tau)
    # Iterate
    Q.update(X, A, alpha, C, gamma, tau, repeat=maxiter)

    #
    # SHOW RESULTS
    #

    # Mean and standard deviation of the posterior
    (f_mean, f_squared) = CX.get_moments()
    f_std = np.sqrt(f_squared - f_mean**2)

    # Plot observations space
    #plt.figure()
    for m in range(M):
        plt.subplot(M,1,m+1)
        plt.plot(y[m,:], 'r.')
        plt.plot(f[m,:], 'b-')
        bpplt.errorplot(y=f_mean[m,:], error=2*f_std[m,:])
    

def run(method):

    # Seed for random number generator
    seed = 495
    np.random.seed(seed)
    print("seed = ", seed)

    # Create data
    M = 10
    N = 200
    (y, f) = simulate_drifting_lssm(M, N)

    # Add missing values randomly
    mask = random.mask(M, N, p=0.3)
    # Add missing values to a period of time
    mask[:,70:120] = False
    #mask[:] = True
    y[~mask] = np.nan # BayesPy doesn't require NaNs, they're just for plotting.

    # Run the method
    if method == 'lssm':
        run_lssm(y, f, mask, 3, 200)
    elif method == 'dlssm':
        run_dlssm(y, f, mask, 3, 2, 200)
    else:
        raise Exception("Unknown method requested")

if __name__ == '__main__':
    run('dlssm')
    plt.show()
