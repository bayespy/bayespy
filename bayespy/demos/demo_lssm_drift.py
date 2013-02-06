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

# TODO/FIXME: THERE IS A BUG SOMEWHERE?!?!?!

import numpy as np
import scipy
import matplotlib.pyplot as plt

from bayespy.inference.vmp.nodes.gaussian_markov_chain import GaussianMarkovChain
from bayespy.inference.vmp.nodes.gaussian import Gaussian
from bayespy.inference.vmp.nodes.gamma import Gamma
from bayespy.inference.vmp.nodes.normal import Normal
from bayespy.inference.vmp.nodes.dot import Dot, MatrixDot

from bayespy.utils import utils
from bayespy.utils import random

from bayespy.inference.vmp.vmp import VB

import bayespy.plot.plotting as bpplt

        
def linear_state_space_model(K=2, D=3, N=100, M=10):

    # W : (D) x (D*K)
    W = Gaussian(np.zeros(D*K),
                 1e-6*np.identity(D*K),
                 plates=(D,),
                 name='W')

    # Dynamics matrix with ARD
    # alpha : (D) x ()
    alpha = Gamma(1e-5,
                  1e-5,
                  plates=(K,),
                  name='alpha')
    # S : (N-1) x (K)
    A_S = Gaussian(np.zeros(K),
                   alpha.as_diagonal_wishart(),
                   plates=(K,),
                   name='A_S')
    model = 0
    if model == 0:
        S = GaussianMarkovChain(np.zeros(K),
                                1e-6*np.identity(K),
                                A_S,
                                np.ones(K),
                                n=N-1,
                                name='S')
        # A : (N-1,D) x (D)
        A = MatrixDot(W, 
                      S.as_gaussian().add_plate_axis(-1), 
                      name='A')
    elif model == 1:
        S = GaussianMarkovChain(np.zeros(K),
                                1e-6*np.identity(K),
                                A_S,
                                np.ones(K),
                                n=1,
                                name='S')
        # A : (N-1,D) x (D)
        A = MatrixDot(W, 
                      S.as_gaussian().add_plate_axis(-1), 
                      name='A')

    # Latent states with dynamics
    # X : () x (N,D)
    X = GaussianMarkovChain(np.zeros(D),         # mean of x0
                            1e-3*np.identity(D), # prec of x0
                            A,                   # dynamics
                            np.ones(D),          # innovation
                            n=N,                 # time instances
                            name='X')

    # Mixing matrix from latent space to observation space using ARD
    # gamma : (D) x ()
    gamma = Gamma(1e-5,
                  1e-5,
                  plates=(D,),
                  name='gamma')
    # C : (M,1) x (D)
    C = Gaussian(np.zeros(D),
                 gamma.as_diagonal_wishart(),
                 plates=(M,1),
                 name='C')

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

    return (Y, CX, X, tau, C, gamma, A, A_S, S, W, alpha)

def run():

    seed = 496#np.random.randint(1000)
    maxiter = 10
        
    print("seed = ", seed)
    np.random.seed(seed)

    # Simulate some data
    K = 2
    D = 3
    M = 10
    N = 200
    c = np.random.randn(M,D)
    w = 0.3
    a = np.array([[np.cos(w), -np.sin(w), 0], 
                  [np.sin(w), np.cos(w),  0], 
                  [0,         0,          1]])
    #a = 0.9*scipy.linalg.orth(np.random.randn(D,D))
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

    # Create the model
    (Y, 
     CX,
     X, 
     tau,
     C, 
     gamma, 
     A, 
     A_S,
     S, 
     W, 
     alpha) = linear_state_space_model(K=2,
                                       D=D, 
                                       N=N,
                                       M=M)

    # Hmm.. does the mask go properly from X to A?
    
    # Add missing values randomly
    mask = random.mask(M, N, p=1.0)
    # Add missing values to a period of time
    mask[:,30:80] = True
    y[~mask] = np.nan # BayesPy doesn't require NaNs, they're just for plotting.
    # Observe the data
    Y.observe(y, mask=mask)
    

    # Initialize nodes (must use some randomness for C and W)
    C.initialize_from_value(np.random.randn(*(C.get_shape(0))))
    ## C.initialize_from_parameters(C.random(),
    ##                              np.identity(D))
    W.initialize_from_value(np.random.randn(*(W.get_shape(0))))
    #W.initialize_from_parameters(W.random(), np.identity(K*D))
    ## A_S.initialize_from_parameters(np.identity(K),
    ##                                1e2*np.identity(K))
    #S.initialize_from_value(np.random.randn(K))
    #S.initialize_from_value(np.random.randn(1,K))
    S.initialize_from_value(np.tile(np.random.randn(K), (N-1,1)))
    #S.initialize_from_value(np.tile(np.random.randn(K), (N-1,1,1)))
    ## S.initialize_from_parameters(np.random.randn(*(S.get_shape(0))),
    ##                              1e-3*np.identity(K),
    ##                              np.identity(K),
    ##                              1e-5*np.ones(K))


    # Run inference
    #print(A.get_moments()[0])
    Q = VB(Y, X, tau, C, gamma, W, A_S, S, alpha)
    #Q.update(X, C, tau, A, repeat=5)
    #print(A.get_moments()[0])
    Q.update(X, C, tau, gamma, W, repeat=3)
    Q.update(X, C, tau, gamma, W, S, A_S, repeat=50)
    #Q.update(X, C, tau, gamma, W, repeat=5)
    for i in range(0):
        Ap = A.get_moments()[0]
        Q.update(X, S, repeat=1)
        An = A.get_moments()[0]

    X_vb = X.u[0]
    varX_vb = utils.diagonal(X.u[1] - X_vb[...,np.newaxis,:] * X_vb[...,:,np.newaxis])

    u_CX = CX.get_moments()
    CX_vb = u_CX[0]
    varCX_vb = u_CX[1] - CX_vb**2

    # Show results
    ## plt.figure(1)
    ## plt.clf()
    ## for d in range(D):
    ##     plt.subplot(D,1,d+1)
    ##     plt.plot(x[:,d], 'r-')
    ##     #plt.figure(2)
    ##     #plt.clf()
    ##     #for d in range(D):
    ##     #plt.subplot(D,1,d)
    ##     bpplt.errorplot(y=X_vb[:,d],
    ##                     error=2*np.sqrt(varX_vb[:,d]))

    ## plt.figure(2)
    ## plt.clf()
    ## lp = Q.get_iteration_by_nodes()
    ## plt.plot(np.diff(lp[Y]), 'r')
    ## plt.plot(np.diff(lp[X]), 'b')
    ## plt.plot(np.diff(lp[C]), 'm')
    ## plt.plot(np.diff(lp[tau]), 'm')
    ## plt.plot(np.diff(lp[gamma]), 'm')
    ## plt.plot(np.diff(lp[S]), 'r')
    ## plt.plot(np.diff(lp[A_S]), 'k')
    ## plt.plot(np.diff(lp[W]), 'g')

    ## print('x', lp[X][-1])
    ## print('s', lp[S][-1])
    ## print('w', lp[W][-1])
    ## print('a_s', lp[A_S][-1])
    ## A_S.show()
    
    plt.figure(1)
    plt.clf()
    for m in range(M):
        plt.subplot(M,1,m+1)
        plt.plot(y[m,:], 'r.')
        plt.plot(f[m,:], 'b-')
        bpplt.errorplot(y=CX_vb[m,:],
                        error=2*np.sqrt(varCX_vb[m,:]))

    ## tau.show()
    ## gamma.show()
    ## alpha.show()

    ## print(A.get_moments()[0])

    
    return
    

if __name__ == '__main__':
    #run_another()
    run()
    plt.show()
