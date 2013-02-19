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
from bayespy.inference.vmp.nodes.gamma import Gamma
from bayespy.inference.vmp.nodes.normal import Normal
from bayespy.inference.vmp.nodes.dot import Dot, MatrixDot

from bayespy.utils import utils
from bayespy.utils import random

from bayespy.inference.vmp.vmp import VB

import bayespy.plot.plotting as bpplt

        
class linear_state_space_model():

    def __init__(self, K=2, D=3, N=100, M=10, drift=True):

        # B : (D) x (D*K)
        beta = Gamma(1e-5,
                     1e-5,
                     name='beta')
        B = Gaussian(np.zeros(D*K),
                     1e-6*np.identity(D*K),
                     plates=(D,),
                     name='W')
        #W.initialize_from_value(np.random.randn(D,D*K))
        #W.initialize_from_value(np.repeat(np.identity(D), K, axis=-1))
        b = np.zeros((D,D,K))
        b[np.arange(D),np.arange(D),np.zeros(D,dtype=int)] = 1
        B.initialize_from_value(np.reshape(1*b, (D,D*K)))
        #B.initialize_from_value(np.random.randn(D,D*K))

        # Dynamics matrix with ARD
        # alpha : (D) x ()
        alpha = Gamma(1e-5,
                      1e-5,
                      plates=(K,),
                      name='alpha')
        # A : (K) x (K)
        A = Gaussian(np.identity(K),
        #np.zeros(K),
                       alpha.as_diagonal_wishart(),
                       plates=(K,),
                       name='A_S')
        A.initialize_from_value(np.identity(K))

        v_s = Gamma(1e-5,
                    1e-5,
                    plates=(K,),
                    name="v_s")
        if drift:
            model = 0
        else:
            model = 3
        if model == 0:
            # S : () x (N-1,K)
            S = GaussianMarkovChain(np.ones(K),
                                    1e-6*np.identity(K),
                                    A,
                                    v_s,
            #np.ones(K),
                                    n=N-1,
                                    name='S')
            S.initialize_from_value(1*np.ones((N-1,K)))
            # A : (N-1,D) x (D)
            BS = MatrixDot(B, 
                           S.as_gaussian().add_plate_axis(-1), 
                           name='A')
        elif model == 1:
            # S : () x (1,K)
            S = GaussianMarkovChain(np.zeros(K),
                                    1e-6*np.identity(K),
                                    A,
                                    np.ones(K),
                                    n=1,
                                    name='S')
            S.initialize_from_value(np.ones((1,K)))
            # A : (1,D) x (D)
            BS = MatrixDot(B,
                           S.as_gaussian().add_plate_axis(-1), 
                           name='A')

        elif model == 2:
            # S : () x (K)
            S = Gaussian(np.zeros(K),
                         np.identity(K),
                         name='S')
            s = np.random.randn(K)
            #print(s)
            S.initialize_from_value(np.ones(K))
            #S.initialize_from_value(s)
            # A : (D) x (D)
            BS = MatrixDot(B, 
                           S,
                           name='A')

        elif model == 3:
            S = Gaussian(np.zeros(K),
                         np.identity(K),
                         plates=(1,),
                         name='S')
            # A : (D) x (D)
            BS = Gaussian(np.zeros(D),
                          np.identity(D),
                          plates=(D,),
                          name='A')
            BS.initialize_from_value(np.identity(D))
            B = BS

        # Possible reasons for bad initializations:
        # - Dynamic matrix is negative.

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
                     gamma.as_diagonal_wishart(),
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

        self.Y = Y
        self.tau = tau
        self.CX = CX
        self.C = C
        self.gamma = gamma
        self.X = X
        self.BS = BS
        self.B = B
        self.beta = beta
        self.S = S
        self.A = A
        self.alpha = alpha
        self.v_s = v_s
        #return (Y, CX, X, tau, C, gamma, A, A_S, S, v_s, W, alpha)

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

def run():

    seed = 495
    maxiter = 10
        
    print("seed = ", seed)
    np.random.seed(seed)

    M = 10
    N = 200
    D = 3
    K = 2

    (y, f) = simulate_drifting_lssm(M, N)

    # Create the model
    
    model = linear_state_space_model(K=K, D=D, N=N, M=M, drift=True)
    Y = model.Y
    tau = model.tau
    CX = model.CX
    C = model.C
    gamma = model.gamma
    X = model.X
    BS = model.BS
    B = model.B
    beta = model.beta
    S = model.S
    A = model.A
    v_s = model.v_s
    alpha = model.alpha

    # Hmm.. does the mask go properly from X to A?
    
    # Add missing values randomly
    mask = random.mask(M, N, p=0.3)
    # Add missing values to a period of time
    mask[:,70:120] = False
    y[~mask] = np.nan # BayesPy doesn't require NaNs, they're just for plotting.
    # Observe the data
    Y.observe(y, mask=mask)
    ## plt.figure()
    ## for m in range(M):
    ##     plt.subplot(M,1,m+1)
    ##     plt.plot(y[m,:], 'r.')
    ##     plt.plot(f[m,:], 'b-')
    ## return
    
    # Run inference
    #print(A.get_moments()[0])
    Q = VB(Y, X, tau, C, gamma, B, beta, A, alpha, S, v_s)
    # First learn the states and static dynamics
    #Q.update(X, C, gamma, tau, W, repeat=2)
    # Then, learn drifting dynamics
    #Q.update(W, S, A_S, alpha, v_s, repeat=2)
    # Then, fine tune by learning everything jointly
    Q.update(X, C, gamma, tau, B, beta, S, A, alpha, v_s, repeat=maxiter)

    
    X_vb = X.u[0]
    varX_vb = utils.diagonal(X.u[1] - X_vb[...,np.newaxis,:] * X_vb[...,:,np.newaxis])

    u_CX = CX.get_moments()
    CX_vb = u_CX[0]
    varCX_vb = u_CX[1] - CX_vb**2

    # Show results

    # Plot observations space
    plt.figure()
    for m in range(M):
        plt.subplot(M,1,m+1)
        plt.plot(y[m,:], 'r.')
        plt.plot(f[m,:], 'b-')
        bpplt.errorplot(y=CX_vb[m,:],
                        error=2*np.sqrt(varCX_vb[m,:]))

    # Plot drifting dynamics
    u_S = S.get_moments()
    s_mu = u_S[0]
    s_ss = u_S[1]
    s_var = np.einsum('...ii->...i', s_ss) - s_mu**2
    plt.figure()
    for k in range(K):
        plt.subplot(K,1,k+1)
        bpplt.errorplot(y=s_mu[:,k],
                        error=2*np.sqrt(s_var[:,k]))

    # Plot dynamics of the drift
    (a_mu, _) = A.get_moments()
    plt.figure()
    bpplt.matrix(a_mu)
    plt.colorbar()
    plt.title("Dynamic matrix of S")
    #plt.imshow(as_mu, interpolation='nearest', cmap=plt.get_cmap('RdBu'))

    ## # Plot matrices from the drift dynamics to dynamic matrix
    ## (w_mu, _) = W.get_moments()
    ## plt.figure()
    ## for d in range(D):
    ##     plt.subplot(D,1,d+1)
    ##     bpplt.matrix(np.reshape(w_mu[d,:], (D,K)))
    ##     plt.colorbar()
    ##     plt.title("W[%d]" % d)

    ## # Plot average dynamics matrix
    ## (a_mu, _) = A.get_moments()
    ## a_mu = np.mean(a_mu, axis=0)
    ## plt.figure()
    ## bpplt.matrix(a_mu)
    ## plt.colorbar()
    ## plt.title("Averaged A")

    alpha.show()
    A.show()
    v_s.show()
    
    return
    

if __name__ == '__main__':
    run()
    plt.show()
