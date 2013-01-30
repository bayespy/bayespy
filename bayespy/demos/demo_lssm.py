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
# under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
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

from bayespy.inference.vmp.nodes.gaussian_markov_chain import GaussianMarkovChain
from bayespy.inference.vmp.nodes.gaussian import Gaussian
from bayespy.inference.vmp.nodes.gamma import Gamma
from bayespy.inference.vmp.nodes.normal import Normal
from bayespy.inference.vmp.nodes.dot import Dot

from bayespy.utils import utils
from bayespy.utils import random

from bayespy.inference.vmp.vmp import VB

import bayespy.plot.plotting as bpplt

def linear_state_space_model(D=3, N=100, M=10):

    # Dynamics matrix with ARD
    alpha = Gamma(1e-5,
                  1e-5,
                  plates=(D,),
                  name='alpha')
    A = Gaussian(np.zeros(D),
                 alpha.as_diagonal_wishart(),
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
    C = Gaussian(np.zeros(D),
                 gamma.as_diagonal_wishart(),
                 plates=(M,1),
                 name='C')

    # Observation noise
    tau = Gamma(1e-5,
                1e-5,
                name='tau')

    # Observations
    CX = Dot(C, X.as_gaussian())
    Y = Normal(CX,
               tau,
               name='Y')

    return (Y, CX, X, tau, C, gamma, A, alpha)

def run():

    seed = np.random.randint(1000)
    print("seed = ", seed)
    np.random.seed(seed)

    # Simulate some data
    D = 3
    M = 6
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
    (Y, CX, X, tau, C, gamma, A, alpha) = linear_state_space_model(D, N, M)

    # Hmm.. does the mask go properly from X to A?
    
    # Add missing values
    mask = random.mask(M, N, p=0.3)
    y[~mask] = np.nan # BayesPy doesn't require this. Just for plotting.
    # Observe the data
    Y.observe(y, mask=mask)
    

    # Initialize nodes (must use some randomness for C)
    C.initialize_from_parameters(C.random(), np.identity(D))

    # Run inference
    Q = VB(Y, X, tau, C, gamma, A, alpha)
    #Q.update(X, C, tau, A, repeat=5)
    Q.update(X, C, tau, gamma, A, alpha, repeat=20)

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
    plt.figure(3)
    plt.clf()
    for m in range(M):
        plt.subplot(M,1,m+1)
        plt.plot(y[m,:], 'r.')
        plt.plot(f[m,:], 'b-')
        bpplt.errorplot(y=CX_vb[m,:],
                        error=2*np.sqrt(varCX_vb[m,:]))

    tau.show()
    gamma.show()
    alpha.show()

    
    return
    

if __name__ == '__main__':
    run()
    plt.show()
