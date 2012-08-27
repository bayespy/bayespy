######################################################################
# Copyright (C) 2011,2012 Jaakko Luttinen
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


import numpy as np
#import scipy as sp
import matplotlib.pyplot as plt
import time

# Profiling stuff
#import profile
import cProfile
import pstats


import imp

import utils

import Nodes.ExponentialFamily as EF
import Nodes.CovarianceFunctions as CF
import Nodes.GaussianProcesses as GP
imp.reload(utils)
imp.reload(EF)
imp.reload(CF)
imp.reload(GP)

# * Model
# * * Nodes
# * * * GaussianScalar
# * * * GaussianVector
# * * * Gamma
# * Inference
# * * VB
# * * * Nodes
# * * * * GaussianScalar
# * * * * GaussianVector
# * * * * Gamma
# * * * * Constant
# * * * * Delta
# * * EP
# * * MCMC
# * * * Nodes
# * * * * Gibbs
# * * * * * GaussianScalar
# * * * * * GaussianVector
# * Plotting

# VB.update(W,X,tau, repeat=20)

# Protocols:
# Value
# VB.GaussianScalar
# VB.GaussianVector
# VB.Gamma
# EP.GaussianScalar
# ...

# input_protocols = [VB.GaussianScalar, VB.Gamma]
# output_protocol = VB.GassianVector
#
# or do i need to check input protocols.. child checks parents' output
# protocols and that's it.

def m_plot(x, Y, style):
    Y = np.atleast_2d(Y)
    M = Y.shape[-2]
    for i in range(M):
        plt.subplot(M,1,i+1)
        plt.plot(x, Y[i], style)

def m_errorplot(x, Y, L, U):
    Y = np.atleast_2d(Y)
    L = np.atleast_2d(L)
    U = np.atleast_2d(U)
    M = Y.shape[-2]
    ## print(np.shape(Y))
    ## print(np.shape(L))
    ## print(np.shape(U))
    ## print(np.shape(M))
    for i in range(M):
        plt.subplot(M,1,i+1)
        lower = Y[i] - L[i]
        upper = Y[i] + U[i]
        #print(upper-lower)
        #if np.any(lower>=upper):
            #print('WTF?!')
        plt.fill_between(x,
                         upper,
                         lower,
                         #where=(upper>=lower),
                         facecolor=(0.6,0.6,0.6,1),
                         edgecolor=(0,0,0,0),
                         #edgecolor=(0.6,0.6,0.6,1),
                         linewidth=0,
                         interpolate=True)
        plt.plot(x, Y[i], color=(0,0,0,1))
        plt.ylabel(str(i))



# MULTIVARIATE GP!!

def test_sparse_gp():
    
    ## Generate data

    # Noisy observations from a sinusoid
    N = 10000
    func = lambda x: np.sin(x*2*np.pi/20)
    x = np.random.uniform(low=0, high=N, size=(N,))
    f = func(x)
    y = f + np.random.normal(0, 0.2, np.shape(f))

    # Plot data
    plt.clf()
    plt.plot(x,y,'r+')

    ## Construct model

    # Covariance function stuff
    ls = EF.NodeConstantScalar(3, name='lengthscale')
    amp = EF.NodeConstantScalar(2.0, name='amplitude')
    noise = EF.NodeConstantScalar(0.6, name='noise')
    # Latent process covariance
    #K_f = CF.SquaredExponential(amp, ls)
    K_f = CF.PiecewisePolynomial2(amp, ls)
    # Noise process covariance
    K_noise = CF.Delta(noise)
    # Observation process covariance
    K_y = CF.Sum(K_f, K_noise)
    # Joint covariance
    #K_joint = CF.Multiple([[K_f, K_f],[K_f,K_y]], sparse=True)

    # Mean function stuff
    M = GP.Constant(lambda x: (x/10-2)*(x/10+1))
    # Means for latent and observation processes
    #M_multi = GP.Multiple([M, M])

    # Gaussian process
    F = GP.GaussianProcess(M, [[K_f, K_f], [K_f, K_y]])
    #F = GP.GaussianProcess(M, [[K_f, K_f], [K_f, K_y]])
    #F = GP.GaussianProcess(M_multi, K_joint)

    ## Inference
    F.observe([[],x], y)
    utils.vb_optimize_nodes(ls, amp, noise)
    F.update()
    u = F.get_parameters()

    ## Show results

    # Print hyperparameters
    print('parameters')
    print(ls.name, ls.u[0])
    print(amp.name, amp.u[0])
    print(noise.name, noise.u[0])

    # Posterior predictions
    xh = np.arange(np.min(x)-5, np.max(x)+10, 0.1)
    (fh, varfh) = u([[],xh], covariance=1)
    #(fh, varfh) = u([xh,[]], covariance=1)

    # Plot predictive distribution
    varfh[varfh<0] = 0
    errfh = np.sqrt(varfh)
    m_errorplot(xh, fh, errfh, errfh)
    
    return


def test_gp():

    # Generate data
    func = lambda x: np.sin(x*2*np.pi/5)
    x = np.random.uniform(low=0, high=10, size=(100,))
    f = func(x)
    f = f + np.random.normal(0, 0.2, np.shape(f))
    plt.clf()
    plt.plot(x,f,'r+')
    #plt.plot(x,y,'r+')

    # Construct model
    ls = EF.NodeConstantScalar(1.5, name='lengthscale')
    amp = EF.NodeConstantScalar(2.0, name='amplitude')
    noise = EF.NodeConstantScalar(0.6, name='noise')
    K = CF.SquaredExponential(amp, ls)
    K_noise = CF.Delta(noise)
    K_sum = CF.Sum(K, K_noise)

    M = GP.Constant(lambda x: (x/10-2)*(x/10+1))

    method = 'multi'

    if method == 'sum':
        # Sum GP
        F = GP.GaussianProcess(M, K_sum)

    elif method == 'multi':
        # Joint for latent function and observation process
        M_multi = GP.Multiple([M, M])

        K_zeros = CF.Zeros()

        
        #K_multi = CF.Multiple([[K, K],[K,K_sum]])
        K_multi1 = CF.Multiple([[K, K],[K,K]])
        #K_multi2 = CF.Multiple([[None, None],[None, K_noise]])
        K_multi2 = CF.Multiple([[K_zeros, K_zeros],[K_zeros,K_noise]])
        
        xp = np.arange(0,5,1)
        F = GP.GaussianProcess(M_multi, K_multi1,
                               k_sparse=K_multi2,
                               #pseudoinputs=[[],xp])
                               pseudoinputs=None)
        #F = GP.GaussianProcess(M_multi, K_multi, pseudoinputs=[[],xp])
        # Observations are from the latter process:
        #xf = np.array([])
        #x_pseudo = [[], x]
        #x_full = [np.array([15, 20]), []]
        #xy = x
        #x = [xf, xy]
        #f = np.concatenate([func(xf), f])
        #f_pseudo = f
        #f_full = func(x_full)
        x = [[], x]
        

    # Inference
    #F.observe(x_pseudo, f_pseudo, pseudo=True)
    F.observe(x, f)
    utils.vb_optimize_nodes(ls, amp, noise)
    F.update()
    u = F.get_parameters()

    print('parameters')
    print(ls.name)
    print(ls.u[0])
    print(amp.name)
    print(amp.u[0])
    print(noise.name)
    print(noise.u[0])

    #print(F.lower_bound_contribution())

    # Posterior predictions
    xh = np.arange(-5, 20, 0.1)
    if method == 'multi':
        # Choose which process you want to examine:
        (fh, varfh) = u([[],xh], covariance=1)
        #(fh, varfh) = u([xh,[]], covariance=1)
    else:
        (fh, varfh) = u(xh, covariance=1)

    #print(fh)
    #print(np.shape(fh))
    #print(np.shape(varfh))

    varfh[varfh<0] = 0
    errfh = np.sqrt(varfh)
    #print(varfh)
    #print(errfh)
    m_errorplot(xh, fh, errfh, errfh)
    
    return
    
    # Construct a GP
    k = gp_cov_se(magnitude=theta1, lengthscale=theta2)
    f = NodeGP(0, k)
    f.observe(x, y)
    f.update()
    (mp, kp) = f.get_parameters()


def test_pca():

    # Dimensionalities
    dataset = 1
    if dataset == 1:
        M = 10
        N = 100
        D_y = 3
        D = 3+2
        # Generate data
        w = np.random.normal(0, 1, size=(M,1,D_y))
        x = np.random.normal(0, 1, size=(1,N,D_y))
        f = sum_product(w, x, axes_to_sum=[-1])#np.einsum('...i,...i', w, x)
        y = f + np.random.normal(0, 0.5, size=(M,N))
    elif dataset == 2:
        # Data from matlab comparison
        f = np.genfromtxt('/home/jluttine/matlab/fa/data_pca_01_f.txt')
        y = np.genfromtxt('/home/jluttine/matlab/fa/data_pca_01_y.txt')
        D = np.genfromtxt('/home/jluttine/matlab/fa/data_pca_01_d.txt')
        (M,N) = np.shape(y)

    # Construct the PCA model with ARD

    alpha = NodeGamma(1e-10, 1e-10, plates=(D,), name='alpha')
    alpha.update()
    diag_alpha = NodeWishartFromGamma(alpha)
    #Lambda = NodeWishart(D, D * np.identity(D), plates=(), name='Lambda')
    #Lambda.update()
    
    X = NodeGaussian(np.zeros(D), np.identity(D), name="X", plates=(1,N))

    X.update()
    X.u[0] = X.random()

    W = NodeGaussian(np.zeros(D), diag_alpha, name="W", plates=(M,1))
    #W = NodeGaussian(np.zeros(D), Lambda, name="W", plates=(M,1))
    #W = NodeGaussian(np.zeros(D), np.identity(D), name="W", plates=(M,1))
    W.update()
    W.u[0] = W.random()

    WX = NodeDot(W,X,S,R)

    tau = NodeGamma(1e-5, 1e-5, name="tau", plates=(M,N))
    tau.update()

    Y = NodeNormal(WX, tau, name="Y", plates=(M,N))
    Y.update()

    # Initialize (from prior)

    # Y.update()
    # mask = True
    # mask = np.ones((M,N), dtype=np.bool)
    mask = np.random.rand(M,N) < 0.4
    mask[:,20:40] = False
    Y.observe(y, mask)

    # Inference
    L_last = -np.inf
    for i in range(100):
        t = time.clock()
        X.update()
        W.update()
        tau.update()
        #Lambda.update()
        alpha.update()

        L_X = X.lower_bound_contribution()
        L_W = W.lower_bound_contribution()
        L_tau = tau.lower_bound_contribution()
        L_Y = Y.lower_bound_contribution()
        L_alpha = alpha.lower_bound_contribution()
        #print("X: %f, W: %f, tau: %f, Y: %f" % (L_X, L_W, L_tau, L_Y))
        L = L_X + L_W + L_tau + L_Y
        print("Iteration %d: loglike=%e (%.3f seconds)" % (i+1, L, time.clock()-t))
        if L_last > L:
            L_diff = (L_last - L)
            #raise Exception("Lower bound decreased %e! Bug somewhere or numerical inaccuracy?" % L_diff)
        if L - L_last < 1e-12:
            print("Converged.")
            #break
        L_last = L

    #return


    #print(shape(yh))
    plt.figure(1)
    plt.clf()
    WX_params = WX.get_parameters()
    fh = WX_params[0] * np.ones(y.shape)
    err_fh = 2*np.sqrt(WX_params[1]) * np.ones(y.shape)
    m_errorplot(np.arange(N), fh, err_fh, err_fh)
    m_plot(np.arange(N), f, 'g')
    m_plot(np.arange(N), y, 'r+')

    #plt.figure(2)
    #plt.clf()

    #alpha.show()
    print(alpha.u[0])
    
        
    tau.show()



def test_normal():

    M = 10
    N = 5

    # mu
    mu = NodeNormal(0.0, 10**-5, name="mu", plates=())
    print("Prior for mu:")
    mu.update()
    mu.show()

    # tau
    tau = NodeGamma(10**-5, 10**-5, plates=(N,), name="tau")
    print("Prior for tau:")
    tau.update()
    tau.show()

    # x
    x = NodeNormal(mu, tau, plates=(M,N), name="x")
    print("Prior for x:")
    x.update()
    x.show()

    # y (generate data)
    y = NodeNormal(x, 1, plates=(M,N), name="y")
    y.observe(random.normal(loc=10, scale=10, size=(M,N)))

    # Inference
    for i in range(50):
        x.update()
        mu.update()
        tau.update()

    print("Posterior for mu:")
    mu.show()
    print("Posterior for tau:")
    tau.show()
    print("Posterior for x:")
    x.show()
    
    return
    
def test_multivariate():    

    D = 3
    N = 100
    M = 200

    # mu
    mu = NodeGaussian(np.zeros(D), 10**(-10)*np.identity(D), plates=(M,1), name='mu')
    print("Prior for mu:")
    mu.update()
    mu.show()

    # Lambda
    Lambda = NodeWishart(D, (10**-10) * np.identity(D), plates=(1,N), name='Lambda')
    print("Prior for Lambda:")
    Lambda.update()
    Lambda.show()

    #Y = NodeGaussian(mu, 10**(-2)*identity(D), plates=(M,N), name='Y')
    Y = NodeGaussian(mu, Lambda, plates=(M,N), name='Y')
    Y.observe(random.normal(loc=10, scale=10, size=(M,N,D)))

    ## # y (generate data)
    ## for i in range(100):
    ##     y = NodeGaussian(mu, Lambda)
    ##     v = random.normal(0,1, D)
    ##     y.fix(v)

    # Inference
    try:
        for i in range(50):
            mu.update()
            Lambda.update()

        print("Posterior for mu:")
        mu.show()
        print("Posterior for Lambda:")
        Lambda.show()
    except Exception:
        pass
    

if __name__ == '__main__':

    # FOR INTERACTIVE SESSIONS, NON-BLOCKING PLOTTING:
    plt.ion()

    test_sparse_gp()
    #test_gp()
    #test_pca()
    #cProfile.run('test_sparse_gp()', 'profile.tmp')
    #S = pstats.Stats('profile.tmp')
    #test_normal()
    #test_multivariate()

