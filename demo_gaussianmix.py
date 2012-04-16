
import numpy as np
#import scipy as sp
import matplotlib.pyplot as plt
import plotting as myplt
import time
#import profile


import imp

import utils

import Nodes.ExponentialFamily as EF
#import Nodes.CovarianceFunctions as CF
#import Nodes.GaussianProcesses as GP
imp.reload(utils)
imp.reload(EF)
imp.reload(myplt)
#imp.reload(CF)
#imp.reload(GP)

def gaussianmix_model(N, M, D):
    # N = number of data vectors
    # M = number of mixtures
    # D = dimensionality
    
    # Construct the Gaussian mixture model

    # M prior weights (for components)
    rho = EF.Dirichlet(np.ones(M),
                       name='rho')
    # N M-dimensional weights (for data)
    alpha = EF.Multinomial(rho,
                           plates=(N,),
                           name='alpha')
    # M D-dimensional component means
    X = EF.Gaussian(np.zeros(D), np.identity(D),
                    plates=(M,),
                    name='X')
    # M D-dimensional component covariances
    Sigma = EF.Wishart(1, np.identity(D),
                       plates(M,),
                       name='Sigma')
    # N D-dimensional observation vectors
    Y = EF.Mixture(X, alpha,
                   name='Y')

    return (Y, WX, W, X, tau, alpha)


def run(M=10, N=100, D_y=3, D=5):
    # Generate data
    w = np.random.normal(0, 1, size=(M,1,D_y))
    x = np.random.normal(0, 1, size=(1,N,D_y))
    f = utils.sum_product(w, x, axes_to_sum=[-1])
    y = f + np.random.normal(0, 0.5, size=(M,N))

    # Construct model
    (Y, WX, W, X, tau, alpha) = pca_model(M, N, D)

    # Initialize nodes (from prior and randomly)
    alpha.update()
    W.update()
    X.update()
    tau.update()
    W.u[0] = W.random()
    X.u[0] = X.random()
    Y.update()

    # Data with missing values
    mask = np.random.rand(M,N) < 0.4 # randomly missing
    mask[:,20:40] = False # gap missing
    Y.observe(y, mask)

    # Inference loop.
    L_last = -np.inf
    for i in range(100):
        t = time.clock()

        # Update nodes
        X.update()
        W.update()
        tau.update()
        alpha.update()

        # Compute lower bound
        L_X = X.lower_bound_contribution()
        L_W = W.lower_bound_contribution()
        L_tau = tau.lower_bound_contribution()
        L_Y = Y.lower_bound_contribution()
        L_alpha = alpha.lower_bound_contribution()
        L = L_X + L_W + L_tau + L_Y

        # Check convergence
        print("Iteration %d: loglike=%e (%.3f seconds)" % (i+1, L, time.clock()-t))
        if L_last > L:
            L_diff = (L_last - L)
            #raise Exception("Lower bound decreased %e! Bug somewhere or numerical inaccuracy?" % L_diff)
        if L - L_last < 1e-12:
            print("Converged.")
            #break
        L_last = L


    plt.ion()
    plt.figure()
    plt.clf()
    WX_params = WX.get_parameters()
    fh = WX_params[0] * np.ones(y.shape)
    err_fh = 2*np.sqrt(WX_params[1]) * np.ones(y.shape)
    for d in range(D):
        myplt.errorplot(np.arange(N), fh[d], err_fh[d], err_fh[d])
        plt.plot(np.arange(N), f[d], 'g')
        plt.plot(np.arange(N), y[d], 'r+')


if __name__ == '__main__':
    # FOR INTERACTIVE SESSIONS, NON-BLOCKING PLOTTING:
    plt.ion()
    run()

