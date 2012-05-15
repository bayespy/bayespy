
import numpy as np
import matplotlib.pyplot as plt
import time

import utils
import plotting as myplt
import Nodes.ExponentialFamily as EF

import imp
imp.reload(utils)
imp.reload(myplt)
imp.reload(EF)

def gaussianmix_model(N, K, D):
    # N = number of data vectors
    # K = number of clusters
    # D = dimensionality
    
    # Construct the Gaussian mixture model

    # K prior weights (for components)
    rho = EF.Dirichlet(np.ones(K),
                       name='rho')
    # N K-dimensional weights (for data)
    alpha = EF.Categorical(rho,
                           plates=(N,),
                           name='alpha')
    # K D-dimensional component means
    X = EF.Gaussian(np.zeros(D), np.identity(D),
                    plates=(K,),
                    name='X')
    # K D-dimensional component covariances
    Sigma = EF.Wishart(D, np.identity(D),
                       plates=(K,),
                       name='Sigma')
    # N D-dimensional observation vectors
    Y = EF.Mixture(EF.Gaussian)(alpha, X, Sigma, plates=(N,), name='Y')
    # TODO: Plates should be learned automatically if not given (it
    # would be the smallest shape broadcasted from the shapes of the
    # parents)

    return (Y, X, Sigma, alpha, rho)


def run(N=50, K=5, D=2):
    # Generate data
    y = np.random.normal(0, 0.5, size=(N,D))

    # Construct model
    (Y, X, Sigma, alpha, rho) = gaussianmix_model(N,K,D)

    # Initialize nodes (from prior and randomly)
    rho.initialize()
    alpha.initialize()
    Sigma.initialize()
    X.initialize()
    Y.initialize()

    # Data with missing values
    ## mask = np.random.rand(M,N) < 0.4 # randomly missing
    ## mask[:,20:40] = False # gap missing
    # Y.observe(y, mask)
    Y.observe(y)

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

