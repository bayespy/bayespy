
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
    alpha = EF.Dirichlet(0.1*np.ones(K),
                         name='alpha')
    # N K-dimensional cluster assignments (for data)
    z = EF.Categorical(alpha,
                       plates=(N,),
                       name='z')
    # K D-dimensional component means
    X = EF.Gaussian(np.zeros(D), 10*np.identity(D),
                    plates=(K,),
                    name='X')
    # K D-dimensional component covariances
    Lambda = EF.Wishart(D, 10*np.identity(D),
                        plates=(K,),
                        name='Lambda')
    # N D-dimensional observation vectors
    Y = EF.Mixture(EF.Gaussian)(z, X, Lambda, plates=(N,), name='Y')
    # TODO: Plates should be learned automatically if not given (it
    # would be the smallest shape broadcasted from the shapes of the
    # parents)

    return (Y, X, Lambda, z, alpha)


def run(N=50, K=5, D=2):

    #np.random.seed(2)
    
    # Generate data
    y = np.random.normal(0, 0.5, size=(N,D))

    # Construct model
    (Y, X, Lambda, z, alpha) = gaussianmix_model(N,K,D)

    # Initialize nodes (from prior and randomly)
    alpha.initialize()
    z.initialize()
    Lambda.initialize()
    X.initialize_random_mean()
    #Y.initialize()

    ## X.show()
    ## return

    # Data with missing values
    ## mask = np.random.rand(M,N) < 0.4 # randomly missing
    ## mask[:,20:40] = False # gap missing
    # Y.observe(y, mask)
    Y.observe(y)

    # Inference loop.
    maxiter = 200
    L_X = np.zeros(maxiter)
    L_Lambda = np.zeros(maxiter)
    L_alpha = np.zeros(maxiter)
    L_z = np.zeros(maxiter)
    L_Y = np.zeros(maxiter)
    L = np.zeros(maxiter)
    L_last = -np.inf
    for i in range(maxiter):
        t = time.clock()

        # Update nodes
        z.update()
        alpha.update()
        X.update()
        Lambda.update()

        # Compute lower bound
        L_X[i] = X.lower_bound_contribution()
        L_Lambda[i] = Lambda.lower_bound_contribution()
        L_alpha[i] = alpha.lower_bound_contribution()
        L_z[i] = z.lower_bound_contribution()
        L_Y[i] = Y.lower_bound_contribution()
        L[i] = L_X[i] + L_Lambda[i] + L_alpha[i] + L_z[i] + L_Y[i]

        # Check convergence
        print("Iteration %d: loglike=%e (%.3f seconds)" % (i+1, L[i], time.clock()-t))
        if L_last - L[i] > 1e-12:
            L_diff = (L_last - L[i])
            raise Exception("Lower bound decreased %e! Bug somewhere or numerical inaccuracy?" % L_diff)
        if L[i] - L_last < 1e-12:
            print("Converged.")
            #break
        L_last = L[i]

    X.show()
    #Lambda.show()
    #alpha.show()
    
    alpha.show()

    #print(y)
    #print(Y.u[0])

    plt.ion()
    plt.clf()
    ax = plt.plot(np.vstack([L_X, L_Lambda, L_z, L_alpha, L_Y]).T)
    plt.legend(ax)
    ## plt.figure()
    ## plt.clf()
    ## WX_params = WX.get_parameters()
    ## fh = WX_params[0] * np.ones(y.shape)
    ## err_fh = 2*np.sqrt(WX_params[1]) * np.ones(y.shape)
    ## for d in range(D):
    ##     myplt.errorplot(np.arange(N), fh[d], err_fh[d], err_fh[d])
    ##     plt.plot(np.arange(N), f[d], 'g')
    ##     plt.plot(np.arange(N), y[d], 'r+')


if __name__ == '__main__':
    # FOR INTERACTIVE SESSIONS, NON-BLOCKING PLOTTING:
    plt.ion()
    run()

