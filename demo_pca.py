
import time
import numpy as np
import matplotlib.pyplot as plt
import plotting as myplt

import utils
import nodes as EF

# Reload everything (helpful for interactive sessions)
import imp
import nodes
import nodes.node
import nodes.variables
import nodes.variables.variable
import nodes.variables.gamma
import nodes.variables.gaussian
import nodes.variables.normal
import nodes.variables.dot
imp.reload(utils)
imp.reload(nodes)
imp.reload(nodes.node)
imp.reload(nodes.variables)
imp.reload(nodes.variables.variable)
imp.reload(nodes.variables.gamma)
imp.reload(nodes.variables.gaussian)
imp.reload(nodes.variables.normal)
imp.reload(nodes.variables.dot)

def pca_model(M, N, D):
    # Construct the PCA model with ARD

    # ARD
    alpha = EF.Gamma(1e-10,
                     1e-10,
                     plates=(D,),
                     name='alpha')

    # Loadings
    W = EF.Gaussian(np.zeros(D),
                    EF.GammaToDiagonalWishart(alpha),
                    name="W",
                    plates=(M,1))

    # States
    X = EF.Gaussian(np.zeros(D),
                    np.identity(D),
                    name="X",
                    plates=(1,N))

    # PCA
    WX = EF.Dot(W,X)

    # Noise
    tau = EF.Gamma(1e-5, 1e-5, name="tau", plates=())

    # Noisy observations
    Y = EF.Normal(WX, tau, name="Y", plates=(M,N))

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
    alpha.initialize_from_prior()
    tau.initialize_from_prior()
    X.initialize_from_prior()
    X.initialize_from_parameters(X.random(), np.identity(D))
    W.initialize_from_prior()
    W.initialize_from_parameters(W.random(), np.identity(D))

    # Data with missing values
    mask = np.random.rand(M,N) < 0.4 # randomly missing
    mask[:,20:40] = False # gap missing
    Y.observe(y, mask)

    # Inference loop.
    maxiter = 10000
    L_X = np.zeros(maxiter)
    L_W = np.zeros(maxiter)
    L_tau = np.zeros(maxiter)
    L_Y = np.zeros(maxiter)
    L_alpha = np.zeros(maxiter)
    L = np.zeros(maxiter)
    L_last = -np.inf
    for i in range(maxiter):
        t = time.clock()

        # Update nodes
        X.update()
        W.update()
        tau.update()
        alpha.update()

        # Compute lower bound
        L_X[i] = X.lower_bound_contribution()
        L_W[i] = W.lower_bound_contribution()
        L_tau[i] = tau.lower_bound_contribution()
        L_Y[i] = Y.lower_bound_contribution()
        L_alpha[i] = alpha.lower_bound_contribution()
        L[i] = L_X[i] + L_W[i] + L_tau[i] + L_Y[i] + L_alpha[i]
        #print('loglike terms:', L_X, L_W, L_tau, L_Y, L_alpha)

        # Check convergence
        print("Iteration %d: loglike=%e (%.3f seconds)" % (i+1, L[i], time.clock()-t))
        if L_last > L[i]:
            L_diff = (L_last - L[i])
            print("Lower bound decreased %e! Bug somewhere or numerical inaccuracy?" % L_diff)
        if L[i] - L_last < 1e-12:
            print("Converged.")
            #break
        L_last = L[i]

    tau.show()

    ## plt.clf()
    ## Z = np.vstack([L_X,L_W,L_tau,L_alpha,L_Y,L]).T
    ## dZ = np.diff(Z, axis=0)
    ## ax = plt.plot(dZ[50:])
    ## plt.legend(ax)
    ## return


    plt.ion()
    #plt.figure()
    plt.clf()
    WX_params = WX.get_parameters()
    fh = WX_params[0] * np.ones(y.shape)
    err_fh = 2*np.sqrt(WX_params[1]) * np.ones(y.shape)
    for m in range(M):
        plt.subplot(M,1,m)
        #errorplot(y, error=None, x=None, lower=None, upper=None):
        myplt.errorplot(fh[m], x=np.arange(N), error=err_fh[m])
        plt.plot(np.arange(N), f[m], 'g')
        plt.plot(np.arange(N), y[m], 'r+')


if __name__ == '__main__':
    # FOR INTERACTIVE SESSIONS, NON-BLOCKING PLOTTING:
    plt.ion()
    run()

