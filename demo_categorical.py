
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

def categorical_model(M, D):

    alpha = EF.Dirichlet(1*np.ones(D))
    p = EF.Categorical(alpha, plates=(M,))
    return (p, alpha)


def run(M=8, D=4):
    ## # Generate data
    ## w = np.random.normal(0, 1, size=(M,1,D_y))
    ## x = np.random.normal(0, 1, size=(1,N,D_y))
    ## f = utils.sum_product(w, x, axes_to_sum=[-1])
    ## y = f + np.random.normal(0, 0.5, size=(M,N))
    y = np.random.randint(D, size=(M,))

    # Construct model
    (p, alpha) = categorical_model(M, D)

    # Initialize nodes (from prior and randomly)
    alpha.update()
    p.update()

    p.observe(y)

    # Data with missing values
    #mask = np.random.rand(M,N) < 0.4 # randomly missing
    #mask[:,20:40] = False # gap missing
    #Y.observe(y, mask)

    # Inference loop.
    L_last = -np.inf
    for i in range(10):
        t = time.clock()

        # Update nodes
        p.update()
        alpha.update()

        # Compute lower bound
        L_p = p.lower_bound_contribution()
        L_alpha = alpha.lower_bound_contribution()
        L = L_p + L_alpha

        # Check convergence
        print("Iteration %d: loglike=%e (%.3f seconds)" % (i+1, L, time.clock()-t))
        if L_last > L:
            L_diff = (L_last - L)
            #raise Exception("Lower bound decreased %e! Bug somewhere or numerical inaccuracy?" % L_diff)
        if L - L_last < 1e-12:
            print("Converged.")
            #break
        L_last = L


    p.show()
    alpha.show()
    
    ## plt.ion()
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

