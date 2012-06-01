
import numpy as np
import matplotlib.pyplot as plt
import time

import utils
import plotting as myplt
import nodes as EF
#import nodes.exponential_family as EF

## import imp
## imp.reload(utils)
## imp.reload(myplt)
## imp.reload(EF)

# Reload everything (helpful for interactive sessions)
import imp
import nodes
import nodes.node
import nodes.variables
import nodes.variables.variable
import nodes.variables.wishart
import nodes.variables.gaussian
import nodes.variables.mixture
import nodes.variables.dirichlet
import nodes.variables.categorical
imp.reload(utils)
imp.reload(nodes)
imp.reload(nodes.node)
imp.reload(nodes.variables)
imp.reload(nodes.variables.variable)
imp.reload(nodes.variables.wishart)
imp.reload(nodes.variables.gaussian)
imp.reload(nodes.variables.mixture)
imp.reload(nodes.variables.dirichlet)
imp.reload(nodes.variables.categorical)

def gaussianmix_model(N, K, D):
    # N = number of data vectors
    # K = number of clusters
    # D = dimensionality
    
    # Construct the Gaussian mixture model

    # K prior weights (for components)
    alpha = EF.Dirichlet(1*np.ones(K),
                         name='alpha')
    # N K-dimensional cluster assignments (for data)
    z = EF.Categorical(alpha,
                       plates=(N,),
                       name='z')
    # K D-dimensional component means
    X = EF.Gaussian(np.zeros(D), 0.01*np.identity(D),
                    plates=(K,),
                    name='X')
    # K D-dimensional component covariances
    Lambda = EF.Wishart(D, 0.01*np.identity(D),
                        plates=(K,),
                        name='Lambda')
    # N D-dimensional observation vectors
    Y = EF.Mixture(EF.Gaussian)(z, X, Lambda, plates=(N,), name='Y')
    # TODO: Plates should be learned automatically if not given (it
    # would be the smallest shape broadcasted from the shapes of the
    # parents)

    return (Y, X, Lambda, z, alpha)


def run(N=50, K=5, D=2):

    plt.ion()
    #17,31
    #np.random.seed(31)
    
    # Generate data
    N1 = np.floor(0.5*N)
    N2 = N - N1
    y = np.vstack([np.random.normal(0, 0.5, size=(N1,D)),
                   np.random.normal(10, 0.5, size=(N2,D))])

    
    # Construct model
    (Y, X, Lambda, z, alpha) = gaussianmix_model(N,K,D)

    # Initialize nodes (from prior and randomly)
    alpha.initialize_from_prior()
    z.initialize_from_prior()
    Lambda.initialize_from_parameters(D, 10*np.identity(D))

    X.initialize_from_prior()
    X.initialize_from_parameters(X.random(), np.identity(D))

    ## X.initialize_from_parameters(np.random.permutation(y)[:K],
    ##                              0.01*np.identity(D))

    #X.initialize_from_random()
    # Initialize means by selecting random data points
    #X.initialize_from_value(np.random.permutation(y)[:K])
    #return
    #X.initialize_random_mean()
    #Y.initialize()

    ## X.show()
    ## return

    # Data with missing values
    ## mask = np.random.rand(M,N) < 0.4 # randomly missing
    ## mask[:,20:40] = False # gap missing
    # Y.observe(y, mask)
    ## alpha.show()
    ## Lambda.show()
    ## z.show()
    X.show()
    #return

    Y.observe(y)

    ## z.update()
    ## X.update()
    ## alpha.update()

    ## X.show()
    ## alpha.show()
    #Lambda.show()
    #return

    ## X.show()
    ## Lambda.show()
    ## z.update()
    ## z.show()
    ## return

    # Inference loop.
    maxiter = 30
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

        #Y.show()

        #z.show()

        # Compute lower bound
        L_X[i] = X.lower_bound_contribution()
        L_Lambda[i] = Lambda.lower_bound_contribution()
        L_alpha[i] = alpha.lower_bound_contribution()
        L_z[i] = z.lower_bound_contribution()
        L_Y[i] = Y.lower_bound_contribution()
        L[i] = L_X[i] + L_Lambda[i] + L_alpha[i] + L_z[i] + L_Y[i]

        #print('terms:', L_X[i], L_Lambda[i], L_alpha[i], L_z[i], L_Y[i])

        # Check convergence
        print("Iteration %d: loglike=%e (%.3f seconds)" % (i+1, L[i], time.clock()-t))
        if L_last - L[i] > 1e-6:
            L_diff = (L_last - L[i])
            print("Lower bound decreased %e! Bug somewhere or numerical inaccuracy?" % L_diff)
            #raise Exception("Lower bound decreased %e! Bug somewhere or numerical inaccuracy?" % L_diff)
        if L[i] - L_last < 1e-12:
            print("Converged.")
            #break
        L_last = L[i]

    # Predictive stuff
    zh = EF.Categorical(alpha, name='zh')
    Yh = EF.Mixture(EF.Gaussian)(zh, X, Lambda, name='Yh')
    # TODO/FIXME: Messages to parents should use the masks such that
    # children don't need to be initialized!
    zh.initialize_from_prior()
    Yh.initialize_from_prior()
    zh.update()
    #zh.show()
    N1 = 400
    N2 = 400
    x1 = np.linspace(-3, 15, N1)
    x2 = np.linspace(-3, 15, N2)
    xh = utils.grid(x1, x2)
    lpdf = Yh.integrated_logpdf_from_parents(xh, 0)
    pdf = np.reshape(np.exp(lpdf), (N2,N1))
    #print(pdf)
    #plt.clf()
    plt.clf()
    #plt.imshow(x1, x2, pdf)
    plt.contourf(x1, x2, pdf, 100)
    plt.scatter(y[:,0], y[:,1])
    print('integrated pdf:', np.sum(pdf)*(18*18)/(N1*N2))
    #return

    X.show()
    alpha.show()

    #print(y)
    #print(Y.u[0])

    plt.clf()
    f = np.vstack([L_X, L_Lambda, L_z, L_alpha, L_Y, L]).T
    #f = np.diff(f, axis=0)
    ax = plt.plot(f)
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

