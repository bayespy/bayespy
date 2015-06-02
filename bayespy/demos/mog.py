################################################################################
# Copyright (C) 2011-2012 Jaakko Luttinen
#
# This file is licensed under the MIT License.
################################################################################


import numpy as np
import matplotlib.pyplot as plt
import time

from bayespy.utils import misc
import bayespy.plot as bpplt
from bayespy.inference.vmp import nodes
from bayespy.inference.vmp.vmp import VB

def gaussianmix_model(N, K, D, covariance='full'):
    # N = number of data vectors
    # K = number of clusters
    # D = dimensionality
    
    # Construct the Gaussian mixture model

    # K prior weights (for components)
    alpha = nodes.Dirichlet(1e-3*np.ones(K),
                            name='alpha')
    # N K-dimensional cluster assignments (for data)
    z = nodes.Categorical(alpha,
                          plates=(N,),
                          name='z')
    # K D-dimensional component means
    X = nodes.GaussianARD(0, 1e-3,
                          shape=(D,),
                          plates=(K,),
                          name='X')
    if covariance.lower() == 'full':
        # K D-dimensional component covariances
        Lambda = nodes.Wishart(D, 0.01*np.identity(D),
                               plates=(K,),
                               name='Lambda')
        # N D-dimensional observation vectors
        Y = nodes.Mixture(z, nodes.Gaussian, X, Lambda, plates=(N,), name='Y')
    elif covariance.lower() == 'diagonal':
        # Inverse variances
        Lambda = nodes.Gamma(1e-3, 1e-3, plates=(K, D), name='Lambda')
        # N D-dimensional observation vectors
        Y = nodes.Mixture(z, nodes.GaussianARD, X, Lambda, plates=(N,), name='Y')
    elif covariance.lower() == 'isotropic':
        # Inverse variances
        Lambda = nodes.Gamma(1e-3, 1e-3, plates=(K, 1), name='Lambda')
        # N D-dimensional observation vectors
        Y = nodes.Mixture(z, nodes.GaussianARD, X, Lambda, plates=(N,), name='Y')

    z.initialize_from_random()

    return VB(Y, X, Lambda, z, alpha)


@bpplt.interactive
def run(N=50, K=5, D=2):

    # Generate data
    N1 = np.floor(0.5*N)
    N2 = N - N1
    y = np.vstack([np.random.normal(0, 0.5, size=(N1,D)),
                   np.random.normal(10, 0.5, size=(N2,D))])

    
    # Construct model
    Q = gaussianmix_model(N,K,D)

    # Observe data
    Q['Y'].observe(y)

    # Run inference
    Q.update(repeat=30)

    # Run predictive model
    zh = nodes.Categorical(Q['alpha'], name='zh')
    Yh = nodes.Mixture(zh, nodes.Gaussian, Q['X'], Q['Lambda'], name='Yh')
    zh.update()

    # Plot predictive pdf
    N1 = 400
    N2 = 400
    x1 = np.linspace(-3, 15, N1)
    x2 = np.linspace(-3, 15, N2)
    xh = misc.grid(x1, x2)
    lpdf = Yh.integrated_logpdf_from_parents(xh, 0)
    pdf = np.reshape(np.exp(lpdf), (N2,N1))
    plt.clf()
    plt.contourf(x1, x2, pdf, 100)
    plt.scatter(y[:,0], y[:,1])
    print('integrated pdf:', np.sum(pdf)*(18*18)/(N1*N2))

    #Q['X'].show()
    #Q['alpha'].show()


if __name__ == '__main__':
    run()
    plt.show()

