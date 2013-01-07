import numpy as np
import matplotlib.pyplot as plt

from bayespy.utils import utils

import bayespy.plot.plotting as bpplt

import imp
imp.reload(utils)
imp.reload(bpplt)

def run():
    # Create some data
    N = 500
    D = 2
    # Initial state
    x0 = np.array([0.5, -0.5])
    # Dynamics
    A = np.array([[.9, -.4], [.4, .9]])
    # Innovation covariance matrix
    V = np.array([[1, 0], [0, 1]])
    # Observation noise covariance matrix
    C = 1*np.array([[1, 0], [0, 1]])

    X = np.empty((N,D))
    Y = np.empty((N,D))

    # Simulate data
    x = x0
    for n in range(N):
        x = np.dot(A,x) + np.random.multivariate_normal(np.zeros(D), V)
        X[n,:] = x
        Y[n,:] = x + np.random.multivariate_normal(np.zeros(D), C)

    U = np.linalg.inv(C)
    UY = np.linalg.solve(C, Y.T).T

    # Create iterators for the static matrices
    U = N*(U,)
    A = N*(A,)
    V = N*(V,)
    
    (Xh, CovXh) = utils.kalman_filter(UY, U, A, V, np.zeros(2), 10*np.identity(2))
    (Xh, CovXh) = utils.rts_smoother(Xh, CovXh, A, V)
    
    plt.clf()
    for d in range(D):
        plt.subplot(D,1,d)
        #plt.plot(Xh[:,d], 'r--')
        bpplt.errorplot(Xh[:,d], error=2*np.sqrt(CovXh[:,d,d]))
        plt.plot(X[:,d], 'r-')
        plt.plot(Y[:,d], '.')
    

#pip install https://github.com/matplotlib/matplotlib/archive/v1.2.0rc3.tar.gz
