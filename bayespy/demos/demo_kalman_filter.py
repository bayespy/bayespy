import numpy as np
import matplotlib.pyplot as plt

from bayespy.utils import utils

def run():
    # Create some data
    N = 100
    D = 1
    # Initial state
    x0 = np.array([0.5])
    # Dynamics
    A = np.array([0.9])
    # Innovation covariance matrix
    V = np.array([1])
    # Observation noise covariance matrix
    C = np.array([0.5])

    X = np.empty((N,D))

    # Simulate data
    x = x0
    for n in range(N):
        x = A*x + np.random.multivariate_normal(np.zeros(D), V)

    plt.clf()
    plt.plot(x)
    

#pip install https://github.com/matplotlib/matplotlib/archive/v1.2.0rc3.tar.gz
