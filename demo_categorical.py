
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

def categorical_model(M, D):

    alpha = EF.Dirichlet(1*np.ones(D))
    p = EF.Categorical(alpha, plates=(M,))
    return (p, alpha)


def run(M=10, D=4):

    # Generate data
    y = np.random.randint(D, size=(M,))

    # Construct model
    (p, alpha) = categorical_model(M, D)

    # Initialize nodes
    alpha.update()
    p.update()

    # Observe the data with randomly missing values
    mask = np.random.rand(M) < 0.5 # randomly missing
    p.observe(y, mask)

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

        if L - L_last < 1e-12:
            print("Converged.")

        L_last = L


    p.show()
    alpha.show()

if __name__ == '__main__':
    run()

