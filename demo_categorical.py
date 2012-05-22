
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

    p = EF.Dirichlet(1*np.ones(D), name='p')
    z = EF.Categorical(p, plates=(M,), name='z')
    return (z, p)


def run(M=30, D=5):

    # Generate data
    y = np.random.randint(D, size=(M,))

    # Construct model
    (z, p) = categorical_model(M, D)

    # Initialize nodes
    p.update()
    z.update()

    # Observe the data with randomly missing values
    mask = np.random.rand(M) < 0.5 # randomly missing
    z.observe(y, mask)

    # Inference loop.
    L_last = -np.inf
    for i in range(100):
        t = time.clock()

        # Update nodes
        p.update()
        z.update()

        # Compute lower bound
        L_p = p.lower_bound_contribution()
        L_z = z.lower_bound_contribution()
        L = L_p + L_z

        # Check convergence
        print("Iteration %d: loglike=%e (%.3f seconds)" % (i+1, L, time.clock()-t))
        if L_last > L:
            L_diff = (L_last - L)

        if L - L_last < 1e-12:
            print("Converged.")

        L_last = L


    z.show()
    p.show()

if __name__ == '__main__':
    run()

