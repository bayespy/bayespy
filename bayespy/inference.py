import numpy as np
import time

class VB():

    def __init__(self, *nodes, tol=1e-6):
        self.model = set(nodes)
        self.iter = 0
        self.L = -np.inf

    def update(self, *nodes, repeat=10):
        for i in range(repeat):
            t = time.clock()
            for node in nodes:
                node.update()
            self.iter += 1
            
            L = self.loglikelihood_lowerbound()
            print("Iteration %d: loglike=%e (%.3f seconds)" 
                  % (self.iter, L, time.clock()-t))

            # Check for errors
            if self.L - L > 1e-6:
                L_diff = (self.L - L)
                print("Lower bound decreased %e! Bug somewhere or numerical inaccuracy?" % L_diff)
                #raise Exception("Lower bound decreased %e! Bug somewhere or numerical inaccuracy?" % L_diff)

            # Check for convergence
            if L - self.L < 1e-12:
                print("Converged.")

            self.L = L

    def loglikelihood_lowerbound(self):
        L = 0
        for node in self.model:
            L += node.lower_bound_contribution()
        return L

