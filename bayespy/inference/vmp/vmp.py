######################################################################
# Copyright (C) 2011,2012 Jaakko Luttinen
#
# This file is licensed under Version 3.0 of the GNU General Public
# License. See LICENSE for a text of the license.
######################################################################

######################################################################
# This file is part of BayesPy.
#
# BayesPy is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# BayesPy is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with BayesPy.  If not, see <http://www.gnu.org/licenses/>.
######################################################################

import numpy as np
import matplotlib.pyplot as plt
import warnings
import time

class VB():

    def __init__(self, *nodes, tol=1e-6):
        self.model = set(nodes)
        self.iter = 0
        self.L = -np.inf

        self.l = dict(zip(self.model, 
                          len(self.model)*[np.array([])]))

    def update(self, *nodes, repeat=1):

        # By default, update all nodes
        if len(nodes) == 0:
            nodes = self.model
            
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
                warnings.warn("Lower bound decreased %e! Bug somewhere or "
                              "numerical inaccuracy?" % L_diff)

            # Check for convergence
            if L - self.L < 1e-12:
                print("Converged.")

            self.L = L

    def loglikelihood_lowerbound(self):
        L = 0
        for node in self.model:
            lp = node.lower_bound_contribution()
            self.l[node] = np.append(self.l[node], lp)
            L += lp
        return L

    def plot_iteration_by_nodes(self):
        """
        Plot the cost function per node during the iteration.

        Handy tool for debugging.
        """
        
        D = len(self.l)
        N = self.iter
        L = np.empty((N,D))
        legends = []
        for (d, node) in enumerate(self.l):
            L[:,d] = self.l[node]
            legends += [node.name]
        plt.plot(np.arange(N)+1, L)
        plt.legend(legends)
        plt.title('Lower bound contributions by nodes')
        plt.xlabel('Iteration')

    def get_iteration_by_nodes(self):
        return self.l

