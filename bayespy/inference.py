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
import time

class VB():

    def __init__(self, *nodes, tol=1e-6):
        self.model = set(nodes)
        self.iter = 0
        self.L = -np.inf

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

