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
import h5py
import datetime
import tempfile

from bayespy import utils

from bayespy.inference.vmp.nodes.node import Node

class VB():

    def __init__(self,
                 *nodes, 
                 tol=1e-6, 
                 autosave_iterations=0, 
                 autosave_filename=None):
        self.model = set(nodes)
        self.iter = 0
        self.L = np.array(())
        self.l = dict(zip(self.model, 
                          len(self.model)*[np.array([])]))
        self.autosave_iterations = autosave_iterations
        if autosave_filename is None or autosave_filename == '':
            date = datetime.datetime.today().strftime('%Y%m%d%H%M%S')
            prefix = 'vb_autosave_%s_' % date
            tmpfile = tempfile.NamedTemporaryFile(prefix=prefix,
                                                  suffix='.hdf5')
            self.autosave_filename = tmpfile.name
        else:
            self.autosave_filename = autosave_filename

    def update(self, *nodes, repeat=1):

        # Append the cost arrays
        self.L = np.append(self.L, utils.utils.nans(repeat))
        for (node, l) in self.l.items():
            self.l[node] = np.append(l, utils.utils.nans(repeat))

        # By default, update all nodes
        if len(nodes) == 0:
            nodes = self.model

        for i in range(repeat):
            t = time.clock()

            # Update nodes
            for node in nodes:
                node.update()
                # Force garbage collection

            # Compute lower bound
            L = self.loglikelihood_lowerbound()
            print("Iteration %d: loglike=%e (%.3f seconds)" 
                  % (self.iter+1, L, time.clock()-t))

            # Check the progress of the iteration
            if self.iter > 0:
                # Check for errors
                if self.L[self.iter-1] - L > 1e-6:
                    L_diff = (self.L[self.iter-1] - L)
                    warnings.warn("Lower bound decreased %e! Bug somewhere or "
                                  "numerical inaccuracy?" % L_diff)

                # Check for convergence
                if L - self.L[self.iter-1] < 1e-12:
                    print("Converged.")

            # Auto-save, if requested
            if (self.autosave_iterations > 0 
                and np.mod(self.iter+1, self.autosave_iterations) == 0):

                self.save(self.autosave_filename)
                print('Auto-saved to %s' % self.autosave_filename)

            self.L[self.iter] = L
            self.iter += 1

            

    def loglikelihood_lowerbound(self):
        L = 0
        for node in self.model:
            lp = node.lower_bound_contribution()
            L += lp
            self.l[node][self.iter] = lp
            
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


    def save(self, filename):
        # Open HDF5 file
        h5f = h5py.File(filename, 'w')
        # Write each node
        nodegroup = h5f.create_group('nodes')
        for node in self.model:
            if node.name == '':
                raise Exception("In order to save nodes, they must have "
                                "(unique) names.")
            node.save(nodegroup.create_group(node.name))
        # Write iteration statistics
        utils.utils.write_to_hdf5(h5f, self.L, 'L')
        utils.utils.write_to_hdf5(h5f, self.iter, 'iter')
        boundgroup = h5f.create_group('boundterms')
        for node in self.model:
            utils.utils.write_to_hdf5(boundgroup, self.l[node], node.name)
        # Close file
        h5f.close()

    def load(self, filename):
        # Open HDF5 file
        h5f = h5py.File(filename, 'r')
        # Read each node
        for node in self.model:
            if node.name == '':
                raise Exception("In order to load nodes, they must have "
                                "(unique) names.")
            node.load(h5f['nodes'][node.name])
        # Read iteration statistics
        self.L = h5f['L'][...]
        self.iter = h5f['iter'][...]
        for node in self.model:
            self.l[node] = h5f['boundterms'][node.name][...]
        # Close file
        h5f.close()
        
