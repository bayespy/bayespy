######################################################################
# Copyright (C) 2011-2013 Jaakko Luttinen
#
# This file is licensed under Version 3.0 of the GNU General Public
# License. See LICENSE for a text of the license.
######################################################################

######################################################################
# This file is part of BayesPy.
#
# BayesPy is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation.
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

from bayespy.utils import misc

from bayespy.inference.vmp.nodes.node import Node

class VB():
    r"""
    Variational Bayesian (VB) inference engine

    Parameters
    ----------

    nodes : nodes
    
        Nodes that form the model. Must include all at least all stochastic
        nodes of the model.
        
    tol : double, optional

        Convergence criterion.  Tolerance for the relative change in the VB
        lower bound.

    autosave_filename : string, optional

        Filename for automatic saving

    autosave_iterations : int, optional

        Iteration interval between each automatic saving

    callback : callable, optional

        Function which is called after each update iteration step

    """

    def __init__(self,
                 *nodes, 
                 tol=1e-5, 
                 autosave_filename=None,
                 autosave_iterations=0, 
                 callback=None):

        for (ind, node) in enumerate(nodes):
            if not isinstance(node, Node):
                raise ValueError("Argument number %d is not a node" % (ind+1))
            
        # Remove duplicate nodes
        self.model = misc.unique(nodes)

        self._figures = {}
        
        self.iter = 0
        self.L = np.array(())
        self.l = dict(zip(self.model, 
                          len(self.model)*[np.array([])]))
        self.autosave_iterations = autosave_iterations
        if not autosave_filename:
            date = datetime.datetime.today().strftime('%Y%m%d%H%M%S')
            prefix = 'vb_autosave_%s_' % date
            tmpfile = tempfile.NamedTemporaryFile(prefix=prefix,
                                                  suffix='.hdf5')
            self.autosave_filename = tmpfile.name
            self.filename = None
        else:
            self.autosave_filename = autosave_filename
            self.filename = autosave_filename

        # Check uniqueness of the node names
        names = [node.name for node in self.model]
        if len(names) != len(self.model):
            raise Exception("Use unique names for nodes.")

        self.callback = callback
        self.callback_output = None
        self.tol = tol

    def set_autosave(self, filename, iterations=None):
        self.autosave_filename = filename
        self.filename = filename
        if iterations is not None:
            self.autosave_iterations = iterations


    def set_callback(self, callback):
        self.callback = callback

    def update(self, *nodes, repeat=1, plot=False, tol=None, verbose=True):

        # TODO/FIXME:
        #
        # If no nodes are given and thus everything is updated, the update order
        # should be from down to bottom. Or something similar..

        # Append the cost arrays
        self.L = np.append(self.L, misc.nans(repeat))
        for (node, l) in self.l.items():
            self.l[node] = np.append(l, misc.nans(repeat))

        # By default, update all nodes
        if len(nodes) == 0:
            nodes = self.model

        converged = False

        for i in range(repeat):
            t = time.clock()

            # Update nodes
            for node in nodes:
                X = self[node]
                if hasattr(X, 'update') and callable(X.update):
                    X.update()
                    if plot:
                        self.plot(X)

            # Call the custom function provided by the user
            if callable(self.callback):
                z = self.callback()
                if z is not None:
                    z = np.array(z)[...,np.newaxis]
                    if self.callback_output is None:
                        self.callback_output = z
                    else:
                        self.callback_output = np.concatenate((self.callback_output,z),
                                                              axis=-1)

            # Compute lower bound
            L = self.loglikelihood_lowerbound()
            if verbose:
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
                if tol is None:
                    tol = self.tol
                div = 0.5 * (abs(L) + abs(self.L[self.iter-1]))
                if (L - self.L[self.iter-1]) / div < tol or L - self.L[self.iter-1] <= 0:
                    converged = True
                    if verbose:
                        print("Converged at iteration %d." % (self.iter+1))

            self.L[self.iter] = L
            self.iter += 1

            # Auto-save, if requested
            if (self.autosave_iterations > 0 
                and np.mod(self.iter, self.autosave_iterations) == 0):

                self.save(self.autosave_filename)
                if verbose:
                    print('Auto-saved to %s' % self.autosave_filename)

            if converged:
                return



    def compute_lowerbound(self):
        L = 0
        for node in self.model:
            L += node.lower_bound_contribution()
        return L

    def compute_lowerbound_terms(self, *nodes):
        if len(nodes) == 0:
            nodes = self.model
        return {node: node.lower_bound_contribution()
                for node in nodes}

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
        plt.legend(legends, loc='lower right')
        plt.title('Lower bound contributions by nodes')
        plt.xlabel('Iteration')

    def get_iteration_by_nodes(self):
        return self.l


    def save(self, filename=None):

        if self.iter == 0:
            # Check HDF5 version.
            if h5py.version.hdf5_version_tuple < (1,8,7): 
                warnings.warn("WARNING! Your HDF5 version is %s. HDF5 versions "
                              "<1.8.7 are not able to save empty arrays, thus "
                              "you may experience problems if you for instance "
                              "try to save before running any iteration steps."
                              % str(h5py.version.hdf5_version_tuple))
            

        # By default, use the same file as for auto-saving
        if not filename:
            if self.filename:
                filename = self.filename
            else:
                raise Exception("Filename must be given.")

        # Open HDF5 file
        h5f = h5py.File(filename, 'w')

        try:
            # Write each node
            nodegroup = h5f.create_group('nodes')
            for node in self.model:
                if node.name == '':
                    raise Exception("In order to save nodes, they must have "
                                    "(unique) names.")
                if hasattr(node, 'save') and callable(node.save):
                    node.save(nodegroup.create_group(node.name))
            # Write iteration statistics
            misc.write_to_hdf5(h5f, self.L, 'L')
            misc.write_to_hdf5(h5f, self.iter, 'iter')
            if self.callback_output is not None:
                misc.write_to_hdf5(h5f, 
                                          self.callback_output,
                                          'callback_output')
            boundgroup = h5f.create_group('boundterms')
            for node in self.model:
                misc.write_to_hdf5(boundgroup, self.l[node], node.name)
        finally:
            # Close file
            h5f.close()

    def load(self, *nodes, filename=None):

        # By default, use the same file as for auto-saving
        if not filename:
            if self.filename:
                filename = self.filename
            else:
                raise Exception("Filename must be given.")
            
        # Open HDF5 file
        h5f = h5py.File(filename, 'r')

        try:
            # Get nodes to load
            if len(nodes) == 0:
                nodes = self.model
            else:
                nodes = [self[node] for node in nodes if node is not None]
            # Read each node
            for node_id in nodes:
                node = self[node_id]
                if node.name == '':
                    h5f.close()
                    raise Exception("In order to load nodes, they must have "
                                    "(unique) names.")
                if hasattr(node, 'load') and callable(node.load):
                    try:
                        node.load(h5f['nodes'][node.name])
                    except KeyError:
                        h5f.close()
                        raise Exception("File does not contain variable %s"
                                        % node.name)
            # Read iteration statistics
            self.L = h5f['L'][...]
            self.iter = h5f['iter'][...]
            for node in self.model:
                self.l[node] = h5f['boundterms'][node.name][...]
            try:
                self.callback_output = h5f['callback_output'][...]
            except KeyError:
                pass

        finally:
            # Close file
            h5f.close()
        
    def __getitem__(self, name):
        if name in self.model:
            return name
        else:
            # Dictionary for mapping node names to nodes
            dictionary = {node.name: node for node in self.model}
            return dictionary[name]        

    def plot(self, *nodes):
        """
        Plot the distribution of the given nodes (or all nodes)
        """
        if len(nodes) == 0:
            nodes = self.model

        if not plt.isinteractive():
            plt.ion()
            redisable = True
        else:
            redisable = False

        for node in nodes:
            node = self[node]
            if node.has_plotter():
                try:
                    plt.figure(self._figures[node])
                except:
                    f = plt.figure()
                    self._figures[node] = f.number
                plt.clf()
                node.plot()
                plt.draw()
                plt.show()

        if redisable:
            plt.ioff()
