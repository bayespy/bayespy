################################################################################
# Copyright (C) 2011-2015 Jaakko Luttinen
#
# This file is licensed under the MIT License.
################################################################################


import numpy as np
import matplotlib.pyplot as plt
import warnings
import time
import h5py
import datetime
import tempfile
import scipy
import logging

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
                 use_logging=False,
                 user_data=None,
                 callback=None):

        self.user_data = user_data

        for (ind, node) in enumerate(nodes):
            if not isinstance(node, Node):
                raise ValueError("Argument number %d is not a node" % (ind+1))

        if use_logging:
            logger = logging.getLogger(__name__)
            self.print = logger.info
        else:
            # By default, don't use logging, just print stuff
            self.print = print
            
        # Remove duplicate nodes
        self.model = misc.unique(nodes)

        self.ignore_bound_checks = False

        self._figures = {}
        
        self.iter = 0
        self.annealing_changed = False
        self.converged = False
        self.L = np.array(())
        self.cputime = np.array(())
        self.l = dict(zip(self.model, 
                          len(self.model)*[np.array([])]))
        self.autosave_iterations = autosave_iterations
        self.autosave_nodes = None
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


    def use_logging(self, use):
        if use_logging:
            logger = logging.getLogger(__name__)
            self.print = logger.info
        else:
            # By default, don't use logging, just print stuff
            self.print = print
        return


    def set_autosave(self, filename, iterations=None, nodes=None):
        self.autosave_filename = filename
        self.filename = filename
        self.autosave_nodes = nodes
        if iterations is not None:
            self.autosave_iterations = iterations


    def set_callback(self, callback):
        self.callback = callback

    def update(self, *nodes, repeat=1, plot=False, tol=None, verbose=True):

        # TODO/FIXME:
        #
        # If no nodes are given and thus everything is updated, the update order
        # should be from down to bottom. Or something similar..

        # By default, update all nodes
        if len(nodes) == 0:
            nodes = self.model
        if plot is True:
            plot_nodes = self.model
        elif plot is False:
            plot_nodes = []
        else:
            plot_nodes = [self[x] for x in plot]

        # Make certain that at least one of the nodes in the model has been
        # observed
        if (not self.ignore_bound_checks
            and all(~np.any(n.observed) for n in self.model)):

            raise Exception("At least one node in the model must be observed.")

        converged = False

        for i in range(repeat):

            t = time.clock()

            # Update nodes
            for node in nodes:
                X = self[node]
                if hasattr(X, 'update') and callable(X.update):
                    X.update()
                if X in plot_nodes:
                    self.plot(X)

            cputime = time.clock() - t
            if self._end_iteration_step(None, cputime, tol=tol, verbose=verbose):
                return


    def has_converged(self, tol=None):
        return self.converged



    def compute_lowerbound(self, ignore_masked=True):
        L = 0
        for node in self.model:
            L += node.lower_bound_contribution(ignore_masked=ignore_masked)
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

    def plot_iteration_by_nodes(self, axes=None, diff=False):
        """
        Plot the cost function per node during the iteration.

        Handy tool for debugging.
        """

        if axes is None:
            axes = plt.gca()
        
        D = len(self.l)
        N = self.iter + 1
        if diff:
            L = np.empty((N-1,D))
            x = np.arange(N-1) + 2
        else:
            L = np.empty((N,D))
            x = np.arange(N) + 1
        legends = []
        for (d, node) in enumerate(self.l):
            if diff:
                L[:,d] = np.diff(self.l[node][:N])
            else:
                L[:,d] = self.l[node][:N]
            legends += [node.name]
        axes.plot(x, L)
        axes.legend(legends, loc='lower right')
        axes.set_title('Lower bound contributions by nodes')
        axes.set_xlabel('Iteration')


    def get_iteration_by_nodes(self):
        return self.l


    def save(self, *nodes, filename=None):

        if len(nodes) == 0:
            nodes = self.model
        else:
            nodes = [self[node] for node in nodes if node is not None]

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
            if self.autosave_filename:
                filename = self.autosave_filename
            else:
                raise Exception("Filename must be given.")

        # Open HDF5 file
        h5f = h5py.File(filename, 'w')

        try:
            # Write each node
            nodegroup = h5f.create_group('nodes')
            for node in nodes:
                if node.name == '':
                    raise Exception("In order to save nodes, they must have "
                                    "(unique) names.")
                if hasattr(node, '_save') and callable(node._save):
                    node._save(nodegroup.create_group(node.name))
            # Write iteration statistics
            misc.write_to_hdf5(h5f, self.L, 'L')
            misc.write_to_hdf5(h5f, self.cputime, 'cputime')
            misc.write_to_hdf5(h5f, self.iter, 'iter')
            misc.write_to_hdf5(h5f, self.converged, 'converged')
            if self.callback_output is not None:
                misc.write_to_hdf5(h5f, 
                                   self.callback_output,
                                   'callback_output')
            boundgroup = h5f.create_group('boundterms')
            for node in nodes:
                misc.write_to_hdf5(boundgroup, self.l[node], node.name)
            # Write user data
            if self.user_data is not None:
                user_data_group = h5f.create_group('user_data')
                for (key, value) in self.user_data.items():
                    user_data_group[key] = value
        finally:
            # Close file
            h5f.close()


    @staticmethod
    def load_user_data(filename):
        f = h5py.File(filename, 'r')
        try:
            group = f['user_data']
            for (key, value) in group.items():
                user_data['key'] = value[...]
        except:
            raise
        finally:
            f.close()
        return


    def load(self, *nodes, filename=None, nodes_only=False):

        # By default, use the same file as for auto-saving
        if not filename:
            if self.autosave_filename:
                filename = self.autosave_filename
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
                        node._load(h5f['nodes'][node.name])
                    except KeyError:
                        h5f.close()
                        raise Exception("File does not contain variable %s"
                                        % node.name)
            # Read iteration statistics
            if not nodes_only:
                self.L = h5f['L'][...]
                self.cputime = h5f['cputime'][...]
                self.iter = h5f['iter'][...]
                self.converged = h5f['converged'][...]
                for node in nodes:
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

    def plot(self, *nodes, **kwargs):
        """
        Plot the distribution of the given nodes (or all nodes)
        """

        if len(nodes) == 0:
            nodes = self.model

        for node in nodes:
            node = self[node]
            if node.has_plotter():

                try:
                    fignum = self._figures[node]
                except KeyError:
                    fig = plt.figure()
                    self._figures[node] = fig.number
                else:
                    fig = plt.figure(num=fignum)

                fig.clf()
                node.plot(fig=fig, **kwargs)
                fig.canvas.draw()


    @property
    def ignore_bound_checks(self):
        return self.__ignore_bound_checks


    @ignore_bound_checks.setter
    def ignore_bound_checks(self, ignore):
        self.__ignore_bound_checks = ignore


    def get_gradients(self, *nodes, euclidian=False):
        """
        Computes gradients (both Riemannian and normal)
        """
        rg = [self[node].get_riemannian_gradient() for node in nodes]
        if euclidian:
            g = [self[node].get_gradient(rg_x)
                 for (node, rg_x) in zip(nodes, rg)]
            return (rg, g)
        else:
            return rg


    def get_parameters(self, *nodes):
        """
        Get parameters of the nodes
        """
        return [self[node].get_parameters()
                for node in nodes]


    def set_parameters(self, x, *nodes):
        """
        Set parameters of the nodes
        """
        for (node, xi) in zip(nodes, x):
            self[node].set_parameters(xi)
        return


    def gradient_step(self, *nodes, scale=1.0):
        """
        Update nodes by taking a gradient ascent step
        """
        p = self.add(self.get_parameters(*nodes),
                     self.get_gradients(*nodes),
                     scale=scale)
        self.set_parameters(p, *nodes)
        return


    def dot(self, x1, x2):
        """
        Computes dot products of given vectors (in parameter format)
        """
        v = 0
        # Loop over nodes
        for (y1, y2) in zip(x1, x2):
            # Loop over parameters
            for (z1, z2) in zip(y1, y2):
                v += np.dot(np.ravel(z1), np.ravel(z2))
        return v


    def add(self, x1, x2, scale=1):
        """
        Add two vectors (in parameter format)
        """
        v = []
        # Loop over nodes
        for (y1, y2) in zip(x1, x2):
            v.append([])
            # Loop over parameters
            for (z1, z2) in zip(y1, y2):
                v[-1].append(z1 + scale*z2)
        return v


    def optimize(self, *nodes, maxiter=10, verbose=True, method='fletcher-reeves',
                 riemannian=True, collapsed=None, tol=None):
        """
        Optimize nodes using Riemannian conjugate gradient
        """

        method = method.lower()

        if collapsed is None:
            collapsed = []

        scale = 1.0
        p = self.get_parameters(*nodes)
        dd_prev = 0

        for i in range(maxiter):

            t = time.clock()

            # Get gradients
            if riemannian and method == 'gradient':
                rg = self.get_gradients(*nodes, euclidian=False)
                g1 = rg
                g2 = rg
            else:
                (rg, g) = self.get_gradients(*nodes, euclidian=True)
                if riemannian:
                    g1 = g
                    g2 = rg
                else:
                    g1 = g
                    g2 = g

            if method == 'gradient':
                b = 0
            elif method == 'fletcher-reeves':
                dd_curr = self.dot(g1, g2)
                if dd_prev == 0:
                    b = 0
                else:
                    b = dd_curr / dd_prev
                dd_prev = dd_curr
            else:
                raise Exception("Unknown optimization method: %s" % (method))

            if b:
                s = self.add(g2, s, scale=b)
            else:
                s = g2

            success = False
            while not success:

                p_new = self.add(p, s, scale=scale)

                try:
                    self.set_parameters(p_new, *nodes)
                except:
                    if verbose:
                        self.print("CG update was unsuccessful, using gradient and resetting CG")
                    if s is g2:
                        scale = scale / 2
                    dd_prev = 0
                    s = g2
                    continue

                # Update collapsed variables
                collapsed_params = self.get_parameters(*collapsed)
                try:
                    for node in collapsed:
                        self[node].update()
                except:
                    self.set_parameters(collapsed_params, *collapsed)
                    if verbose:
                        self.print("Collapsed node update node failed, reset CG")
                    if s is g2:
                        scale = scale / 2
                    dd_prev = 0
                    s = g2
                    continue

                L = self.compute_lowerbound()

                bound_decreased = (
                    self.iter > 0 and
                    L < self.L[self.iter-1] and
                    not np.allclose(L, self.L[self.iter-1], rtol=1e-8)
                )

                if np.isnan(L) or bound_decreased:

                    # Restore the state of the collapsed nodes to what it was
                    # before updating them
                    self.set_parameters(collapsed_params, *collapsed)
                    if s is g2:
                        scale = scale / 2
                        if verbose:
                            self.print(
                                "Gradient ascent decreased lower bound from {0} to {1}, halfing step length"
                                .format(
                                    self.L[self.iter-1],
                                    L,
                                )
                            )
                    else:
                        if scale < 2 ** (-10):
                            if verbose:
                                self.print(
                                    "CG decreased lower bound from {0} to {1}, reset CG."
                                    .format(
                                        self.L[self.iter-1],
                                        L,
                                    )
                                )
                            dd_prev = 0
                            s = g2
                        else:
                            scale = scale / 2
                            if verbose:
                                self.print(
                                    "CG decreased lower bound from {0} to {1}, halfing step length"
                                    .format(
                                        self.L[self.iter-1],
                                        L,
                                    )
                                )
                    continue

                success = True

            scale = scale * np.sqrt(2)
            p = p_new

            cputime = time.clock() - t
            if self._end_iteration_step('OPT', cputime, tol=tol, verbose=verbose):
                break


    def pattern_search(self, *nodes, collapsed=None, maxiter=3):
        """Perform simple pattern search :cite:`Honkela:2003`.

        Some of the variables can be collapsed.
        """

        if collapsed is None:
            collapsed = []

        t = time.clock()

        # Update all nodes
        for x in nodes:
            self[x].update()
        for x in collapsed:
            self[x].update()

        # Current parameter values
        p0 = self.get_parameters(*nodes)

        # Update optimized nodes
        for x in nodes:
            self[x].update()

        # New parameter values
        p1 = self.get_parameters(*nodes)

        # Search direction
        dp = self.add(p1, p0, scale=-1)

        # Cost function for pattern search
        def cost(alpha):
            p_new = self.add(p1, dp, scale=alpha)
            try:
                self.set_parameters(p_new, *nodes)
            except:
                return np.inf
            # Update collapsed nodes
            for x in collapsed:
                self[x].update()
            return -self.compute_lowerbound()

        # Optimize step length
        res = scipy.optimize.minimize_scalar(cost, bracket=[0, 3], options={'maxiter':maxiter})

        # Set found parameter values
        p_new = self.add(p1, dp, scale=res.x)
        self.set_parameters(p_new, *nodes)

        # Update collapsed nodes
        for x in collapsed:
            self[x].update()

        cputime = time.clock() - t
        self._end_iteration_step('PS', cputime)


    def set_annealing(self, annealing):
        """
        Set deterministic annealing from range (0, 1].

        With 1, no annealing, standard updates.

        With smaller values, entropy has more weight and model
        probability equations less.  With 0, one would obtain improper
        uniform distributions.
        """
        for node in self.model:
            node.annealing = annealing
        self.annealing_changed = True
        self.converged = False
        return


    def _append_iterations(self, iters):
        """
        Append some arrays for more iterations
        """
        self.L = np.append(self.L, misc.nans(iters))
        self.cputime = np.append(self.cputime, misc.nans(iters))
        for (node, l) in self.l.items():
            self.l[node] = np.append(l, misc.nans(iters))
        return


    def _end_iteration_step(self, method, cputime, tol=None, verbose=True, bound_cpu_time=True):
        """
        Do some routines after each iteration step
        """

        if self.iter >= len(self.L):
            self._append_iterations(100)

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

        t = time.clock()
        L = self.loglikelihood_lowerbound()
        if bound_cpu_time:
            cputime += time.clock() - t

        self.cputime[self.iter] = cputime
        self.L[self.iter] = L

        if verbose:
            if method:
                self.print("Iteration %d (%s): loglike=%e (%.3f seconds)"
                           % (self.iter+1, method, L, cputime))
            else:
                self.print("Iteration %d: loglike=%e (%.3f seconds)"
                           % (self.iter+1, L, cputime))

        # Check the progress of the iteration
        self.converged = False
        if not self.ignore_bound_checks and not self.annealing_changed and self.iter > 0:
            # Check for errors
            if self.L[self.iter-1] - L > 1e-6:
                L_diff = (self.L[self.iter-1] - L)
                warnings.warn("Lower bound decreased %e! Bug somewhere or "
                              "numerical inaccuracy?" % L_diff)

            # Check for convergence
            L0 = self.L[self.iter-1]
            L1 = self.L[self.iter]
            if tol is None:
                tol = self.tol
            div = 0.5 * (abs(L0) + abs(L1))
            if (L1 - L0) / div < tol:
            #if (L1 - L0) / div < tol or L1 - L0 <= 0:
                if verbose:
                    self.print("Converged at iteration %d." % (self.iter+1))
                self.converged = True

        # Auto-save, if requested
        if (self.autosave_iterations > 0 
            and np.mod(self.iter+1, self.autosave_iterations) == 0):

            if self.autosave_nodes is not None:
                self.save(*self.autosave_nodes, filename=self.autosave_filename)
            else:
                self.save(filename=self.autosave_filename)
            if verbose:
                self.print('Auto-saved to %s' % self.autosave_filename)

        self.annealing_changed = False

        self.iter += 1

        return self.converged
