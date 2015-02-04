######################################################################
# Copyright (C) 2014 Jaakko Luttinen
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

"""
Demonstrate Riemannian conjugate gradient
"""

import numpy as np

from bayespy.nodes import (Beta,
                           Bernoulli,
                           GaussianARD,
                           Gaussian,
                           SumMultiply)

from bayespy.utils import random

from bayespy.inference.vmp.vmp import VB

import bayespy.plot as bpplt


def get_gradient(*nodes):
    """
    Computes gradients (both Riemannian and normal)
    """
    rg = [X.get_riemannian_gradient() for X in nodes]
    g = [X.get_gradient(rg_x) for (X, rg_x) in zip(nodes, rg)]
    return (rg, g)


def dot(x1, x2):
    """
    Computes dot products of given vectors
    """
    v = 0
    # Loop over nodes
    for (y1, y2) in zip(x1, x2):
        # Loop over parameters
        for (z1, z2) in zip(y1, y2):
            v += np.dot(np.ravel(z1), np.ravel(z2))
    return v


def add(x1, x2, scale=1):
    """
    Computes dot products of given vectors
    """
    v = []
    # Loop over nodes
    for (y1, y2) in zip(x1, x2):
        v.append([])
        # Loop over parameters
        for (z1, z2) in zip(y1, y2):
            v[-1].append(z1 + scale*z2)
    return v


def update_direction(d, s, b):
    """
    Updates direction d with direction s weighted by b.

    d + b*s
    """
    y = []
    for (di, si) in zip(d, s):
        y.append([])
        for (dij, sij) in zip(di, si):
            y[-1].append(dij + b*sij)
    return y


def set_parameters(nodes, p):
    """
    Update parameters by taking a step into the given direction
    """
    for (node, pi) in zip(nodes, p):
        node.set_parameters(pi)
    return


def get_parameters(*nodes):
    """
    Update parameters by taking a step into the given direction
    """
    return [node.get_parameters() for node in nodes]


def step(nodes, s, scale=1):
    """
    Update parameters by taking a step into the given direction
    """
    for (node, si) in zip(nodes, s):
        node.update_parameters(si, scale=scale)
    return


def optimize(Q, *nodes, maxiter=10):

    print("Start CG optimization")

    print(Q.compute_lowerbound())

    # Get gradients
    p = get_parameters(*nodes)
    (rg, g) = get_gradient(*nodes)

    dd_prev = dot(g, rg)

    p_new = add(p, rg)
    set_parameters(nodes, p_new)
    p = p_new
    #step(nodes, rg)
    print(Q.compute_lowerbound())

    s = rg
    
    for i in range(maxiter):

        (rg, g) = get_gradient(*nodes)

        dd_curr = dot(g, rg)
        b = dd_curr / dd_prev
        dd_prev = dd_curr

        s = update_direction(rg, s, b)

        p_new = add(p, s)
        try:
            set_parameters(nodes, p_new)
        except:
            print("WARNING! CG update was unsuccessful, using gradient and resetting CG")
            s = rg
            p_new = add(p, rg)
            set_parameters(nodes, p_new)
        p = p_new

        #step(nodes, s, scale=1)

        print("Iteration %d: loglike=%e" % (i+1, Q.compute_lowerbound()))


def pca():

    np.random.seed(41)

    M = 4
    N = 3
    D = 2

    # Construct the model
    X = Gaussian(np.zeros(D), np.identity(D), plates=(1,N))
    W = Gaussian(np.zeros(D), np.identity(D), plates=(M,1))
    #X = GaussianARD(0, 1, plates=(1,N), shape=(D,))
    #W = GaussianARD(0, 1, plates=(M,1), shape=(D,))
    W.initialize_from_random()
    F = SumMultiply('d,d->', W, X)
    #F = SumMultiply(',->', W, X)
    Y = GaussianARD(F, 1e4)

    # Observe data
    data = np.sum(W.random() * X.random(), axis=-1)
    Y.observe(data)

    # Initialize VB engine
    Q = VB(Y, X, W)

    # Take one update step (so phi is ok)
    Q.update()

    # Store the state
    x = X.get_parameters()
    w = W.get_parameters()

    # Run VB-EM
    Q.update(repeat=20)

    # Restore the state
    X.set_parameters(x)
    W.set_parameters(w)

    # Run Riemannian conjugate gradient
    #Q.update(repeat=10)
    optimize(Q, X, W, maxiter=20)
    ## d_X = X.get_gradient()
    ## d_W = W.get_gradient()
    ## dd_previous = dot([d_X, d_W], [d_X, d_W])

    ## X.update_parameters(d_X)
    ## W.update_parameters(d_W)

    ## s_X = d_X
    ## s_W = d_W



def beta():

    # Simple model
    p = Beta([1, 1], plates=(3,))
    Z = Bernoulli(p, plates=(100,3))

    # Observe data
    data = Z.random()
    Z.observe(data)

    # Learn p using Riemannian gradient
    for i in range(50):
        d = p.get_gradient()
        p.update_parameters(d, scale=0.5)
        print(p.u)

    p.update()
    print(p.u)


if __name__ == "__main__":
    pca()
