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
                           SumMultiply)

from bayespy.utils import random

from bayespy.inference.vmp.vmp import VB

import bayespy.plot as bpplt


def get_gradient(*nodes):
    return [X.get_gradient() for X in nodes]


def dot(x1, x2):
    v = 0
    for (y1, y2) in zip(x1, x2):
        for (z1, z2) in zip(y1, y2):
            v += np.dot(np.ravel(z1), np.ravel(z2))
    return v


def update_direction(d, s, b):
    y = []
    for (di, si) in zip(d, s):
        y.append([])
        for (dij, sij) in zip(di, si):
            y[-1].append(dij + b*sij)
    return y


def step(nodes, s):
    for (node, si) in zip(nodes, s):
        node.update_parameters(si, scale=1)
    return


def optimize(Q, *nodes):

    print(Q.compute_lowerbound())

    # Get gradients
    grad = get_gradient(*nodes)

    dd_prev = dot(grad, grad)

    step(nodes, grad)
    print(Q.compute_lowerbound())

    s = grad
    
    for i in range(10):

        grad = get_gradient(*nodes)

        dd_curr = dot(grad, grad)
        b = dd_curr / dd_prev
        dd_prev = dd_curr

        s = update_direction(grad, s, b)

        step(nodes, s)

        print(Q.compute_lowerbound())


def pca():

    np.random.seed(41)

    M = 1
    N = 1
    D = 1

    # Construct the model
    X = GaussianARD(0, 1, plates=(1,N), shape=(D,))
    W = GaussianARD(0, 1, plates=(M,1), shape=(D,))
    W.initialize_from_random()
    F = SumMultiply('d,d->', W, X)
    #F = SumMultiply(',->', W, X)
    Y = GaussianARD(F, 1000)

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
    Q.update(repeat=10)

    # Restore the state
    X.set_parameters(x)
    W.set_parameters(w)

    # Run Riemannian conjugate gradient
    #Q.update(repeat=10)
    optimize(Q, X, W)
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
