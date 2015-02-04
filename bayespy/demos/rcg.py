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


def pca():

    np.random.seed(41)

    M = 20
    N = 100
    D = 5

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
    p = Q.get_parameters(X, W)

    # Run VB-EM
    Q.update(repeat=30)

    # Restore the state
    Q.set_parameters(p, X, W)

    # Run Riemannian conjugate gradient
    Q.optimize(X, W, maxiter=30)



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
