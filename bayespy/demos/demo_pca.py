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
from bayespy.plot import plotting as myplt

from bayespy import utils
from bayespy import nodes

from bayespy.inference.vmp.vmp import VB
from bayespy.inference.vmp import transformations

from bayespy.inference.vmp.nodes.gamma import diagonal

import bayespy.plot.plotting as bpplt

def pca_model(M, N, D):
    # Construct the PCA model with ARD

    # ARD
    alpha = nodes.Gamma(1e-2,
                        1e-2,
                        plates=(D,),
                        name='alpha')

    # Loadings
    W = nodes.Gaussian(np.zeros(D),
                       diagonal(alpha),
                       name="W",
                       plates=(M,1))

    # States
    X = nodes.Gaussian(np.zeros(D),
                       np.identity(D),
                       name="X",
                       plates=(1,N))

    # PCA
    F = nodes.SumMultiply('i,i', W, X, name="F")

    # Noise
    tau = nodes.Gamma(1e-2, 1e-2, name="tau", plates=())

    # Noisy observations
    Y = nodes.Normal(F, tau, name="Y", plates=(M,N))

    return (Y, F, W, X, tau, alpha)


def run(M=10, N=100, D_y=3, D=5, seed=42):

    if seed is not None:
        np.random.seed(seed)
    
    # Generate data
    w = np.random.normal(0, 1, size=(M,1,D_y))
    x = np.random.normal(0, 1, size=(1,N,D_y))
    f = utils.utils.sum_product(w, x, axes_to_sum=[-1])
    y = f + np.random.normal(0, 0.2, size=(M,N))

    # Construct model
    (Y, F, W, X, tau, alpha) = pca_model(M, N, D)

    # Data with missing values
    mask = utils.random.mask(M, N, p=0.5) # randomly missing
    y[~mask] = np.nan
    Y.observe(y, mask=mask)

    # Construct inference machine
    Q = VB(Y, W, X, tau, alpha)

    # Initialize some nodes randomly
    X.initialize_from_random()
    W.initialize_from_random()

    # Inference loop.
    # TODO/FIXME: Use rotations.
    Q.update(repeat=100)

    # Plot results
    plt.figure()
    bpplt.timeseries_normal(F)
    bpplt.timeseries(f, 'g-')
    bpplt.timeseries(y, 'r+')
    plt.show()

if __name__ == '__main__':
    run()

