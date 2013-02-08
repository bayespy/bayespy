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


import time
import numpy as np
import matplotlib.pyplot as plt

from bayespy.plot import plotting as myplt

from bayespy import utils
from bayespy.inference.vmp import nodes

from bayespy.inference.vmp.vmp import VB

def pca_model(M, N, D):
    # Construct the PCA model with ARD

    # ARD
    alpha = nodes.Gamma(1e-2,
                        1e-2,
                        plates=(D,),
                        name='alpha')

    # Loadings
    W = nodes.Gaussian(np.zeros(D),
                       alpha.as_diagonal_wishart(),
                       name="W",
                       plates=(M,1))

    # States
    X = nodes.Gaussian(np.zeros(D),
                       np.identity(D),
                       name="X",
                       plates=(1,N))

    # PCA
    WX = nodes.Dot(W, X, name="WX")

    # Noise
    tau = nodes.Gamma(1e-2, 1e-2, name="tau", plates=())

    # Noisy observations
    Y = nodes.Normal(WX, tau, name="Y", plates=(M,N))

    return (Y, WX, W, X, tau, alpha)


def run(M=10, N=100, D_y=3, D=5):
    seed = 45
    print('seed =', seed)
    np.random.seed(seed)
    # Generate data
    w = np.random.normal(0, 1, size=(M,1,D_y))
    x = np.random.normal(0, 1, size=(1,N,D_y))
    f = utils.utils.sum_product(w, x, axes_to_sum=[-1])
    y = f + np.random.normal(0, 0.5, size=(M,N))

    # Construct model
    (Y, WX, W, X, tau, alpha) = pca_model(M, N, D)

    # Data with missing values
    mask = utils.random.mask(M, N, p=0.9) # randomly missing
    mask[:,20:40] = False # gap missing
    #mask[2,:] = False # gap missing
    y[~mask] = np.nan
    Y.observe(y, mask=mask)

    # Construct inference machine
    #Q = VB(Y, WX, W, X, tau, alpha)
    Q = VB(Y, W, X, tau, alpha)

    # Initialize nodes (from prior and randomly)
    alpha.initialize_from_prior()
    tau.initialize_from_prior()
    X.initialize_from_prior()
    W.initialize_from_prior()
    X.initialize_from_parameters(X.random(), np.identity(D))
    W.initialize_from_parameters(W.random(), np.identity(D))

    # Inference loop.
    Q.update(X, W, repeat=1)
    Q.update(alpha, tau, repeat=1)
    Q.update(X, W, alpha, tau, repeat=300)

    plt.clf()
    WX_params = WX.get_parameters()
    fh = WX_params[0] * np.ones(y.shape)
    err_fh = 2*np.sqrt(WX_params[1] + 1/tau.get_moments()[0]) * np.ones(y.shape)
    for m in range(M):
        plt.subplot(M,1,m+1)
        #errorplot(y, error=None, x=None, lower=None, upper=None):
        myplt.errorplot(fh[m], x=np.arange(N), error=err_fh[m])
        plt.plot(np.arange(N), f[m], 'g')
        plt.plot(np.arange(N), y[m], 'r+')

    plt.figure()
    Q.plot_iteration_by_nodes()

    plt.figure()
    plt.subplot(2,2,1)
    myplt.binary_matrix(W.mask)
    plt.subplot(2,2,2)
    myplt.binary_matrix(X.mask)
    plt.subplot(2,2,3)
    #myplt.binary_matrix(WX.get_mask())
    plt.subplot(2,2,4)
    myplt.binary_matrix(Y.mask)

    tau.show()
    alpha.show()

    plt.show()

if __name__ == '__main__':
    # FOR INTERACTIVE SESSIONS, NON-BLOCKING PLOTTING:
    #plt.ion()
    run()

