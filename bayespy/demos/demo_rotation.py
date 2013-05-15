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
import scipy
import matplotlib.pyplot as plt

from bayespy.plot import plotting as myplt

from bayespy import utils
from bayespy.inference.vmp import nodes

from bayespy.inference.vmp.vmp import VB
from bayespy.inference.vmp import transformations

from bayespy.inference.vmp.nodes.gamma import diagonal

"""
Demonstrate the effect of speed-up transformations for PCA model.
"""

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
    WX = nodes.Dot(W, X, name="WX")

    # Noise
    tau = nodes.Gamma(1e-2, 1e-2, name="tau")

    # Noisy observations
    Y = nodes.Normal(WX, tau, name="Y")

    return (Y, WX, W, X, tau, alpha)


def run(M=50, N=200, D_y=10, D=20, maxiter=100):
    seed = 45
    print('seed =', seed)
    np.random.seed(seed)
    
    # Generate data (covariance eigenvalues: 1,1,...,1,2^2,3^2,...,(D_y+1)^2
    (q,r) = scipy.linalg.qr(np.random.randn(M,M))
    C = np.diag(np.arange(2,2+D_y))
    C = np.ones(M)
    C[:D_y] += np.arange(1,1+D_y)
    y = C[:,np.newaxis] * np.random.randn(M,N)
    y = np.dot(q, y)

    # Construct model
    (Y, WX, W, X, tau, alpha) = pca_model(M, N, D)

    # Data with missing values
    mask = utils.random.mask(M, N, p=0.9) # randomly missing
    mask[:,20:40] = False # gap missing
    y[~mask] = np.nan
    Y.observe(y, mask=mask)

    # Construct inference machine
    Q = VB(Y, W, X, tau, alpha,
           autosave_filename=utils.utils.tempfile())

    # Initialize nodes (from prior and randomly)
    X.initialize_from_value(X.random())
    W.initialize_from_value(W.random())

    Q.update(repeat=1)
    Q.save()

    #
    # Run inference with rotations.
    #
    rotX = transformations.RotateGaussian(X)
    rotW = transformations.RotateGaussianARD(W, alpha)
    R = transformations.RotationOptimizer(rotX, rotW, D)

    for ind in range(maxiter):
        Q.update()
        R.rotate(check_gradient=False, 
                 maxiter=10, 
                 verbose=False,
                 check_bound=Q.compute_lowerbound,
                 check_bound_terms=Q.compute_lowerbound_terms)

    L_rot = Q.L

    #
    # Re-run inference without rotations.
    #
    Q.load()
    Q.update(repeat=maxiter)
    L_norot = Q.L

    #
    # Plot comparison
    #
    plt.plot(L_rot)
    plt.plot(L_norot)
    plt.legend(['With rotations', 'Without rotations'], loc='lower right')
    plt.show()

if __name__ == '__main__':
    run()

