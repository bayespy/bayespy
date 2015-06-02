################################################################################
# Copyright (C) 2011-2013 Jaakko Luttinen
#
# This file is licensed under the MIT License.
################################################################################


import numpy as np
import matplotlib.pyplot as plt

import h5py
import tempfile

import bayespy.plot as bpplt

from bayespy.utils import misc
from bayespy.utils import random
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
    Y = nodes.GaussianARD(WX, tau, name="Y", plates=(M,N))

    return (Y, WX, W, X, tau, alpha)


@bpplt.interactive
def run(M=10, N=100, D_y=3, D=5):
    seed = 45
    print('seed =', seed)
    np.random.seed(seed)

    # Check HDF5 version.
    if h5py.version.hdf5_version_tuple < (1,8,7): 
        print("WARNING! Your HDF5 version is %s. HDF5 versions <1.8.7 are not "
              "able to save empty arrays, thus you may experience problems if "
              "you for instance try to save before running any iteration steps."
              % str(h5py.version.hdf5_version_tuple))
    
    # Generate data
    w = np.random.normal(0, 1, size=(M,1,D_y))
    x = np.random.normal(0, 1, size=(1,N,D_y))
    f = misc.sum_product(w, x, axes_to_sum=[-1])
    y = f + np.random.normal(0, 0.5, size=(M,N))

    # Construct model
    (Y, WX, W, X, tau, alpha) = pca_model(M, N, D)

    # Data with missing values
    mask = random.mask(M, N, p=0.9) # randomly missing
    mask[:,20:40] = False # gap missing
    y[~mask] = np.nan
    Y.observe(y, mask=mask)

    # Construct inference machine
    Q = VB(Y, W, X, tau, alpha, autosave_iterations=5)

    # Initialize some nodes randomly
    X.initialize_from_value(X.random())
    W.initialize_from_value(W.random())

    # Save the state into a HDF5 file
    filename = tempfile.NamedTemporaryFile(suffix='hdf5').name
    Q.update(X, W, alpha, tau, repeat=1)
    Q.save(filename=filename)

    # Inference loop.
    Q.update(X, W, alpha, tau, repeat=10)

    # Reload the state from the HDF5 file
    Q.load(filename=filename)

    # Inference loop again.
    Q.update(X, W, alpha, tau, repeat=10)

    # NOTE: Saving and loading requires that you have the model
    # constructed. "Save" does not store the model structure nor does "load"
    # read it. They are just used for reading and writing the contents of the
    # nodes. Thus, if you want to load, you first need to construct the same
    # model that was used for saving and then use load to set the states of the
    # nodes.

    plt.clf()
    WX_params = WX.get_parameters()
    fh = WX_params[0] * np.ones(y.shape)
    err_fh = 2*np.sqrt(WX_params[1] + 1/tau.get_moments()[0]) * np.ones(y.shape)
    for m in range(M):
        plt.subplot(M,1,m+1)
        #errorplot(y, error=None, x=None, lower=None, upper=None):
        bpplt.errorplot(fh[m], x=np.arange(N), error=err_fh[m])
        plt.plot(np.arange(N), f[m], 'g')
        plt.plot(np.arange(N), y[m], 'r+')

    plt.figure()
    Q.plot_iteration_by_nodes()

    plt.figure()
    plt.subplot(2,2,1)
    bpplt.binary_matrix(W.mask)
    plt.subplot(2,2,2)
    bpplt.binary_matrix(X.mask)
    plt.subplot(2,2,3)
    #bpplt.binary_matrix(WX.get_mask())
    plt.subplot(2,2,4)
    bpplt.binary_matrix(Y.mask)


if __name__ == '__main__':
    run()
    plt.show()

