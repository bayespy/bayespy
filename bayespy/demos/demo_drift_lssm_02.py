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
Demo: Drifting linear state-space model for SPDE problem.

The observed process is a stochastic advection-diffusion spatio-temporal
process. The velocity field changes in time so standard linear state-space model
is not able to learn the process accurately. Drifting linear state-space model
may use drift for either loadings or dynamics or both, but the learning is good
only when using fully drifting model, that is, drifting dynamics and loadings.

Some observations about the performance of the drifting LSSM (more like
hypotheses than proved facts):

    * The number of stations and the latent space dimensionality must be quite
      large so that the dynamics can be learnt (applies for both LSSM versions).

    * If latent space dimensionality D and/or(?) drifting space dimensionality
      are large, the drifting weights might be regularized too much by VB. That
      is, the drifting weights time series S might perform badly over periods of
      missing values. If this is true, it is a bit disappointing, because it
      would be nice that the method would not start performing worse if the
      model complexity is increased but rather the unnecessary components are
      just pruned out. Or the number of time instances must be large enough to
      learn the dynamics of the dynamics.
"""

# Nicely working experiments:
# Data: 
#     no-dynamic
#     M=100
#     D=50
#     burnin=100
#     thin=20
#     velocity=1e-1
#     innovation_noise=1e-4
#     noise_ratio=1e-1
#     lengthscale=0.6
# Data: 
#     no-dynamic
#     M=100
#     D=50
#     burnin=100
#     thin=20
#     velocity=1e-1
#     innovation_noise=1e-3
#     noise_ratio=5e-1
#     lengthscale=0.6
# Data, quite ok results: 
#     no-dynamic
#     M=70
#     D=50
#     burnin=100
#     thin=20
#     velocity=1e-1
#     innovation_noise=1e-3
#     noise_ratio=5e-1
#     lengthscale=0.6

# Data:
## resolution=resolution,
## burnin=1000,
## thin=20,
## velocity=4e-2,
## diffusion=1e-4,
## decay= 1.0 - 5e-3,
## innovation_noise=1e-4,
## innovation_lengthscale=1.0,
## noise_ratio=5e-1)


import numpy as np
import scipy
import matplotlib.pyplot as plt

from bayespy.nodes import GaussianMarkovChain
from bayespy.nodes import DriftingGaussianMarkovChain
from bayespy.nodes import GaussianArrayARD
from bayespy.nodes import Gamma
from bayespy.nodes import SumMultiply

from bayespy.utils import utils
from bayespy.utils import random

from bayespy.inference.vmp.vmp import VB
from bayespy.inference.vmp import transformations

import bayespy.plot.plotting as bpplt

from bayespy.utils.covfunc.covariance import covfunc_se as covfunc

from bayespy.demos import demo_drift_lssm_01

def simulate_process(M=100, N=100, T=100, velocity=1e-3, diffusion=1e-5,
                     lengthscale=0.6,
                     noise=1e0, decay=0.9995):
    """
    Simulate advection-diffusion PDE on a unit square.

    The boundaries are cylindrical, that is, the object is a torus.
    """
    dy = 1.0/M
    dx = 1.0/N
    dt = 1.0/T

    MN = M*N

    # Transform the torus space into an 4-D Euclidean space
    xh = np.arange(M) / M
    yh = np.arange(N) / N
    # The covariance of the spatial innovation noise
    Kx = covfunc(1.0,
                 lengthscale, 
                 np.array([np.sin(2*np.pi*xh),
                           np.cos(2*np.pi*xh)]).T,
                 np.array([np.sin(2*np.pi*xh),
                           np.cos(2*np.pi*xh)]).T)
    Ky = covfunc(1.0,
                 lengthscale,
                 np.array([np.sin(2*np.pi*yh),
                           np.cos(2*np.pi*yh)]).T,
                 np.array([np.sin(2*np.pi*yh),
                           np.cos(2*np.pi*yh)]).T)
    Lx = np.linalg.cholesky(Kx+1e-6*np.identity(M))
    Ly = np.linalg.cholesky(Ky+1e-6*np.identity(N))
    draw_R = lambda : noise * np.ravel(np.dot(Lx, 
                                              np.dot(np.random.randn(M,N), 
                                                     Ly.T)))
    
                 
    
    # Flow field
    #v = velocity*np.array([1, 1]) #np.random.randn(2)
    # Log-diffusion
    logD = np.log(diffusion) #+ np.random.randn()
    # Source?
    #noise = noise*np.sqrt(dt/(dx*dy))
    R = draw_R() #noise*np.random.randn(MN)

    # Initial state
    (x, y) = np.meshgrid(np.arange(N), np.arange(M))
    u = np.sin(3*2*np.pi*x/N) * np.sin(3*2*np.pi*y/M)
    u = np.ravel(u)
    #u = R
    #u = np.sin(2*np.pi/(M*N) * np.mod(np.arange(M*N), M*N)) #np.random.randn(M*N)
    #u = np.sin(2*np.pi/(M*N) * np.mod(np.arange(M*N), M*N)) #np.random.randn(M*N)


    U = np.empty((T, M, N))

    o = np.ones(MN)
    i = np.tile(np.arange(MN), 5)
    j = np.empty(5*MN)
    d = np.empty(5*MN)
    
    # Initial state from: -nabla u = R
    D = np.exp(logD)
    v = np.zeros(2)

    v = 2*velocity*np.ones(2)
    #v = velocity*np.random.randn(2)

    
    j[:MN] = np.mod(np.arange(MN), MN)
    d[:MN] = 0
    j[MN:2*MN] = np.mod(np.arange(MN)+1, MN)
    d[MN:2*MN] = 1
    j[2*MN:3*MN] = np.mod(np.arange(MN)+N, MN)
    d[2*MN:3*MN] = 2
    j[3*MN:4*MN] = np.mod(np.arange(MN)-1, MN)
    d[3*MN:4*MN] = 3
    j[4*MN:5*MN] = np.mod(np.arange(MN)-N, MN)
    d[4*MN:5*MN] = 4
    A = scipy.sparse.csc_matrix((d, (i,j)))
    ind0 = (A.data == 0)
    ind1 = (A.data == 1)
    ind2 = (A.data == 2)
    ind3 = (A.data == 3)
    ind4 = (A.data == 4)
    
    for t in range(T):

        #v = np.sqrt(decay)*v + np.sqrt(1-decay)*velocity*np.random.randn(2)
        v = decay*v + np.sqrt(1-decay**2)*velocity*np.random.randn(2)
        D = np.exp(logD)

        R = draw_R()
        
        # Form the system matrix
        A.data[ind0] = 1 + 2*D*dt/dx**2 + 2*D*dt/dy**2
        A.data[ind1] = -(D*dt/dx**2 - v[0]*dt/(2*dx))
        A.data[ind2] = -(D*dt/dy**2 - v[1]*dt/(2*dy))
        A.data[ind3] = -(D*dt/dx**2 + v[0]*dt/(2*dx))
        A.data[ind4] = -(D*dt/dy**2 + v[1]*dt/(2*dy))

        # Solve the system
        u = scipy.sparse.linalg.spsolve(A, u + R)

        # Store the solution
        U[t] = np.reshape(u, (M,N))

        print('\rSimulating SPDE data... %d%%' % (int(100.0*(t+1)/T)), end="")

    print('\rSimulating SPDE data... Done.')

    return U


def simulate_data(filename=None, 
                  resolution=30,
                  M=50,
                  N=1000, 
                  diffusion=1e-6,
                  velocity=4e-3,
                  innovation_noise=1e-3,
                  innovation_lengthscale=0.6,
                  noise_ratio=1e-1,
                  decay=0.9997,
                  burnin=1000,
                  thin=20):

    # Simulate the process

    # Because simulate_process simulates a unit square over unit time step we
    # have to multiply parameters by the time length in order to get the effect
    # of a long time :)
    #
    # Thin-parameter affects the temporal resolution.
    diffusion *= (N + burnin/thin)
    velocity *= (N + burnin/thin)
    decay = decay ** (1/thin)
    innovation_noise *= np.sqrt(N)
    U = simulate_process(resolution,
                         resolution,
                         burnin+N*thin, 
                         diffusion=diffusion,
                         velocity=velocity,
                         noise=innovation_noise,
                         lengthscale=innovation_lengthscale,
                         decay=decay)

    # Put some stations randomly
    x1x2 = np.random.permutation(resolution*resolution)[:M]
    x1 = np.arange(resolution)[np.mod(x1x2, resolution)]
    x2 = np.arange(resolution)[(x1x2/resolution).astype(int)]
    #x1 = np.random.randint(0, resolution, M)
    #x2 = np.random.randint(0, resolution, M)

    # Get noisy observations
    U = U[burnin::thin]
    F = U[:, x1, x2].T
    std = np.std(F)
    Y = F + noise_ratio*std*np.random.randn(*np.shape(F))
    X = np.array([x1, x2]).T

    # Save data
    if filename is not None:
        f = h5py.File(filename, 'w')
        try:
            utils.write_to_hdf5(f, U, 'U')
            utils.write_to_hdf5(f, diffusion/T, 'diffusion')
            utils.write_to_hdf5(f, velocity/T, 'velocity')
            utils.write_to_hdf5(f, decay, 'decay')
            utils.write_to_hdf5(f, innovation_noise/np.sqrt(T), 'innovation_noise')
            utils.write_to_hdf5(f, noise_ratio, 'noise_ratio')
            utils.write_to_hdf5(f, Y, 'Y')
            utils.write_to_hdf5(f, F, 'F')
            utils.write_to_hdf5(f, X, 'X')
        finally:
            f.close()

    return (U, Y, F, X)

def run(M=100, N=2000, D=20, K=4, rotate=False, maxiter=200, seed=42,
        debug=False, precompute=False, resolution=30, dynamic=True):
    
    # Seed for random number generator
    if seed is not None:
        np.random.seed(seed)

    # Create data
    if dynamic:
        decay = 1.0 - 5e-3
    else:
        decay = 1.0

    (U, y, f, X) = simulate_data(M=M, 
                                 N=N,
                                 resolution=resolution,
                                 burnin=1000,
                                 thin=20,
                                 velocity=4e-2,
                                 diffusion=1e-4,
                                 decay=decay,
                                 innovation_noise=1e-4,
                                 innovation_lengthscale=1.0,
                                 noise_ratio=5e-1)

    plt.ion()
    plt.plot(X[:,0], X[:,1], 'kx')
    bpplt.matrix_animation(U)
    plt.show()
    plt.ioff()

    ## bpplt.timeseries(f, 'b-')
    ## bpplt.timeseries(y, 'r.')

    # Missing values
    mask = random.mask(M, N, p=0.8)
    # Create some gaps
    gap = 15
    interval = 100
    for m in range(100, N, interval):
        start = m
        end = min(m+gap, N-1)
        mask[:,start:end] = False

    #mask[:] = True # DEBUG
    # Remove the observations
    y[~mask] = np.nan # BayesPy doesn't require NaNs, they're just for plotting.

    # Run the method
    if K is not None:
        demo_drift_lssm_01.run_dlssm(y, f, mask, D, K, maxiter,
                                     rotate=rotate,
                                     debug=debug,
                                     precompute=precompute,
                                     plot_X=True,
                                     plot_Y=True,
                                     plot_S=True)
                                     ## plot_X=plot_X,
                                     ## plot_Y=plot_Y,
                                     ## plot_S=plot_S)
    else:
        demo_drift_lssm_01.run_lssm(y, f, mask, D, maxiter, 
                                    rotate=rotate, 
                                    debug=debug,
                                    precompute=precompute,
                                    plot_X=True,
                                    plot_Y=True)
        

    plt.show()

    return

if __name__ == '__main__':
    import sys, getopt, os
    try:
        opts, args = getopt.getopt(sys.argv[1:],
                                   "",
                                   [
                                       "m=",
                                       "n=",
                                       "d=",
                                       "k=",
                                       "resolution=",
                                       "seed=",
                                       "maxiter=",
                                       "no-dynamic",
                                       "debug",
                                       "precompute",
                                       "plot-y",
                                       "plot-x",
                                       "plot-s",
                                       "rotate"
                                   ])
    except getopt.GetoptError:
        print('python demo_lssm_drift.py <options>')
        print('--m=<INT>           Dimensionality of data vectors')
        print('--n=<INT>           Number of data vectors')
        print('--d=<INT>           Dimensionality of the latent vectors in the model')
        print('--k=<INT>           Dimensionality of the latent drift space')
        print('--resolution=<INT>  Grid resolution for the SPDE simulation')
        print('--no-dynamic        Velocity field of the SPDE does not change')
        print('--rotate            Apply speed-up rotations')
        print('--maxiter=<INT>     Maximum number of VB iterations')
        print('--seed=<INT>        Seed (integer) for the random number generator')
        print('--debug             Check that the rotations are implemented correctly')
        print('--plot-y            Plot Y')
        print('--plot-x            Plot X')
        print('--plot-s            Plot S')
        print('--precompute        Precompute some moments when rotating. May '
              'speed up or slow down.')
        sys.exit(2)

    kwargs = {}
    for opt, arg in opts:
        if opt == "--rotate":
            kwargs["rotate"] = True
        elif opt == "--maxiter":
            kwargs["maxiter"] = int(arg)
        elif opt == "--debug":
            kwargs["debug"] = True
        elif opt == "--precompute":
            kwargs["precompute"] = True
        elif opt == "--no-dynamic":
            kwargs["dynamic"] = False
        elif opt == "--seed":
            kwargs["seed"] = int(arg)
        elif opt == "--m":
            kwargs["M"] = int(arg)
        elif opt == "--n":
            kwargs["N"] = int(arg)
        elif opt == "--d":
            kwargs["D"] = int(arg)
        elif opt == "--k":
            if int(arg) == 0:
                kwargs["K"] = None
            else:
                kwargs["K"] = int(arg)
        elif opt == "--resolution":
            kwargs["resolution"] = int(arg)
        elif opt == "--plot-x":
            kwargs["plot_X"] = True
        elif opt == "--plot-s":
            kwargs["plot_S"] = True
        elif opt == "--plot-y":
            kwargs["plot_Y"] = True
        else:
            raise ValueError("Unhandled argument given")

    run(**kwargs)
