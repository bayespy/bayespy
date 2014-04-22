######################################################################
# Copyright (C) 2013-2014 Jaakko Luttinen
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
Demo: Linear state-space model with drifting dynamics for 1-D signal.

The observation is 1-D signal with changing frequency. The frequency oscillates
so it can be learnt too. Missing values are used to create a few gaps in the
data so the task is to reconstruct the gaps. You can learn either standard LSSM
or drifting LSSM. Standard LSSM is unable to predict the system because it is
not static. On the other hand, drifting LSSM can (if learning succeeds) give
very good estimations.
"""

import numpy as np
import scipy
import matplotlib.pyplot as plt

from bayespy.nodes import GaussianMarkovChain
from bayespy.nodes import DriftingGaussianMarkovChain
from bayespy.nodes import Gamma
from bayespy.nodes import SumMultiply

from bayespy.utils import utils
from bayespy.utils import random

from bayespy.inference.vmp.vmp import VB
from bayespy.inference.vmp import transformations

import bayespy.plot.plotting as bpplt

from bayespy.demos.model_lssm import run_lssm
from bayespy.demos.demo_switching_linear_state_space_model import run_slssm

def simulate_drifting_lssm(M, N):

    t = np.tile(np.arange(N), (M,1))
    a = 0.1 * 2*np.pi # base frequency
    b = 0.01 * 2*np.pi # frequency of the frequency-change
    c = 8
    #c = 0.5/c # derivative of the inner sin: |-b*c*cos(c*t)| <= |b*c|
    f = np.sin( a * (t + c*np.sin(b*t)) )
    y = f + (0.1*np.sqrt(M))*np.random.randn(M,N)

    return (y, f)


def run(M=1, N=1000, D=5, K=4, seed=42, maxiter=200, 
        rotate=True, debug=False, precompute=False,
        drift=False, switch=False,
        plot_X=False, plot_Y=True, plot_S=False):

    # Seed for random number generator
    if seed is not None:
        np.random.seed(seed)

    # Create data
    (y, f) = simulate_drifting_lssm(M, N)

    # Create some gaps
    mask_gaps = utils.trues((M,N))
    for m in range(100, N, 140):
        start = m
        end = min(m+15, N-1)
        mask_gaps[:,start:end] = False
    # Missing values
    mask_random = np.logical_or(random.mask(M, N, p=0.8),
                                np.logical_not(mask_gaps))
    # Remove the observations
    mask = np.logical_and(mask_gaps, mask_random)
    # Remove the observations
    y[~mask] = np.nan # BayesPy doesn't require NaNs, they're just for plotting.
    
    # Plot observations
    if plot_Y:
        plt.figure()
        bpplt.timeseries(f, 'b-')
        bpplt.timeseries(y, 'r.')
        plt.ylim([-2, 2])
    
    # Run the method
    if switch:
        Q = run_slssm(y, D, K,
                      mask=mask, 
                      maxiter=maxiter,
                      rotate=rotate,
                      debug=debug,
                      update_hyper=10,
                      monitor=True)
    else:
        Q = run_lssm(y, D, 
                     mask=mask, 
                     K=K, 
                     maxiter=maxiter,
                     rotate=rotate,
                     debug=debug,
                     precompute=precompute,
                     drift_A=drift,
                     update_hyper=10,
                     start_rotating_drift=20,
                     monitor=True)

    # Plot observations
    if plot_Y:
        plt.figure()
        bpplt.timeseries_normal(Q['F'], scale=2)
        bpplt.timeseries(f, 'b-')
        bpplt.timeseries(y, 'r.')
        plt.ylim([-2, 2])
    
    # Plot latent space
    if plot_X:
        Q.plot('X')
    
    # Plot drift space
    if plot_S and drift:
        Q.plot('S')

    # Compute RMSE
    rmse_random = utils.rmse(Q['Y'].get_moments()[0][~mask_random], 
                             f[~mask_random])
    rmse_gaps = utils.rmse(Q['Y'].get_moments()[0][~mask_gaps],
                           f[~mask_gaps])
    print("RMSE for randomly missing values: %f" % rmse_random)
    print("RMSE for gap values: %f" % rmse_gaps)

    plt.show()

if __name__ == '__main__':
    import sys, getopt, os
    try:
        opts, args = getopt.getopt(sys.argv[1:],
                                   "",
                                   [
                                    "n=",
                                    "d=",
                                    "k=",
                                    "model=",
                                    "seed=",
                                    "maxiter=",
                                    "debug",
                                    "precompute",
                                    "plot-y",
                                    "plot-x",
                                    "plot-s",
                                    "no-rotation"])
    except getopt.GetoptError:
        print('python demo_lssm_drift.py <options>')
        print('--n=<INT>        Number of data vectors')
        print('--d=<INT>        Dimensionality of the latent vectors in the model')
        print('--k=<INT>        Dimensionality of the latent drift space or number of mixtures in switching model')
        print('--model          [static] / drift / switch')
        print('--no-rotation    Do not apply speed-up rotations')
        print('--maxiter=<INT>  Maximum number of VB iterations')
        print('--seed=<INT>     Seed (integer) for the random number generator')
        print('--debug          Check that the rotations are implemented correctly')
        print('--plot-y         Plot Y')
        print('--plot-x         Plot X')
        print('--plot-s         Plot S')
        print('--precompute     Precompute some moments when rotating. May '
              'speed up or slow down.')
        sys.exit(2)

    print("By default, this function runs the static LSSM with speed-up "
          "rotations")
    print("You may also choose --model=drift or --model=switch")
    print("See --help for more help")

    kwargs = {}
    for opt, arg in opts:
        if opt == "--no-rotation":
            kwargs["rotate"] = False
        elif opt == "--maxiter":
            kwargs["maxiter"] = int(arg)
        elif opt == "--debug":
            kwargs["debug"] = True
        elif opt == "--precompute":
            kwargs["precompute"] = True
        elif opt == "--seed":
            kwargs["seed"] = int(arg)
        elif opt == "--n":
            kwargs["N"] = int(arg)
        elif opt == "--d":
            kwargs["D"] = int(arg)
        elif opt == "--k":
            if int(arg) == 0:
                kwargs["K"] = None
            else:
                kwargs["K"] = int(arg)
        elif opt == "--model":
            if arg == "drift":
                kwargs["drift"] = True
            elif arg == "switch":
                kwargs["switch"] = True
        elif opt == "--plot-x":
            kwargs["plot_X"] = True
        elif opt == "--plot-s":
            kwargs["plot_S"] = True
        elif opt == "--plot-y":
            kwargs["plot_Y"] = True
        else:
            raise ValueError("Unhandled argument given")

    run(M=1, **kwargs)
