######################################################################
# Copyright (C) 2011,2012 Jaakko Luttinen
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
#import scipy as sp
import matplotlib.pyplot as plt
import plotting as myplt
import time

# Profiling stuff
#import profile
#import cProfile
#import pstats


import imp

import utils

import Nodes.ExponentialFamily as EF
import Nodes.CovarianceFunctions as CF
import Nodes.GaussianProcesses as GP
imp.reload(myplt)
imp.reload(utils)
imp.reload(EF)
imp.reload(CF)
imp.reload(GP)

# MULTIVARIATE GP!!

def gpfa_model(x_a, x_s, n_a, n_s, D):
    # This?
    for i in range(D):
        amp[i] = EF.Delta(name='amplitude-'+str(i))
        ls[i] = EF.Delta(name='lengthscale-'+str(i))
        cf[i] = CF.PiecewisePolynomial2(amp[i], ls[i])
    cf_a = CF.Multiple(cf)
    a = GP.GaussianProcess(0, cf_a, name='a')
    indices = np.arange(D*n_a).reshape((n_a,D))  # maybe some better way to express this?
    x_A = [x_a] * D
    A = GP.ProcessToVector(a, x_A, indices)
    # This?
    amp = EF.Delta(name='amplitude', plates=(D,))
    ls = EF.Delta(name='lengthscale', plates=(D,))
    cf = CF.PiecewisePolynomial2(amp, ls)
    cf_a = CF.Multiple(cf)
    a = GP.GaussianProcess(0, cf_a, name='a')
    A = GP.ProcessToVector(a, x_a)

def run():
    
    ## Generate data

    # Noisy observations from a sinusoid
    N = 100
    #func = lambda x: np.sin(x*2*np.pi/50)
    x = np.random.uniform(low=0, high=N, size=(N,))
    f = func(x)
    y = f + np.random.normal(0, 0.2, np.shape(f))

    # Plot data
    plt.clf()
    plt.plot(x,y,'r+')

    ## Construct model

    # Covariance function stuff
    ls = EF.NodeConstantScalar(3, name='lengthscale')
    amp = EF.NodeConstantScalar(2.0, name='amplitude')
    noise = EF.NodeConstantScalar(0.6, name='noise')
    # Latent process covariance
    #K_f = CF.SquaredExponential(amp, ls)
    K_f = CF.PiecewisePolynomial2(amp, ls)
    # Noise process covariance
    K_noise = CF.Delta(noise)
    # Observation process covariance
    K_y = CF.Sum(K_f, K_noise)
    # Joint covariance
    #K_joint = CF.Multiple([[K_f, K_f],[K_f,K_y]], sparse=True)

    # Mean function stuff
    M = GP.Constant(lambda x: (x/10-2)*(x/10+1))
    # Means for latent and observation processes
    #M_multi = GP.Multiple([M, M])

    # Gaussian process
    F = GP.GaussianProcess(M, [[K_f, K_f], [K_f, K_y]])
    #F = GP.GaussianProcess(M, [[K_f, K_f], [K_f, K_y]])
    #F = GP.GaussianProcess(M_multi, K_joint)

    ## Inference
    F.observe([[],x], y)
    utils.vb_optimize_nodes(ls, amp, noise)
    F.update()
    u = F.get_parameters()

    ## Show results

    # Print hyperparameters
    print('parameters')
    print(ls.name, ls.u[0])
    print(amp.name, amp.u[0])
    print(noise.name, noise.u[0])

    # Posterior predictions
    xh = np.arange(np.min(x)-5, np.max(x)+10, 0.1)
    (fh, varfh) = u([[],xh], covariance=1)
    #(fh, varfh) = u([xh,[]], covariance=1)

    # Plot predictive distribution
    varfh[varfh<0] = 0
    errfh = np.sqrt(varfh)
    myplt.errorplot(xh, fh, errfh, errfh)
    
    return


if __name__ == '__main__':
    run()

