######################################################################
# Copyright (C) 2013 Jaakko Luttinen
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

from bayespy.utils import utils

import bayespy.plot.plotting as bpplt

import imp
imp.reload(utils)
imp.reload(bpplt)

def run():
    # Create some data
    N = 500
    D = 2
    # Initial state
    x0 = np.array([0.5, -0.5])
    # Dynamics (time varying)
    A0 = np.array([[.9, -.4], [.4, .9]])
    A1 = np.array([[.98, -.1], [.1, .98]])
    l = np.linspace(0, 1, N).reshape((-1,1,1))
    A = (1-l)*A0 + l*A1
    # Innovation covariance matrix
    V = np.array([[1, 0], [0, 1]])
    # Observation noise covariance matrix
    C0 = 10*np.array([[1, 0], [0, 1]])
    C1 = 0.01*np.array([[1, 0], [0, 1]])
    C = (1-l)**2*C0 + l**2*C1

    X = np.empty((N,D))
    Y = np.empty((N,D))

    # Simulate data
    x = x0
    for n in range(N):
        x = np.dot(A[n,:,:],x) + np.random.multivariate_normal(np.zeros(D), V)
        X[n,:] = x
        Y[n,:] = x + np.random.multivariate_normal(np.zeros(D), C[n,:,:])

    U = np.empty((N,D,D))
    UY = np.empty((N,D))
    for n in range(N):
        U[n,:,:] = np.linalg.inv(C[n,:,:])
        UY[n,:] = np.linalg.solve(C[n,:,:], Y[n,:])

    # Create iterators for the static matrices
    #U = N*(U,)
    #A = N*(A,)
    V = N*(V,)
    
    (Xh, CovXh) = utils.kalman_filter(UY, U, A, V, np.zeros(2), 10*np.identity(2))
    (Xh, CovXh) = utils.rts_smoother(Xh, CovXh, A, V)
    
    plt.clf()
    for d in range(D):
        plt.subplot(D,1,d)
        #plt.plot(Xh[:,d], 'r--')
        bpplt.errorplot(Xh[:,d], error=2*np.sqrt(CovXh[:,d,d]))
        plt.plot(X[:,d], 'r-')
        plt.plot(Y[:,d], '.')
    plt.show()
    
if __name__ == '__main__':
    # FOR INTERACTIVE SESSIONS, NON-BLOCKING PLOTTING:
    #plt.ion()
    run()
