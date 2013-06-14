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
import scipy.sparse as sp
import matplotlib.pyplot as plt
from matplotlib import animation
#from matplotlib.pyplot import *

from bayespy import utils

def timeseries_gaussian(X, axis=-1):
    """
    Parameters:
    X : node
        Node with Gaussian moments.
    axis : int
        The index of the time axis.
    """
    u_X = X.get_moments()
    x = u_X[0]
    xx = u_X[1]
    std = np.sqrt(np.einsum('...ii->...i', xx) - x**2)
    
    _timeseries_mean_and_std(x, std, axis=axis)
    
def timeseries_normal(X, axis=-1):
    """
    Parameters:
    X : node
        Node with Gaussian moments.
    axis : int
        The index of the time axis.
    """
    u_X = X.get_moments()
    x = u_X[0]
    xx = u_X[1]
    std = np.sqrt(xx - x**2)
    _timeseries_mean_and_std(x, std, axis=axis)


def timeseries(x, axis=-1):
    return _timeseries_mean_and_std(x, 0*x, axis=axis)

def _timeseries_mean_and_std(y, std, axis=-1, **kwargs):
    # TODO/FIXME: You must multiply by ones(plates) in order to plot
    # broadcasted plates properly
    
    shape = list(np.shape(y))

    # Get and remove the length of the time axis
    T = shape.pop(axis)

    # Move time axis to first
    y = np.rollaxis(y, axis)
    std = np.rollaxis(std, axis)
    
    y = np.reshape(y, (T, -1))
    std = np.reshape(std, (T, -1))

    # Remove 1s
    shape = [s for s in shape if s > 1]

    # Calculate number of rows and columns
    shape = utils.utils.multiply_shapes(shape, (1,1))
    if len(shape) > 2:
        raise Exception("Can plot only in 2 dimensions (rows and columns)")
    (M, N) = shape

    # Prefer plotting to rows
    if M == 1:
        M = N
        N = 1

    # Plot each timeseries
    #M = 2
    for i in range(M*N):
        plt.subplot(M,N,i+1)
        errorplot(y=y[:,i], error=2*std[:,i], **kwargs)

def matrix(A):
    A = np.atleast_2d(A)
    vmax = np.max(np.abs(A))
    return  plt.imshow(A, 
                       interpolation='nearest', 
                       cmap='RdBu_r',
                       vmin=-vmax,
                       vmax=vmax)

def matrix_animation(A, **kwargs):

    A = np.atleast_3d(A)
    vmax = np.max(np.abs(A))
    x = plt.imshow(A[0],
                   interpolation='nearest',
                   cmap='RdBu_r',
                   vmin=-vmax,
                   vmax=vmax,
                   **kwargs)
    s = plt.title('t = %d' % 0)

    for (t, a) in enumerate(A):
        x.set_array(a)
        s.set_text('t = %d' % t)
        plt.draw()

def matrix_movie(A, filename, 
                 fps=25,
                 title='Matrix Movie',
                 artist='BayesPy',
                 dpi=100,
                 **kwargs):

    FFMpegWriter = animation.writers['ffmpeg']
    metadata = dict(title=title,
                    artist=artist)
    writer = FFMpegWriter(fps=fps, metadata=metadata)        
        
    A = np.atleast_3d(A)
    vmax = np.max(np.abs(A))
    x = plt.imshow(A[0],
                   interpolation='nearest',
                   cmap='RdBu_r',
                   vmin=-vmax,
                   vmax=vmax,
                   **kwargs)
    s = plt.title('t = %d' % 0)
    fig = plt.gcf()
    
    with writer.saving(fig, filename, dpi):
        for (t, a) in enumerate(A):
            x.set_array(a)
            s.set_text('t = %d' % t)
            plt.draw()
            writer.grab_frame()

def binary_matrix(A):
    A = np.atleast_2d(A)
    G = np.zeros(np.shape(A) + (3,))
    G[A] = [0,0,0]
    G[np.logical_not(A)] = [1,1,1]
    plt.imshow(G, interpolation='nearest')

def gaussian_mixture(w, mu, Sigma):
    pass

def gaussian_mixture_logpdf(x, w, mu, Sigma):
    # Shape(x)      = (N, D)
    # Shape(w)      = (K,)
    # Shape(mu)     = (K, D)
    # Shape(Sigma)  = (K, D, D)
    # Shape(result) = (N,)

    # Dimensionality
    D = np.shape(x)[-1]

    # Cholesky decomposition of the covariance matrix
    U = utils.m_chol(Sigma)

    # Reshape x:
    # Shape(x)     = (N, 1, D)
    x = np.expand_dims(x, axis=-2)

    # (x-mu) and (x-mu)'*inv(Sigma)*(x-mu):
    # Shape(v)     = (N, K, D)
    # Shape(z)     = (N, K)
    v = x - mu
    z = np.einsum('...i,...i', v, utils.m_chol_solve(U, v))

    # Log-determinant of Sigma:
    # Shape(ldet)  = (K,)
    ldet = utils.m_chol_logdet(U)

    # Compute log pdf for each cluster:
    # Shape(lpdf)  = (N, K)
    lpdf = utils.gaussian_logpdf(z, 0, 0, ldet, D)
    
    

def matrixplot(A, colorbar=False):
    if sp.issparse(A):
        A = A.toarray()
    plt.imshow(A, interpolation='nearest')
    if colorbar:
        plt.colorbar()


def contourplot(x1, x2, y, colorbar=False, filled=True):
    """ Plots 2D contour plot. x1 and x2 are 1D vectors, y contains
    the function values. y.size must be x1.size*x2.size. """
    
    y = np.reshape(y, (len(x2),len(x1)))
    if filled:
        plt.contourf(x1, x2, y)
    else:
        plt.contour(x1, x2, y)
    if colorbar:
        plt.colorbar()
        

def errorplot(y=None, error=None, x=None, lower=None, upper=None):

    # Default inputs
    if x is None:
        x = np.arange(np.size(y))

    # Parse errors (lower=lower/error/upper, upper=upper/error/lower)
    if lower is None:
        if error is not None:
            lower = error
        elif upper is not None:
            lower = upper
    if upper is None:
        if error is not None:
            upper = error
        elif lower is not None:
            upper = lower

    # Plot errors
    if (lower is not None) and (upper is not None):
        #print(np.max(lower))
        #print(np.max(upper))
        l = y - lower
        u = y + upper
        plt.fill_between(x,
                         u,
                         l,
                         facecolor=(0.6,0.6,0.6,1),
                         edgecolor=(0,0,0,0),
                         linewidth=0,
                         interpolate=True)
    # Plot function
    plt.plot(x, y, color=(0,0,0,1))

#def multiplot(plot_function, *args, **kwargs):
    

def m_plot(x, Y, style):
    Y = np.atleast_2d(Y)
    M = Y.shape[-2]
    for i in range(M):
        plt.subplot(M,1,i+1)
        plt.plot(x, Y[i], style)

## def multi_errorplot(Y, error=None, x=None, lower=None, upper=None):

##     for m in range(M):
##         for n in range(N):
##             plt.subplot(M,N,m*N+n)
##             errorplot(Y[m][n],
##                       error=error[m][n],
##                       x=x[m][n],
##                       lower=lower[m][n],
##                       upper=upper[m][n])

def m_errorplot(x, Y, L, U):
    Y = np.atleast_2d(Y)
    L = np.atleast_2d(L)
    U = np.atleast_2d(U)
    M = Y.shape[-2]
    ## print(np.shape(Y))
    ## print(np.shape(L))
    ## print(np.shape(U))
    ## print(np.shape(M))
    for i in range(M):
        plt.subplot(M,1,i+1)
        lower = Y[i] - L[i]
        upper = Y[i] + U[i]
        #print(upper-lower)
        #if np.any(lower>=upper):
            #print('WTF?!')
        plt.fill_between(x,
                         upper,
                         lower,
                         #where=(upper>=lower),
                         facecolor=(0.6,0.6,0.6,1),
                         edgecolor=(0,0,0,0),
                         #edgecolor=(0.6,0.6,0.6,1),
                         linewidth=0,
                         interpolate=True)
        plt.plot(x, Y[i], color=(0,0,0,1))
        plt.ylabel(str(i))

