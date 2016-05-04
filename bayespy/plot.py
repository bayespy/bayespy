################################################################################
# Copyright (C) 2011-2013 Jaakko Luttinen
#
# This file is licensed under the MIT License.
################################################################################


"""
Functions for plotting nodes.

Functions
=========

.. currentmodule:: bayespy.plot

.. autosummary::
   :toctree: generated/

   pdf
   contour
   plot
   hinton
   gaussian_mixture_2d

Plotters
========

.. autosummary::
   :toctree: generated/

   Plotter
   PDFPlotter
   ContourPlotter
   HintonPlotter
   FunctionPlotter
   GaussianTimeseriesPlotter
   CategoricalMarkovChainPlotter
"""


import os, sys

############################################################################
# A STUPID WORKAROUND FOR A MATPLOTLIB 1.4.0 BUG RELATED TO INTERACTIVE MODE
# See: https://github.com/matplotlib/matplotlib/issues/3505
import __main__
if hasattr(__main__, '__file__'):
    sys.ps1 = ('WORKAROUND FOR A BUG #3505 IN MATPLOTLIB.\n'
               'IF YOU SEE THIS MESSAGE, TRY MATPLOTLIB!=1.4.0.')
# This workaround does not work on Python shell, only on stand-alone scripts
# and IPython. A better solution: require MPL!=1.4.0.
#############################################################################

import numpy as np
import scipy.sparse as sp
import scipy
from scipy import special
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib import colors
#from matplotlib.pyplot import *

from bayespy.inference.vmp.nodes.categorical import CategoricalMoments
from bayespy.inference.vmp.nodes.gaussian import (GaussianMoments,
                                                  GaussianWishartMoments)
from bayespy.inference.vmp.nodes.beta import BetaMoments
from bayespy.inference.vmp.nodes.beta import DirichletMoments
from bayespy.inference.vmp.nodes.bernoulli import BernoulliMoments
from bayespy.inference.vmp.nodes.categorical import CategoricalMoments
from bayespy.inference.vmp.nodes.gamma import GammaMoments
from bayespy.inference.vmp.nodes.node import Node, Moments

from bayespy.utils import (misc,
                           random,
                           linalg)


# Users can use pyplot via this module
import matplotlib
mpl = matplotlib
pyplot = plt


def interactive(function):
    """A decorator for forcing functions to use the interactive mode.

    Parameters
    ----------

    function : callable
        The function to be decorated
    """

    def new_function(*args, **kwargs):
        if mpl.is_interactive():
            was_interactive = True
        else:
            was_interactive = False
            mpl.interactive(True)

        retval = function(*args, **kwargs)

        if not was_interactive:
            mpl.interactive(False)

        return retval

    return new_function


def _subplots(plotfunc, *args, fig=None, kwargs=None):
    """Create a collection of subplots

    Each subplot is created with the same plotting function.

    Inputs are given as pairs:

    (x, 3), (y, 2), ...

    where x,y,... are the input arrays and 3,2,... are the ndim
    parameters.  The last ndim axes of each array are interpreted as a
    single element to the plotting function.

    All high-level plotting functions should wrap low-level plotting
    functions with this function in order to generate subplots for
    plates.
    """

    if kwargs is None:
        kwargs = {}

    if fig is None:
        fig = plt.gcf()

    # Parse shape and plates of each input array
    shapes = [np.shape(x)[-n:] if n > 0 else ()
              for (x,n) in args]
    plates = [np.shape(x)[:-n] if n > 0 else np.shape(x)
              for (x,n) in args]

    # Get the full grid shape of the subplots
    broadcasted_plates = misc.broadcasted_shape(*plates)

    # Subplot indexing layout
    M = np.prod(broadcasted_plates[-2::-2])
    N = np.prod(broadcasted_plates[-1::-2])
    strides_subplot = [np.prod(broadcasted_plates[(j+2)::2]) * N
                       if ((len(broadcasted_plates)-j) % 2) == 0 else
                       np.prod(broadcasted_plates[(j+2)::2])
                       for j in range(len(broadcasted_plates))]

    # Plot each subplot
    for ind in misc.nested_iterator(broadcasted_plates):

        # Get the list of inputs for this subplot
        broadcasted_args = []
        for n in range(len(args)):
            i = misc.safe_indices(ind, plates[n])
            broadcasted_args.append(args[n][0][i])

        # Plot the subplot using the given function
        ind_subplot = np.einsum('i,i', ind, strides_subplot)
        axes = fig.add_subplot(M, N, ind_subplot+1)
        plotfunc(*broadcasted_args, axes=axes, **kwargs)


def pdf(Z, x, *args, name=None, axes=None, fig=None, **kwargs):
    """
    Plot probability density function of a scalar variable.

    Parameters
    ----------

    Z : node or function
        Stochastic node or log pdf function

    x : array
        Grid points
    """

    # TODO: Make it possible to plot a plated variable using _subplots function.

    if axes is None and fig is None:
        axes = plt.gca()
    else:
        if fig is None:
            fig = plt.gcf()
        axes = fig.add_subplot(111)

    try:
        lpdf = Z.logpdf(x)
    except AttributeError:
        lpdf = Z(x)
    p = np.exp(lpdf)
    retval = axes.plot(x, p, *args, **kwargs)

    if name is None:
        try:
            name = Z.name
        except AttributeError:
            pass

    if name:
        axes.set_title(r'$q(%s)$' % (name))
        axes.set_xlabel(r'$%s$' % (name))
        
    return retval


def contour(Z, x, y, n=None, axes=None, fig=None, **kwargs):
    """
    Plot 2-D probability density function of a 2-D variable.

    Parameters
    ----------

    Z : node or function
        Stochastic node or log pdf function

    x : array
        Grid points on x axis

    y : array
        Grid points on y axis
    """

    # TODO: Make it possible to plot a plated variable using _subplots function.

    if axes is None and fig is None:
        axes = plt.gca()
    else:
        if fig is None:
            fig = plt.gcf()
        axes = fig.add_subplot(111)

    XY = misc.grid(x, y)
    try:
        lpdf = Z.logpdf(XY)
    except AttributeError:
        lpdf = Z(XY)
    p = np.exp(lpdf)
    shape = (np.size(x), np.size(y))
    X = np.reshape(XY[:,0], shape)
    Y = np.reshape(XY[:,1], shape)
    P = np.reshape(p, shape)
    if n is not None:
        levels = np.linspace(0, np.amax(P), num=n+2)[1:-1]
        return axes.contour(X, Y, P, levels, **kwargs)
    else:
        return axes.contour(X, Y, P, **kwargs)
        


def plot_gaussian_mc(X, scale=2, **kwargs):
    """
    Plot Gaussian Markov chain as a 1-D function
    
    Parameters
    ----------
    X : node
        Node with Gaussian Markov chain moments.
    """
    timeseries_gaussian(X, axis=-2, scale=scale, **kwargs)


def plot_bernoulli(X, axis=-1, scale=2, **kwargs):
    """
    Plot Bernoulli node as a 1-D function
    """
    X = X._ensure_moments(X, BernoulliMoments)
    u_X = X.get_moments()
    z = u_X[0]
    return _timeseries_mean_and_error(z, None, axis=axis, **kwargs)


def plot_gaussian(X, axis=-1, scale=2, **kwargs):
    """
    Plot Gaussian node as a 1-D function
    
    Parameters
    ----------
    X : node
        Node with Gaussian moments.
    axis : int
        The index of the time axis.
    """
    X = X._ensure_moments(X, GaussianMoments, ndim=0)
    u_X = X.get_moments()
    x = u_X[0]
    xx = misc.get_diag(u_X[1], ndim=len(X.dims[0]))
    std = scale * np.sqrt(xx - x**2)
    #std = scale * np.sqrt(np.einsum('...ii->...i', xx) - x**2)
    
    return _timeseries_mean_and_error(x, std, axis=axis, **kwargs)


def plot(Y, axis=-1, scale=2, center=False, **kwargs):
    """
    Plot a variable or an array as 1-D function with errorbars
    """
    if misc.is_numeric(Y):
        return _timeseries_mean_and_error(Y, None, axis=axis, center=center, **kwargs)

    if isinstance(Y, Node):

        # Try Bernoulli plotting
        try:
            Y = Y._ensure_moments(Y, BernoulliMoments)
        except BernoulliMoments.NoConverterError:
            pass
        else:
            return plot_bernoulli(Y, axis=axis, scale=scale, center=center, **kwargs)

        # Try Gaussian plotting
        try:
            Y = Y._ensure_moments(Y, GaussianMoments, ndim=0)
        except GaussianMoments.NoConverterError:
            pass
        else:
            return plot_gaussian(Y, axis=axis, scale=scale, center=center, **kwargs)

    (mu, var) = Y.get_mean_and_variance()
    std = np.sqrt(var)
    
    return _timeseries_mean_and_error(mu, std, 
                                      axis=axis,
                                      scale=scale,
                                      center=center, 
                                      **kwargs)


# Some backward compatibility
def timeseries_gaussian_mc(*args, center=True, **kwargs):
    return plot_gaussian_mc(*args, center=center, **kwargs)
def timeseries_gaussian(*args, center=True, **kwargs):
    return plot_gaussian(*args, center=center, **kwargs)
timeseries_normal = timeseries_gaussian
def timeseries(*args, center=True, **kwargs):
    return plot(*args, center=center, **kwargs)


def _timeseries_mean_and_error(y, std, *args, axis=-1, center=True, fig=None, axes=None, **kwargs):
    # TODO/FIXME: You must multiply by ones(plates) in order to plot
    # broadcasted plates properly

    if fig is None:
        fig = plt.gcf()

    y = np.atleast_1d(y)
    shape = list(np.shape(y))

    # Get and remove the length of the time axis
    T = shape.pop(axis)

    # Move time axis to first
    y = np.rollaxis(y, axis)
    if std is not None:
        std = np.rollaxis(std, axis)
    
    y = np.reshape(y, (T, -1))
    if std is not None:
        std = np.reshape(std, (T, -1))

    # Remove 1s
    shape = [s for s in shape if s > 1]

    # Calculate number of rows and columns
    shape = misc.multiply_shapes(shape, (1,1))
    if len(shape) > 2:
        raise Exception("Can plot only in 2 dimensions (rows and columns)")
    (M, N) = shape

    # Prefer plotting to rows
    if M == 1:
        M = N
        N = 1

    # Plot each timeseries
    if axes is None:
        ax0 = fig.add_subplot(M, N, 1)
    for i in range(M*N):
        if axes is None:
            if i > 0:
                # Share x axis between all subplots
                ax = fig.add_subplot(M, N, i+1, sharex=ax0)
            else:
                ax = ax0

            # Autoscale the axes to data and use tight y and x axes
            ax.autoscale(enable=True, tight=True)
            ax.set_ylim(auto=True)

            if i < (M-1)*N:
                # Remove x tick labels from other than the last row
                plt.setp(ax.get_xticklabels(), visible=False)

        else:
            ax = axes[i]

        if std is None:
            errorplot(y=y[:,i], axes=ax, **kwargs)
        else:
            if len(args) > 0:
                raise Exception("Can't handle extra arguments")
            errorplot(y=y[:,i], error=std[:,i], axes=ax, **kwargs)

        if center:
            # Center the zero level on y-axis
            ylim = ax.get_ylim()
            vmax = np.max(np.abs(ylim))
            ax.set_ylim([-vmax, vmax])

    if axes is None:
        # Remove height space between subplots
        fig.subplots_adjust(hspace=0)


def _blob(axes, x, y, area, colour):
    """
    Draws a square-shaped blob with the given area (< 1) at
    the given coordinates.
    """
    hs = np.sqrt(area) / 2
    xcorners = np.array([x - hs, x + hs, x + hs, x - hs])
    ycorners = np.array([y - hs, y - hs, y + hs, y + hs])
    axes.fill(xcorners, ycorners, colour, edgecolor=colour)

def _rectangle(axes, x, y, width, height, **kwargs):
    _x = x - width/2
    _y = y - height/2
    rectangle = plt.Rectangle((_x, _y), 
                              width,
                              height,
                              **kwargs)
    axes.add_patch(rectangle)
    return


def gaussian_mixture_2d(X, alpha=None, scale=2, fill=False, axes=None, **kwargs):
    """
    Plot Gaussian mixture as ellipses in 2-D

    Parameters
    ----------

    X : Mixture node

    alpha : Dirichlet-like node (optional)
       Probabilities for the clusters

    scale : float (optional)
       Scale for the covariance ellipses (by default, 2)
    """
        
    if axes is None:
        axes = plt.gca()

    mu_Lambda = X._ensure_moments(X.parents[1], GaussianWishartMoments)

    (mu, _, Lambda, _) = mu_Lambda.get_moments()
    mu = np.linalg.solve(Lambda, mu)

    if len(mu_Lambda.plates) != 1:
        raise NotImplementedError("Not yet implemented for more plates")
    
    K = mu_Lambda.plates[0]

    width = np.zeros(K)
    height = np.zeros(K)
    angle = np.zeros(K)

    for k in range(K):
        m = mu[k]
        L = Lambda[k]
        (u, W) = scipy.linalg.eigh(L)
        u[0] = np.sqrt(1/u[0])
        u[1] = np.sqrt(1/u[1])
        width[k] = 2*u[0]
        height[k] = 2*u[1]
        angle[k] = np.arctan(W[0,1] / W[0,0])

    angle = 180 * angle / np.pi
    mode_height = 1 / (width * height)

    # Use cluster probabilities to adjust alpha channel
    if alpha is not None:
        # Compute the normalized probabilities in a numerically stable way
        logsum_p = misc.logsumexp(alpha.u[0], axis=-1, keepdims=True)
        logp = alpha.u[0] - logsum_p
        p = np.exp(logp)
        # Visibility is based on cluster mode peak height
        visibility = mode_height * p
        visibility /= np.amax(visibility)
    else:
        visibility = np.ones(K)

    for k in range(K):
        ell = mpl.patches.Ellipse(mu[k], scale*width[k], scale*height[k],
                                  angle=(180+angle[k]),
                                  fill=fill,
                                  alpha=visibility[k],
                                  **kwargs)
        axes.add_artist(ell)

    plt.axis('equal')

    # If observed, plot the data too
    if np.any(X.observed):
        mask = np.array(X.observed) * np.ones(X.plates, dtype=np.bool)
        y = X.u[0][mask]
        plt.plot(y[:,0], y[:,1], 'r.')

    return
    
    
def _hinton(W, error=None, vmax=None, square=False, axes=None):
    """
    Draws a Hinton diagram for visualizing a weight matrix. 

    Temporarily disables matplotlib interactive mode if it is on, 
    otherwise this takes forever.

    Originally copied from
    http://wiki.scipy.org/Cookbook/Matplotlib/HintonDiagrams
    """

    if axes is None:
        axes = plt.gca()

    W = misc.atleast_nd(W, 2)
    (height, width) = W.shape
    if not vmax:
        #vmax = 2**np.ceil(np.log(np.max(np.abs(W)))/np.log(2))
        if error is not None:
            vmax = np.max(np.abs(W) + error)
        else:
            vmax = np.max(np.abs(W))

    axes.fill(0.5+np.array([0,width,width,0]),
              0.5+np.array([0,0,height,height]),
              'gray')
    if square:
        axes.set_aspect('equal')
    axes.set_ylim(0.5, height+0.5)
    axes.set_xlim(0.5, width+0.5)
    axes.set_xticks([])
    axes.set_yticks([])
    axes.invert_yaxis()
    for x in range(width):
        for y in range(height):
            _x = x+1
            _y = y+1
            w = W[y,x]
            _w = np.abs(w)
            if w > 0:
                _c = 'white'
            else:
                _c = 'black'
            if error is not None:
                e = error[y,x]
                if e < 0:
                    print(e, _w, vmax)
                    raise Exception("BUG? Negative error")
                if _w + e > vmax:
                    print(e, _w, vmax)
                    raise Exception("BUG? Value+error greater than max")
                _rectangle(axes,
                           _x,
                           _y, 
                           min(1, np.sqrt((_w+e)/vmax)),
                           min(1, np.sqrt((_w+e)/vmax)),
                           edgecolor=_c,
                           fill=False)
            _blob(axes, _x, _y, min(1, _w/vmax), _c)
                

def matrix(A, axes=None):

    if axes is None:
        axes = plt.gca()

    A = np.atleast_2d(A)
    vmax = np.max(np.abs(A))
    return  axes.imshow(A, 
                        interpolation='nearest', 
                        cmap='RdBu_r',
                        vmin=-vmax,
                        vmax=vmax)

def new_matrix(A, vmax=None):
    A = np.atleast_2d(A)
    if vmax is None:
        vmax = np.max(np.abs(A))

    (M, N) = np.shape(A)

    for i in range(M):
        for j in range(N):
            pass
    
def gaussian_hinton(X, rows=None, cols=None, scale=1, fig=None):
    """
    Plot the Hinton diagram of a Gaussian node
    """

    if fig is None:
        fig = plt.gcf()

    # Get mean and second moment
    X = X._ensure_moments(X, GaussianMoments, ndim=0)
    (x, xx) = X.get_moments()
    ndim = len(X.dims[0])
    shape = X.get_shape(0)
    size = len(X.get_shape(0))

    # Compute standard deviation
    xx = misc.get_diag(xx, ndim=ndim)
    std = np.sqrt(xx - x**2)

    # Force explicit elements when broadcasting
    x = x * np.ones(shape)
    std = std * np.ones(shape)

    if rows is None:
        rows = np.nan
    if cols is None:
        cols = np.nan

    # Preprocess the axes to 0,...,ndim
    if rows < 0:
        rows += size
    if cols < 0:
        cols += size
    if rows < 0 or rows >= size:
        raise ValueError("Row axis invalid")
    if cols < 0 or cols >= size:
        raise ValueError("Column axis invalid")

    # Remove non-row and non-column axes that have length 1
    squeezed_shape = list(shape)
    for i in reversed(range(len(shape))):
        if shape[i] == 1 and i != rows and i != cols:
            squeezed_shape.pop(i)
            if i < cols:
                cols -= 1
            if i < rows:
                rows -= 1
    x = np.reshape(x, squeezed_shape)
    std = np.reshape(std, squeezed_shape)

    if np.ndim(x) < 2:
        cols += 2 - np.ndim(x)
        rows += 2 - np.ndim(x)
        x = np.atleast_2d(x)
        std = np.atleast_2d(std)

    size = np.ndim(x)
    if np.isnan(cols):
        if rows != size - 1:
            cols = size - 1
        else:
            cols = size - 2
    if np.isnan(rows):
        if cols != size - 1:
            rows = size - 1
        else:
            rows = size - 2

    # Put the row and column axes to the end
    axes = [i for i in range(size) if i not in (rows, cols)] + [rows, cols]
    x = np.transpose(x, axes=axes)
    std = np.transpose(std, axes=axes)

    vmax = np.max(np.abs(x) + scale*std)
    
    if scale == 0:
        _subplots(_hinton, (x, 2), fig=fig, kwargs=dict(vmax=vmax))
    else:
        def plotfunc(z, e, **kwargs):
            return _hinton(z, error=e, **kwargs)

        _subplots(plotfunc, (x, 2), (scale*std, 2), fig=fig, kwargs=dict(vmax=vmax))


def _hinton_figure(x, rows=None, cols=None, fig=None, square=True):
    """
    Plot the Hinton diagram of a Gaussian node
    """

    scale = 0
    std = 0

    if fig is None:
        fig = plt.gcf()

    # Get mean and second moment
    shape = np.shape(x)
    size = np.ndim(x)

    if rows is None:
        rows = np.nan
    if cols is None:
        cols = np.nan

    # Preprocess the axes to 0,...,ndim
    if rows < 0:
        rows += size
    if cols < 0:
        cols += size
    if rows < 0 or rows >= size:
        raise ValueError("Row axis invalid")
    if cols < 0 or cols >= size:
        raise ValueError("Column axis invalid")

    # Remove non-row and non-column axes that have length 1
    squeezed_shape = list(shape)
    for i in reversed(range(len(shape))):
        if shape[i] == 1 and i != rows and i != cols:
            squeezed_shape.pop(i)
            if i < cols:
                cols -= 1
            if i < rows:
                rows -= 1
    x = np.reshape(x, squeezed_shape)

    size = np.ndim(x)
    if np.isnan(cols):
        if rows != size - 1:
            cols = size - 1
        else:
            cols = size - 2
    if np.isnan(rows):
        if cols != size - 1:
            rows = size - 1
        else:
            rows = size - 2

    # Put the row and column axes to the end
    if np.ndim(x) >= 2:
        axes = [i for i in range(size) if i not in (rows, cols)] + [rows, cols]
        x = np.transpose(x, axes=axes)
        #std = np.transpose(std, axes=axes)

    vmax = np.max(np.abs(x) + scale*std)

    kw = dict(vmax=vmax, square=square)
    if scale == 0:
        _subplots(_hinton, (x, 2), fig=fig, kwargs=kw)
    else:
        def plotfunc(z, e, **kwargs):
            return _hinton(z, error=e, **kwargs)

        _subplots(plotfunc, (x, 2), (scale*std, 2), fig=fig, kwargs=kw)


# For backwards compatibility:
gaussian_array = gaussian_hinton


def timeseries_categorical_mc(Z, fig=None):

    if fig is None:
        fig = plt.gcf()

    # Make sure that the node is categorical
    Z = Z._ensure_moments(Z, CategoricalMoments, categories=None)

    # Get expectations (and broadcast explicitly)
    z = Z._message_to_child()[0] * np.ones(Z.get_shape(0))

    # Compute the subplot layout
    z = misc.atleast_nd(z, 4)
    if np.ndim(z) != 4:
        raise ValueError("Can not plot arrays with over 4 axes")
    M = np.shape(z)[0]
    N = np.shape(z)[1]

    # Plot Hintons
    for i in range(M):
        for j in range(N):
            axes = fig.add_subplot(M, N, i*N+j+1)
            _hinton(z[i,j].T, vmax=1.0, square=False, axes=axes)


def gamma_hinton(alpha, square=True, **kwargs):
    """
    Plot a beta distributed random variable as a Hinton diagram
    """

    # Make sure that the node is beta
    alpha = alpha._ensure_moments(alpha, GammaMoments)

    # Compute exp( <log p> )
    x = alpha.get_moments()[0]

    # Explicit broadcasting
    x = x * np.ones(alpha.plates)

    # Plot Hinton diagram
    return _hinton_figure(x, square=square, **kwargs)


def beta_hinton(P, square=True):
    """
    Plot a beta distributed random variable as a Hinton diagram
    """

    # Make sure that the node is beta
    P = P._ensure_moments(P, BetaMoments)

    # Compute exp( <log p> )
    p = np.exp(P._message_to_child()[0][...,0])

    # Explicit broadcasting
    p = p * np.ones(P.plates)

    # Plot Hinton diagram
    return _hinton(p, vmax=1.0, square=square)
    

def dirichlet_hinton(P, square=True):
    """
    Plot a beta distributed random variable as a Hinton diagram
    """

    # Make sure that the node is beta
    P = P._ensure_moments(P, DirichletMoments)

    # Compute exp( <log p> )
    p = np.exp(P._message_to_child()[0])

    # Explicit broadcasting
    p = p * np.ones(P.plates+(1,))

    # Plot Hinton diagram
    return _hinton(p, vmax=1.0, square=square)

    
def bernoulli_hinton(Z, square=True):
    """
    Plot a Bernoulli distributed random variable as a Hinton diagram
    """

    # Make sure that the node is Bernoulli
    Z = Z._ensure_moments(Z, BernoulliMoments)

    # Get <Z>
    z = Z._message_to_child()[0]

    # Explicit broadcasting
    z = z * np.ones(Z.plates)

    # Plot Hinton diagram
    return _hinton(z, vmax=1.0, square=square)
    

def categorical_hinton(Z, square=True):
    """
    Plot a Bernoulli distributed random variable as a Hinton diagram
    """

    # Make sure that the node is Bernoulli
    Z = Z._ensure_moments(Z, CategoricalMoments, categories=None)

    # Get <Z>
    z = Z._message_to_child()[0]

    # Explicit broadcasting
    z = z * np.ones(Z.plates+(1,))

    # Plot Hinton diagram
    return _hinton(np.squeeze(z), vmax=1.0, square=square)
    

def hinton(X, **kwargs):
    r"""
    Plot the Hinton diagram of a node

    The keyword arguments depend on the node type.  For some node types, the
    diagram also shows uncertainty with non-filled rectangles.  Currently,
    beta-like, Gaussian-like and Dirichlet-like nodes are supported.

    Parameters
    ----------

    X : node

    """

    if hasattr(X, "_ensure_moments"):

        try:
            X = X._ensure_moments(X, GaussianMoments, ndim=0)
        except Moments.NoConverterError:
            pass
        else:
            return gaussian_hinton(X, **kwargs)

        try:
            X = X._ensure_moments(X, GammaMoments)
        except Moments.NoConverterError:
            pass
        else:
            return gamma_hinton(X, **kwargs)

        try:
            X = X._ensure_moments(X, BetaMoments)
        except Moments.NoConverterError:
            pass
        else:
            return beta_hinton(X, **kwargs)

        try:
            X = X._ensure_moments(X, DirichletMoments)
        except Moments.NoConverterError:
            pass
        else:
            return dirichlet_hinton(X, **kwargs)

        try:
            X = X._ensure_moments(X, BernoulliMoments)
        except Moments.NoConverterError:
            pass
        else:
            return bernoulli_hinton(X, **kwargs)

        try:
            X = X._ensure_moments(X, CategoricalMoments, categories=None)
        except Moments.NoConverterError:
            pass
        else:
            return categorical_hinton(X, **kwargs)

    return _hinton_figure(X, **kwargs)
    

class Plotter():
    r"""
    Wrapper for plotting functions and base class for node plotters

    The purpose of this class is to collect all the parameters needed by a
    plotting function and provide a callable interface which needs only the node
    as the input.
    
    Plotter instances are callable objects that plot a given node using a
    specified plotting function.

    Parameters
    ----------

    plotter : function

        Plotting function to use

    args : defined by the plotting function

        Additional inputs needed by the plotting function

    kwargs : defined by the plotting function

        Additional keyword arguments supported by the plotting function

    Examples
    --------

    First, create a gamma variable:
    
    >>> import numpy as np
    >>> from bayespy.nodes import Gamma
    >>> x = Gamma(4, 5)

    The probability density function can be plotted as:

    >>> import bayespy.plot as bpplt
    >>> bpplt.pdf(x, np.linspace(0.1, 10, num=100))         # doctest: +ELLIPSIS
    [<matplotlib.lines.Line2D object at 0x...>]

    However, this can be problematic when one needs to provide a
    plotting function for the inference engine as the inference engine
    gives only the node as input.  Thus, we need to create a simple
    plotter wrapper:
    
    >>> p = bpplt.Plotter(bpplt.pdf, np.linspace(0.1, 10, num=100))

    Now, this callable object ``p`` needs only the node as the input:

    >>> p(x)                                                # doctest: +ELLIPSIS
    [<matplotlib.lines.Line2D object at 0x...>]

    Thus, it can be given to the inference engine to use as a plotting function:

    >>> x = Gamma(4, 5, plotter=p)
    >>> x.plot()                                            # doctest: +ELLIPSIS
    [<matplotlib.lines.Line2D object at 0x...>]
    """
    

    def __init__(self, plotter, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs
        self._plotter = plotter


    def __call__(self, X, fig=None, **kwargs):
        """
        Plot the node using the specified plotting function

        Parameters
        ----------

        X : node

            The plotted node
        """
        kwargs_all = self._kwargs.copy()
        kwargs_all.update(kwargs)
        return self._plotter(X, *self._args, fig=fig, **kwargs_all)

        
class PDFPlotter(Plotter):
    r"""
    Plotter of probability density function of a scalar node

    Parameters
    ----------

    x_grid : array

        Numerical grid on which the density function is computed and
        plotted

    See also
    --------
    pdf
    """
    def __init__(self, x_grid, **kwargs):
        super().__init__(pdf, x_grid, **kwargs)


class ContourPlotter(Plotter):
    r"""
    Plotter of probability density function of a two-dimensional node

    Parameters
    ----------

    x1_grid : array

        Grid for the first dimension

    x2_grid : array

        Grid for the second dimension

    See also
    --------
    contour
    """
    def __init__(self, x1_grid, x2_grid, **kwargs):
        super().__init__(contour, x1_grid, x2_grid, **kwargs)


class HintonPlotter(Plotter):
    r"""
    Plotter of the Hinton diagram of a node

    See also
    --------
    hinton
    """
    def __init__(self, **kwargs):
        super().__init__(hinton, **kwargs)


class FunctionPlotter(Plotter):
    r"""
    Plotter of a node as a 1-dimensional function

    See also
    --------
    plot
    """
    def __init__(self, **kwargs):
        super().__init__(plot, **kwargs)


class GaussianMarkovChainPlotter(Plotter):
    r"""
    Plotter of a Gaussian Markov chain as a timeseries
    """
    def __init__(self, **kwargs):
        super().__init__(timeseries_gaussian_mc, **kwargs)


class GaussianTimeseriesPlotter(Plotter):
    r"""
    Plotter of a Gaussian node as a timeseries
    """
    def __init__(self, **kwargs):
        super().__init__(timeseries_gaussian, **kwargs)


class GaussianHintonPlotter(Plotter):
    r"""
    Plotter of a Gaussian node as a Hinton diagram
    """
    def __init__(self, **kwargs):
        super().__init__(gaussian_array, **kwargs)


class CategoricalMarkovChainPlotter(Plotter):
    r"""
    Plotter of a Categorical timeseries
    """
    def __init__(self, **kwargs):
        super().__init__(timeseries_categorical_mc, **kwargs)


def matrix_animation(A, filename=None, fps=25, fig=None, **kwargs):

    if fig is None:
        fig = plt.gcf()

    axes = fig.add_subplot(111)

    A = np.atleast_3d(A)
    vmax = np.max(np.abs(A))
    x = axes.imshow(A[0],
                    interpolation='nearest',
                    cmap='RdBu_r',
                    vmin=-vmax,
                    vmax=vmax,
                    **kwargs)
    s = axes.set_title('t = %d' % 0)

    def animate(nframe):
        s.set_text('t = %d' % nframe)
        x.set_array(A[nframe])
        return (x, s)

    anim = animation.FuncAnimation(fig, animate,
                                   frames=np.shape(A)[0],
                                   interval=1000/fps,
                                   blit=False,
                                   repeat=False)
        
    return anim


def save_animation(anim, filename, fps=25, bitrate=5000, fig=None):

    # A bug in numpy/matplotlib causes this not to work in python3.3:
    # https://github.com/matplotlib/matplotlib/issues/1891
    #
    # So the following command does not work currently..
    #
    # anim.save(filename, fps=fps)

    if fig is None:
        fig = plt.gcf()

    writer = animation.FFMpegFileWriter(fps=fps, bitrate=bitrate)
    writer.setup(fig, filename, 100)
    anim.save(filename, 
              fps=fps,
              writer=writer,
              bitrate=bitrate)
    return


def binary_matrix(A, axes=None):
    if axes is None:
        axes = plt.gca()

    A = np.atleast_2d(A)
    G = np.zeros(np.shape(A) + (3,))
    G[A] = [0,0,0]
    G[np.logical_not(A)] = [1,1,1]
    axes.imshow(G, interpolation='nearest')


def gaussian_mixture_logpdf(x, w, mu, Sigma):
    # Shape(x)      = (N, D)
    # Shape(w)      = (K,)
    # Shape(mu)     = (K, D)
    # Shape(Sigma)  = (K, D, D)
    # Shape(result) = (N,)

    # Dimensionality
    D = np.shape(x)[-1]

    # Cholesky decomposition of the covariance matrix
    U = linalg.chol(Sigma)

    # Reshape x:
    # Shape(x)     = (N, 1, D)
    x = np.expand_dims(x, axis=-2)

    # (x-mu) and (x-mu)'*inv(Sigma)*(x-mu):
    # Shape(v)     = (N, K, D)
    # Shape(z)     = (N, K)
    v = x - mu
    z = np.einsum('...i,...i', v, linalg.chol_solve(U, v))

    # Log-determinant of Sigma:
    # Shape(ldet)  = (K,)
    ldet = linalg.chol_logdet(U)

    # Compute log pdf for each cluster:
    # Shape(lpdf)  = (N, K)
    lpdf = misc.gaussian_logpdf(z, 0, 0, ldet, D)
    
    

def matrixplot(A, colorbar=False, axes=None):
    if axes is None:
        axes = plt.gca()

    if sp.issparse(A):
        A = A.toarray()
    axes.imshow(A, interpolation='nearest')
    if colorbar:
        plt.colorbar(ax=axes)


def contourplot(x1, x2, y, colorbar=False, filled=True, axes=None):
    """ Plots 2D contour plot. x1 and x2 are 1D vectors, y contains
    the function values. y.size must be x1.size*x2.size. """
    
    if axes is None:
        axes = plt.gca()

    y = np.reshape(y, (len(x2),len(x1)))
    if filled:
        axes.contourf(x1, x2, y)
    else:
        axes.contour(x1, x2, y)
    if colorbar:
        plt.colorbar(ax=axes)
        

def errorplot(y=None, error=None, x=None, lower=None, upper=None,
              color=(0,0,0,1), fillcolor=None, axes=None, **kwargs):

    if axes is None:
        axes = plt.gca()

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
        l = y - lower
        u = y + upper
        if fillcolor is None:
            color = colors.ColorConverter().to_rgba(color)
            fillcolor = tuple(color[:3]) + (0.2 * color[3],)
        axes.fill_between(x,
                          l,
                          u,
                          facecolor=fillcolor,
                          edgecolor=(0, 0, 0, 0),
                          linewidth=1,
                          interpolate=True)
    # Plot function
    axes.plot(x, y, color=color, **kwargs)


def plotmatrix(X):
    """
    Creates a matrix of marginal plots.

    On diagonal, are marginal plots of each variable. Off-diagonal plot (i,j)
    shows the joint marginal density of x_i and x_j.
    """
    return X.plotmatrix()

    
def _pdf_t(mu, s2, nu, axes=None, scale=4, color='k'):
    """
    """
    if axes is None:
        axes = plt.gca()

    s = np.sqrt(s2)
    x = np.linspace(mu-scale*s, mu+scale*s, num=100)
    y2 = (x-mu)**2 / s2
    lpdf = random.t_logpdf(y2, np.log(s2), nu, 1)
    p = np.exp(lpdf)
    return axes.plot(x, p, color=color)


def _pdf_gamma(a, b, axes=None, scale=4, color='k'):
    """
    """
    if axes is None:
        axes = plt.gca()

    if np.size(a) != 1 or np.size(b) != 1:
        raise ValueError("Parameters must be scalars")
    mean = a/b
    v = scale*np.sqrt(a/b**2)
    m = max(0, mean-v)
    n = mean + v
    x = np.linspace(m, n, num=100)
    logx = np.log(x)
    lpdf = random.gamma_logpdf(b*x,
                               logx,
                               a*logx,
                               a*np.log(b),
                               special.gammaln(a))
    p = np.exp(lpdf)
    return axes.plot(x, p, color=color)


def _contour_t(mu, Cov, nu, axes=None, scale=4, transpose=False, colors='k'):
    """
    """
    if axes is None:
        axes = plt.gca()

    if np.shape(mu) != (2,) or np.shape(Cov) != (2,2) or np.shape(nu) != ():
        print(np.shape(mu), np.shape(Cov), np.shape(nu))
        raise ValueError("Only 2-d t-distribution allowed")
    
    if transpose:
        mu = mu[[1,0]]
        Cov = Cov[np.ix_([1,0],[1,0])]

    s = np.sqrt(np.diag(Cov))
    x0 = np.linspace(mu[0]-scale*s[0], mu[0]+scale*s[0], num=100)
    x1 = np.linspace(mu[1]-scale*s[1], mu[1]+scale*s[1], num=100)
    X0X1 = misc.grid(x0, x1)
    Y = X0X1 - mu
    L = linalg.chol(Cov)
    logdet_Cov = linalg.chol_logdet(L)
    Z = linalg.chol_solve(L, Y)
    Z = linalg.inner(Y, Z, ndim=1)
    lpdf = random.t_logpdf(Z, logdet_Cov, nu, 2)
    p = np.exp(lpdf)
    shape = (np.size(x0), np.size(x1))
    X0 = np.reshape(X0X1[:,0], shape)
    X1 = np.reshape(X0X1[:,1], shape)
    P = np.reshape(p, shape)
    return axes.contour(X0, X1, P, colors=colors)


def _contour_gaussian_gamma(mu, s2, a, b, axes=None, transpose=False):
    """
    """
    pass
