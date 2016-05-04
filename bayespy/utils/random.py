################################################################################
# Copyright (C) 2013 Jaakko Luttinen
#
# This file is licensed under the MIT License.
################################################################################


r"""
General functions random sampling and distributions.
"""

import numpy as np
from scipy import special

from . import linalg
from . import misc


def intervals(N, length, amount=1, gap=0):
    r"""
    Return random non-overlapping parts of a sequence.

    For instance, N=16, length=2 and amount=4:
    [0, |1, 2|, 3, 4, 5, |6, 7|, 8, 9, |10, 11|, |12, 13|, 14, 15]
    that is,
    [1,2,6,7,10,11,12,13]

    However, the function returns only the indices of the beginning of the
    sequences, that is, in the example:
    [1,6,10,12]
    """

    if length * amount + gap * (amount-1) > N:
        raise ValueError("Too short sequence")

    # In practice, we draw the sizes of the gaps between the sequences
    total_gap = N - length*amount - gap*(amount-1)
    gaps = np.random.multinomial(total_gap, np.ones(amount+1)/(amount+1))

    # And then we get the beginning index of each sequence
    intervals = np.cumsum(gaps[:-1]) + np.arange(amount)*(length+gap)

    return intervals

def mask(*shape, p=0.5):
    r"""
    Return a boolean array of the given shape.

    Parameters
    ----------
    d0, d1, ..., dn : int
        Shape of the output.
    p : value in range [0,1]
        A probability that the elements are `True`.
    """
    return np.random.rand(*shape) < p

def wishart(nu, V):
    r"""
    Draw a random sample from the Wishart distribution.

    Parameters
    ----------
    nu : int
    """
    # TODO/FIXME: Are these correct..
    D = np.shape(V)[0]
    if nu < D:
        raise ValueError("Degrees of freedom must be equal or greater than the "
                         "dimensionality of the matrix.")
    X = np.random.multivariate_normal(np.zeros(D), V, size=nu)
    return np.dot(X, X.T)

wishart_rand = wishart

def invwishart_rand(nu, V):
    # TODO/FIXME: Are these correct..
    return np.linalg.inv(wishart_rand(nu, V))

def covariance(D, size=(), nu=None):
    r"""
    Draw a random covariance matrix.

    Draws from inverse-Wishart distribution. The distribution of each element is
    independent of the dimensionality of the matrix.

    C ~ Inv-W(I, D)

    Parameters
    ----------
    D : int
        Dimensionality of the covariance matrix.

    Returns:
    --------
    C : (D,D) ndarray
        Positive-definite symmetric :math:`D\times D` matrix.
    """

    if nu is None:
        nu = D

    if nu < D:
        raise ValueError("nu must be greater than or equal to D")

    try:
        size = tuple(size)
    except TypeError:
        size = (size,)
    shape = size + (D,nu)
    C = np.random.randn(*shape)
    C = linalg.dot(C, np.swapaxes(C, -1, -2)) / nu
    return linalg.inv(C)
#return np.linalg.inv(np.dot(C, C.T))

def correlation(D):
    r"""
    Draw a random correlation matrix.
    """
    X = np.random.randn(D,D);
    s = np.sqrt(np.sum(X**2, axis=-1, keepdims=True))
    X = X / s
    return np.dot(X, X.T)


def gaussian_logpdf(yVy, yVmu, muVmu, logdet_V, D):
    r"""
    Log-density of a Gaussian distribution.

    :math:`\mathcal{G}(\mathbf{y}|\boldsymbol{\mu},\mathbf{V}^{-1})`

    Parameters
    -----------
    yVy : ndarray or double
        :math:`\mathbf{y}^T\mathbf{Vy}`
    yVmu : ndarray or double
        :math:`\mathbf{y}^T\mathbf{V}\boldsymbol{\mu}`
    muVmu : ndarray or double
        :math:`\boldsymbol{\mu}^T\mathbf{V}\boldsymbol{\mu}`
    logdet_V : ndarray or double
        Log-determinant of the precision matrix, :math:`\log|\mathbf{V}|`.
    D : int
        Dimensionality of the distribution.
    """
    return -0.5*yVy + yVmu - 0.5*muVmu + 0.5*logdet_V - 0.5*D*np.log(2*np.pi)

def gaussian_entropy(logdet_V, D):
    r"""
    Compute the entropy of a Gaussian distribution.

    If you want to get the gradient, just let each parameter be a gradient of
    that term.

    Parameters
    ----------
    logdet_V : ndarray or double
        The log-determinant of the precision matrix.
    D : int
        The dimensionality of the distribution.
    """
    return -0.5*logdet_V + 0.5*D + 0.5*D*np.log(2*np.pi)

def gamma_logpdf(bx, logx, a_logx, a_logb, gammaln_a):
    r"""
    Log-density of :math:`\mathcal{G}(x|a,b)`.
    
    If you want to get the gradient, just let each parameter be a gradient of
    that term.

    Parameters
    ----------
    bx : ndarray
        :math:`bx`
    logx : ndarray
        :math:`\log(x)`
    a_logx : ndarray
        :math:`a \log(x)`
    a_logb : ndarray
        :math:`a \log(b)`
    gammaln_a : ndarray
        :math:`\log\Gamma(a)`
    """
    return a_logb - gammaln_a + a_logx - logx - bx
#def gamma_logpdf(a, log_b, gammaln_a, 

def gamma_entropy(a, log_b, gammaln_a, psi_a, a_psi_a):
    r"""
    Entropy of :math:`\mathcal{G}(a,b)`.

    If you want to get the gradient, just let each parameter be a gradient of
    that term.

    Parameters
    ----------
    a : ndarray
        :math:`a`
    log_b : ndarray
        :math:`\log(b)`
    gammaln_a : ndarray
        :math:`\log\Gamma(a)`
    psi_a : ndarray
        :math:`\psi(a)`
    a_psi_a : ndarray
        :math:`a\psi(a)`
    """
    return a - log_b + gammaln_a + psi_a - a_psi_a

def orth(D):
    r"""
    Draw random orthogonal matrix.
    """
    Q = np.random.randn(D,D)
    (Q, _) = np.linalg.qr(Q)
    return Q

def svd(s):
    r"""
    Draw a random matrix given its singular values.
    """
    D = len(s)
    U = orth(D) * s
    V = orth(D)
    return np.dot(U, V.T)
    
def sphere(N=1):
    r"""
    Draw random points uniformly on a unit sphere.

    Returns (latitude,longitude) in degrees.
    """
    lon = np.random.uniform(-180, 180, N)
    lat = (np.arccos(np.random.uniform(-1, 1, N)) * 180 / np.pi) - 90
    return (lat, lon)


def bernoulli(p, size=None):
    r"""
    Draw random samples from the Bernoulli distribution.
    """
    if isinstance(size, int):
        size = (size,)
    if size is None:
        size = np.shape(p)
    return (np.random.rand(*size) < p)


def categorical(p, size=None):
    r"""
    Draw random samples from a categorical distribution.
    """
    if size is None:
        size = np.shape(p)[:-1]
    if isinstance(size, int):
        size = (size,)

    if np.any(np.asanyarray(p)<0):
        raise ValueError("Array contains negative probabilities")

    if not misc.is_shape_subset(np.shape(p)[:-1], size):
        raise ValueError("Probability array shape and requested size are "
                         "inconsistent")

    size = tuple(size)

    # Normalize probabilities
    p = p / np.sum(p, axis=-1, keepdims=True)

    # Compute cumulative probabilities (p_1, p_1+p_2, ..., p_1+...+p_N):
    P = np.cumsum(p, axis=-1)

    # Draw samples from interval [0,1]
    x = np.random.rand(*size)

    # For simplicity, repeat p to the size of the output (plus probability axis)
    K = np.shape(p)[-1]
    P = P * np.ones(tuple(size)+(K,))

    if size == ():
        z = np.searchsorted(P, x)
    else:
        # Seach the indices
        z = np.zeros(size)
        inds = misc.nested_iterator(size)
        for ind in inds:
            z[ind] = np.searchsorted(P[ind], x[ind])

    return z.astype(np.int)


def multinomial(n, p, size=None):

    plates_n = np.shape(n)
    plates_p = np.shape(p)[:-1]
    k = np.shape(p)[-1]

    if size is None:
        size = misc.broadcasted_shape(plates_n, plates_p)

    if not misc.is_shape_subset(plates_n, size):
        raise ValueError("Shape of n does not broadcast to the given size")

    if not misc.is_shape_subset(plates_p, size):
        raise ValueError("Shape of p does not broadcast to the given size")

    # This isn't a very efficient implementation. One could use NumPy's
    # multinomial once for all those plates for which n and p is the same.

    n = np.broadcast_to(n, size)
    p = np.broadcast_to(p, size + (k,))

    x = np.empty(size + (k,))

    for i in misc.nested_iterator(size):
        x[i] = np.random.multinomial(n[i], p[i])

    return x.astype(np.int)


def gamma(a, b, size=None):
    x = np.random.gamma(a, b, size=size)
    if np.any(x == 0):
        raise RuntimeError(
            "Numerically zero samples. Try using a larger shape parameter in "
            "the gamma distribution."
        )
    return x


def dirichlet(alpha, size=None):
    r"""
    Draw random samples from the Dirichlet distribution.
    """
    if isinstance(size, int):
        size = (size,)
    if size is None:
        size = np.shape(alpha)
    else:
        size = size + np.shape(alpha)[-1:]
    p = np.random.gamma(alpha, size=size)
    sump = np.sum(p, axis=-1, keepdims=True)
    if np.any(sump == 0):
        raise RuntimeError(
            "Numerically zero samples. Try using a larger Dirichlet "
            "concentration parameter value."
        )
    p /= sump
    return p


def logodds_to_probability(x):
    r"""
    Solves p from log(p/(1-p))
    """
    return 1 / (1 + np.exp(-x)) 
    

def alpha_beta_recursion(logp0, logP):
    r"""
    Compute alpha-beta recursion for Markov chain

    Initial state log-probabilities are in `p0` and state transition
    log-probabilities are in P. The probabilities do not need to be scaled to
    sum to one, but they are interpreted as below:

    logp0 = log P(z_0) + log P(y_0|z_0)
    logP[...,n,:,:] = log P(z_{n+1}|z_n) + log P(y_{n+1}|z_{n+1})
    """

    logp0 = misc.atleast_nd(logp0, 1)
    logP = misc.atleast_nd(logP, 3)
    
    D = np.shape(logp0)[-1]
    N = np.shape(logP)[-3]
    plates = misc.broadcasted_shape(np.shape(logp0)[:-1], np.shape(logP)[:-3])

    if np.shape(logP)[-2:] != (D,D):
        raise ValueError("Dimension mismatch %s != %s"
                         % (np.shape(logP)[-2:],
                            (D,D)))

    #
    # Run the recursion algorithm
    #

    # Allocate memory
    logalpha = np.zeros(plates+(N,D))
    logbeta = np.zeros(plates+(N,D))
    g = np.zeros(plates)

    # Forward recursion
    logalpha[...,0,:] = logp0
    for n in range(1,N):
        # Compute: P(z_{n-1},z_n|x_1,...,x_n)
        v = logalpha[...,n-1,:,None] + logP[...,n-1,:,:]
        c = misc.logsumexp(v, axis=(-1,-2))
        # Sum over z_{n-1} to get: log P(z_n|x_1,...,x_n)
        logalpha[...,n,:] = misc.logsumexp(v - c[...,None,None], axis=-2)
        g -= c

    # Compute the normalization of the last term
    v = logalpha[...,N-1,:,None] + logP[...,N-1,:,:]
    g -= misc.logsumexp(v, axis=(-1,-2))

    # Backward recursion 
    logbeta[...,N-1,:] = 0
    for n in reversed(range(N-1)):
        v = logbeta[...,n+1,None,:] + logP[...,n+1,:,:]
        c = misc.logsumexp(v, axis=(-1,-2))
        logbeta[...,n,:] = misc.logsumexp(v - c[...,None,None], axis=-1)

    v = logalpha[...,:,:,None] + logbeta[...,:,None,:] + logP[...,:,:,:]
    c = misc.logsumexp(v, axis=(-1,-2))
    zz = np.exp(v - c[...,None,None])

    # The logsumexp normalization is not numerically accurate, so do
    # normalization again:
    zz /= np.sum(zz, axis=(-1,-2), keepdims=True)

    z0 = np.sum(zz[...,0,:,:], axis=-1)
    z0 /= np.sum(z0, axis=-1, keepdims=True)

    return (z0, zz, g)


def gaussian_gamma_to_t(mu, Cov, a, b, ndim=1):
    r"""
    Integrates gamma distribution to obtain parameters of t distribution
    """
    alpha = a/b
    nu = 2*a
    S = Cov / misc.add_trailing_axes(alpha, 2*ndim)
    return (mu, S, nu)


def t_logpdf(z2, logdet_cov, nu, D):
    r"""
    """
    return (special.gammaln((nu+D)/2)
            - special.gammaln(nu/2)
            - 0.5 * D * np.log(nu*np.pi)
            - 0.5 * logdet_cov
            - 0.5 * (nu+D) * np.log(1 + z2/nu))
