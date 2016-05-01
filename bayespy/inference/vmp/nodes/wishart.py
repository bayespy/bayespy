################################################################################
# Copyright (C) 2011-2012,2014 Jaakko Luttinen
#
# This file is licensed under the MIT License.
################################################################################


import numpy as np
import scipy.special as special

from bayespy.utils import misc, linalg

from .expfamily import ExponentialFamily
from .expfamily import ExponentialFamilyDistribution
from .expfamily import useconstructor
from .constant import Constant
from .deterministic import Deterministic
from .gamma import GammaMoments

from .node import Moments, Node


class WishartPriorMoments(Moments):


    def __init__(self, k):
        self.k = k
        self.dims = ( (), () )
        return


    def compute_fixed_moments(self, n):
        """ Compute moments for fixed x. """
        u0 = np.asanyarray(n)
        u1 = special.multigammaln(0.5*u0, self.k)
        return [u0, u1]


    @classmethod
    def from_values(cls, x, d):
        """ Compute the dimensions of phi or u. """
        return cls(d)


class WishartMoments(Moments):


    def __init__(self, shape):
        self.shape = shape
        self.ndim = len(shape)
        self.dims = ( 2 * shape, () )
        return


    def compute_fixed_moments(self, Lambda, gradient=None):
        """ Compute moments for fixed x. """
        L = linalg.chol(Lambda, ndim=self.ndim)
        ldet = linalg.chol_logdet(L, ndim=self.ndim)
        u = [Lambda,
             ldet]

        if gradient is None:
            return u

        du0 = gradient[0]
        du1 = (
            misc.add_trailing_axes(gradient[1], 2*self.ndim)
            * linalg.chol_inv(L, ndim=self.ndim)
        )

        du = du0 + du1

        return (u, du)


    def plates_from_shape(self, shape):
        if self.ndim == 0:
            return shape
        else:
            return shape[:-2*self.ndim]


    def shape_from_plates(self, plates):
        return plates + self.shape + self.shape


    def get_instance_conversion_kwargs(self):
        return dict(ndim=self.ndim)


    def get_instance_converter(self, ndim):
        if ndim != self.ndim:
            raise NotImplementedError(
                "No conversion between different ndim implemented for "
                "WishartMoments yet"
            )
        return None


    @classmethod
    def from_values(cls, x, ndim):
        """ Compute the dimensions of phi and u. """
        if np.ndim(x) < 2 * ndim:
            raise ValueError("Values for Wishart distribution must be at least "
                             "2-D arrays.")
        if np.shape(x)[-ndim:] != np.shape(x)[-2*ndim:-ndim]:
            raise ValueError("Values for Wishart distribution must be square "
                             "matrices, thus the two last axes must have equal "
                             "length.")
        shape = np.shape(x)[-ndim:]
        return cls(shape)


class WishartDistribution(ExponentialFamilyDistribution):
    """
    Sub-classes implement distribution specific computations.

    Distribution for :math:`k \times k` symmetric positive definite matrix.

    .. math::

        \Lambda \sim \mathcal{W}(n, V)

    Note: :math:`V` is inverse scale matrix.

    .. math::

        p(\Lambda | n, V) = ..
    """


    def compute_message_to_parent(self, parent, index, u_self, u_n, u_V):
        if index == 0:
            raise NotImplementedError("Message from Wishart to degrees of "
                                      "freedom parameter (first parent) "
                                      "not yet implemented")
        elif index == 1:
            Lambda = u_self[0]
            n = u_n[0]
            return [-0.5 * Lambda,
                    0.5 * n]
        else:
            raise ValueError("Invalid parent index {0}".format(index))

    def compute_phi_from_parents(self, u_n, u_V, mask=True):
        r"""
        Compute natural parameters

        .. math::

            \phi(n, V) =
            \begin{bmatrix}
              -\frac{1}{2} V
              \\
              \frac{1}{2} n
            \end{bmatrix}
        """
        return [-0.5 * u_V[0],
                0.5 * u_n[0]]

    def compute_moments_and_cgf(self, phi, mask=True):
        r"""
        Return moments and cgf for given natural parameters

        .. math::

            \langle u \rangle =
            \begin{bmatrix}
              \phi_2 (-\phi_1)^{-1}
              \\
              -\log|-\phi_1| + \psi_k(\phi_2)
            \end{bmatrix}
            \\
            g(\phi) = \phi_2 \log|-\phi_1| - \log \Gamma_k(\phi_2)
        """
        U = linalg.chol(-phi[0])
        k = np.shape(phi[0])[-1]
        #k = self.dims[0][0]
        logdet_phi0 = linalg.chol_logdet(U)
        u0 = phi[1][...,np.newaxis,np.newaxis] * linalg.chol_inv(U)
        u1 = -logdet_phi0 + misc.multidigamma(phi[1], k)
        u = [u0, u1]
        g = phi[1] * logdet_phi0 - special.multigammaln(phi[1], k)
        return (u, g)

    def compute_cgf_from_parents(self, u_n, u_V):
        r"""
        CGF from parents

        .. math::

            g(n, V) = \frac{n}{2} \log|V| - \frac{nk}{2} \log 2 -
            \log \Gamma_k(\frac{n}{2})
        """
        n = u_n[0]
        gammaln_n = u_n[1]
        V = u_V[0]
        logdet_V = u_V[1]
        k = np.shape(V)[-1]
        g = 0.5*n*logdet_V - 0.5*k*n*np.log(2) - gammaln_n
        return g

    def compute_fixed_moments_and_f(self, Lambda, mask=True):
        r"""
        Compute u(x) and f(x) for given x.

        .. math:

            u(\Lambda) =
            \begin{bmatrix}
              \Lambda
              \\
              \log |\Lambda|
            \end{bmatrix}
        """
        k = np.shape(Lambda)[-1]
        ldet = linalg.chol_logdet(linalg.chol(Lambda))
        u = [Lambda,
             ldet]
        f = -(k+1)/2 * ldet
        return (u, f)


class Wishart(ExponentialFamily):
    r"""
    Node for Wishart random variables.

    The random variable :math:`\mathbf{\Lambda}` is a :math:`D\times{}D`
    positive-definite symmetric matrix.

    .. math::

        p(\mathbf{\Lambda}) = \mathrm{Wishart}(\mathbf{\Lambda} | N,
        \mathbf{V})

    Parameters
    ----------

    n : scalar or array

        :math:`N`, degrees of freedom, :math:`N>D-1`.

    V : Wishart-like node or (...,D,D)-array

        :math:`\mathbf{V}`, scale matrix.
    """

    _distribution = WishartDistribution()


    def __init__(self, n, V, **kwargs):
        """
        Create Wishart node.
        """
        super().__init__(n, V, **kwargs)


    @classmethod
    def _constructor(cls, n, V, **kwargs):
        """
        Constructs distribution and moments objects.
        """

        # Make V a proper parent node and get the dimensionality of the matrix
        V = cls._ensure_moments(V, WishartMoments, ndim=1)
        D = V.dims[0][-1]

        n = cls._ensure_moments(n, WishartPriorMoments, d=D)

        moments = WishartMoments((D,))

        # Parent node message types
        parent_moments = (n._moments, V._moments)

        parents = [n, V]

        return (parents,
                kwargs,
                moments.dims,
                cls._total_plates(kwargs.get('plates'),
                                  cls._distribution.plates_from_parent(0, n.plates),
                                  cls._distribution.plates_from_parent(1, V.plates)),
                cls._distribution,
                moments,
                parent_moments)


    def scale(self, scalar, **kwargs):
        return _ScaledWishart(self, scalar, **kwargs)


    def __str__(self):
        n = 2*self.phi[1]
        A = 0.5 * self.u[0] / self.phi[1][...,np.newaxis,np.newaxis]
        return ("%s ~ Wishart(n, A)\n"
                "  n =\n"
                "%s\n"
                "  A =\n"
                "%s\n"
                % (self.name, n, A))


class _ScaledWishart(Deterministic):


    def __init__(self, Lambda, alpha, ndim=None, **kwargs):

        if ndim is None:
            try:
                ndim = Lambda._moments.ndim
            except AttributeError:
                raise ValueError("Give explicit ndim argument. (ndim=1 for normal matrix)")

        Lambda = self._ensure_moments(Lambda, WishartMoments, ndim=ndim)
        alpha = self._ensure_moments(alpha, GammaMoments)

        dims = Lambda.dims

        self._moments = Lambda._moments
        self._parent_moments = (Lambda._moments, alpha._moments)

        return super().__init__(Lambda, alpha, dims=dims, **kwargs)


    def _compute_moments(self, u_Lambda, u_alpha):

        Lambda = u_Lambda[0]
        logdet_Lambda = u_Lambda[1]

        alpha = misc.add_trailing_axes(u_alpha[0], 2*self._moments.ndim)
        logalpha = u_alpha[1]

        u0 = Lambda * alpha
        u1 = logdet_Lambda + np.prod(self._moments.shape) * logalpha

        return [u0, u1]


    def _compute_message_to_parent(self, index, m, u_Lambda, u_alpha):

        if index == 0:
            alpha = misc.add_trailing_axes(u_alpha[0], 2*self._moments.ndim)
            logalpha = u_alpha[1]
            m0 = m[0] * alpha
            m1 = m[1]
            return [m0, m1]

        if index == 1:
            Lambda = u_Lambda[0]
            logdet_Lambda = u_Lambda[1]
            m0 = linalg.inner(m[0], Lambda, ndim=2*self._moments.ndim)
            m1 = m[1] * np.prod(self._moments.shape)
            return [m0, m1]

        raise IndexError()
