################################################################################
# Copyright (C) 2015 Jaakko Luttinen
#
# This file is licensed under the MIT License.
################################################################################


import numpy as np
import functools

from .deterministic import Deterministic
from .gaussian import Gaussian, GaussianMoments

from bayespy.utils import linalg


class Add(Deterministic):
    r"""
    Node for computing sums of Gaussian nodes: :math:`X+Y+Z`.

    Examples
    --------

    >>> import numpy as np
    >>> from bayespy import nodes
    >>> X = nodes.Gaussian(np.zeros(2), np.identity(2), plates=(3,))
    >>> Y = nodes.Gaussian(np.ones(2), np.identity(2))
    >>> Z = nodes.Add(X, Y)
    >>> print("Mean:\n", Z.get_moments()[0])
    Mean:
     [[ 1.  1.]]
    >>> print("Second moment:\n", Z.get_moments()[1])
    Second moment:
     [[[ 3.  1.]
      [ 1.  3.]]]

    Notes
    -----

    Shapes of the nodes must be identical. Plates are broadcasted.

    This node sums nodes that are independent in the posterior
    approximation. However, summing variables puts a strong coupling among the
    variables, which is lost in this construction.  Thus, it is usually better
    to use a single Gaussian node to represent the set of the summed variables
    and use SumMultiply node to compute the sum.  In that way, the correlation
    between the variables is not lost. However, in some cases it is necessary or
    useful to use Add node.

    See also
    --------

    Dot, SumMultiply
    """

    def __init__(self, *nodes, **kwargs):
        """
        Add(X1, X2, ...)
        """

        ndim = None
        for node in nodes:
            try:
                node = self._ensure_moments(node, GaussianMoments, ndim=None)
            except ValueError:
                pass
            else:
                ndim = node._moments.ndim
                break

        nodes = [self._ensure_moments(node, GaussianMoments, ndim=ndim)
                 for node in nodes]

        N = len(nodes)
        if N < 2:
            raise ValueError("Give at least two parents")

        nodes = list(nodes)

        for n in range(N-1):
            if nodes[n].dims != nodes[n+1].dims:
                raise ValueError("Nodes do not have identical shapes")

        ndim = len(nodes[0].dims[0])
        dims = tuple(nodes[0].dims)
        shape = dims[0]

        self._moments = GaussianMoments(shape)
        self._parent_moments = N * [GaussianMoments(shape)]

        self.ndim = ndim
        self.N = N

        super().__init__(*nodes, dims=dims, **kwargs)


    def _compute_moments(self, *u_parents):
        """
        Compute the moments of the sum
        """

        u0 = functools.reduce(np.add,
                              (u_parent[0] for u_parent in u_parents))
        u1 = functools.reduce(np.add,
                              (u_parent[1] for u_parent in u_parents))

        for i in range(self.N):
            for j in range(i+1, self.N):
                xi_xj = linalg.outer(u_parents[i][0], u_parents[j][0], ndim=self.ndim)
                xj_xi = linalg.transpose(xi_xj, ndim=self.ndim)
                u1 = u1 + xi_xj + xj_xi
                                                                     
        return [u0, u1]


    def _compute_message_to_parent(self, index, m, *u_parents):
        """
        Compute the message to a parent node.

        .. math::

           (\sum_i \mathbf{x}_i)^T \mathbf{M}_2 (\sum_j \mathbf{x}_j)
           + (\sum_i \mathbf{x}_i)^T \mathbf{m}_1

        Moments of the parents are

        .. math::

           u_1^{(i)} = \langle \mathbf{x}_i \rangle
           \\
           u_2^{(i)} = \langle \mathbf{x}_i \mathbf{x}_i^T \rangle

        Thus, the message for :math:`i`-th parent is

        .. math::
        
           \phi_{x_i}^{(1)} = \mathbf{m}_1 + 2 \mathbf{M}_2 \sum_{j\neq i} \mathbf{x}_j
           \\
           \phi_{x_i}^{(2)} = \mathbf{M}_2
        """

        # Remove the moments of the parent that receives the message
        u_parents = u_parents[:index] + u_parents[(index+1):]

        m0 = (m[0] +
              linalg.mvdot(
                  2*m[1],
                  functools.reduce(np.add,
                                   (u_parent[0] for u_parent in u_parents)),
                  ndim=self.ndim))

        m1 = m[1]
            
        return [m0, m1]
