import numpy as np

from bayespy.utils import misc
from bayespy.utils import linalg
from .gaussian import GaussianMoments
from .deterministic import Deterministic


class ConcatGaussian(Deterministic):
    r"""Concatenate Gaussian vectors along the variable axis (not plate axis)

    NOTE: This concatenates on the variable axis! That is, the dimensionality
    of the resulting Gaussian vector is the sum of the dimensionalities of the
    input Gaussian vectors.

    TODO: Add support for Gaussian arrays and arbitrary concatenation axis.
    """


    def __init__(self, *nodes, **kwargs):

        # Number of nodes to concatenate
        N = len(nodes)

        # This is stuff that will be useful when implementing arbitrary
        # concatenation. That is, first determine ndim.
        #
        # # Convert nodes to Gaussians (if they are not nodes, don't worry)
        # nodes_gaussian = []
        # for node in nodes:
        #     try:
        #         node_gaussian = node._convert(GaussianMoments)
        #     except AttributeError: # Moments.NoConverterError:
        #         nodes_gaussian.append(node)
        #     else:
        #         nodes_gaussian.append(node_gaussian)
        # nodes = nodes_gaussian
        #
        # # Determine shape from the first Gaussian node
        # shape = None
        # for node in nodes:
        #     try:
        #         shape = node.dims[0]
        #     except AttibuteError:
        #         pass
        #     else:
        #         break
        # if shape is None:
        #     raise ValueError("Couldn't determine shape from the input nodes")
        #
        # ndim = len(shape)

        nodes = [self._ensure_moments(node, GaussianMoments, ndim=1)
                 for node in nodes]

        D = sum(node.dims[0][0] for node in nodes)

        shape = (D,)

        self._moments = GaussianMoments(shape)

        self._parent_moments = [node._moments for node in nodes]

        # Make sure all parents are Gaussian vectors
        if any(len(node.dims[0]) != 1 for node in nodes):
            raise ValueError("Input nodes must be (Gaussian) vectors")

        self.slices = tuple(np.cumsum([0] + [node.dims[0][0] for node in nodes]))
        D = self.slices[-1]

        return super().__init__(*nodes, dims=((D,), (D, D)), **kwargs)


    def _compute_moments(self, *u_nodes):
        x = misc.concatenate(*[u[0] for u in u_nodes], axis=-1)
        xx = misc.block_diag(*[u[1] for u in u_nodes])

        # Explicitly broadcast xx to plates of x
        x_plates = np.shape(x)[:-1]
        xx = np.ones(x_plates)[...,None,None] * xx

        # Compute the cross-covariance terms using the means of each variable
        # (because covariances are zero for factorized nodes in the VB
        # approximation)
        i_start = 0
        for m in range(len(u_nodes)):
            i_end = i_start + np.shape(u_nodes[m][0])[-1]
            j_start = 0
            for n in range(m):
                j_end = j_start + np.shape(u_nodes[n][0])[-1]
                xm_xn = linalg.outer(u_nodes[m][0], u_nodes[n][0], ndim=1)
                xx[...,i_start:i_end,j_start:j_end] = xm_xn
                xx[...,j_start:j_end,i_start:i_end] = misc.T(xm_xn)
                j_start = j_end
            i_start = i_end

        return [x, xx]


    def _compute_message_to_parent(self, i, m, *u_nodes):
        r = self.slices

        # Pick the proper parts from the message array
        m0 = m[0][...,r[i]:r[i+1]]
        m1 = m[1][...,r[i]:r[i+1],r[i]:r[i+1]]

        # Handle cross-covariance terms by using the mean of the covariate node
        for (j, u) in enumerate(u_nodes):
            if j != i:
                m0 = m0 + 2 * np.einsum(
                    '...ij,...j->...i',
                    m[1][...,r[i]:r[i+1],r[j]:r[j+1]],
                    u[0]
                )

        return [m0, m1]
