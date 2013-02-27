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
# under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
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

from bayespy.utils import utils

from .node import Node
from .deterministic import Deterministic
from .constant import Constant
from .gaussian import Gaussian

def _message_sum_multiply(plates_parent, dims_parent, *arrays):
    """
    Compute message to parent and sum over plates.
    """
    # The shape of the full message
    shapes = [np.shape(array) for array in arrays]
    shape_full = utils.broadcasted_shape(*shapes)
    # Find axes that should be summed
    shape_parent = plates_parent + dims_parent
    sum_axes = utils.axes_to_collapse(shape_full, shape_parent)
    # Compute the multiplier for cancelling the
    # plate-multiplier.  Because we are summing over the
    # dimensions already in this function (for efficiency), we
    # need to cancel the effect of the plate-multiplier
    # applied in the message_to_parent function.
    r = 1
    for j in sum_axes:
        if j >= 0 and j < len(plates_parent):
            r *= shape_full[j]
        elif j < 0 and j < -len(dims_parent):
            r *= shape_full[j]
    # Compute the sum-product
    m = utils.sum_multiply(*arrays,
                           axis=sum_axes,
                           sumaxis=True,
                           keepdims=True) / r
    # Remove extra axes
    m = utils.squeeze_to_dim(m, len(shape_parent))
    return m

def _mask_sum(plates_parent, mask):

    # Compute the mask
    axes = utils.axes_to_collapse(np.shape(mask), plates_parent)
    mask = np.any(mask, axis=axes, keepdims=True)
    mask = utils.squeeze_to_dim(mask, len(plates_parent))
    return mask


class Dot(Deterministic):
    """
    A deterministic node for computing vector product of two Gaussians.

    See also:
    ---------
    MatrixDot
    """

    # This node satisfies Normal-protocol to children and
    # Gaussian-protocol to parents

    # This node is deterministic, just handles message processing.

    # y(i0,i1,...,in) = sum_d prod_k xk(i0,i1,...,in,d)

    def __init__(self, *parents, **kwargs):
        # For now, do not use plates other than from the parents,
        # although it would be possible (it would mean that you'd
        # create several "datasets" with identical "PCA
        # distribution"). Maybe it is better to create such plates in
        # children nodes?
        #plates = []

        parents = list(parents)
        # Convert constant arrays to constant nodes
        for n in range(len(parents)):
            if utils.is_numeric(parents[n]):
                parents[n] = Constant(Gaussian)(parents[n])

        parent_dims = [parent.dims for parent in parents]
        parent_plates = [parent.plates for parent in parents]

        # Check that the parents have equal dimensions
        for n in range(len(parent_dims)-1):
            if parent_dims[n] != parent_dims[n+1]:
                raise ValueError("Dimensions of the Gaussians do not "
                                 "match: %s" % (parent_dims,))

        ## try:
        ##     plates = utils.broadcasted_shape(*parent_plates)
        ## except ValueError:
        ##     raise ValueError("The plates of the parents are "
        ##                      "incompatible: %s" % (parent_plates,))
        
        super().__init__(*parents,
        #plates=tuple(plates), 
                         dims=((),()),
                         **kwargs)

            

    def _compute_moments(self, *u_parents):
        if len(self.parents) == 0:
            return [0, 0]

        str1 = '...i' + ',...i' * (len(self.parents)-1)
        str2 = '...ij' + ',...ij' * (len(self.parents)-1)

        u1 = list()
        u2 = list()
        for u in u_parents:
            u1.append(u[0])
            u2.append(u[1])

        x = [np.einsum(str1, *u1),
             np.einsum(str2, *u2)]

        return x
        


    def get_parameters(self):
        # Compute mean and variance
        u = self.get_moments()
        u[1] -= u[0]**2
        return u
        

    def _compute_message_and_mask_to_parent(self, index, m, *u_parents):

        # Normally we don't need to care about masks when computing the
        # message. However, in this node we want to avoid computing huge message
        # arrays so we sum some axis already here. Thus, we need to apply the
        # mask

        mask = self.mask
        parent = self.parents[index]

        # Compute both messages
        for i in range(2):

            # Add extra axes to the message from children
            m[i] = utils.add_trailing_axes(m[i], i+1)

            # List of elements to multiply together
            A = [m[i]]
            for k in range(len(u_parents)):
                if k != index:
                    A.append(u_parents[k][i])

            # Compute the sum over some axes already here in order to avoid huge
            # message matrices.
            m[i] = _message_sum_multiply(parent.plates, parent.dims[i], *A)

        # Compute the mask
        s = utils.axes_to_collapse(np.shape(mask), parent.plates)
        mask = np.any(mask, axis=s, keepdims=True)
        mask = utils.squeeze_to_dim(mask, len(parent.plates))

        return (m, mask)



class MatrixDot(Deterministic):
    """
    A deterministic node for computing matrix-vector product of Gaussians.

    The matrix is given in vector form.  Therefore, the first parent should have
    dimensionality that is a multiple of the dimensionality of the second
    parent. The dimensionality of the result is that multiplier:
        A : (M*N)-dimensional 
        X : N-dimensional 
        result : M-dimensional

    `Dot` computes vector-vector product of Gaussians, thus the result is scalar
    Normal. This node assumes that you have a full joint Gaussian distribution
    for all elements of a matrix, thus the result is vector Gaussian.

    See also
    --------
    Dot
    """

    def __init__(self, A, X, **kwargs):

        # Convert arrays to constant nodes
        if utils.is_numeric(A):
            A = Constant(Gaussian)(A)
        if utils.is_numeric(X):
            X = Constant(Gaussian)(X)

        MN = A.dims[0][0]
        N = X.dims[0][0]

        # Check parents
        if A.dims != ( (MN,), (MN, MN) ):
            raise ValueError("Invalid dimensionality of the first parent.")
        if X.dims != ( (N,), (N, N) ):
            raise ValueError("Invalid dimensionality of the second parent.")
        if (MN % N) != 0:
            raise ValueError("The dimensionality of the first parent should be "
                             "a multiple of the second parent.")

        # Dimensionality of the output
        M = int(MN / N)
        dims = ( (M,), (M,M) )

        Node.__init__(self, A, X, dims=dims, **kwargs)

            

    def _compute_moments(self, u_A, u_X):
        """
        Get the moments of the Gaussian output.
        """

        # Helpful variables to clarify the code
        A = u_A[0]
        AA = u_A[1]
        X = u_X[0]
        XX = u_X[1]

        (A, AA) = self._reshape_to_matrix(A, AA)
        #(A, AA) = self._reshape_to_matrix(A, AA, X.shape[-1])

        # Compute matrix-vector products
        Y = np.einsum('...ij,...j->...i', A, X)
        YY = np.einsum('...jlik,...lk->...ij', AA, XX)

        return [Y, YY]

    def _compute_message_and_mask_to_parent(self, index, m, u_A, u_X):
        """
        Compute the message to a parent node.
        """

        mask = self.mask
        parent = self.parents[index]

        # Message from children
        VY = m[0] # (..., D)
        V = m[1]  # (..., D, D)
        D = np.shape(V)[-1]

        if index == 0: # Message to A
            X = u_X[0]  # (..., K)
            XX = u_X[1] # (..., K, K)
            K = np.shape(X)[-1]

            # m0 : (..., D,K)     -> (...,D*K)
            # m1 : (..., D,K,D,K) -> (...,D*K,D*K)

            # In order to avoid huge memory consumption, compute the sum-product
            # already in this method (instead of Node._message_to_parent).  That
            # is, some axes are summed over already here.
            m0 = _message_sum_multiply(parent.plates, (D,K),
                                       VY[...,:,np.newaxis],
                                       X[...,np.newaxis,:])

            m1 = _message_sum_multiply(parent.plates, (D,K,D,K),
                                       V[...,:,np.newaxis,:,np.newaxis],
                                       XX[...,np.newaxis,:,np.newaxis,:])
            # Do the same for the mask
            mask = _mask_sum(parent.plates, mask)

            (m0, m1) = self._reshape_to_vector(m0, m1)

        elif index == 1: # Message to X
            (A, AA) = self._reshape_to_matrix(u_A[0], u_A[1])
            #K = parent.dims[0][0]
            K = np.shape(A)[-1]
            
            # A : (...,D,K)
            # AA : (...,D,K,D,K)

            # In order to avoid huge memory consumption, compute the sum-product
            # already in this method (instead of Node._message_to_parent).  That
            # is, some axes are summed over already here.
            m0 = _message_sum_multiply(parent.plates, (1,K),
                                       A,
                                       VY[...,:,np.newaxis])
            m0 = m0[...,0,:]
            
            m1 = _message_sum_multiply(parent.plates, (1,K,1,K),
                                       AA,
                                       V[...,:,np.newaxis,:,np.newaxis])
            m1 = m1[...,0,:,0,:]
            # Do the same for the mask
            mask = _mask_sum(parent.plates, mask)
            #m0 = np.einsum('...ij,...i->...j', A, VY)
            #m1 = np.einsum('...kilj,...kl->...ij', AA, V)

        return ([m0,m1], mask)

    
    def _reshape_to_matrix(self, A, AA):
        """
        Reshape vector form moments to matrix form.

        A : (...,M*N)
        AA : (...,M*N,M*N)

        Reshape to:
        A : (...,M,N)
        AA : (...,M,N,M,N)
        """
        # The dimensionalities
        MN = self.parents[0].dims[0][0]
        N = self.parents[1].dims[0][0]
        M = int(MN / N)

        # Reshape vector A to a matrix
        sh_A = np.shape(A)[:-1] + (M,N)
        sh_AA = np.shape(A)[:-1] + (M,N,M,N)
        A = A.reshape(sh_A)
        AA = AA.reshape(sh_AA)

        return (A, AA)

    def _reshape_to_vector(self, A, AA):
        """
        Reshape matrix form moments to vector form.

        A : (...,M,N)
        AA : (...,M,N,M,N)

        Reshape to:
        A : (...,M*N)
        AA : (...,M*N,M*N)
        """

        MN = A.shape[-2] * A.shape[-1]
        sh_A = A.shape[:-2] + (MN,)
        sh_AA = AA.shape[:-4] + (MN,MN)

        A = A.reshape(sh_A)
        AA = AA.reshape(sh_AA)

        return (A, AA)

