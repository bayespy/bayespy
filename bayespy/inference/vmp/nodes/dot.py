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

            

    def get_moments(self):
        if len(self.parents) == 0:
            return [0, 0]

        str1 = '...i' + ',...i' * (len(self.parents)-1)
        str2 = '...ij' + ',...ij' * (len(self.parents)-1)

        u1 = list()
        u2 = list()
        for parent in self.parents:
            u = parent._message_to_child()
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
        #def get_message(self, index, u_parents):
        
        #(m, mask) = self.message_from_children()

        # Normally we don't need to care about masks when computing the
        # message. However, in this node we want to avoid computing huge message
        # arrays so we sum some axis already here. Thus, we need to apply the
        # mask

        mask = self.mask
        parent = self.parents[index]

        # Compute both messages
        for i in range(2):

            # Add extra axes to the message from children
            #m_shape = np.shape(m[i]) + (1,) * (i+1)
            #m[i] = np.reshape(m[i], m_shape)

            # Put masked elements to zero
            np.copyto(m[i], 0, where=np.logical_not(mask))
                
            # Add extra axes to the mask from children
            #mask_shape = np.shape(mask) + (1,) * (i+1)
            #mask_i = np.reshape(mask, mask_shape)

            #mask_i = mask
            m[i] = utils.add_trailing_axes(m[i], i+1)
            #for k in range(i+1):
                #m[i] = np.expand_dims(m[i], axis=-1)
                #mask_i = np.expand_dims(mask_i, axis=-1)

            # List of elements to multiply together
            A = [m[i]]
            for k in range(len(u_parents)):
                if k != index:
                    A.append(u_parents[k][i])

            # Find out which axes are summed over. Also, 
            full_shape = utils.broadcasted_shape_from_arrays(*A)
            axes = utils.axes_to_collapse(full_shape, parent.get_shape(i))
            # Compute the multiplier for cancelling the
            # plate-multiplier.  Because we are summing over the
            # dimensions already in this function (for efficiency), we
            # need to cancel the effect of the plate-multiplier
            # applied in the message_to_parent function.
            r = 1
            for j in axes:
                r *= full_shape[j]

            # Compute dot product (and cancel plate-multiplier)
            m[i] = utils.sum_product(*A, axes_to_sum=axes, keepdims=True) / r

        # Compute the mask
        s = utils.axes_to_collapse(np.shape(mask), parent.plates)
        mask = np.any(mask, axis=s, keepdims=True)
        mask = utils.squeeze_to_dim(mask, len(parent.plates))

        return (m, mask)
    #return (m, mask)




class MatrixDot(Node):
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

            

    def get_moments(self):
        """
        Get the moments of the Gaussian output.
        """

        # Get parents' moments
        u_A = self.parents[0].message_to_child()
        u_X = self.parents[1].message_to_child()

        # Helpful variables to clarify the code
        A = u_A[0]
        AA = u_A[1]
        X = u_X[0]
        XX = u_X[1]

        (A, AA) = self._reshape_to_matrix(A, AA, X.shape[-1])

        #print('debug in matrixdot.moments', np.shape(A), np.shape(AA), np.shape(X), np.shape(XX))

        # Compute matrix-vector products
        Y = np.einsum('...ij,...j->...i', A, X)
        YY = np.einsum('...jlik,...lk->...ij', AA, XX)

        return [Y, YY]

    def get_message(self, index, u_parents):
        """
        Compute the message to a parent node.
        """

        # Get the message from children
        (m, mask) = self.message_from_children()
        VY = m[0]
        V = m[1]

        if index == 0: # Message to A
            X = u_parents[1][0]
            XX = u_parents[1][1]
            m0 = VY[...,:,np.newaxis] * X[...,np.newaxis,:]
            m1 = np.einsum('...ik,...jl->...ijkl', V, XX)
            (m0, m1) = self._reshape_to_vector(m0, m1)
        elif index == 1: # Message to X
            (A, AA) = self._reshape_to_matrix(u_parents[0][0],
                                              u_parents[0][1],
                                              self.parents[1].dims[0][0])
            m0 = np.einsum('...ij,...i->...j', A, VY)
            m1 = np.einsum('...kilj,...kl->...ij', AA, V)

        m = [m0, m1]
        return (m, mask)

    
    def _reshape_to_matrix(self, A, AA, N):
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

