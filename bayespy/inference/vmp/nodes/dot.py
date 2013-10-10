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



class SumMultiply(Deterministic):
    """
    Compute the sum-product of Gaussian nodes similarly to numpy.einsum.
    
    For instance, the equivalent of
    
        np.einsum('abc,bd,ca->da', X, Y, Z)
        
    would be given as
    
        SumMultiply(X, [0,1,2], Y, [1,3], Z, [2,0], [3,0])
        
    which is similar to the other syntax of numpy.einsum.

    This node operates similarly as numpy.einsum. However, you must use all the
    elements of each node, that is, an operation like np.einsum('ii->i',X) is
    not allowed. Thus, for each node, each axis must be given unique id. The id
    identifies which axes correspond to which axes between the different
    nodes. Also, Ellipsis ('...') is not yet supported for simplicity.

    Each output axis must appear in the input mappings.

    The convenient string notation of numpy.einsum is not yet implemented.

    Examples
    --------

    Sum over the rows:
    'ij->j'
    
    Inner product of three vectors:
    'i,i,i'

    Matrix-vector product:
    'ij,j->i'

    Matrix-matrix product:
    'ik,kj->ij'

    Outer product:
    'i,j->ij'

    Vector-matrix-vector product:
    'i,ij,j'

    Note
    ----

    This operation can be extremely slow if not used wisely. For large and
    complex operations, it is sometimes more efficient to split the operation
    into multiple nodes. For instance, the example above could probably be
    computed faster by

        XZ = SumMultiply(X, [0,1,2], Z, [2,0], [0,1])
        SumMultiply(XZ, [0,1], Y, [1,2], [2,0])

    because the third axis ('c') could be summed out already in the first
    operation. This same effect applies also to numpy.einsum in general.
    """

    def __init__(self, *args, **kwargs):
        """
        SumMultiply(Node1, map1, Node2, map2, ..., NodeN, mapN [, map_out])
        """

        args = list(args)

        if len(args) < 2:
            raise ValueError("Not enough inputs")

        # Two different parsing methods, depends on how the arguments are given
        if utils.is_string(args[0]):
            # This is the format:
            # SumMultiply('ik,k,kj->ij', X, Y, Z)
            strings = args[0]
            nodes = args[1:]
            # Remove whitespace
            strings = utils.remove_whitespace(strings)
            # Split on '->' (should contain only one '->' or none)
            strings = strings.split('->')
            if len(strings) > 2:
                raise ValueError('The string contains too many ->')
            strings_in = strings[0]
            if len(strings) == 2:
                string_out = strings[1]
            else:
                string_out = ''
            # Split former part on ',' (the number of parts should be equal to
            # nodes)
            strings_in = strings_in.split(',')
            if len(strings_in) != len(nodes):
                raise ValueError('Number of given input nodes is different '
                                 'from the input keys in the string')
            # Split strings into key lists using single character keys
            keysets = [list(string_in) for string_in in strings_in]
            keys_out = list(string_out)
            
            
        else:
            # This is the format:
            # SumMultiply(X, [0,2], Y, [2], Z, [2,1], [0,1])

            # If given, the output mapping is the last argument
            if len(args) % 2 == 0:
                keys_out = []
            else:
                keys_out = args.pop(-1)

            # Node and axis mapping are given in turns
            nodes = args[::2]
            keysets = args[1::2]
            
        # Find all the keys (store only once each)
        full_keyset = []
        for keyset in keysets:
            full_keyset += keyset
            #full_keyset += list(keyset.keys())
        full_keyset = list(set(full_keyset))

        #
        # Check the validity of each node
        #
        for n in range(len(nodes)):
            # Convert constant arrays to constant nodes
            if utils.is_numeric(nodes[n]):
                # TODO/FIXME: Use GaussianArray and infer the dimensionality
                # from the length of the axis mapping!
                nodes[n] = Constant(Gaussian)(nodes[n])
            # Check that the maps and the size of the variable are consistent
            if len(nodes[n].dims[0]) != len(keysets[n]):
                raise ValueError("Wrong number of keys (%d) for the node "
                                 "number %d with %d dimensions"
                                 % (len(keysets[n]),
                                    n,
                                    len(nodes[n].dims[0])))
            # Check that the keys are unique
            if len(set(keysets[n])) != len(keysets[n]):
                raise ValueError("Axis keys for node number %d are not unique"
                                 % n)
            # Check that the dims are proper Gaussians
            if len(nodes[n].dims) != 2:
                raise ValueError("Node %d is not Gaussian" % n)
            if nodes[n].dims[0] + nodes[n].dims[0] != nodes[n].dims[1]:
                raise ValueError("Node %d is not Gaussian" % n)

        # Check the validity of output keys: each output key must be included in
        # the input keys
        if len(keys_out) != len(set(keys_out)):
            raise ValueError("Output keys are not unique")
        for key in keys_out:
            if key not in full_keyset:
                raise ValueError("Output key %s does not appear in any input"
                                 % key)

        # Check the validity of the nodes with respect to the key mapping.
        # Check that the node dimensions map and broadcast properly, that is,
        # all the nodes using the same key for axes must have equal size for
        # those axes (or size 1).
        broadcasted_size = {}
        for key in full_keyset:
            broadcasted_size[key] = 1
            for (node, keyset) in zip(nodes, keysets):
                try:
                    # Find the axis for the key
                    index = keyset.index(key)
                except ValueError:
                    # OK, this node doesn't use this key for any axis
                    pass
                else:
                    # Length of the axis for that key
                    node_size = node.dims[0][index]
                    if node_size != broadcasted_size[key]:
                        if broadcasted_size[key] == 1:
                            # Apply broadcasting
                            broadcasted_size[key] = node_size
                        elif node_size != 1:
                            # Different sizes and neither has size 1
                            raise ValueError("Axes using key %s do not "
                                             "broadcast properly"
                                             % key)
                                             

        # Compute the shape of the output
        dim0 = [broadcasted_size[key] for key in keys_out]
        dim1 = dim0 + dim0

        # Rename the keys to [0,1,...,N-1] where N is the total number of keys
        self.N_keys = len(full_keyset)
        self.out_keys = [full_keyset.index(key) for key in keys_out]
        self.in_keys = [ [full_keyset.index(key) for key in keyset]
                         for keyset in keysets ]

        super().__init__(*nodes,
                         dims=(tuple(dim0),tuple(dim1)),
                         **kwargs)

            

    def _compute_moments(self, *u_parents):


        # Compute the number of plate axes for each node
        plate_counts0 = [(np.ndim(u_parent[0]) - len(keys))
                         for (keys,u_parent) in zip(self.in_keys, u_parents)]
        plate_counts1 = [(np.ndim(u_parent[1]) - 2*len(keys))
                         for (keys,u_parent) in zip(self.in_keys, u_parents)]
        # The number of plate axes for the output
        N0 = max(plate_counts0)
        N1 = max(plate_counts1)
        # The total number of unique keys used (keys are 0,1,...,N_keys-1)
        D = self.N_keys

        #
        # Compute the mean
        #
        out_all_keys = list(range(D+N0-1, D-1, -1)) + self.out_keys
        #nodes_dim_keys = self.nodes_dim_keys
        in_all_keys = [list(range(D+plate_count-1, D-1, -1)) + keys
                      for (plate_count, keys) in zip(plate_counts0, 
                                                     self.in_keys)]
        u0 = [u[0] for u in u_parents]
        
        args = utils.zipper_merge(u0, in_all_keys) + [out_all_keys]
        x0 = np.einsum(*args)

        #
        # Compute the covariance
        #
        out_all_keys = (list(range(2*D+N1-1, 2*D-1, -1)) 
                        + [D+key for key in self.out_keys] 
                        + self.out_keys)
        #nodes_dim_keys = self.nodes_dim_keys
        in_all_keys = [list(range(2*D+plate_count-1, 2*D-1, -1)) 
                       + [D+key for key in node_keys]
                       + node_keys
                       for (plate_count, node_keys) in zip(plate_counts1, 
                                                           self.in_keys)]
        u1 = [u[1] for u in u_parents]
        args = utils.zipper_merge(u1, in_all_keys) + [out_all_keys]
        print(in_all_keys, np.shape(u_parents[0][1]), out_all_keys)
        x1 = np.einsum(*args)

        return [x0, x1]

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
    A deterministic node for computing sum-products of Gaussian matrices.

    Each Gaussian matrix is two-dimensional and the dimensions must be identical
    or at least broadcastable. For instance:
        X1 : (10,1)
        X2 : (10,10)
        X3 : (1,10)
        X4 : (10,10)
        X5 : (1,1)
        
    The keywords ``sumrows`` and ``sumcols`` define the axes which are summed
    over. If both dimensions are summed, the result is a scalar Normal
    message. If only one dimension is summed, the result is a Gaussian (vector)
    message. If none of the dimensions is summed, the result is a GaussianMatrix
    message. The resulting messages can also be returned as iterators over some
    plate dimensions?
    
    See also
    --------
    Dot, GaussianMatrixARD, Gaussian
    """

    def __init__(self, A, X, sumrows=True, sumcols=True, return_iterator=False, **kwargs):

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




























class MatrixDotDEPRECATED(Deterministic):
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

