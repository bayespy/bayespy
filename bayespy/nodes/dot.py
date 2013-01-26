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

class Dot(Node):

    # This node satisfies Normal-protocol to children and
    # Gaussian-protocol to parents

    # This node is deterministic, just handles message processing.

    # y(i0,i1,...,in) = sum_d prod_k xk(i0,i1,...,in,d)

    def __init__(self, *args, **kwargs):
        # For now, do not use plates other than from the parents,
        # although it would be possible (it would mean that you'd
        # create several "datasets" with identical "PCA
        # distribution"). Maybe it is better to create such plates in
        # children nodes?
        plates = []
        for x in args:
            # Convert constant matrices to nodes
            if np.isscalar(x) or isinstance(x, np.ndarray):
                x = NodeConstantGaussian(x)
            # Dimensionality of the Gaussian(s). You should check that
            # all the parents have the same dimensionality!
            self.d = x.dims[0]
            # Check consistency of plates (broadcasting rules!)
            for ind in range(min(len(plates),len(x.plates))):
                if plates[-ind-1] == 1:
                    plates[-ind-1] = x.plates[-ind-1]
                elif x.plates[-ind-1] != 1 and plates[-ind-1] != x.plates[-ind-1]:
                    raise Exception('Plates do not match')
            # Add new extra plates
            plates = list(x.plates[:(len(x.plates)-len(plates))]) + plates

        Node.__init__(self, *args, plates=tuple(plates), dims=[(),()], **kwargs)

            

    def get_moments(self):
        if len(self.parents) == 0:
            return [0, 0]

        str1 = '...i' + ',...i' * (len(self.parents)-1)
        str2 = '...ij' + ',...ij' * (len(self.parents)-1)

        u1 = list()
        u2 = list()
        for parent in self.parents:
            u = parent.message_to_child()
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
        

    def get_message(self, index, u_parents):
        
        (m, mask) = self.message_from_children()

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


    def OLD_get_message(self, index, u_parents):
        
        (m, mask) = self.message_from_children()

        parent = self.parents[index]

        # Compute both messages
        for i in range(2):

            # Add extra axes to the message from children
            #m_shape = np.shape(m[i]) + (1,) * (i+1)
            #m[i] = np.reshape(m[i], m_shape)

            # Add extra axes to the mask from children
            mask_shape = np.shape(mask) + (1,) * (i+1)
            mask_i = np.reshape(mask, mask_shape)

            mask_i = mask
            for k in range(i+1):
                m[i] = np.expand_dims(m[i], axis=-1)
                mask_i = np.expand_dims(mask_i, axis=-1)

            # List of elements to multiply together
            A = [m[i], mask_i]
            for k in range(len(u_parents)):
                if k != index:
                    A.append(u_parents[k][i])

            # Find out which axes are summed over. Also, because
            # we are summing over the dimensions already in this
            # function (for efficiency), we need to cancel the
            # effect of the plate-multiplier applied in the
            # message_to_parent function.
            full_shape = utils.broadcasted_shape_from_arrays(*A)
            axes = utils.axes_to_collapse(full_shape, parent.get_shape(i))
            r = 1
            for j in axes:
                r *= full_shape[j]

            # Compute dot product
            m[i] = utils.sum_product(*A, axes_to_sum=axes, keepdims=True) / r

        # Compute the mask
        s = utils.axes_to_collapse(np.shape(mask), parent.plates)
        mask = np.any(mask, axis=s, keepdims=True)
        mask = utils.squeeze_to_dim(mask, len(parent.plates))

        return (m, mask)


