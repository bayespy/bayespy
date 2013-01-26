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

'''
General class for a node in a Bayesian network. The node can be
stochastic or deterministic.
'''

import numpy as np

class Node:

    def __init__(self, *args, dims=None, plates=(), name=""):

        if dims is None:
            raise Exception("You need to specify the dimensionality" \
                            + " of the distribution for class" \
                            + str(self.__class__))
        self.dims = dims
        self.plates = plates
        self.name = name

        # Parents
        self.parents = args
        # Inform parent nodes
        for (index,parent) in enumerate(self.parents):
            if parent:
                parent.add_child(self, index)
        # Children
        self.children = list()

    def add_child(self, child, index):
        self.children.append((child, index))

    def plates_to_parent(self, index):
        return self.plates

    #def get_shape(self, ind):
    #    return self.plates + self.dims[ind]

    ## @staticmethod
    ## def plate_multiplier(plates, *args):
    ##     # Check broadcasting of the shapes
    ##     for arg in args:
    ##         utils.broadcasted_shape(plates, arg)
            
    ##     r = 1
    ##     for j in range(-len(plates),0):
    ##         mult = True
    ##         for arg in args:
    ##             if not (-j > len(arg) or arg[j] == 1):
    ##                 mult = False
    ##         if mult:
    ##             r *= plates[j]
    ##     return r




## class NodeConstant(Node):
##     def __init__(self, u, **kwargs):
##         self.u = u
##         Node.__init__(self, **kwargs)

##     def message_to_child(self, gradient=False):
##         if gradient:
##             return (self.u, [])
##         else:
##             return self.u


## class NodeConstantScalar(NodeConstant):
##     @staticmethod
##     def compute_fixed_u_and_f(x):
##         """ Compute u(x) and f(x) for given x. """
##         return ([x], 0)

##     def __init__(self, a, **kwargs):
##         NodeConstant.__init__(self,
##                               [a],
##                               plates=np.shape(a),
##                               dims=[()],
##                               **kwargs)

    
