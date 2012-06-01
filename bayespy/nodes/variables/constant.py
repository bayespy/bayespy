import itertools
import numpy as np
import scipy as sp
import scipy.linalg.decomp_cholesky as decomp
import scipy.linalg as linalg
import scipy.special as special
import scipy.spatial.distance as distance

import imp

import utils
imp.reload(utils)

from ..node import Node
#import Node

def Constant(distribution):
    class _Constant(Node):

        @staticmethod
        def compute_fixed_moments(x):
            """ Compute u(x) for given x. """
            return distribution.compute_fixed_moments(x)
        
        @staticmethod
        def compute_fixed_u_and_f(x):
            """ Compute u(x) and f(x) for given x. """
            return distribution.compute_fixed_u_and_f(x)
        
        def __init__(self, x, **kwargs):
            # Compute moments
            self.u = distribution.compute_fixed_moments(x)
            # Dimensions of the moments
            dims = distribution.compute_dims_from_values(x)
            # Number of plate axes
            plates_ndim = np.ndim(x) - distribution.ndim_observations
            plates = np.shape(x)[:plates_ndim]
            # Parent constructor
            super().__init__(dims=dims, plates=plates, **kwargs)
            
        def get_moments(self):
            return self.u
        
    return _Constant
        
