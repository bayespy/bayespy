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

from .variable import Variable
from .constant import Constant

class Gamma(Variable):

    ndims = (0, 0)

    @staticmethod
    def compute_phi_from_parents(u_parents):
        return [-u_parents[1][0],
                1*u_parents[0][0]]

    @staticmethod
    def compute_g_from_parents(u_parents):
        a = u_parents[0][0]
        gammaln_a = special.gammaln(a)
        b = u_parents[1][0]
        log_b = u_parents[1][1]
        g = a * log_b - gammaln_a
        return g

    @staticmethod
    def compute_u_and_g(phi, mask=True):
        log_b = np.log(-phi[0])
        u0 = phi[1] / (-phi[0])
        u1 = special.digamma(phi[1]) - log_b
        u = [u0, u1]
        g = phi[1] * log_b - special.gammaln(phi[1])
        return (u, g)
        

    @staticmethod
    def compute_message(index, u, u_parents):
        """ . """
        if index == 0:
            raise Exception("No analytic solution exists")
        elif index == 1:
            return [-u[0],
                    u_parents[0][0]]

    @staticmethod
    def compute_dims(*parents):
        """ Compute the dimensions of phi/u. """
        return [(), ()]

    def __init__(self, a, b, plates=(), **kwargs):

        # TODO: USE asarray(a)

        # Check for constant a
        if np.isscalar(a) or isinstance(a, np.ndarray):
            a = NodeConstantScalar(a)

        # Check for constant b
        if np.isscalar(b) or isinstance(b, np.ndarray):
            b = NodeConstant([b, np.log(b)], plates=np.shape(b), dims=[(),()])

        # Construct
        super().__init__(a, b, plates=plates, **kwargs)

    def show(self):
        a = self.phi[1]
        b = -self.phi[0]
        print("Gamma(" + str(a) + ", " + str(b) + ")")
