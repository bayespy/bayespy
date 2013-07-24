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

from .variable import Variable
#from .constant import Constant
#from .wishart import Wishart

class Gaussian(Variable):

    #ndims = (1, 2)
    #ndims_parents = [(1, 2), (2, 0)]
    # Observations are vectors (1-D):
    ndim_observations = 1

    # Gaussian(mu, inv(Lambda))

    def __init__(self, mu, Lambda, plates=(), **kwargs):

        ## # Check for constant mu
        ## if np.isscalar(mu) or isinstance(mu, np.ndarray):
        ##     mu = Constant(Gaussian)(mu)

        ## # Check for constant Lambda
        ## if np.isscalar(Lambda) or isinstance(Lambda, np.ndarray):
        ##     Lambda = Constant(Wishart)(Lambda)

        ## # You could check whether the dimensions of mu and Lambda
        ## # match (and Lambda is square)
        ## if Lambda.dims[0][-1] != mu.dims[0][-1]:
        ##     raise Exception("Dimensionalities of mu and Lambda do not match.")

        # Construct
        super().__init__(mu, Lambda,
                         plates=plates,
                         **kwargs)
