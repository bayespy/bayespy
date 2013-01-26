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
import matplotlib.pyplot as plt
import time

from bayespy.utils import utils
from bayespy.plot import plotting as myplt
import bayespy.inference.vmp.nodes as EF

import imp
imp.reload(utils)
imp.reload(myplt)
imp.reload(EF)

def categorical_model(M, D):

    p = EF.Dirichlet(1*np.ones(D), name='p')
    z = EF.Categorical(p, plates=(M,), name='z')
    return (z, p)


def run(M=30, D=5):

    # Generate data
    y = np.random.randint(D, size=(M,))

    # Construct model
    (z, p) = categorical_model(M, D)

    # Initialize nodes
    p.update()
    z.update()

    # Observe the data with randomly missing values
    mask = np.random.rand(M) < 0.5 # randomly missing
    z.observe(y, mask)

    # Inference loop.
    L_last = -np.inf
    for i in range(10):
        t = time.clock()

        # Update nodes
        p.update()
        z.update()

        # Compute lower bound
        L_p = p.lower_bound_contribution()
        L_z = z.lower_bound_contribution()
        L = L_p + L_z

        # Check convergence
        print("Iteration %d: loglike=%e (%.3f seconds)" % (i+1, L, time.clock()-t))
        if L_last > L:
            L_diff = (L_last - L)

        if L - L_last < 1e-12:
            print("Converged.")

        L_last = L


    z.show()
    p.show()

if __name__ == '__main__':
    run()

