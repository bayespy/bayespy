######################################################################
# Copyright (C) 2011-2013 Jaakko Luttinen
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
from scipy import optimize

def minimize(f, x0):
    """
    Simple wrapper for SciPy's optimize.

    The given function must return a tuple: (value, gradient).
    """
    opt = optimize.minimize(f, x0, jac=True)
    return opt.x

def check_gradient(f, x0, verbose=True):
    """
    Simple wrapper for SciPy's gradient checker.

    The given function must return a tuple: (value, gradient).
    """
    err = optimize.check_grad(lambda x: f(x)[0],
                              lambda x: f(x)[1],
                              np.atleast_1d(x0))
    if verbose:
        print("Gradient error = %g" % err)
    return err
    
