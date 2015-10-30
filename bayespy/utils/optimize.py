################################################################################
# Copyright (C) 2011-2013 Jaakko Luttinen
#
# This file is licensed under the MIT License.
################################################################################


import numpy as np
from scipy import optimize

def minimize(f, x0, maxiter=None, verbose=False):
    """
    Simple wrapper for SciPy's optimize.

    The given function must return a tuple: (value, gradient).
    """
    options = {'disp': verbose}
    if maxiter is not None:
        options['maxiter'] = maxiter
    opt = optimize.minimize(f, x0, jac=True, method='CG', options=options)
    return opt.x

def check_gradient(f, x0, verbose=True, epsilon=optimize.optimize._epsilon, return_abserr=False):
    """
    Simple wrapper for SciPy's gradient checker.

    The given function must return a tuple: (value, gradient).

    Returns absolute and relative errors
    """
    df = f(x0)[1]
    df_num = optimize.approx_fprime(x0, 
                                    lambda x: f(x)[0], 
                                    epsilon)
    abserr = np.linalg.norm(df-df_num)
    norm_num = np.linalg.norm(df_num)
    if abserr == 0 and norm_num == 0:
        err = 0
    else:
        err = abserr / norm_num
    if verbose:
        print("Norm of numerical gradient: %g" % np.linalg.norm(df_num))
        print("Norm of function gradient:  %g" % np.linalg.norm(df))
        print("Gradient relative error = %g and absolute error = %g" % 
              (err,
               abserr))

    return (abserr, err)

