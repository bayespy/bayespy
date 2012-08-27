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


import time
import numpy as np
import matplotlib.pyplot as plt
import bayespy.plotting as myplt

import bayespy.utils as utils
import bayespy.nodes as EF

# Reload everything (helpful for interactive sessions)
import imp
import bayespy.nodes
import bayespy.nodes.node
import bayespy.nodes.variables
import bayespy.nodes.variables.variable
import bayespy.nodes.variables.gamma
import bayespy.nodes.variables.gaussian
import bayespy.nodes.variables.normal
import bayespy.nodes.variables.dot
imp.reload(utils)
imp.reload(bayespy.nodes)
imp.reload(bayespy.nodes.node)
imp.reload(bayespy.nodes.variables)
imp.reload(bayespy.nodes.variables.variable)
imp.reload(bayespy.nodes.variables.gamma)
imp.reload(bayespy.nodes.variables.gaussian)
imp.reload(bayespy.nodes.variables.normal)
imp.reload(bayespy.nodes.variables.dot)

def rotate_pca(W, X, alpha, maxiter=10):
    # Rotate: W*R, X*R^-1

    W.start_rotation(to_children=False, to_parents=True)
    X.start_rotation(to_children=False, to_parents=True)

    # TODO: Sum over plates
    u_q_W = W.u
    u_q_alpha = alpha.u
    phi_q_alpha = alpha.phi

    phi_p_X = X.phi_of_p()
    g_p_X = 0

    phi_p_alpha = alpha.phi_of_p()
    g_p_alpha = 0

    f_X = X.get_rotation_cost_function(parents_transformed=False)
    f_W = W.get_rotation_cost_function(parents_transformed=False)
    (transform_x, cost_x) = X.start_rotation(..)
    (transform_w, cost_w) = W.start_rotation(..)
    # (transform_alpha, cost_alpha) = alpha.start_something(..)

    transform_x(invR, svd=(V.T,invS,U.T))
    transform_w(R, svd=(U,S,V))
    # transform_alpha(..)

    cost_x(gradient=True)
    cost_w(gradient=True)
    # cost_alpha()

    f_X.rotate(invR)
    f_W.rotate(R)
    # alpha.transform(..)
    # mu.transform(..)

    f_X.
    
    X.rotate(invR)
    W.rotate(R)
    alpha.translate_b(..)

    (transform_x, cost_x, gradient_x) = X.start_rotation()
    .. alpha.start_optimization_translate_b()

    # Transform
    transform_x(invR)
    transform_w(R)
    db = W.get_moments() ...
    transform_alpha(db)

    # Compute cost
    l += cost_x()
    l += cost_w()
    l += cost_alpha()
    
    # Compute gradient
    dR = gradient_x()
    

    def get_rotation_cost_gaussian(X):
        # TODO: Sum over plates
        u_q_X = X.get_rotateable_moments()
        #u_q_X = X.u
        
        # Assume that the parents are not optimized at the same time,
        # so we can keep their moments fixed:
        phi_p_X = X.phi_of_p()

        def rotation_cost(R, svd=None, gradient=False):
            if svd is None:
                (U,S,V) = np.svd(R)
            else:
                (U,S,V) = svd
            # Rotate the moments and compute the cost
            u_qh_X = X.rotate_moments(u_q_X, R)
            log_qh_X = N_X * X.rotation_cost(U,S,V, gradient=True)
            log_ph_X = X.compute_logpdf(u_qh_X,
                                        phi_p_X,
                                        0,
                                        0,
                                        gradient=True)
            l = log_qh_X + log_ph_X
            return l

        return rotation_cost

    def get_rotation_cost_gaussian_ard(X, alpha):
        # TODO: Sum over plates. Ignore observations and ignorable
        # plates? You shouldn't rotate observed values! Actually,
        # variables shouldn't have observed/fixed values at all.
        u_q_X = X.u
        u_q_mu = X.parents[0].u
        u_q_alpha = alpha.u
        phi_q_alpha = alpha.phi
        phi_p_alpha = alpha.phi_of_p()

        def rotation_cost(R, svd=None, gradient=False):
            if svd is None:
                (U,S,V) = np.svd(R)
            else:
                (U,S,V) = svd
            # Rotate X
            # ???: What if something is observed?
            u_qh_X = X.rotate_moments(u_q_X, R)
            # Scale alpha, i.e., change parameter b
            phi_qh_alpha = [phi_q_alpha[0],
                            phi_q_alpha[1] + xhxh - xx]
            u_qh_alpha = alpha.compute_moments(phi_qh_alpha)
            # Compute terms for <p(X|alpha)> over transformed alpha.
            phi_ph_X = X.compute_phi_from_parents([u_q_mu,
                                                   u_qh_alpha])
            g_ph_X = X.compute_g_from_parents([u_q_mu,
                                               u_qh_alpha])
            # Cost for X
            log_qh_X = N_X * X.rotation_cost(U,S,V)
            log_ph_X = X.compute_logpdf(u_qh_X,
                                        phi_ph_X,
                                        g_ph_X,
                                        0)
            # Cost for alpha
            log_qh_alpha = N_alpha * alpha_w.scaling_entropy(scale_alpha)
            log_ph_alpha = alpha_w.compute_logpdf(u_qh_alpha,
                                                  phi_p_alpha,
                                                  0,
                                                  0)

        return rotation_cost
            
            

    def rotation_cost(R):
        # SVD of rotation
        (u,s,v) = np.svd(R)

        # Transform moments of W
        #uh_W = [np.dot(u_W[0],R.T),
        #        np.dot(np.dot(R, u_W[1]), R.T)]
        u_qh_W = W.rotate_moments(uh_W, R, gradient=True)
        u_qh_X = X.rotate_moments(uh_X, invR)

        # Transform moments of alpha
        # ???
        phi_qh_alpha = [phi_q_alpha[0],
                        phi_q_alpha[1] + whwh - ww]
        #uh_aw = alpha_w.moments_from_translated_parameters(0,whwh-ww)
        u_qh_alpha = alpha.compute_moments(phi_qh_alpha)
        scale_alpha = u_qh_alpha[0] / u_q_alpha[0]
        #scale_w = (aw - ww + whwh) / aw
        #uh_aw = [scale_w * u_aw[0],
        #         np.log(scale_w) + u_aw[1]
        #uh_aw = alpha_w.scale_moments(u_aw, scale_w)
        
        phi_ph_W = W.compute_phi_from_parents([..])
        g_ph_W = W.compute_g_from_parents([..])
        
                
        log_qh_W = N_W * W.rotation_entropy(u,s,v, gradient=True)
        # gradient?
        log_ph_W = W.compute_logpdf(u_qh_W,
                                  phi_ph_W,
                                  g_ph_W,
                                  0)

        log_qh_X = X.rotation_cost(u,s,v)
        log_ph_X = X.compute_logpdf(u_qh_X,
                                  phi_p_X,
                                  g_p_X,
                                  0)

        # These must be Wishart nodes?

        log_qh_alpha = N_alpha * alpha_w.scaling_entropy(scale_alpha)
        log_ph_alpha = alpha_w.compute_logpdf(u_qh_alpha,
                                              phi_p_alpha,
                                              g_p_alpha,
                                              0)


    # Find optimal rotation
    R = np.identity(D)
    R = optimize(rotation_cost, R, maxiter=maxiter)

    # Apply rotation
    (u,s,v) = np.svd(R)
    W.rotate(R)
    X.rotate(invR)
    alpha_w.scale(??)
    alpha_x.scale(??)

def pca_model(M, N, D):
    # Construct the PCA model with ARD

    # ARD
    alpha = EF.Gamma(1e-10,
                     1e-10,
                     plates=(D,),
                     name='alpha')

    # Loadings
    W = EF.Gaussian(np.zeros(D),
                    EF.GammaToDiagonalWishart(alpha),
                    name="W",
                    plates=(M,1))

    # States
    X = EF.Gaussian(np.zeros(D),
                    np.identity(D),
                    name="X",
                    plates=(1,N))

    # PCA
    WX = EF.Dot(W,X)

    # Noise
    tau = EF.Gamma(1e-5, 1e-5, name="tau", plates=())

    # Noisy observations
    Y = EF.Normal(WX, tau, name="Y", plates=(M,N))

    return (Y, WX, W, X, tau, alpha)


def run(M=10, N=100, D_y=3, D=5):
    # Generate data
    w = np.random.normal(0, 1, size=(M,1,D_y))
    x = np.random.normal(0, 1, size=(1,N,D_y))
    f = utils.sum_product(w, x, axes_to_sum=[-1])
    y = f + np.random.normal(0, 0.5, size=(M,N))

    # Construct model
    (Y, WX, W, X, tau, alpha) = pca_model(M, N, D)

    # Initialize nodes (from prior and randomly)
    alpha.initialize_from_prior()
    tau.initialize_from_prior()
    X.initialize_from_prior()
    X.initialize_from_parameters(X.random(), np.identity(D))
    W.initialize_from_prior()
    W.initialize_from_parameters(W.random(), np.identity(D))

    # Data with missing values
    mask = np.random.rand(M,N) < 0.4 # randomly missing
    mask[:,20:40] = False # gap missing
    Y.observe(y, mask)

    # Inference loop.
    maxiter = 10000
    L_X = np.zeros(maxiter)
    L_W = np.zeros(maxiter)
    L_tau = np.zeros(maxiter)
    L_Y = np.zeros(maxiter)
    L_alpha = np.zeros(maxiter)
    L = np.zeros(maxiter)
    L_last = -np.inf
    for i in range(maxiter):
        t = time.clock()

        # Update nodes
        X.update()
        W.update()
        tau.update()
        alpha.update()

        # Compute lower bound
        L_X[i] = X.lower_bound_contribution()
        L_W[i] = W.lower_bound_contribution()
        L_tau[i] = tau.lower_bound_contribution()
        L_Y[i] = Y.lower_bound_contribution()
        L_alpha[i] = alpha.lower_bound_contribution()
        L[i] = L_X[i] + L_W[i] + L_tau[i] + L_Y[i] + L_alpha[i]
        #print('loglike terms:', L_X, L_W, L_tau, L_Y, L_alpha)

        # Check convergence
        print("Iteration %d: loglike=%e (%.3f seconds)" % (i+1, L[i], time.clock()-t))
        if L_last > L[i]:
            L_diff = (L_last - L[i])
            print("Lower bound decreased %e! Bug somewhere or numerical inaccuracy?" % L_diff)
        if L[i] - L_last < 1e-12:
            print("Converged.")
            #break
        L_last = L[i]

    if False:
        # Optimization:
        # - initialize for optimization
        # - iteration:
        #   - set new values to nodes
        #   - compute cost and gradient
        # - terminate optimization
        
        # Rotation pseudo code:
        utils.rotate_pca(W, X, alpha, maxiter=10)
        rot = Rotation(W, X, alpha)
        R = rot.optimize(maxiter=50)
        cw = W.get_rotation_cost_function()
        cx = X.get_rotation_cost_function()
        ca = alpha.get_cost_function()
        ca = @(R) ca(
        cost = cw + cx + ca
        R = optimize(cost, identity(D))
        W.rotate(R)
        X.rotate(R)
        alpha.???(R)

    tau.show()

    ## plt.clf()
    ## Z = np.vstack([L_X,L_W,L_tau,L_alpha,L_Y,L]).T
    ## dZ = np.diff(Z, axis=0)
    ## ax = plt.plot(dZ[50:])
    ## plt.legend(ax)
    ## return


    plt.ion()
    #plt.figure()
    plt.clf()
    WX_params = WX.get_parameters()
    fh = WX_params[0] * np.ones(y.shape)
    err_fh = 2*np.sqrt(WX_params[1]) * np.ones(y.shape)
    for m in range(M):
        plt.subplot(M,1,m)
        #errorplot(y, error=None, x=None, lower=None, upper=None):
        myplt.errorplot(fh[m], x=np.arange(N), error=err_fh[m])
        plt.plot(np.arange(N), f[m], 'g')
        plt.plot(np.arange(N), y[m], 'r+')


if __name__ == '__main__':
    # FOR INTERACTIVE SESSIONS, NON-BLOCKING PLOTTING:
    plt.ion()
    run()

