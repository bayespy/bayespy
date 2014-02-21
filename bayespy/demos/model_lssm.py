######################################################################
# Copyright (C) 2014 Jaakko Luttinen
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

"""
Linear state space models
"""

import numpy as np
import scipy
import matplotlib.pyplot as plt

from bayespy.nodes import GaussianMarkovChain
from bayespy.nodes import DriftingGaussianMarkovChain
from bayespy.nodes import GaussianArrayARD
from bayespy.nodes import Gamma
from bayespy.nodes import SumMultiply

from bayespy.utils import utils
from bayespy.utils import random

from bayespy.inference.vmp.vmp import VB
from bayespy.inference.vmp import transformations

import bayespy.plot.plotting as bpplt

def lssm(M, N, D, K=1, drift_C=False, drift_A=False):

    if (drift_C or drift_A) and not K > 0:
        raise ValueError("K must be positive integer when using drift")

    # Drift weights
    if drift_A or drift_C:
        # Dynamics matrix with ARD
        # beta : (K) x ()
        beta = Gamma(1e-5,
                     1e-5,
                     plates=(K,),
                     name='beta')
        # B : (K) x (K)
        B = GaussianArrayARD(np.identity(K),
                             beta,
                             shape=(K,),
                             plates=(K,),
                             name='B',
                             initialize=False)
        B.initialize_from_value(np.identity(K))

        # State of the drift, that is, temporal weights for dynamics matrices
        # S : () x (N,K)
        S = GaussianMarkovChain(np.ones(K),
                                1e-6*np.identity(K),
                                B,
                                np.ones(K),
                                n=N,
                                name='S',
                                initialize=False)
        S.initialize_from_value(np.ones((N,K))+0.01*np.random.randn(N,K))

    if not drift_A:

        # Dynamic matrix
        # alpha: (D) x ()
        alpha = Gamma(1e-5,
                      1e-5,
                      plates=(D,),
                      name='alpha')
        # A : (D) x (D)
        A = GaussianArrayARD(0,
                             alpha,
                             shape=(D,),
                             plates=(D,),
                             name='A',
                             initialize=False)
        A.initialize_from_value(np.identity(D))

        # Latent states with dynamics
        # X : () x (N,D)
        X = GaussianMarkovChain(np.zeros(D),         # mean of x0
                                1e-3*np.identity(D), # prec of x0
                                A,                   # dynamics
                                np.ones(D),          # innovation
                                n=N,                 # time instances
                                name='X',
                                initialize=False)
        X.initialize_from_value(np.random.randn(N,D))

    else:
        
        # Projection matrix of the dynamics matrix
        # alpha : (K) x ()
        alpha = Gamma(1e-5,
                      1e-5,
                      plates=(D,K),
                      name='alpha')
        # A : (D) x (D,K)
        a = np.zeros((D,D,K))
        a[np.arange(D),np.arange(D),np.zeros(D,dtype=int)] = 1
        A = GaussianArrayARD(0,
                             alpha,
                             shape=(D,K),
                             plates=(D,),
                             name='A',
                             initialize=False)

        # Initialize S and B such that B*S is almost an identity matrix
        A.initialize_from_value(np.reshape(1*a, (D,D,K)))

        # Latent states with dynamics
        # X : () x (N,D)
        X = DriftingGaussianMarkovChain(np.zeros(D),         # mean of x0
                                        1e-3*np.identity(D), # prec of x0
                                        A,                   # dynamics matrices
                                        S.as_gaussian()[1:], # temporal weights
                                        np.ones(D),          # innovation
                                        n=N,                 # time instances
                                        name='X',
                                        initialize=False)
        X.initialize_from_value(np.random.randn(N,D))

    if not drift_C:
        # Mixing matrix from latent space to observation space using ARD
        # gamma : (D) x ()
        gamma = Gamma(1e-5,
                      1e-5,
                      plates=(D,),
                      name='gamma')
        # C : (M,1) x (D)
        C = GaussianArrayARD(0,
                             gamma,
                             shape=(D,),
                             plates=(M,1),
                             name='C',
                             initialize=False)
        C.initialize_from_random()

        # Noiseless process
        # F : (M,N) x ()
        F = SumMultiply('d,d',
                        C,
                        X.as_gaussian(),
                        name='F')
    else:
        # Mixing matrix from latent space to observation space using ARD
        # gamma : (D,K) x ()
        gamma = Gamma(1e-5,
                      1e-5,
                      plates=(D,K),
                      name='gamma')
        # C : (M,1) x (D,K)
        C = GaussianArrayARD(0,
                             gamma,
                             shape=(D,K),
                             plates=(M,1),
                             name='C',
                             initialize=False)
        C.initialize_from_random()

        # Noiseless process
        # F : (M,N) x ()
        F = SumMultiply('dk,d,k',
                        C,
                        X.as_gaussian(),
                        S.as_gaussian(),
                        name='F')
        
                  
    # Observation noise
    # tau : () x ()
    tau = Gamma(1e-5,
                1e-5,
                name='tau')

    # Observations
    # Y: (M,N) x ()
    Y = GaussianArrayARD(F,
                         tau,
                         name='Y')

    # Construct inference machine
    if drift_C or drift_A:
        Q = VB(Y, F, X, S, B, beta, A, alpha, C, gamma, tau)
    else:
        Q = VB(Y, F, X, A, alpha, C, gamma, tau)

    return Q
    

def run_lssm(y, D, 
             K=1,
             mask=True, 
             maxiter=100,
             rotate=False, 
             debug=False, 
             precompute=False,
             drift_C=False,
             drift_A=False):
    
    """
    Run VB inference for linear state-space model with drifting dynamics.
    """
        
    (M, N) = np.shape(y)

    # Construct the model
    Q = lssm(M, N, D, K=K, drift_C=drift_C, drift_A=drift_A)

    # Observe data
    Q['Y'].observe(y, mask=mask)

    # Set up rotation speed-up
    if rotate:
        
        # Rotate the D-dimensional state space (X, A, C)
        rotA = transformations.RotateGaussianArrayARD(Q['A'], 
                                                      Q['alpha'],
                                                      axis=0,
                                                      precompute=precompute)
        if drift_A:
            rotX = transformations.RotateDriftingMarkovChain(Q['X'], 
                                                             Q['A'], 
                                                             Q['S'].as_gaussian()[...,1:,None], 
                                                             rotA)
        else:
            rotX = transformations.RotateGaussianMarkovChain(Q['X'], 
                                                             rotA)
        rotC = transformations.RotateGaussianArrayARD(Q['C'],
                                                      Q['gamma'],
                                                      axis=0,
                                                      precompute=precompute)
        R_X = transformations.RotationOptimizer(rotX, rotC, D)

        # Rotate the K-dimensional latent dynamics space (S, A, C)
        if drift_A or drift_C:
            rotB = transformations.RotateGaussianArrayARD(Q['B'],
                                                          Q['beta'], 
                                                          precompute=precompute)
            rotS = transformations.RotateGaussianMarkovChain(Q['S'], rotB)

            if drift_A:
                rotA = transformations.RotateGaussianArrayARD(Q['A'],
                                                              Q['alpha'],
                                                              axis=-1,
                                                              precompute=precompute)
            if drift_C:
                rotC = transformations.RotateGaussianArrayARD(Q['C'],
                                                              Q['gamma'],
                                                              axis=-1,
                                                              precompute=precompute)

            if drift_A and not drift_C:
                rotAC = rotA
            elif not drift_A and drift_C:
                rotAC = rotC
            else:
                rotAC = transformations.RotateMultiple(rotA, rotC)

            R_S = transformations.RotationOptimizer(rotS, rotAC, K)
            
        if debug:
            rotate_kwargs = {'check_bound': True,
                             'check_gradient': True}
        else:
            rotate_kwargs = {}

    # Run inference using rotations
    for ind in range(maxiter):
        Q.update()
        if rotate:
            R_X.rotate(**rotate_kwargs)
            if drift_A or drift_C:
                R_S.rotate(**rotate_kwargs)

    # Return the posterior approximation
    return Q
