################################################################################
# Copyright (C) 2013-2014 Jaakko Luttinen
#
# This file is licensed under the MIT License.
################################################################################


"""
Unit tests for gaussian_markov_chain module.
"""

import numpy as np

from ..gaussian_markov_chain import GaussianMarkovChain
from ..gaussian_markov_chain import VaryingGaussianMarkovChain
from ..gaussian import Gaussian, GaussianMoments
from ..gaussian import GaussianARD
from ..wishart import Wishart, WishartMoments
from ..gamma import Gamma, GammaMoments

from bayespy.utils import random
from bayespy.utils import linalg
from bayespy.utils import misc

from bayespy.utils.misc import TestCase


def kalman_filter(y, U, A, V, mu0, Cov0, out=None):
    """
    Perform Kalman filtering to obtain filtered mean and covariance.

    The parameters of the process may vary in time, thus they are
    given as iterators instead of fixed values.

    Parameters
    ----------
    y : (N,D) array
        "Normalized" noisy observations of the states, that is, the
        observations multiplied by the precision matrix U (and possibly
        other transformation matrices).
    U : (N,D,D) array or N-list of (D,D) arrays
        Precision matrix (i.e., inverse covariance matrix) of the observation 
        noise for each time instance.
    A : (N-1,D,D) array or (N-1)-list of (D,D) arrays
        Dynamic matrix for each time instance.
    V : (N-1,D,D) array or (N-1)-list of (D,D) arrays
        Covariance matrix of the innovation noise for each time instance.

    Returns
    -------
    mu : array
        Filtered mean of the states.
    Cov : array
        Filtered covariance of the states.

    See also
    --------
    rts_smoother
    """
    mu = mu0
    Cov = Cov0

    # Allocate memory for the results
    (N,D) = np.shape(y)
    X = np.empty((N,D))
    CovX = np.empty((N,D,D))

    # Update step for t=0
    M = np.dot(np.dot(Cov, U[0]), Cov) + Cov
    L = linalg.chol(M)
    mu = np.dot(Cov, linalg.chol_solve(L, np.dot(Cov,y[0]) + mu))
    Cov = np.dot(Cov, linalg.chol_solve(L, Cov))
    X[0,:] = mu
    CovX[0,:,:] = Cov

    #for (yn, Un, An, Vn) in zip(y, U, A, V):
    for n in range(len(y)-1): #(yn, Un, An, Vn) in zip(y, U, A, V):
        # Prediction step
        mu = np.dot(A[n], mu)
        Cov = np.dot(np.dot(A[n], Cov), A[n].T) + V[n]
        # Update step
        M = np.dot(np.dot(Cov, U[n+1]), Cov) + Cov
        L = linalg.chol(M)
        mu = np.dot(Cov, linalg.chol_solve(L, np.dot(Cov,y[n+1]) + mu))
        Cov = np.dot(Cov, linalg.chol_solve(L, Cov))

        # Force symmetric covariance (for numeric inaccuracy)
        Cov = 0.5*Cov + 0.5*Cov.T

        # Store results
        X[n+1,:] = mu
        CovX[n+1,:,:] = Cov

    return (X, CovX)


def rts_smoother(mu, Cov, A, V, removethis=None):
    """
    Perform Rauch-Tung-Striebel smoothing to obtain the posterior.

    The function returns the posterior mean and covariance of each
    state. The parameters of the process may vary in time, thus they
    are given as iterators instead of fixed values.

    Parameters
    ----------
    mu : (N,D) array
        Mean of the states from Kalman filter.
    Cov : (N,D,D) array
        Covariance of the states from Kalman filter. 
    A : (N-1,D,D) array or (N-1)-list of (D,D) arrays
        Dynamic matrix for each time instance.
    V : (N-1,D,D) array or (N-1)-list of (D,D) arrays
        Covariance matrix of the innovation noise for each time instance.

    Returns
    -------
    mu : array
        Posterior mean of the states.
    Cov : array
        Posterior covariance of the states.

    See also
    --------
    kalman_filter
    """

    N = len(mu)
    #n = N-1

    # Start from the last time instance and smoothen backwards
    x = mu[-1,:]
    Covx = Cov[-1,:,:]

    for n in reversed(range(N-1)):#(An, Vn) in zip(reversed(A), reversed(V)):

        #n = n - 1
        #if n <= 0:
        #    break

        # The predicted value of n
        x_p = np.dot(A[n], mu[n,:])
        Cov_p = np.dot(np.dot(A[n], Cov[n,:,:]), A[n].T) + V[n]

        # Temporary variable
        S = np.linalg.solve(Cov_p, np.dot(A[n], Cov[n,:,:]))

        # Smoothed value of n
        x = mu[n,:] + np.dot(S.T, x-x_p)
        Covx = Cov[n,:,:] + np.dot(np.dot(S.T, Covx-Cov_p), S)

        # Force symmetric covariance (for numeric inaccuracy)
        Covx = 0.5*Covx + 0.5*Covx.T

        # Store results
        mu[n,:] = x
        Cov[n,:] = Covx


    return (mu, Cov)


class TestGaussianMarkovChain(TestCase):

    def create_model(self, N, D):

        # Construct the model
        Mu = Gaussian(np.random.randn(D),
                      np.identity(D))
        Lambda = Wishart(D,
                         random.covariance(D))
        A = Gaussian(np.random.randn(D,D),
                     np.identity(D))
        V = Gamma(D,
                  np.random.rand(D))
        X = GaussianMarkovChain(Mu, Lambda, A, V, n=N)
        Y = Gaussian(X, np.identity(D))

        return (Y, X, Mu, Lambda, A, V)
        

    def test_plates(self):
        """
        Test that plates are handled correctly.
        """

    def test_message_to_mu0(self):
        pass

    def test_message_to_Lambda0(self):
        pass

    def test_message_to_A(self):
        pass

    def test_message_to_v(self):
        pass

    def test_message_to_child(self):
        """
        Test the updating of GaussianMarkovChain.

        Check that the moments and the lower bound contribution are computed
        correctly.
        """

        # TODO: Add plates and missing values!

        # Dimensionalities
        D = 3
        N = 5
        (Y, X, Mu, Lambda, A, V) = self.create_model(N, D)

        # Inference with arbitrary observations
        y = np.random.randn(N,D)
        Y.observe(y)
        X.update()
        (x_vb, xnxn_vb, xpxn_vb) = X.get_moments()

        # Get parameter moments
        (mu0, mumu0) = Mu.get_moments()
        (icov0, logdet0) = Lambda.get_moments()
        (a, aa) = A.get_moments()
        (icov_x, logdetx) = V.get_moments()
        icov_x = np.diag(icov_x)
        # Prior precision
        Z = np.einsum('...kij,...kk->...ij', aa, icov_x)
        U_diag = [icov0+Z] + (N-2)*[icov_x+Z] + [icov_x]
        U_super = (N-1) * [-np.dot(a.T, icov_x)]
        U = misc.block_banded(U_diag, U_super)
        # Prior mean
        mu_prior = np.zeros(D*N)
        mu_prior[:D] = np.dot(icov0,mu0)
        # Data 
        Cov = np.linalg.inv(U + np.identity(D*N))
        mu = np.dot(Cov, mu_prior + y.flatten())
        # Moments
        xx = mu[:,np.newaxis]*mu[np.newaxis,:] + Cov
        mu = np.reshape(mu, (N,D))
        xx = np.reshape(xx, (N,D,N,D))

        # Check results
        self.assertAllClose(x_vb, mu,
                            msg="Incorrect mean")
        for n in range(N):
            self.assertAllClose(xnxn_vb[n,:,:], xx[n,:,n,:],
                                msg="Incorrect second moment")
        for n in range(N-1):
            self.assertAllClose(xpxn_vb[n,:,:], xx[n,:,n+1,:],
                                msg="Incorrect lagged second moment")


        # Compute the entropy H(X)
        ldet = linalg.logdet_cov(Cov)
        H = random.gaussian_entropy(-ldet, N*D)
        # Compute <log p(X|...)>
        xx = np.reshape(xx, (N*D, N*D))
        mu = np.reshape(mu, (N*D,))
        ldet = -logdet0 - np.sum(np.ones((N-1,D))*logdetx)
        P = random.gaussian_logpdf(np.einsum('...ij,...ij', 
                                                   xx, 
                                                   U),
                                         np.einsum('...i,...i', 
                                                   mu, 
                                                   mu_prior),
                                         np.einsum('...ij,...ij', 
                                                   mumu0,
                                                   icov0),
                                         -ldet,
                                         N*D)
                                                   
        # The VB bound from the net
        l = X.lower_bound_contribution()

        self.assertAllClose(l, H+P)
                                                   

        # Compute the true bound <log p(X|...)> + H(X)


        #
        # Simple tests
        #

        def check(N, D, plates=None, mu=None, Lambda=None, A=None, V=None):
            if mu is None:
                mu = np.random.randn(D)
            if Lambda is None:
                Lambda = random.covariance(D)
            if A is None:
                A = np.random.randn(D,D)
            if V is None:
                V = np.random.rand(D)
            X = GaussianMarkovChain(mu,
                                    Lambda,
                                    A,
                                    V,
                                    plates=plates,
                                    n=N)
            (u0, u1, u2) = X._message_to_child()
            (mu, mumu) = Gaussian._ensure_moments(mu, GaussianMoments, ndim=1).get_moments()
            (Lambda, _) = Wishart._ensure_moments(Lambda, WishartMoments, ndim=1).get_moments()
            (a, aa) = Gaussian._ensure_moments(A, GaussianMoments, ndim=1).get_moments()
            a = a * np.ones((N-1,D,D))     # explicit broadcasting for simplicity
            aa = aa * np.ones((N-1,D,D,D)) # explicit broadcasting for simplicity
            (v, _) = Gamma._ensure_moments(V, GammaMoments).get_moments()
            v = v * np.ones((N-1,D))
            plates_C = X.plates
            plates_mu = X.plates
            C = np.zeros(plates_C + (N,D,N,D))
            plates_mu = np.shape(mu)[:-1]
            m = np.zeros(plates_mu + (N,D))
            m[...,0,:] = np.einsum('...ij,...j->...i', Lambda, mu)
            C[...,0,:,0,:] = Lambda + np.einsum('...dij,...d->...ij',
                                                aa[...,0,:,:,:],
                                                v[...,0,:])
            for n in range(1,N-1):
                C[...,n,:,n,:] = (np.einsum('...dij,...d->...ij',
                                            aa[...,n,:,:,:],
                                            v[...,n,:])
                                  + v[...,n,:,None] * np.identity(D))
            for n in range(N-1):
                C[...,n,:,n+1,:] = -np.einsum('...di,...d->...id',
                                              a[...,n,:,:],
                                              v[...,n,:])
                C[...,n+1,:,n,:] = -np.einsum('...di,...d->...di',
                                              a[...,n,:,:],
                                              v[...,n,:])
            C[...,-1,:,-1,:] = v[...,-1,:,None]*np.identity(D)
            C = np.reshape(C, plates_C+(N*D,N*D))
            Cov = np.linalg.inv(C)
            Cov = np.reshape(Cov, plates_C+(N,D,N,D))
            m0 = np.einsum('...minj,...nj->...mi', Cov, m)
            m1 = np.zeros(plates_C+(N,D,D))
            m2 = np.zeros(plates_C+(N-1,D,D))
            for n in range(N):
                m1[...,n,:,:] = Cov[...,n,:,n,:] + np.einsum('...i,...j->...ij',
                                                             m0[...,n,:],
                                                             m0[...,n,:])
            for n in range(N-1):
                m2[...,n,:,:] = Cov[...,n,:,n+1,:] + np.einsum('...i,...j->...ij',
                                                               m0[...,n,:],
                                                               m0[...,n+1,:])
            self.assertAllClose(m0, u0*np.ones(np.shape(m0)))
            self.assertAllClose(m1, u1*np.ones(np.shape(m1)))
            self.assertAllClose(m2, u2*np.ones(np.shape(m2)))

            pass

        check(4,1)
        check(4,3)

        #
        # Test mu
        #

        # Simple
        check(4,3,
              mu=Gaussian(np.random.randn(3),
                          random.covariance(3)))
        # Plates
        check(4,3,
              mu=Gaussian(np.random.randn(5,6,3),
                          random.covariance(3),
                          plates=(5,6)))
        # Plates with moments broadcasted over plates
        check(4,3,
              mu=Gaussian(np.random.randn(3),
                          random.covariance(3),
                          plates=(5,)))
        check(4,3,
              mu=Gaussian(np.random.randn(1,3),
                          random.covariance(3),
                          plates=(5,)))
        # Plates broadcasting
        check(4,3,
              plates=(5,),
              mu=Gaussian(np.random.randn(3),
                          random.covariance(3),
                          plates=()))
        check(4,3,
              plates=(5,),
              mu=Gaussian(np.random.randn(1,3),
                          random.covariance(3),
                          plates=(1,)))

        #
        # Test Lambda
        #
            
        # Simple
        check(4,3,
              Lambda=Wishart(10+np.random.rand(),
                             random.covariance(3)))
        # Plates
        check(4,3,
              Lambda=Wishart(10+np.random.rand(),
                             random.covariance(3),
                             plates=(5,6)))
        # Plates with moments broadcasted over plates
        check(4,3,
              Lambda=Wishart(10+np.random.rand(),
                             random.covariance(3),
                             plates=(5,)))
        check(4,3,
              Lambda=Wishart(10+np.random.rand(1),
                             random.covariance(3),
                             plates=(5,)))
        # Plates broadcasting
        check(4,3,
              plates=(5,),
              Lambda=Wishart(10+np.random.rand(),
                             random.covariance(3),
                             plates=()))
        check(4,3,
              plates=(5,),
              Lambda=Wishart(10+np.random.rand(),
                             random.covariance(3),
                             plates=(1,)))

        #
        # Test A
        #

        # Simple
        check(4,3,
              A=GaussianARD(np.random.randn(3,3),
                            np.random.rand(3,3),
                            shape=(3,),
                            plates=(3,)))
        # Plates on time axis
        check(5,3,
              A=GaussianARD(np.random.randn(4,3,3),
                            np.random.rand(4,3,3),
                            shape=(3,),
                            plates=(4,3)))
        # Plates on time axis with broadcasted moments
        check(5,3,
              A=GaussianARD(np.random.randn(1,3,3),
                            np.random.rand(1,3,3),
                            shape=(3,),
                            plates=(4,3)))
        check(5,3,
              A=GaussianARD(np.random.randn(3,3),
                            np.random.rand(3,3),
                            shape=(3,),
                            plates=(4,3)))
        # Plates
        check(4,3,
              A=GaussianARD(np.random.randn(5,6,1,3,3),
                            np.random.rand(5,6,1,3,3),
                            shape=(3,),
                            plates=(5,6,1,3)))
        # Plates with moments broadcasted over plates
        check(4,3,
              A=GaussianARD(np.random.randn(3,3),
                            np.random.rand(3,3),
                            shape=(3,),
                            plates=(5,1,3)))
        check(4,3,
              A=GaussianARD(np.random.randn(1,1,3,3),
                            np.random.rand(1,1,3,3),
                            shape=(3,),
                            plates=(5,1,3)))
        # Plates broadcasting
        check(4,3,
              plates=(5,),
              A=GaussianARD(np.random.randn(3,3),
                            np.random.rand(3,3),
                            shape=(3,),
                            plates=(3,)))
        check(4,3,
              plates=(5,),
              A=GaussianARD(np.random.randn(3,3),
                            np.random.rand(3,3),
                            shape=(3,),
                            plates=(1,1,3)))

        #
        # Test v
        #
        
        # Simple
        check(4,3,
              V=Gamma(np.random.rand(1,3),
                      np.random.rand(1,3),
                      plates=(1,3)))
        check(4,3,
              V=Gamma(np.random.rand(3),
                      np.random.rand(3),
                      plates=(3,)))
        # Plates
        check(4,3,
              V=Gamma(np.random.rand(5,6,1,3),
                      np.random.rand(5,6,1,3),
                      plates=(5,6,1,3)))
        # Plates with moments broadcasted over plates
        check(4,3,
              V=Gamma(np.random.rand(1,3),
                      np.random.rand(1,3),
                      plates=(5,1,3)))
        check(4,3,
              V=Gamma(np.random.rand(1,1,3),
                      np.random.rand(1,1,3),
                      plates=(5,1,3)))
        # Plates broadcasting
        check(4,3,
              plates=(5,),
              V=Gamma(np.random.rand(1,3),
                      np.random.rand(1,3),
                      plates=(1,3)))
        check(4,3,
              plates=(5,),
              V=Gamma(np.random.rand(1,1,3),
                      np.random.rand(1,1,3),
                      plates=(1,1,3)))

        #
        # Check with input signals
        #

        mu = 2
        Lambda = 3
        A = 4
        B = 5
        v = 6
        inputs = [[-2], [3]]
        X = GaussianMarkovChain([mu], [[Lambda]], [[A,B]], [v], inputs=inputs)
        V = (np.array([[v*A**2, -v*A,    0],
                       [-v*A,    v*A**2, -v*A],
                       [0,       -v*A,   0]]) +
             np.array([[Lambda, 0, 0],
                       [0,      v, 0],
                       [0,      0, v]]))
        m = (np.array([Lambda*mu, 0, 0]) +
             np.array([0, v*B*inputs[0][0], v*B*inputs[1][0]]) -
             np.array([v*A*B*inputs[0][0], v*A*B*inputs[1][0], 0]))
        Cov = np.linalg.inv(V)
        mean = np.dot(Cov, m)

        X.update()
        u = X.get_moments()

        self.assertAllClose(u[0], mean[:,None])
        self.assertAllClose(u[1] - u[0][...,None,:]*u[0][...,:,None],
                            Cov[(0,1,2),(0,1,2),None,None])
        self.assertAllClose(u[2] - u[0][...,:-1,:,None]*u[0][...,1:,None,:],
                            Cov[(0,1),(1,2),None,None])

        pass
        
        

    def test_smoothing(self):
        """
        Test the posterior estimation of GaussianMarkovChain.

        Create time-variant dynamics and compare the results of BayesPy VB
        inference and standard Kalman filtering & smoothing.

        This is not that useful anymore, because the moments are checked much
        better in another test method.
        """

        #
        # Set up an artificial system
        #

        # Dimensions
        N = 500
        D = 2
        # Dynamics (time varying)
        A0 = np.array([[.9, -.4], [.4, .9]])
        A1 = np.array([[.98, -.1], [.1, .98]])
        l = np.linspace(0, 1, N-1).reshape((-1,1,1))
        A = (1-l)*A0 + l*A1
        # Innovation covariance matrix (time varying)
        v = np.random.rand(D)
        V = np.diag(v)
        # Observation noise covariance matrix
        C = np.identity(D)

        #
        # Simulate data
        #
        
        X = np.empty((N,D))
        Y = np.empty((N,D))

        x = np.array([0.5, -0.5])
        X[0,:] = x
        Y[0,:] = x + np.random.multivariate_normal(np.zeros(D), C)
        for n in range(N-1):
            x = np.dot(A[n,:,:],x) + np.random.multivariate_normal(np.zeros(D), V)
            X[n+1,:] = x
            Y[n+1,:] = x + np.random.multivariate_normal(np.zeros(D), C)

        #
        # BayesPy inference
        #

        # Construct VB model
        Xh = GaussianMarkovChain(np.zeros(D), np.identity(D), A, 1/v, n=N)
        Yh = Gaussian(Xh, np.identity(D), plates=(N,))
        # Put data 
        Yh.observe(Y)
        # Run inference
        Xh.update()
        # Store results
        Xh_vb = Xh.u[0]
        CovXh_vb = Xh.u[1] - Xh_vb[...,np.newaxis,:] * Xh_vb[...,:,np.newaxis]

        #
        # "The ground truth" using standard Kalman filter and RTS smoother
        #
        V = N*(V,)
        UY = Y
        U = N*(C,)
        (Xh, CovXh) = kalman_filter(UY, U, A, V, np.zeros(D), np.identity(D))
        (Xh, CovXh) = rts_smoother(Xh, CovXh, A, V)

        #
        # Check results
        #
        self.assertTrue(np.allclose(Xh_vb, Xh))
        self.assertTrue(np.allclose(CovXh_vb, CovXh))
        

class TestVaryingGaussianMarkovChain(TestCase):

    def test_plates_from_parents(self):
        """
        Test that VaryingGaussianMarkovChain deduces plates correctly
        """
        def check(plates_X,
                  plates_mu=(),
                  plates_Lambda=(),
                  plates_B=(),
                  plates_S=(),
                  plates_v=()):
            
            D = 3
            K = 2
            N = 4

            np.random.seed(42)
            mu = Gaussian(np.random.randn(*(plates_mu+(D,))),
                          random.covariance(D))
            Lambda = Wishart(D+np.ones(plates_Lambda),
                             random.covariance(D))
            B = GaussianARD(np.random.randn(*(plates_B+(D,D,K))),
                            1+np.random.rand(*(plates_B+(D,D,K))),
                            shape=(D,K),
                            plates=plates_B+(D,))
            S = GaussianARD(np.random.randn(*(plates_S+(N,K))),
                            1+np.random.rand(*(plates_S+(N,K))),
                            shape=(K,),
                            plates=plates_S+(N,))
            v = Gamma(1+np.random.rand(*(plates_v+(1,D))),
                      1+np.random.rand(*(plates_v+(1,D))))
            X = VaryingGaussianMarkovChain(mu, Lambda, B, S, v, name="X")
            self.assertEqual(plates_X, X.plates,
                             msg="Incorrect plates deduced")
            pass

        check(())
        check((2,3),
              plates_mu=(2,3))
        check((6,7),
              plates_Lambda=(6,7))
        check((2,3),
              plates_B=(2,3))
        check((2,3),
              plates_S=(2,3))
        check((2,3),
              plates_v=(2,3))
        pass

    def test_message_to_child(self):

        # A very simple check before the more complex ones:
        # 1-D process, k=1, fixed constant parameters
        m = 1.0
        l = 4.0
        b = 2.0
        s = [3.0, 8.0]
        v = 5.0
        X = VaryingGaussianMarkovChain([m],
                                       [[l]],
                                       [[[b]]],
                                       [[s[0]],[s[1]]],
                                       [v])
        (u0, u1, u2) = X._message_to_child()
        C = np.array([[l+b**2*s[0]**2*v,        -b*s[0]*v,         0],
                      [       -b*s[0]*v, v+b**2*s[1]**2*v, -b*s[1]*v],
                      [               0,        -b*s[1]*v,         v]])
        Cov = np.linalg.inv(C)
        m0 = np.dot(Cov, [[l*m], [0], [0]])
        m1 = np.diag(Cov)[:,None,None] + m0[:,:,None]**2
        m2 = np.diag(Cov, k=1)[:,None,None] + m0[1:,:,None]*m0[:-1,:,None]
        self.assertAllClose(m0, u0)
        self.assertAllClose(m1, u1)
        self.assertAllClose(m2, u2)

        def check(N, D, K, plates=None, mu=None, Lambda=None, B=None, S=None, V=None):
            if mu is None:
                mu = np.random.randn(D)
            if Lambda is None:
                Lambda = random.covariance(D)
            if B is None:
                B = np.random.randn(D,D,K)
            if S is None:
                S = np.random.randn(N-1,K)
            if V is None:
                V = np.random.rand(D)
            X = VaryingGaussianMarkovChain(mu,
                                           Lambda,
                                           B,
                                           S,
                                           V,
                                           plates=plates,
                                           n=N)
            (u0, u1, u2) = X._message_to_child()
            (mu, mumu) = X.parents[0].get_moments()
            (Lambda, _) = X.parents[1].get_moments()
            (b, bb) = X.parents[2].get_moments()
            (s, ss) = X.parents[3].get_moments()
            (v, _) = X.parents[4].get_moments()
            v = v * np.ones((N-1,D))
            #V = np.atleast_3d(v)[...,-1,:,None]*np.identity(D)
            plates_C = X.plates
            plates_mu = X.plates
            C = np.zeros(plates_C + (N,D,N,D))
            plates_mu = np.shape(mu)[:-1]
            m = np.zeros(plates_mu + (N,D))
            m[...,0,:] = np.einsum('...ij,...j->...i', Lambda, mu)
            #m = np.reshape(m, plates_mu + (N*D,))
            A = np.einsum('...dik,...nk->...ndi', b, s)
            AA = np.einsum('...dikjl,...nkl->...ndij', bb, ss)
            C[...,0,:,0,:] = Lambda + np.einsum('...dij,...d->...ij',
                                                AA[...,0,:,:,:],
                                                v[...,0,:])
            for n in range(1,N-1):
                C[...,n,:,n,:] = (np.einsum('...dij,...d->...ij',
                                            AA[...,n,:,:,:],
                                            v[...,n,:])
                                  + v[...,n,:,None] * np.identity(D))
            for n in range(N-1):
                C[...,n,:,n+1,:] = -np.einsum('...di,...d->...id',
                                              A[...,n,:,:],
                                              v[...,n,:])
                C[...,n+1,:,n,:] = -np.einsum('...di,...d->...di',
                                              A[...,n,:,:],
                                              v[...,n,:])
            C[...,-1,:,-1,:] = v[...,-1,:,None]*np.identity(D)
            C = np.reshape(C, plates_C+(N*D,N*D))
            Cov = np.linalg.inv(C)
            Cov = np.reshape(Cov, plates_C+(N,D,N,D))
            m0 = np.einsum('...minj,...nj->...mi', Cov, m)
            m1 = np.zeros(plates_C+(N,D,D))
            m2 = np.zeros(plates_C+(N-1,D,D))
            for n in range(N):
                m1[...,n,:,:] = Cov[...,n,:,n,:] + np.einsum('...i,...j->...ij',
                                                             m0[...,n,:],
                                                             m0[...,n,:])
            for n in range(N-1):
                m2[...,n,:,:] = Cov[...,n,:,n+1,:] + np.einsum('...i,...j->...ij',
                                                               m0[...,n,:],
                                                               m0[...,n+1,:])
            self.assertAllClose(m0, u0*np.ones(np.shape(m0)))
            self.assertAllClose(m1, u1*np.ones(np.shape(m1)))
            self.assertAllClose(m2, u2*np.ones(np.shape(m2)))

            pass

        check(2,1,1)
        check(2,3,1)
        check(2,1,3)
        check(4,3,2)

        #
        # Test mu
        #

        # Simple
        check(4,3,2,
              mu=Gaussian(np.random.randn(3),
                          random.covariance(3)))
        # Plates
        check(4,3,2,
              mu=Gaussian(np.random.randn(5,6,3),
                          random.covariance(3),
                          plates=(5,6)))
        # Plates with moments broadcasted over plates
        check(4,3,2,
              mu=Gaussian(np.random.randn(3),
                          random.covariance(3),
                          plates=(5,)))
        check(4,3,2,
              mu=Gaussian(np.random.randn(1,3),
                          random.covariance(3),
                          plates=(5,)))
        # Plates broadcasting
        check(4,3,2,
              plates=(5,),
              mu=Gaussian(np.random.randn(3),
                          random.covariance(3),
                          plates=()))
        check(4,3,2,
              plates=(5,),
              mu=Gaussian(np.random.randn(1,3),
                          random.covariance(3),
                          plates=(1,)))

        #
        # Test Lambda
        #
            
        # Simple
        check(4,3,2,
              Lambda=Wishart(10+np.random.rand(),
                             random.covariance(3)))
        # Plates
        check(4,3,2,
              Lambda=Wishart(10+np.random.rand(),
                             random.covariance(3),
                             plates=(5,6)))
        # Plates with moments broadcasted over plates
        check(4,3,2,
              Lambda=Wishart(10+np.random.rand(),
                             random.covariance(3),
                             plates=(5,)))
        check(4,3,2,
              Lambda=Wishart(10+np.random.rand(1),
                             random.covariance(3),
                             plates=(5,)))
        # Plates broadcasting
        check(4,3,2,
              plates=(5,),
              Lambda=Wishart(10+np.random.rand(),
                             random.covariance(3),
                             plates=()))
        check(4,3,2,
              plates=(5,),
              Lambda=Wishart(10+np.random.rand(),
                             random.covariance(3),
                             plates=(1,)))

        #
        # Test B
        #

        # Simple
        check(4,3,2,
              B=GaussianARD(np.random.randn(3,3,2),
                            np.random.rand(3,3,2),
                            shape=(3,2),
                            plates=(3,)))
        # Plates
        check(4,3,2,
              B=GaussianARD(np.random.randn(5,6,3,3,2),
                            np.random.rand(5,6,3,3,2),
                            shape=(3,2),
                            plates=(5,6,3)))
        # Plates with moments broadcasted over plates
        check(4,3,2,
              B=GaussianARD(np.random.randn(3,3,2),
                            np.random.rand(3,3,2),
                            shape=(3,2),
                            plates=(5,3)))
        check(4,3,2,
              B=GaussianARD(np.random.randn(1,3,3,2),
                            np.random.rand(1,3,3,2),
                            shape=(3,2),
                            plates=(5,3)))
        # Plates broadcasting
        check(4,3,2,
              plates=(5,),
              B=GaussianARD(np.random.randn(3,3,2),
                            np.random.rand(3,3,2),
                            shape=(3,2),
                            plates=(3,)))
        check(4,3,2,
              plates=(5,),
              B=GaussianARD(np.random.randn(3,3,2),
                            np.random.rand(3,3,2),
                            shape=(3,2),
                            plates=(1,3)))

        #
        # Test S
        #
            
        # Simple
        check(4,3,2,
              S=GaussianARD(np.random.randn(4-1,2),
                            np.random.rand(4-1,2),
                            shape=(2,),
                            plates=(4-1,)))
        # Plates
        check(4,3,2,
              S=GaussianARD(np.random.randn(5,6,4-1,2),
                            np.random.rand(5,6,4-1,2),
                            shape=(2,),
                            plates=(5,6,4-1,)))
        # Plates with moments broadcasted over plates
        check(4,3,2,
              S=GaussianARD(np.random.randn(4-1,2),
                            np.random.rand(4-1,2),
                            shape=(2,),
                            plates=(5,4-1,)))
        check(4,3,2,
              S=GaussianARD(np.random.randn(1,4-1,2),
                            np.random.rand(1,4-1,2),
                            shape=(2,),
                            plates=(5,4-1,)))
        # Plates broadcasting
        check(4,3,2,
              plates=(5,),
              S=GaussianARD(np.random.randn(4-1,2),
                            np.random.rand(4-1,2),
                            shape=(2,),
                            plates=(4-1,)))
        check(4,3,2,
              plates=(5,),
              S=GaussianARD(np.random.randn(4-1,2),
                            np.random.rand(4-1,2),
                            shape=(2,),
                            plates=(1,4-1,)))

        #
        # Test v
        #
        
        # Simple
        check(4,3,2,
              V=Gamma(np.random.rand(1,3),
                      np.random.rand(1,3),
                      plates=(1,3)))
        check(4,3,2,
              V=Gamma(np.random.rand(3),
                      np.random.rand(3),
                      plates=(3,)))
        # Plates
        check(4,3,2,
              V=Gamma(np.random.rand(5,6,1,3),
                      np.random.rand(5,6,1,3),
                      plates=(5,6,1,3)))
        # Plates with moments broadcasted over plates
        check(4,3,2,
              V=Gamma(np.random.rand(1,3),
                      np.random.rand(1,3),
                      plates=(5,1,3)))
        check(4,3,2,
              V=Gamma(np.random.rand(1,1,3),
                      np.random.rand(1,1,3),
                      plates=(5,1,3)))
        # Plates broadcasting
        check(4,3,2,
              plates=(5,),
              V=Gamma(np.random.rand(1,3),
                      np.random.rand(1,3),
                      plates=(1,3)))
        check(4,3,2,
              plates=(5,),
              V=Gamma(np.random.rand(1,1,3),
                      np.random.rand(1,1,3),
                      plates=(1,1,3)))

        #
        # Uncertainty in both B and S
        #
        check(4,3,2,
              B=GaussianARD(np.random.randn(3,3,2),
                            np.random.rand(3,3,2),
                            shape=(3,2),
                            plates=(3,)),
              S=GaussianARD(np.random.randn(4-1,2),
                            np.random.rand(4-1,2),
                            shape=(2,),
                            plates=(4-1,)))
                            
        pass

    def test_message_to_mu(self):
        # TODO
        pass

    def test_message_to_Lambda(self):
        # TODO
        pass

    def test_message_to_B(self):
        # TODO
        pass

    def test_message_to_S(self):
        # TODO
        pass

    def test_message_to_v(self):
        # TODO
        pass


