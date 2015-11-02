################################################################################
# Copyright (C) 2014 Jaakko Luttinen
#
# This file is licensed under the MIT License.
################################################################################


"""
Unit tests for `transformations` module.
"""

import numpy as np

from bayespy.inference.vmp.nodes.gaussian import GaussianARD
from bayespy.inference.vmp.nodes.gaussian import Gaussian
from bayespy.inference.vmp.nodes.gamma import Gamma
from bayespy.inference.vmp.nodes.wishart import Wishart
from bayespy.inference.vmp.nodes.dot import SumMultiply
from bayespy.inference.vmp.nodes.gaussian_markov_chain import GaussianMarkovChain

from bayespy.utils import linalg
from bayespy.utils import random
from bayespy.utils import optimize

from ..transformations import RotateGaussianARD
from ..transformations import RotateGaussianMarkovChain
from ..transformations import RotateVaryingMarkovChain

from bayespy.utils.misc import TestCase

class TestRotateGaussianARD(TestCase):

    def test_cost_function(self):
        """
        Test the speed-up rotation of Gaussian ARD arrays.
        """

        # Use seed for deterministic testing
        np.random.seed(42)

        def test(shape, plates, 
                 axis=-1, 
                 alpha_plates=None, 
                 plate_axis=None,
                 mu=3):

            if plate_axis is not None:
                precomputes = [False, True]
            else:
                precomputes = [False]
                
            for precompute in precomputes:
                # Construct the model
                D = shape[axis]
                if alpha_plates is not None:
                    alpha = Gamma(2, 2,
                                  plates=alpha_plates)
                    alpha.initialize_from_random()
                else:
                    alpha = 2
                X = GaussianARD(mu, alpha,
                                shape=shape,
                                plates=plates)

                # Some initial learning and rotator constructing
                X.initialize_from_random()
                Y = GaussianARD(X, 1)
                Y.observe(np.random.randn(*(Y.get_shape(0))))
                X.update()
                if alpha_plates is not None:
                    alpha.update()
                    true_cost0_alpha = alpha.lower_bound_contribution()
                    rotX = RotateGaussianARD(X, alpha, 
                                             axis=axis,
                                             precompute=precompute)
                else:
                    rotX = RotateGaussianARD(X, 
                                             axis=axis,
                                             precompute=precompute)
                true_cost0_X = X.lower_bound_contribution()

                # Rotation matrices
                I = np.identity(D)
                R = np.random.randn(D, D)
                if plate_axis is not None:
                    C = plates[plate_axis]
                    Q = np.random.randn(C, C)
                    Ic = np.identity(C)
                else:
                    Q = None
                    Ic = None

                # Compute bound terms
                rotX.setup(plate_axis=plate_axis)
                rot_cost0 = rotX.get_bound_terms(I, Q=Ic)
                rot_cost1 = rotX.get_bound_terms(R, Q=Q)
                self.assertAllClose(sum(rot_cost0.values()),
                                    rotX.bound(I, Q=Ic)[0],
                                    msg="Bound terms and total bound differ")
                self.assertAllClose(sum(rot_cost1.values()),
                                    rotX.bound(R, Q=Q)[0],
                                    msg="Bound terms and total bound differ")
                # Perform rotation
                rotX.rotate(R, Q=Q)
                # Check bound terms
                true_cost1_X = X.lower_bound_contribution()
                self.assertAllClose(true_cost1_X - true_cost0_X,
                                    rot_cost1[X] - rot_cost0[X],
                                    msg="Incorrect rotation cost for X")
                if alpha_plates is not None:
                    true_cost1_alpha = alpha.lower_bound_contribution()
                    self.assertAllClose(true_cost1_alpha - true_cost0_alpha,
                                        rot_cost1[alpha] - rot_cost0[alpha],
                                        msg="Incorrect rotation cost for alpha")
            return

        # Rotating a vector (zero mu)
        test( (3,), (),    axis=-1,                       mu=0)
        test( (3,), (),    axis=-1, alpha_plates=(1,),    mu=0)
        test( (3,), (),    axis=-1, alpha_plates=(3,),    mu=0)
        test( (3,), (2,4), axis=-1,                       mu=0)
        test( (3,), (2,4), axis=-1, alpha_plates=(1,),    mu=0)
        test( (3,), (2,4), axis=-1, alpha_plates=(3,),    mu=0)
        test( (3,), (2,4), axis=-1, alpha_plates=(2,4,3), mu=0)
        test( (3,), (2,4), axis=-1, alpha_plates=(1,4,3), mu=0)

        # Rotating a vector (full mu)
        test( (3,), (),    axis=-1,                       mu=3*np.ones((3,)))
        test( (3,), (),    axis=-1, alpha_plates=(),      mu=3*np.ones((3,)))
        test( (3,), (),    axis=-1, alpha_plates=(1,),    mu=3*np.ones((3,)))
        test( (3,), (),    axis=-1, alpha_plates=(3,),    mu=3*np.ones((3,)))
        test( (3,), (2,4), axis=-1,                       mu=3*np.ones((2,4,3)))
        test( (3,), (2,4), axis=-1, alpha_plates=(1,),    mu=3*np.ones((2,4,3)))
        test( (3,), (2,4), axis=-1, alpha_plates=(3,),    mu=3*np.ones((2,4,3)))
        test( (3,), (2,4), axis=-1, alpha_plates=(2,4,3), mu=3*np.ones((2,4,3)))
        test( (3,), (2,4), axis=-1, alpha_plates=(1,4,3), mu=3*np.ones((2,4,3)))

        # Rotating a vector (broadcast mu)
        test( (3,), (),      axis=-1,                         mu=3*np.ones((1,)))
        test( (3,), (),      axis=-1, alpha_plates=(1,),      mu=3*np.ones((1,)))
        test( (3,), (),      axis=-1, alpha_plates=(3,),      mu=3*np.ones((1,)))
        test( (3,), (2,4,5), axis=-1,                         mu=3*np.ones((4,1,1)))
        test( (3,), (2,4,5), axis=-1, alpha_plates=(1,),      mu=3*np.ones((4,1,1)))
        test( (3,), (2,4,5), axis=-1, alpha_plates=(3,),      mu=3*np.ones((4,1,1)))
        test( (3,), (2,4,5), axis=-1, alpha_plates=(2,4,5,3), mu=3*np.ones((4,1,1))) #!!
        test( (3,), (2,4,5), axis=-1, alpha_plates=(1,4,5,3), mu=3*np.ones((4,1,1)))

        # Rotating an array
        test( (2,3,4), (),    axis=-1)
        test( (2,3,4), (),    axis=-2)
        test( (2,3,4), (),    axis=-3)
        test( (2,3,4), (5,6), axis=-1)
        test( (2,3,4), (5,6), axis=-2)
        test( (2,3,4), (5,6), axis=-3)
        test( (2,3,4), (5,6), axis=-1, alpha_plates=(3,1))
        test( (2,3,4), (5,6), axis=-2, alpha_plates=(3,1))
        test( (2,3,4), (5,6), axis=-3, alpha_plates=(3,1))
        test( (2,3), (4,5,6), axis=-1, alpha_plates=(5,1,2,1))
        test( (2,3), (4,5,6), axis=-2, alpha_plates=(5,1,2,1))

        # Test mu array broadcasting
        test( (2,3,4), (5,6,7), axis=-2,
              mu=GaussianARD(3, 1,
                             shape=(3,1),
                             plates=(6,1,1)))
        test( (2,3,4), (5,6,7), axis=-3,
              mu=GaussianARD(3, 1,
                             shape=(3,1),
                             plates=(6,1,1)))
        test( (2,3,4), (5,6,7), axis=-2, alpha_plates=(5,1,7,2,1,1),
              mu=GaussianARD(3, 1,
                             shape=(3,1),
                             plates=(6,1,1)))

        # Plate rotation
        test( (3,), (5,), axis=-1, plate_axis=-1)
        test( (3,), (4,5,6), axis=-1, plate_axis=-2)
        test( (2,3), (4,5,6), axis=-2, plate_axis=-2)
        test( (2,3,4), (5,6,7), axis=-2, plate_axis=-3)

        # Plate rotation with alpha
        test( (2,3,4), (5,6,7), axis=-2, alpha_plates=(3,1), plate_axis=-2)
        test( (2,3,4), (5,6,7), axis=-2, alpha_plates=(6,1,2,1,4), plate_axis=-3)

        # Plate rotation with mu
        test( (2,3,4), (5,6,7), axis=-2, plate_axis=-2,
              mu=GaussianARD(3, 1,
                             shape=(3,1),
                             plates=(6,1,1)))
        test( (2,3,4), (5,6,7), axis=-3, plate_axis=-2,
              mu=GaussianARD(3, 1,
                             shape=(3,1),
                             plates=(6,1,1)))
        test( (2,3,4), (5,6,7), axis=-2, alpha_plates=(5,1,7,2,1,1), plate_axis=-2,
              mu=GaussianARD(3, 1,
                             shape=(3,1),
                             plates=(6,1,1)))

        #
        # Plate rotation with mu and alpha
        #

        # Basic, matching sizes
        test( (3,), (4,), axis=-1, plate_axis=-1,
              alpha_plates=(4,3),
              mu=GaussianARD(3, 1,
                             shape=(3,),
                             plates=(4,)))
        # Broadcast for mu
        test( (3,), (4,), axis=-1, plate_axis=-1,
              alpha_plates=(4,3),
              mu=GaussianARD(3, 1,
                             shape=(1,),
                             plates=(4,)))
        test( (3,), (4,), axis=-1, plate_axis=-1,
              alpha_plates=(4,3),
              mu=GaussianARD(3, 1,
                             shape=(),
                             plates=(1,)))
        test( (3,), (4,), axis=-1, plate_axis=-1,
              alpha_plates=(4,3),
              mu=GaussianARD(3, 1,
                             shape=(3,),
                             plates=(1,)))
        # Broadcast for alpha
        test( (3,), (4,), axis=-1, plate_axis=-1,
              alpha_plates=(4,1),
              mu=GaussianARD(3, 1,
                             shape=(3,),
                             plates=(4,)))
        test( (3,), (4,), axis=-1, plate_axis=-1,
              alpha_plates=(3,),
              mu=GaussianARD(3, 1,
                             shape=(3,),
                             plates=(4,)))
        # Several variable dimensions
        test( (3,4,5), (2,), axis=-2, plate_axis=-1,
              alpha_plates=(2,3,4,5),
              mu=GaussianARD(3, 1,
                             shape=(3,4,5),
                             plates=(2,)))
        test( (3,4,5), (2,), axis=-2, plate_axis=-1,
              alpha_plates=(2,3,1,5),
              mu=GaussianARD(3, 1,
                             shape=(4,1),
                             plates=(2,1)))
        # Several plate dimensions
        test( (5,), (2,3,4), axis=-1, plate_axis=-2,
              alpha_plates=(2,3,4,5),
              mu=GaussianARD(3, 1,
                             shape=(5,),
                             plates=(2,3,4)))
        # Several plate dimensions, rotated plate broadcasted in alpha
        test( (5,), (2,3,4), axis=-1, plate_axis=-2,
              alpha_plates=(2,1,4,5),
              mu=GaussianARD(3, 1,
                             shape=(5,),
                             plates=(2,3,4)))
        test( (5,), (2,3,4), axis=-1, plate_axis=-2,
              alpha_plates=(4,5),
              mu=GaussianARD(3, 1,
                             shape=(5,),
                             plates=(2,3,4)))
        # Several plate dimensions, rotated plate broadcasted in mu
        test( (5,), (2,3,4), axis=-1, plate_axis=-2,
              alpha_plates=(2,3,4,5),
              mu=GaussianARD(3, 1,
                             shape=(5,),
                             plates=(2,1,4)))
        test( (5,), (2,3,4), axis=-1, plate_axis=-2,
              alpha_plates=(2,3,4,5),
              mu=GaussianARD(3, 1,
                             shape=(5,),
                             plates=(4,)))
        # Several plate dimensions, rotated plate broadcasted in mu and alpha
        test( (5,), (2,3,4), axis=-1, plate_axis=-2,
              alpha_plates=(2,1,4,5),
              mu=GaussianARD(3, 1,
                             shape=(5,),
                             plates=(2,1,4)))
        test( (5,), (2,3,4), axis=-1, plate_axis=-2,
              alpha_plates=(4,5),
              mu=GaussianARD(3, 1,
                             shape=(5,),
                             plates=(4,)))

        # TODO: Missing values
        
        pass

    def test_cost_gradient(self):
        """
        Test gradient of the rotation cost function for Gaussian ARD arrays.
        """

        # Use seed for deterministic testing
        np.random.seed(42)

        def test(shape, plates, 
                 axis=-1, 
                 alpha_plates=None, 
                 plate_axis=None,
                 mu=3):
            
            if plate_axis is not None:
                precomputes = [False, True]
            else:
                precomputes = [False]
                
            for precompute in precomputes:
                # Construct the model
                D = shape[axis]
                if alpha_plates is not None:
                    alpha = Gamma(3, 5,
                                  plates=alpha_plates)
                    alpha.initialize_from_random()
                else:
                    alpha = 2
                X = GaussianARD(mu, alpha,
                                shape=shape,
                                plates=plates)

                # Some initial learning and rotator constructing
                X.initialize_from_random()
                Y = GaussianARD(X, 1)
                Y.observe(np.random.randn(*(Y.get_shape(0))))
                X.update()
                if alpha_plates is not None:
                    alpha.update()
                    rotX = RotateGaussianARD(X, alpha, 
                                             axis=axis,
                                             precompute=precompute)
                else:
                    rotX = RotateGaussianARD(X, 
                                             axis=axis,
                                             precompute=precompute)
                try:
                    mu.update()
                except:
                    pass

                # Rotation matrices
                R = np.random.randn(D, D)
                if plate_axis is not None:
                    C = plates[plate_axis]
                    Q = np.random.randn(C, C)
                else:
                    Q = None

                # Compute bound terms
                rotX.setup(plate_axis=plate_axis)

                if plate_axis is None:
                    def f_r(r):
                        (b, dr) = rotX.bound(np.reshape(r, np.shape(R)))
                        return (b, np.ravel(dr))
                else:
                    def f_r(r):
                        (b, dr, dq) = rotX.bound(np.reshape(r, np.shape(R)),
                                             Q=Q)
                        return (b, np.ravel(dr))

                    def f_q(q):
                        (b, dr, dq) = rotX.bound(R,
                                             Q=np.reshape(q, np.shape(Q)))
                        return (b, np.ravel(dq))

                # Check gradient with respect to R
                err = optimize.check_gradient(f_r, 
                                              np.ravel(R), 
                                              verbose=False)[1]
                self.assertAllClose(err, 0, 
                                    atol=1e-4,
                                    msg="Gradient incorrect for R")

                # Check gradient with respect to Q
                if plate_axis is not None:
                    err = optimize.check_gradient(f_q, 
                                                  np.ravel(Q), 
                                                  verbose=False)[1]
                    self.assertAllClose(err, 0,
                                        atol=1e-4,
                                        msg="Gradient incorrect for Q")

            return

        #
        # Basic rotation
        #
        test((3,), (), axis=-1)
        test((2,3,4), (), axis=-1)
        test((2,3,4), (), axis=-2)
        test((2,3,4), (), axis=-3)
        test((2,3,4), (5,6), axis=-2)

        #
        # Rotation with mu
        #

        # Simple
        test((1,), (), axis=-1,
             mu=GaussianARD(2, 4,
                            shape=(1,),
                            plates=()))
        test((3,), (), axis=-1,
             mu=GaussianARD(2, 4,
                            shape=(3,),
                            plates=()))
        # Broadcast mu over rotated dim
        test((3,), (), axis=-1,
             mu=GaussianARD(2, 4,
                            shape=(1,),
                            plates=()))
        test((3,), (), axis=-1,
             mu=GaussianARD(2, 4,
                            shape=(),
                            plates=()))
        # Broadcast mu over dim when multiple dims
        test((2,3), (), axis=-1,
             mu=GaussianARD(2, 4,
                            shape=(1,3),
                            plates=()))
        test((2,3), (), axis=-1,
             mu=GaussianARD(2, 4,
                            shape=(3,),
                            plates=()))
        # Broadcast mu over rotated dim when multiple dims
        test((2,3), (), axis=-2,
             mu=GaussianARD(2, 4,
                            shape=(1,3),
                            plates=()))
        test((2,3), (), axis=-2,
             mu=GaussianARD(2, 4,
                            shape=(3,),
                            plates=()))
        # Broadcast mu over plates
        test((3,), (4,5), axis=-1,
             mu=GaussianARD(2, 4,
                            shape=(3,),
                            plates=(4,1)))
        test((3,), (4,5), axis=-1,
             mu=GaussianARD(2, 4,
                            shape=(3,),
                            plates=(5,)))

        #
        # Rotation with alpha
        #

        # Simple
        test((1,), (), axis=-1,
             alpha_plates=())
        test((3,), (), axis=-1,
             alpha_plates=(3,))
        # Broadcast alpha over rotated dim
        test((3,), (), axis=-1,
             alpha_plates=())
        test((3,), (), axis=-1,
             alpha_plates=(1,))
        # Broadcast alpha over dim when multiple dims
        test((2,3), (), axis=-1,
             alpha_plates=(1,3))
        test((2,3), (), axis=-1,
             alpha_plates=(3,))
        # Broadcast alpha over rotated dim when multiple dims
        test((2,3), (), axis=-2,
             alpha_plates=(1,3))
        test((2,3), (), axis=-2,
             alpha_plates=(3,))
        # Broadcast alpha over plates
        test((3,), (4,5), axis=-1,
             alpha_plates=(4,1,3))
        test((3,), (4,5), axis=-1,
             alpha_plates=(5,3))

        #
        # Rotation with alpha and mu
        #

        # Simple
        test((1,), (), axis=-1,
             alpha_plates=(1,),
             mu=GaussianARD(2, 4,
                            shape=(1,),
                            plates=()))
        test((3,), (), axis=-1,
             alpha_plates=(3,),
             mu=GaussianARD(2, 4,
                            shape=(3,),
                            plates=()))
        # Broadcast mu over rotated dim
        test((3,), (), axis=-1,
             alpha_plates=(3,),
             mu=GaussianARD(2, 4,
                            shape=(1,),
                            plates=()))
        test((3,), (), axis=-1,
             alpha_plates=(3,),
             mu=GaussianARD(2, 4,
                            shape=(),
                            plates=()))
        # Broadcast alpha over rotated dim
        test((3,), (), axis=-1,
             alpha_plates=(1,),
             mu=GaussianARD(2, 4,
                            shape=(3,),
                            plates=()))
        test((3,), (), axis=-1,
             alpha_plates=(),
             mu=GaussianARD(2, 4,
                            shape=(3,),
                            plates=()))
        # Broadcast both mu and alpha over rotated dim
        test((3,), (), axis=-1,
             alpha_plates=(1,),
             mu=GaussianARD(2, 4,
                            shape=(1,),
                            plates=()))
        test((3,), (), axis=-1,
             alpha_plates=(),
             mu=GaussianARD(2, 4,
                            shape=(),
                            plates=()))
        # Broadcast mu over plates
        test((3,), (4,5), axis=-1,
             alpha_plates=(4,5,3),
             mu=GaussianARD(2, 4,
                            shape=(3,),
                            plates=(4,1)))
        test((3,), (4,5), axis=-1,
             alpha_plates=(4,5,3),
             mu=GaussianARD(2, 4,
                            shape=(3,),
                            plates=(5,)))
        # Broadcast alpha over plates
        test((3,), (4,5), axis=-1,
             alpha_plates=(4,1,3),
             mu=GaussianARD(2, 4,
                            shape=(3,),
                            plates=(4,5)))
        test((3,), (4,5), axis=-1,
             alpha_plates=(5,3),
             mu=GaussianARD(2, 4,
                            shape=(3,),
                            plates=(4,5)))
        # Broadcast both mu and alpha over plates
        test((3,), (4,5), axis=-1,
             alpha_plates=(4,1,3),
             mu=GaussianARD(2, 4,
                            shape=(3,),
                            plates=(4,1)))
        test((3,), (4,5), axis=-1,
             alpha_plates=(5,3),
             mu=GaussianARD(2, 4,
                            shape=(3,),
                            plates=(5,)))
        # Broadcast both mu and alpha over plates but different plates
        test((3,), (4,5), axis=-1,
             alpha_plates=(4,1,3),
             mu=GaussianARD(2, 4,
                            shape=(3,),
                            plates=(5,)))
        test((3,), (4,5), axis=-1,
             alpha_plates=(5,3),
             mu=GaussianARD(2, 4,
                            shape=(3,),
                            plates=(4,1)))

        #
        # Rotation with missing values
        #

        # TODO

        #
        # Plate rotation
        #

        # Simple
        test((2,), (3,),    axis=-1, plate_axis=-1)
        test((2,), (3,4,5), axis=-1, plate_axis=-1)
        test((2,), (3,4,5), axis=-1, plate_axis=-2)
        test((2,), (3,4,5), axis=-1, plate_axis=-3)
        test((2,3), (4,5),  axis=-2, plate_axis=-2)

        # With mu
        test((2,), (3,), axis=-1, plate_axis=-1,
             mu=GaussianARD(3, 4,
                            shape=(2,),
                            plates=(3,)))
        # With mu broadcasted
        test((2,), (3,), axis=-1, plate_axis=-1,
             mu=GaussianARD(3, 4,
                            shape=(2,),
                            plates=(1,)))
        test((2,), (3,), axis=-1, plate_axis=-1,
             mu=GaussianARD(3, 4,
                            shape=(2,),
                            plates=()))
        # With mu multiple plates
        test((2,), (3,4,5), axis=-1, plate_axis=-2,
             mu=GaussianARD(3, 4,
                            shape=(2,),
                            plates=(3,4,5)))
        # With mu multiple dims
        test((2,3,4), (5,), axis=-2, plate_axis=-1,
             mu=GaussianARD(3, 4,
                            shape=(2,3,4),
                            plates=(5,)))

        #
        # With alpha
        #
        print("Test: Plate rotation with alpha. Scalars.")
        test((1,), (1,), axis=-1, plate_axis=-1,
             alpha_plates=(1,1),
             mu=0)
        print("Test: Plate rotation with alpha. Plates.")
        test((1,), (3,), axis=-1, plate_axis=-1,
             alpha_plates=(3,1),
             mu=0)
        print("Test: Plate rotation with alpha. Dims.")
        test((3,), (1,), axis=-1, plate_axis=-1,
             alpha_plates=(1,3),
             mu=0)
        print("Test: Plate rotation with alpha. Broadcast alpha over rotated plates.")
        test((1,), (3,), axis=-1, plate_axis=-1,
             alpha_plates=(1,1),
             mu=0)
        test((1,), (3,), axis=-1, plate_axis=-1,
             alpha_plates=(1,),
             mu=0)
        print("Test: Plate rotation with alpha. Broadcast alpha over dims.")
        test((3,), (1,), axis=-1, plate_axis=-1,
             alpha_plates=(1,1),
             mu=0)
        test((3,), (1,), axis=-1, plate_axis=-1,
             alpha_plates=(),
             mu=0)
        print("Test: Plate rotation with alpha. Multiple dims.")
        test((2,3,4,5), (6,), axis=-2, plate_axis=-1,
             alpha_plates=(6,2,3,4,5),
             mu=0)
        print("Test: Plate rotation with alpha. Multiple plates.")
        test((2,), (3,4,5), axis=-1, plate_axis=-1,
             alpha_plates=(3,4,5,2),
             mu=0)
        test((2,), (3,4,5), axis=-1, plate_axis=-2,
             alpha_plates=(3,4,5,2),
             mu=0)
        test((2,), (3,4,5), axis=-1, plate_axis=-3,
             alpha_plates=(3,4,5,2),
             mu=0)

        #
        # With alpha and mu
        #
        print("Test: Plate rotation with alpha and mu. Scalars.")
        test((1,), (1,), axis=-1, plate_axis=-1,
             alpha_plates=(1,1),
             mu=GaussianARD(2, 3,
                            shape=(1,),
                            plates=(1,)))
        print("Test: Plate rotation with alpha and mu. Plates.")
        test((1,), (3,), axis=-1, plate_axis=-1,
             alpha_plates=(3,1),
             mu=GaussianARD(2, 3,
                            shape=(1,),
                            plates=(3,)))
        print("Test: Plate rotation with alpha and mu. Dims.")
        test((3,), (1,), axis=-1, plate_axis=-1,
             alpha_plates=(1,3),
             mu=GaussianARD(2, 3,
                            shape=(3,),
                            plates=(1,)))
        print("Test: Plate rotation with alpha and mu. Broadcast over rotated "
              "plates.")
        test((1,), (3,), axis=-1, plate_axis=-1,
             alpha_plates=(1,1),
             mu=GaussianARD(2, 3,
                            shape=(1,),
                            plates=(1,)))
        test((1,), (3,), axis=-1, plate_axis=-1,
             alpha_plates=(1,),
             mu=GaussianARD(2, 3,
                            shape=(1,),
                            plates=()))
        print("Test: Plate rotation with alpha and mu. Broadcast over dims.")
        test((3,), (1,), axis=-1, plate_axis=-1,
             alpha_plates=(1,1),
             mu=GaussianARD(2, 3,
                            shape=(1,),
                            plates=(1,)))
        test((3,), (1,), axis=-1, plate_axis=-1,
             alpha_plates=(),
             mu=GaussianARD(2, 3,
                            shape=(),
                            plates=(1,)))
        print("Test: Plate rotation with alpha and mu. Multiple dims.")
        test((2,3,4,5), (6,), axis=-2, plate_axis=-1,
             alpha_plates=(6,2,3,4,5),
             mu=GaussianARD(2, 3,
                            shape=(2,3,4,5),
                            plates=(6,)))
        print("Test: Plate rotation with alpha and mu. Multiple plates.")
        test((2,), (3,4,5), axis=-1, plate_axis=-1,
             alpha_plates=(3,4,5,2),
             mu=GaussianARD(2, 3,
                            shape=(2,),
                            plates=(3,4,5,)))
        test((2,), (3,4,5), axis=-1, plate_axis=-2,
             alpha_plates=(3,4,5,2),
             mu=GaussianARD(2, 3,
                            shape=(2,),
                            plates=(3,4,5,)))
        test((2,), (3,4,5), axis=-1, plate_axis=-3,
             alpha_plates=(3,4,5,2),
             mu=GaussianARD(2, 3,
                            shape=(2,),
                            plates=(3,4,5,)))

        # TODO: With missing values
        
        pass

class TestRotateGaussianMarkovChain(TestCase):

    def test_cost_function(self):
        """
        Test the cost function of the speed-up rotation for Markov chain
        """

        np.random.seed(42)

        def check(D, N, mu=None, Lambda=None, rho=None, A=None):
            if mu is None:
                mu = np.zeros(D)
            if Lambda is None:
                Lambda = np.identity(D)
            if rho is None:
                rho = np.ones(D)
            if A is None:
                A = GaussianARD(3, 5,
                                shape=(D,),
                                plates=(D,))
                
            V = np.identity(D) + np.ones((D,D))

            # Construct model
            X = GaussianMarkovChain(mu,
                                    Lambda,
                                    A,
                                    rho,
                                    n=N+1,
                                    initialize=False)
            Y = Gaussian(X,
                         V,
                         initialize=False)

            # Posterior estimation
            Y.observe(np.random.randn(*(Y.get_shape(0))))
            X.update()
            try:
                A.update()
            except:
                pass
            try:
                mu.update()
            except:
                pass
            try:
                Lambda.update()
            except:
                pass
            try:
                rho.update()
            except:
                pass

            # Construct rotator
            rotA = RotateGaussianARD(A, axis=-1)
            rotX = RotateGaussianMarkovChain(X, rotA)

            # Rotation
            true_cost0 = X.lower_bound_contribution()
            rotX.setup()
            I = np.identity(D)
            R = np.random.randn(D, D)
            rot_cost0 = rotX.get_bound_terms(I)
            rot_cost1 = rotX.get_bound_terms(R)
            self.assertAllClose(sum(rot_cost0.values()),
                                rotX.bound(I)[0],
                                    msg="Bound terms and total bound differ")
            self.assertAllClose(sum(rot_cost1.values()),
                                rotX.bound(R)[0],
                                msg="Bound terms and total bound differ")
            rotX.rotate(R)
            true_cost1 = X.lower_bound_contribution()
            self.assertAllClose(true_cost1 - true_cost0,
                                rot_cost1[X] - rot_cost0[X],
                                msg="Incorrect rotation cost for X")
            
            return

        self._run_checks(check)

        pass

    def test_cost_gradient(self):
        """
        Test the gradient of the speed-up rotation for Markov chain
        """

        # Use seed for deterministic testing
        np.random.seed(42)

        def check(D, N, mu=None, Lambda=None, rho=None, A=None):
            if mu is None:
                mu = np.zeros(D)
            if Lambda is None:
                Lambda = np.identity(D)
            if rho is None:
                rho = np.ones(D)
            if A is None:
                A = GaussianARD(3, 5,
                                shape=(D,),
                                plates=(D,))
                
            V = np.identity(D) + np.ones((D,D))

            # Construct model
            X = GaussianMarkovChain(mu,
                                    Lambda,
                                    A,
                                    rho,
                                    n=N+1,
                                    initialize=False)
            Y = Gaussian(X,
                         V,
                         initialize=False)

            # Posterior estimation
            Y.observe(np.random.randn(*(Y.get_shape(0))))
            X.update()
            try:
                A.update()
            except:
                pass
            try:
                mu.update()
            except:
                pass
            try:
                Lambda.update()
            except:
                pass
            try:
                rho.update()
            except:
                pass

            # Construct rotator
            rotA = RotateGaussianARD(A, axis=-1)
            rotX = RotateGaussianMarkovChain(X, rotA)
            rotX.setup()

            # Check gradient with respect to R
            R = np.random.randn(D, D)
            def cost(r):
                (b, dr) = rotX.bound(np.reshape(r, np.shape(R)))
                return (b, np.ravel(dr))

            err = optimize.check_gradient(cost, 
                                          np.ravel(R), 
                                          verbose=False)[1]
            self.assertAllClose(err, 0, 
                                atol=1e-5,
                                msg="Gradient incorrect")
            
            return

        self._run_checks(check)

        pass

    def _run_checks(self, check):
        
        # Basic test
        check(2, 3)

        # Test mu
        check(2, 3,
              mu=GaussianARD(2, 4,
                             shape=(2,),
                             plates=()))
        check(2, 3,
              mu=GaussianARD(2, 4,
                             shape=(2,),
                             plates=(5,)))

        # Test Lambda
        check(2, 3,
              Lambda=Wishart(3, random.covariance(2)))
        check(2, 3,
              Lambda=Wishart(3, random.covariance(2),
                             plates=(5,)))

        # Test A
        check(2, 3,
              A=GaussianARD(2, 4,
                            shape=(2,),
                            plates=(2,)))
        check(2, 3,
              A=GaussianARD(2, 4,
                            shape=(2,),
                            plates=(3,2)))
        check(2, 3,
              A=GaussianARD(2, 4,
                            shape=(2,),
                            plates=(5,3,2)))

        # Test Lambda and mu
        check(2, 3,
              mu=GaussianARD(2, 4,
                             shape=(2,),
                             plates=()),
              Lambda=Wishart(2, random.covariance(2)))
        check(2, 3,
              mu=GaussianARD(2, 4,
                             shape=(2,),
                             plates=(5,)),
              Lambda=Wishart(2, random.covariance(2),
                             plates=(5,)))

        # Test mu and A
        check(2, 3,
              mu=GaussianARD(2, 4,
                             shape=(2,),
                             plates=()),
              A=GaussianARD(2, 4,
                            shape=(2,),
                            plates=(2,)))
        check(2, 3,
              mu=GaussianARD(2, 4,
                             shape=(2,),
                             plates=(5,)),
              A=GaussianARD(2, 4,
                            shape=(2,),
                            plates=(5,1,2,)))

        # Test Lambda and A
        check(2, 3,
              Lambda=Wishart(2, random.covariance(2)),
              A=GaussianARD(2, 4,
                            shape=(2,),
                            plates=(2,)))
        check(2, 3,
              Lambda=Wishart(2, random.covariance(2),
                             plates=(5,)),
              A=GaussianARD(2, 4,
                            shape=(2,),
                            plates=(5,1,2,)))

        # Test mu, Lambda and A
        check(2, 3,
              mu=GaussianARD(2, 4,
                             shape=(2,),
                             plates=()),
              Lambda=Wishart(2, random.covariance(2)),
              A=GaussianARD(2, 4,
                            shape=(2,),
                            plates=(2,)))
        check(2, 3,
              mu=GaussianARD(2, 4,
                             shape=(2,),
                             plates=(5,)),
              Lambda=Wishart(2, random.covariance(2),
                             plates=(5,)),
              A=GaussianARD(2, 4,
                            shape=(2,),
                            plates=(5,1,2,)))

        pass


    
class TestRotateVaryingMarkovChain(TestCase):

    def test_cost_function(self):
        """
        Test the speed-up rotation of Markov chain with time-varying dynamics
        """

        # Use seed for deterministic testing
        np.random.seed(42)

        def check(D, N, K,
                  mu=None,
                  Lambda=None,
                  rho=None):

            if mu is None:
                mu = np.zeros(D)
            if Lambda is None:
                Lambda = np.identity(D)
            if rho is None:
                rho = np.ones(D)

            V = np.identity(D) + np.ones((D,D))

            # Construct model
            B = GaussianARD(3, 5,
                            shape=(D,K),
                            plates=(1,D))
            S = GaussianARD(2, 4,
                            shape=(K,),
                            plates=(N,1))
            A = SumMultiply('dk,k->d', B, S)
            X = GaussianMarkovChain(mu,
                                    Lambda,
                                    A,
                                    rho,
                                    n=N+1,
                                    initialize=False)
            Y = Gaussian(X,
                         V,
                         initialize=False)

            # Posterior estimation
            Y.observe(np.random.randn(N+1,D))
            X.update()
            B.update()
            S.update()
            try:
                mu.update()
            except:
                pass
            try:
                Lambda.update()
            except:
                pass
            try:
                rho.update()
            except:
                pass

            # Construct rotator
            rotB = RotateGaussianARD(B, axis=-2)
            rotX = RotateVaryingMarkovChain(X, B, S, rotB)

            # Rotation
            true_cost0 = X.lower_bound_contribution()
            rotX.setup()
            I = np.identity(D)
            R = np.random.randn(D, D)
            rot_cost0 = rotX.get_bound_terms(I)
            rot_cost1 = rotX.get_bound_terms(R)
            self.assertAllClose(sum(rot_cost0.values()),
                                rotX.bound(I)[0],
                                    msg="Bound terms and total bound differ")
            self.assertAllClose(sum(rot_cost1.values()),
                                rotX.bound(R)[0],
                                msg="Bound terms and total bound differ")
            rotX.rotate(R)
            true_cost1 = X.lower_bound_contribution()
            self.assertAllClose(true_cost1 - true_cost0,
                                rot_cost1[X] - rot_cost0[X],
                                msg="Incorrect rotation cost for X")
            
            return

        self._run_checks(check)
        
        pass


    def test_cost_gradient(self):
        """
        Test the gradient of the rotation for MC with time-varying dynamics
        """

        # Use seed for deterministic testing
        np.random.seed(42)

        def check(D, N, K,
                  mu=None,
                  Lambda=None,
                  rho=None):

            if mu is None:
                mu = np.zeros(D)
            if Lambda is None:
                Lambda = np.identity(D)
            if rho is None:
                rho = np.ones(D)

            V = np.identity(D) + np.ones((D,D))

            # Construct model
            B = GaussianARD(3, 5,
                            shape=(D,K),
                            plates=(1,D))
            S = GaussianARD(2, 4,
                            shape=(K,),
                            plates=(N,1))
            A = SumMultiply('dk,k->d', B, S)
            X = GaussianMarkovChain(mu,
                                    Lambda,
                                    A,
                                    rho,
                                    n=N+1,
                                    initialize=False)
            Y = Gaussian(X,
                         V,
                         initialize=False)

            # Posterior estimation
            Y.observe(np.random.randn(N+1,D))
            X.update()
            B.update()
            S.update()
            try:
                mu.update()
            except:
                pass
            try:
                Lambda.update()
            except:
                pass
            try:
                rho.update()
            except:
                pass

            # Construct rotator
            rotB = RotateGaussianARD(B, axis=-2)
            rotX = RotateVaryingMarkovChain(X, B, S, rotB)
            rotX.setup()

            # Check gradient with respect to R
            R = np.random.randn(D, D)
            def cost(r):
                (b, dr) = rotX.bound(np.reshape(r, np.shape(R)))
                return (b, np.ravel(dr))

            err = optimize.check_gradient(cost, 
                                          np.ravel(R), 
                                          verbose=False)[1]
            self.assertAllClose(err, 0, 
                                atol=1e-6,
                                msg="Gradient incorrect")
            
            return

        self._run_checks(check)
        
        pass

    def _run_checks(self, check):
        
        # Basic test
        check(1, 1, 1)
        check(2, 1, 1)
        check(1, 2, 1)
        check(1, 1, 2)
        check(3, 4, 2)

        # Test mu
        check(2, 3, 4,
              mu=GaussianARD(2, 4,
                             shape=(2,),
                             plates=()))

        # Test Lambda
        check(2, 3, 4,
              Lambda=Wishart(3, random.covariance(2)))

        # Test Lambda and mu
        check(2, 3, 4,
              mu=GaussianARD(2, 4,
                             shape=(2,),
                             plates=()),
              Lambda=Wishart(2, random.covariance(2)))

        # TODO: Test plates

        pass

