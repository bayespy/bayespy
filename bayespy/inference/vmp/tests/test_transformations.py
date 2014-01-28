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
Unit tests for `transformations` module.
"""

import numpy as np

from bayespy.inference.vmp.nodes.gaussian import GaussianArrayARD
from bayespy.inference.vmp.nodes.gamma import Gamma

from bayespy.utils import utils
from bayespy.utils import linalg
from bayespy.utils import random
from bayespy.utils import optimize

from ..transformations import RotateGaussianArrayARD

from bayespy.utils.utils import TestCase

class TestRotateGaussianArrayARD(TestCase):

    def test_cost_function(self):
        """
        Test the speed-up rotation of Gaussian ARD arrays.
        """

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
                X = GaussianArrayARD(mu, alpha,
                                     shape=shape,
                                     plates=plates)

                # Some initial learning and rotator constructing
                X.initialize_from_random()
                Y = GaussianArrayARD(X, 1)
                Y.observe(np.random.randn(*(Y.get_shape(0))))
                X.update()
                if alpha_plates is not None:
                    alpha.update()
                    true_cost0_alpha = alpha.lower_bound_contribution()
                    rotX = RotateGaussianArrayARD(X, alpha, 
                                                  axis=axis,
                                                  precompute=precompute)
                else:
                    rotX = RotateGaussianArrayARD(X, 
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
              mu=GaussianArrayARD(3, 1,
                                  shape=(3,1),
                                  plates=(6,1)))
        test( (2,3,4), (5,6,7), axis=-3,
              mu=GaussianArrayARD(3, 1,
                                  shape=(3,1),
                                  plates=(6,1)))
        test( (2,3,4), (5,6,7), axis=-2, alpha_plates=(5,1,7,2,1,1),
              mu=GaussianArrayARD(3, 1,
                                  shape=(3,1),
                                  plates=(6,1)))

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
              mu=GaussianArrayARD(3, 1,
                                  shape=(3,1),
                                  plates=(6,1)))
        test( (2,3,4), (5,6,7), axis=-3, plate_axis=-2,
              mu=GaussianArrayARD(3, 1,
                                  shape=(3,1),
                                  plates=(6,1)))
        test( (2,3,4), (5,6,7), axis=-2, alpha_plates=(5,1,7,2,1,1), plate_axis=-2,
              mu=GaussianArrayARD(3, 1,
                                  shape=(3,1),
                                  plates=(6,1)))

        #
        # Plate rotation with mu and alpha
        #

        # Basic, matching sizes
        test( (3,), (4,), axis=-1, plate_axis=-1,
              alpha_plates=(4,3),
              mu=GaussianArrayARD(3, 1,
                                  shape=(3,),
                                  plates=(4,)))
        # Broadcast for mu
        test( (3,), (4,), axis=-1, plate_axis=-1,
              alpha_plates=(4,3),
              mu=GaussianArrayARD(3, 1,
                                  shape=(1,),
                                  plates=(4,)))
        test( (3,), (4,), axis=-1, plate_axis=-1,
              alpha_plates=(4,3),
              mu=GaussianArrayARD(3, 1,
                                  shape=(),
                                  plates=(1,)))
        test( (3,), (4,), axis=-1, plate_axis=-1,
              alpha_plates=(4,3),
              mu=GaussianArrayARD(3, 1,
                                  shape=(3,),
                                  plates=(1,)))
        # Broadcast for alpha
        test( (3,), (4,), axis=-1, plate_axis=-1,
              alpha_plates=(4,1),
              mu=GaussianArrayARD(3, 1,
                                  shape=(3,),
                                  plates=(4,)))
        test( (3,), (4,), axis=-1, plate_axis=-1,
              alpha_plates=(3,),
              mu=GaussianArrayARD(3, 1,
                                  shape=(3,),
                                  plates=(4,)))
        # Several variable dimensions
        test( (3,4,5), (2,), axis=-2, plate_axis=-1,
              alpha_plates=(2,3,4,5),
              mu=GaussianArrayARD(3, 1,
                                  shape=(3,4,5),
                                  plates=(2,)))
        test( (3,4,5), (2,), axis=-2, plate_axis=-1,
              alpha_plates=(2,3,1,5),
              mu=GaussianArrayARD(3, 1,
                                  shape=(4,1),
                                  plates=(2,)))
        # Several plate dimensions
        test( (5,), (2,3,4), axis=-1, plate_axis=-2,
              alpha_plates=(2,3,4,5),
              mu=GaussianArrayARD(3, 1,
                                  shape=(5,),
                                  plates=(2,3,4)))
        # Several plate dimensions, rotated plate broadcasted in alpha
        test( (5,), (2,3,4), axis=-1, plate_axis=-2,
              alpha_plates=(2,1,4,5),
              mu=GaussianArrayARD(3, 1,
                                  shape=(5,),
                                  plates=(2,3,4)))
        test( (5,), (2,3,4), axis=-1, plate_axis=-2,
              alpha_plates=(4,5),
              mu=GaussianArrayARD(3, 1,
                                  shape=(5,),
                                  plates=(2,3,4)))
        # Several plate dimensions, rotated plate broadcasted in mu
        test( (5,), (2,3,4), axis=-1, plate_axis=-2,
              alpha_plates=(2,3,4,5),
              mu=GaussianArrayARD(3, 1,
                                  shape=(5,),
                                  plates=(2,1,4)))
        test( (5,), (2,3,4), axis=-1, plate_axis=-2,
              alpha_plates=(2,3,4,5),
              mu=GaussianArrayARD(3, 1,
                                  shape=(5,),
                                  plates=(4,)))
        # Several plate dimensions, rotated plate broadcasted in mu and alpha
        test( (5,), (2,3,4), axis=-1, plate_axis=-2,
              alpha_plates=(2,1,4,5),
              mu=GaussianArrayARD(3, 1,
                                  shape=(5,),
                                  plates=(2,1,4)))
        test( (5,), (2,3,4), axis=-1, plate_axis=-2,
              alpha_plates=(4,5),
              mu=GaussianArrayARD(3, 1,
                                  shape=(5,),
                                  plates=(4,)))

        # TODO: Missing values
        
        pass

    def test_cost_gradient(self):
        """
        Test gradient of the rotation cost function for Gaussian ARD arrays.
        """


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
                X = GaussianArrayARD(mu, alpha,
                                     shape=shape,
                                     plates=plates)

                # Some initial learning and rotator constructing
                X.initialize_from_random()
                Y = GaussianArrayARD(X, 1)
                Y.observe(np.random.randn(*(Y.get_shape(0))))
                X.update()
                if alpha_plates is not None:
                    alpha.update()
                    rotX = RotateGaussianArrayARD(X, alpha, 
                                                  axis=axis,
                                                  precompute=precompute)
                else:
                    rotX = RotateGaussianArrayARD(X, 
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
                                              verbose=False)
                self.assertAllClose(err, 0, 
                                    atol=1e-4,
                                    msg="Gradient incorrect for R")

                # Check gradient with respect to Q
                if plate_axis is not None:
                    err = optimize.check_gradient(f_q, 
                                                  np.ravel(Q), 
                                                  verbose=False)
                    self.assertAllClose(err, 0,
                                        atol=1e-4,
                                        msg="Gradient incorrect for Q")

            return

        #
        # Basic rotation
        #
        test((3,), (), axis=-1)
        test((3,4,5), (), axis=-1)
        test((3,4,5), (), axis=-2)
        test((3,4,5), (), axis=-3)
        test((3,4,5), (6,7), axis=-2)

        #
        # Rotation with mu
        #

        # Simple
        test((1,), (), axis=-1,
             mu=GaussianArrayARD(2, 4,
                                 shape=(1,),
                                 plates=()))
        test((3,), (), axis=-1,
             mu=GaussianArrayARD(2, 4,
                                 shape=(3,),
                                 plates=()))
        # Broadcast mu over rotated dim
        test((3,), (), axis=-1,
             mu=GaussianArrayARD(2, 4,
                                 shape=(1,),
                                 plates=()))
        test((3,), (), axis=-1,
             mu=GaussianArrayARD(2, 4,
                                 shape=(),
                                 plates=()))
        # Broadcast mu over dim when multiple dims
        test((2,3), (), axis=-1,
             mu=GaussianArrayARD(2, 4,
                                 shape=(1,3),
                                 plates=()))
        test((2,3), (), axis=-1,
             mu=GaussianArrayARD(2, 4,
                                 shape=(3,),
                                 plates=()))
        # Broadcast mu over rotated dim when multiple dims
        test((2,3), (), axis=-2,
             mu=GaussianArrayARD(2, 4,
                                 shape=(1,3),
                                 plates=()))
        test((2,3), (), axis=-2,
             mu=GaussianArrayARD(2, 4,
                                 shape=(3,),
                                 plates=()))
        # Broadcast mu over plates
        test((3,), (4,5), axis=-1,
             mu=GaussianArrayARD(2, 4,
                                 shape=(3,),
                                 plates=(4,1)))
        test((3,), (4,5), axis=-1,
             mu=GaussianArrayARD(2, 4,
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
             mu=GaussianArrayARD(2, 4,
                                 shape=(1,),
                                 plates=()))
        test((3,), (), axis=-1,
             alpha_plates=(3,),
             mu=GaussianArrayARD(2, 4,
                                 shape=(3,),
                                 plates=()))
        # Broadcast mu over rotated dim
        test((3,), (), axis=-1,
             alpha_plates=(3,),
             mu=GaussianArrayARD(2, 4,
                                 shape=(1,),
                                 plates=()))
        test((3,), (), axis=-1,
             alpha_plates=(3,),
             mu=GaussianArrayARD(2, 4,
                                 shape=(),
                                 plates=()))
        # Broadcast alpha over rotated dim
        test((3,), (), axis=-1,
             alpha_plates=(1,),
             mu=GaussianArrayARD(2, 4,
                                 shape=(3,),
                                 plates=()))
        test((3,), (), axis=-1,
             alpha_plates=(),
             mu=GaussianArrayARD(2, 4,
                                 shape=(3,),
                                 plates=()))
        # Broadcast both mu and alpha over rotated dim
        test((3,), (), axis=-1,
             alpha_plates=(1,),
             mu=GaussianArrayARD(2, 4,
                                 shape=(1,),
                                 plates=()))
        test((3,), (), axis=-1,
             alpha_plates=(),
             mu=GaussianArrayARD(2, 4,
                                 shape=(),
                                 plates=()))
        # Broadcast mu over plates
        test((3,), (4,5), axis=-1,
             alpha_plates=(4,5,3),
             mu=GaussianArrayARD(2, 4,
                                 shape=(3,),
                                 plates=(4,1)))
        test((3,), (4,5), axis=-1,
             alpha_plates=(4,5,3),
             mu=GaussianArrayARD(2, 4,
                                 shape=(3,),
                                 plates=(5,)))
        # Broadcast alpha over plates
        test((3,), (4,5), axis=-1,
             alpha_plates=(4,1,3),
             mu=GaussianArrayARD(2, 4,
                                 shape=(3,),
                                 plates=(4,5)))
        test((3,), (4,5), axis=-1,
             alpha_plates=(5,3),
             mu=GaussianArrayARD(2, 4,
                                 shape=(3,),
                                 plates=(4,5)))
        # Broadcast both mu and alpha over plates
        test((3,), (4,5), axis=-1,
             alpha_plates=(4,1,3),
             mu=GaussianArrayARD(2, 4,
                                 shape=(3,),
                                 plates=(4,1)))
        test((3,), (4,5), axis=-1,
             alpha_plates=(5,3),
             mu=GaussianArrayARD(2, 4,
                                 shape=(3,),
                                 plates=(5,)))
        # Broadcast both mu and alpha over plates but different plates
        test((3,), (4,5), axis=-1,
             alpha_plates=(4,1,3),
             mu=GaussianArrayARD(2, 4,
                                 shape=(3,),
                                 plates=(5,)))
        test((3,), (4,5), axis=-1,
             alpha_plates=(5,3),
             mu=GaussianArrayARD(2, 4,
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
             mu=GaussianArrayARD(3, 4,
                                 shape=(2,),
                                 plates=(3,)))
        # With mu broadcasted
        test((2,), (3,), axis=-1, plate_axis=-1,
             mu=GaussianArrayARD(3, 4,
                                 shape=(2,),
                                 plates=(1,)))
        test((2,), (3,), axis=-1, plate_axis=-1,
             mu=GaussianArrayARD(3, 4,
                                 shape=(2,),
                                 plates=()))
        # With mu multiple plates
        test((2,), (3,4,5), axis=-1, plate_axis=-2,
             mu=GaussianArrayARD(3, 4,
                                 shape=(2,),
                                 plates=(3,4,5)))
        # With mu multiple dims
        test((2,3,4), (5,), axis=-2, plate_axis=-1,
             mu=GaussianArrayARD(3, 4,
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
             mu=GaussianArrayARD(2, 3,
                                 shape=(1,),
                                 plates=(1,)))
        print("Test: Plate rotation with alpha and mu. Plates.")
        test((1,), (3,), axis=-1, plate_axis=-1,
             alpha_plates=(3,1),
             mu=GaussianArrayARD(2, 3,
                                 shape=(1,),
                                 plates=(3,)))
        print("Test: Plate rotation with alpha and mu. Dims.")
        test((3,), (1,), axis=-1, plate_axis=-1,
             alpha_plates=(1,3),
             mu=GaussianArrayARD(2, 3,
                                 shape=(3,),
                                 plates=(1,)))
        print("Test: Plate rotation with alpha and mu. Broadcast over rotated "
              "plates.")
        test((1,), (3,), axis=-1, plate_axis=-1,
             alpha_plates=(1,1),
             mu=GaussianArrayARD(2, 3,
                                 shape=(1,),
                                 plates=(1,)))
        test((1,), (3,), axis=-1, plate_axis=-1,
             alpha_plates=(1,),
             mu=GaussianArrayARD(2, 3,
                                 shape=(1,),
                                 plates=()))
        print("Test: Plate rotation with alpha and mu. Broadcast over dims.")
        test((3,), (1,), axis=-1, plate_axis=-1,
             alpha_plates=(1,1),
             mu=GaussianArrayARD(2, 3,
                                 shape=(1,),
                                 plates=(1,)))
        test((3,), (1,), axis=-1, plate_axis=-1,
             alpha_plates=(),
             mu=GaussianArrayARD(2, 3,
                                 shape=(),
                                 plates=(1,)))
        print("Test: Plate rotation with alpha and mu. Multiple dims.")
        test((2,3,4,5), (6,), axis=-2, plate_axis=-1,
             alpha_plates=(6,2,3,4,5),
             mu=GaussianArrayARD(2, 3,
                                 shape=(2,3,4,5),
                                 plates=(6,)))
        print("Test: Plate rotation with alpha and mu. Multiple plates.")
        test((2,), (3,4,5), axis=-1, plate_axis=-1,
             alpha_plates=(3,4,5,2),
             mu=GaussianArrayARD(2, 3,
                                 shape=(2,),
                                 plates=(3,4,5,)))
        test((2,), (3,4,5), axis=-1, plate_axis=-2,
             alpha_plates=(3,4,5,2),
             mu=GaussianArrayARD(2, 3,
                                 shape=(2,),
                                 plates=(3,4,5,)))
        test((2,), (3,4,5), axis=-1, plate_axis=-3,
             alpha_plates=(3,4,5,2),
             mu=GaussianArrayARD(2, 3,
                                 shape=(2,),
                                 plates=(3,4,5,)))

        # TODO: With missing values
        
        pass
